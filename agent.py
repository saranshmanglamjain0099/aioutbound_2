import asyncio
import json
import logging
import os
import ssl
import certifi
from typing import Optional

from dotenv import load_dotenv

# Patch SSL before any network import
_orig_ssl = ssl.create_default_context
def _certifi_ssl(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
    if not kwargs.get("cafile") and not kwargs.get("capath") and not kwargs.get("cadata"):
        kwargs["cafile"] = certifi.where()
    return _orig_ssl(purpose, **kwargs)
ssl.create_default_context = _certifi_ssl

from livekit import agents, api, rtc
from livekit.agents import Agent, AgentSession, RoomInputOptions
try:
    from livekit.agents import RoomOptions as _RoomOptions
    _HAS_ROOM_OPTIONS = True
except ImportError:
    _HAS_ROOM_OPTIONS = False
from livekit.plugins import noise_cancellation, silero

from db import init_db, log_error, get_enabled_tools
from prompts import build_prompt
from tools import AppointmentTools

load_dotenv(".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outbound-agent")

SIP_DOMAIN = os.getenv("VOBIZ_SIP_DOMAIN", "")


async def _log(level: str, msg: str, detail: str = "") -> None:
    if level == "info":      logger.info(msg)
    elif level == "warning": logger.warning(msg)
    else:                    logger.error(msg)
    try:
        await log_error("agent", msg, detail, level)
    except Exception:
        pass


def load_db_settings_to_env() -> None:
    """Load Supabase settings table into os.environ before worker starts."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        return
    try:
        from supabase import create_client
        client = create_client(url, key)
        result = client.table("settings").select("key, value").execute()
        for row in (result.data or []):
            if row.get("value"):
                os.environ[row["key"]] = row["value"]
    except Exception as exc:
        logger.warning("Could not load settings from Supabase: %s", exc)


# ── Import Google plugin paths ───────────────────────────────────────────────
_google_realtime = None
_google_beta_realtime = None
_google_llm = None
_google_tts = None
_google_stt = None
_deepgram_stt = None
_openai_llm = None

try:
    from livekit.plugins import google as _gp
    try:
        _google_realtime = _gp.realtime.RealtimeModel
        logger.info("Loaded google.realtime.RealtimeModel (stable path)")
    except AttributeError:
        pass
    try:
        _google_beta_realtime = _gp.beta.realtime.RealtimeModel
        logger.info("Loaded google.beta.realtime.RealtimeModel (beta path)")
    except AttributeError:
        pass
    try:
        _google_llm = _gp.LLM
        _google_tts = _gp.TTS
        _google_stt = _gp.STT
    except AttributeError:
        pass
except ImportError:
    logger.warning("livekit-plugins-google not installed")

try:
    from livekit.plugins import deepgram as _dg
    _deepgram_stt = _dg.STT
except ImportError:
    pass

try:
    from livekit.plugins import openai as _oa
    _openai_llm = _oa.LLM
except ImportError:
    pass

_sarvam_tts = None
try:
    from livekit.plugins import sarvam as _sa
    import livekit.plugins.sarvam.tts as _sa_tts
    # Monkey-patch Sarvam's strict validation so users can pass custom voice IDs
    _sa_tts.validate_model_speaker_compatibility = lambda m, s: True
    _sarvam_tts = _sa.TTS
except ImportError:
    pass


# ── Session factory ──────────────────────────────────────────────────────────

def _build_session(tools: list, system_prompt: str) -> AgentSession:
    """
    Build AgentSession with Gemini Live or pipeline fallback.

    CRITICAL SILENCE-PREVENTION CONFIG — all 3 required:
    1. SessionResumptionConfig(transparent=True) → auto-reconnects after timeout
    2. ContextWindowCompressionConfig → sliding window prevents token limit freeze
    3. RealtimeInputConfig(END_SENSITIVITY_LOW) → less aggressive VAD, 2s silence threshold

    ⚠️ EndSensitivity MUST use full string form: END_SENSITIVITY_LOW (not .LOW — AttributeError!)
    """
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
    gemini_voice = os.getenv("GEMINI_TTS_VOICE", "Aoede")
    use_realtime = os.getenv("USE_GEMINI_REALTIME", "true").lower() != "false"

    # Sarvam override
    is_sarvam = gemini_voice.startswith("sarvam:")
    custom_sarvam_id = None
    if is_sarvam:
        use_realtime = False
        parts = gemini_voice.split(":")
        if len(parts) >= 3 and parts[1] == "custom":
            custom_sarvam_id = ":".join(parts[2:])
            gemini_voice = custom_sarvam_id
        else:
            gemini_voice = gemini_voice.replace("sarvam:", "")

    RealtimeClass = _google_realtime or (_google_beta_realtime if use_realtime else None)

    if use_realtime and RealtimeClass is not None:
        logger.info("SESSION MODE: Gemini Live realtime (%s, voice=%s)", gemini_model, gemini_voice)
        try:
            from google.genai import types as _gt
            _realtime_input_cfg = _gt.RealtimeInputConfig(
                automatic_activity_detection=_gt.AutomaticActivityDetection(
                    end_of_speech_sensitivity=_gt.EndSensitivity.END_SENSITIVITY_LOW,
                    silence_duration_ms=2000,
                    prefix_padding_ms=200,
                ),
            )
            _session_resumption_cfg = _gt.SessionResumptionConfig(transparent=True)
            _ctx_compression_cfg = _gt.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=_gt.SlidingWindow(target_tokens=12800),
            )
            logger.info("Silence-prevention config applied (VAD LOW, transparent resumption, context compression)")
        except Exception as _cfg_err:
            logger.warning("Could not build silence-prevention config: %s", _cfg_err)
            _realtime_input_cfg = None
            _session_resumption_cfg = None
            _ctx_compression_cfg = None

        realtime_kwargs: dict = dict(model=gemini_model, voice=gemini_voice, instructions=system_prompt)
        if _realtime_input_cfg is not None:
            realtime_kwargs["realtime_input_config"]      = _realtime_input_cfg
            realtime_kwargs["session_resumption"]         = _session_resumption_cfg
            realtime_kwargs["context_window_compression"] = _ctx_compression_cfg

        return AgentSession(llm=RealtimeClass(**realtime_kwargs), tools=tools)

    # Support for OpenRouter models via OpenAI plugin
    if "gemini" not in gemini_model.lower() and _openai_llm is not None:
        logger.info(f"Using OpenRouter/OpenAI for model: {gemini_model}")
        llm = _openai_llm(model=gemini_model, base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY", ""))
    elif _google_llm is not None:
        llm = _google_llm(model=gemini_model)
    else:
        raise RuntimeError("No Google or OpenAI backend. Run: pip install 'livekit-plugins-google>=1.0'")

    logger.info("SESSION MODE: pipeline (STT + LLM + TTS)")
    if _deepgram_stt and os.getenv("DEEPGRAM_API_KEY", "") and os.getenv("DEEPGRAM_API_KEY") != "your_deepgram_key":
        logger.info("Using Deepgram STT")
        stt = _deepgram_stt(model="nova-3", language="hi") # 'hi' natively supports Hinglish code-switching best in Deepgram
    elif _google_stt:
        logger.info("Using Google STT")
        stt = _google_stt()
    else:
        stt = None
    if is_sarvam and _sarvam_tts:
        # Determine language code (Sarvam requires target_language_code)
        lang = "hi-IN"
        if gemini_voice.lower() in ("amelia", "sophia"):
            lang = "en-IN"
        
        sarvam_model = "bulbul:v3"
        # If it's a custom ID or an older voice, v3 might reject it serverside. We'll try v3.
        # But if it's explicitly 'meera', 'amartya', etc., they only exist in v2.
        if gemini_voice.lower() in ("meera", "amartya", "anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"):
            sarvam_model = "bulbul:v2"

        sarvam_key = os.getenv("SARVAM_API_KEY", "")
        if not sarvam_key:
            logger.error("⚠️  SARVAM_API_KEY not set — Sarvam TTS will fail!")
        logger.info(f"Using Sarvam TTS: {gemini_voice} ({lang}) [Model: {sarvam_model}]")
        tts = _sarvam_tts(speaker=gemini_voice, target_language_code=lang, model=sarvam_model, api_key=sarvam_key)
    else:
        tts = _google_tts(voice=gemini_voice) if _google_tts else None
    # ── FIX A: Tune VAD for SIP telephony noise ─────────────────────────────
    # SIP lines have constant background hiss/static. Default threshold (0.5)
    # causes VAD to think the human is always talking → AI stays silent → call drops.
    # Raising activation_threshold to 0.65 ignores low-energy SIP noise.
    # Raising min_silence_duration to 0.8s avoids premature end-of-speech on brief pauses.
    # Using sample_rate=8000 matches SIP telephony codec (G.711 μ-law).
    vad = silero.VAD.load(
        activation_threshold=0.65,     # Higher = ignore SIP static (default 0.5)
        min_silence_duration=0.80,     # Wait longer before end-of-speech (default 0.55)
        min_speech_duration=0.10,      # Require 100ms of speech to trigger (default 0.05)
        prefix_padding_duration=0.30,  # Capture start of speech cleanly
        sample_rate=8000,              # Match SIP telephony sample rate
    )
    logger.info("VAD loaded with SIP-tuned params: threshold=0.65, silence=0.8s, rate=8kHz")

    # ── FIX C: Validate Deepgram STT connectivity ───────────────────────────
    # If Deepgram silently fails or the API key is invalid, the LLM never
    # receives transcription → AI never responds → call drops.
    if stt and _deepgram_stt and isinstance(stt, _deepgram_stt):
        dg_key = os.getenv("DEEPGRAM_API_KEY", "")
        if not dg_key or dg_key == "your_deepgram_key":
            logger.error("⚠️  DEEPGRAM_API_KEY is missing or placeholder — STT will fail silently!")
            asyncio.get_event_loop().create_task(
                _log("error", "Deepgram API key missing/placeholder", "STT will not transcribe any audio")
            )
        else:
            logger.info("Deepgram STT configured (key=%s...)", dg_key[:8])

    return AgentSession(stt=stt, llm=llm, tts=tts, vad=vad, tools=tools)


class OutboundAssistant(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)


async def entrypoint(ctx: agents.JobContext) -> None:
    """
    Main entrypoint. Called per job. Reads metadata JSON from ctx.job.metadata.

    DIAL-FIRST PATTERN — CRITICAL:
    Start Gemini Live ONLY after create_sip_participant(wait_until_answered=True) completes.
    If you start the session during ring time (~20-30s), the Gemini idle timeout fires
    and the session dies silently before the call is even answered.

    NO close_on_disconnect — SIP legs have brief audio dropouts that look like disconnects.
    Instead, watch participant_disconnected event for the specific SIP identity.
    """
    await _log("info", f"Job started — room: {ctx.room.name}")

    phone_number: Optional[str] = None
    lead_name = "there"
    business_name = "our company"
    service_type = "our service"
    custom_prompt: Optional[str] = None
    voice_override: Optional[str] = None
    model_override: Optional[str] = None
    tools_override: Optional[str] = None

    if ctx.job.metadata:
        try:
            data = json.loads(ctx.job.metadata)
            phone_number   = data.get("phone_number")
            lead_name      = data.get("lead_name", lead_name)
            business_name  = data.get("business_name", business_name)
            service_type   = data.get("service_type", service_type)
            custom_prompt  = data.get("system_prompt")
            voice_override = data.get("voice_override")
            model_override = data.get("model_override")
            tools_override = data.get("tools_override")
        except (json.JSONDecodeError, AttributeError):
            await _log("warning", "Invalid JSON in job metadata")

    await _log("info", f"Call job received — phone={phone_number} lead={lead_name} biz={business_name}")

    system_prompt = build_prompt(lead_name=lead_name, business_name=business_name,
                                  service_type=service_type, custom_prompt=custom_prompt)
    tool_ctx = AppointmentTools(ctx, phone_number, lead_name)

    if voice_override:
        os.environ["GEMINI_TTS_VOICE"] = voice_override
    if model_override:
        os.environ["GEMINI_MODEL"] = model_override

    if tools_override:
        try:
            enabled_tools = json.loads(tools_override)
        except Exception:
            enabled_tools = await get_enabled_tools()
    else:
        enabled_tools = await get_enabled_tools()

    # ── Connect ──────────────────────────────────────────────────────────────
    await ctx.connect()
    await _log("info", f"Connected to LiveKit room: {ctx.room.name}")

    # ── Dial — MUST come before session.start() ──────────────────────────────
    if phone_number:
        trunk_id = os.getenv("OUTBOUND_TRUNK_ID")
        if not trunk_id:
            await _log("error", "OUTBOUND_TRUNK_ID not set — cannot place outbound call")
            ctx.shutdown()
            return
        await _log("info", f"Dialing {phone_number} via SIP trunk {trunk_id}")

        # ── FIX D: SIP retry loop with exponential backoff ───────────────────
        # Vobiz may rate-limit concurrent outbound dials (486 Busy / 403 Forbidden).
        # Retry up to 3 times with increasing delay instead of failing immediately.
        MAX_SIP_RETRIES = 3
        sip_connected = False
        for attempt in range(1, MAX_SIP_RETRIES + 1):
            try:
                await _log("info", f"SIP dial attempt {attempt}/{MAX_SIP_RETRIES} for {phone_number}")
                await ctx.api.sip.create_sip_participant(
                    api.CreateSIPParticipantRequest(
                        room_name=ctx.room.name,
                        sip_trunk_id=trunk_id,
                        sip_call_to=phone_number,
                        participant_identity=f"sip_{phone_number}",
                        wait_until_answered=True,
                    )
                )
                sip_connected = True
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = any(code in exc_str for code in ["486", "403", "busy", "forbidden", "rate", "limit", "throttl"])
                if is_rate_limit and attempt < MAX_SIP_RETRIES:
                    backoff = 2 ** attempt  # 2s, 4s
                    await _log("warning", f"SIP dial attempt {attempt} rate-limited: {exc}. Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                else:
                    await _log("error", f"SIP dial FAILED for {phone_number} (attempt {attempt}): {exc}")
                    ctx.shutdown()
                    return

        if not sip_connected:
            await _log("error", f"SIP dial EXHAUSTED all {MAX_SIP_RETRIES} retries for {phone_number}")
            ctx.shutdown()
            return

        await _log("info", f"Call ANSWERED — {phone_number} picked up, starting AI session now")

    # ── Build and start Gemini Live ──────────────────────────────────────────
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
    await _log("info", f"Building AI session — model={gemini_model}")
    active_tools = tool_ctx.build_tool_list(enabled_tools)
    await _log("info", f"Tools loaded: {[t.__name__ for t in active_tools]}")
    session = _build_session(tools=active_tools, system_prompt=system_prompt)

    # Use RoomOptions if available (non-deprecated), else fall back
    # NEVER use close_on_disconnect=True with SIP — drops on any audio blip
    if _HAS_ROOM_OPTIONS:
        from livekit.agents import RoomOptions as _RO
        _session_kwargs = dict(
            room=ctx.room,
            agent=OutboundAssistant(instructions=system_prompt),
            room_options=_RO(input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony())),
        )
    else:
        _session_kwargs = dict(
            room=ctx.room,
            agent=OutboundAssistant(instructions=system_prompt),
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony()),
        )

    await session.start(**_session_kwargs)
    await _log("info", "Agent session started — AI ready, generating greeting")

    # ── Optional S3 recording ────────────────────────────────────────────────
    if phone_number:
        _aws_key    = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID", "")
        _aws_secret = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        _aws_bucket = os.getenv("S3_BUCKET") or os.getenv("AWS_BUCKET_NAME", "")
        _s3_endpoint = os.getenv("S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT", "")
        _s3_region  = os.getenv("S3_REGION") or os.getenv("AWS_REGION", "ap-northeast-1")
        if _aws_key and _aws_secret and _aws_bucket:
            try:
                _recording_path = f"recordings/{ctx.room.name}.ogg"
                _egress_req = api.RoomCompositeEgressRequest(
                    room_name=ctx.room.name, audio_only=True,
                    file_outputs=[api.EncodedFileOutput(
                        file_type=api.EncodedFileType.OGG, filepath=_recording_path,
                        s3=api.S3Upload(access_key=_aws_key, secret=_aws_secret,
                                        bucket=_aws_bucket, region=_s3_region, endpoint=_s3_endpoint),
                    )],
                )
                _egress = await ctx.api.egress.start_room_composite_egress(_egress_req)
                _s3_ep = _s3_endpoint.rstrip("/")
                tool_ctx.recording_url = (f"{_s3_ep}/{_aws_bucket}/{_recording_path}"
                                           if _s3_ep else f"s3://{_aws_bucket}/{_recording_path}")
                await _log("info", f"Recording started: egress={_egress.egress_id}")
            except Exception as _exc:
                await _log("warning", f"Recording start failed (non-fatal): {_exc}")

    # ── Audio stabilization delay ───────────────────────────────────────────
    # SIP audio tracks take a moment to stabilize after answer.
    # Without this delay, the first generate_reply may produce audio before
    # the SIP participant's track is ready → audio is silently discarded.
    if phone_number:
        await _log("info", "Waiting 1.5s for SIP audio track to stabilize...")
        await asyncio.sleep(1.5)

    # ── Greeting ─────────────────────────────────────────────────────────────
    # ALWAYS force a greeting — never rely on Gemini to speak autonomously.
    # Gemini native-audio mode *can* greet from system prompt, but often
    # doesn't on SIP calls (audio track not ready, session init delay, etc).
    # Forcing generate_reply guarantees the human hears something immediately.
    _active_model = os.getenv("GEMINI_MODEL", "")
    _use_realtime = os.getenv("USE_GEMINI_REALTIME", "true").lower() != "false"
    _voice = os.getenv("GEMINI_TTS_VOICE", "")
    _is_pipeline = not _use_realtime or _voice.startswith("sarvam:")

    greeting = (
        f"The call just connected. Greet the lead warmly and ask if you're speaking with {lead_name}. Be natural and conversational."
        if phone_number else "Greet the caller warmly."
    )

    try:
        await session.generate_reply(instructions=greeting)
        await _log("info", "Initial greeting generated successfully")
    except Exception as _gr_exc:
        await _log("warning", f"generate_reply failed: {_gr_exc}")
        # Fallback: try a simpler greeting
        try:
            await session.generate_reply(instructions="Say: Hello! How are you today?")
        except Exception:
            await _log("error", "Both greeting attempts failed — AI will be silent")

    # ── Dead-air watchdog ────────────────────────────────────────────────────
    # If the greeting failed silently (no audio actually sent), the human
    # hears dead air. After 8 seconds, force a backup greeting.
    if phone_number:
        async def _dead_air_watchdog():
            await asyncio.sleep(8)
            try:
                await session.generate_reply(
                    instructions=f"The lead may not have heard you. Say again: Hi {lead_name}, this is a call from {os.getenv('GEMINI_TTS_VOICE', 'your assistant')}. Can you hear me?"
                )
                await _log("info", "Dead-air watchdog triggered backup greeting")
            except Exception:
                pass
        asyncio.create_task(_dead_air_watchdog())

    # ── Keep session alive until SIP participant actually leaves ─────────────
    # Without this block, the entrypoint returns and the process spins down.
    # We watch participant_disconnected for the specific SIP identity.
    if phone_number:
        _sip_identity = f"sip_{phone_number}"
        _disconnect_event = asyncio.Event()

        def _on_participant_disconnected(participant: rtc.RemoteParticipant):
            if participant.identity == _sip_identity:
                _disconnect_event.set()
        def _on_disconnected():
            _disconnect_event.set()

        ctx.room.on("participant_disconnected", _on_participant_disconnected)
        ctx.room.on("disconnected", _on_disconnected)

        try:
            await asyncio.wait_for(_disconnect_event.wait(), timeout=3600)
        except asyncio.TimeoutError:
            await _log("warning", "Call reached 1-hour safety timeout — shutting down")

        await _log("info", f"SIP participant disconnected — ending session for {phone_number}")
        await session.aclose()
    else:
        _done = asyncio.Event()
        ctx.room.on("disconnected", lambda: _done.set())
        try:
            await asyncio.wait_for(_done.wait(), timeout=3600)
        except asyncio.TimeoutError:
            pass


if __name__ == "__main__":
    init_db()
    load_db_settings_to_env()
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="outbound-caller")
    )
