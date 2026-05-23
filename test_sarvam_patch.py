import asyncio
import os
from dotenv import load_dotenv
from livekit.plugins.sarvam import TTS
import livekit.plugins.sarvam.tts as stts

load_dotenv(".env")

# Monkey-patch to bypass validation
stts.validate_model_speaker_compatibility = lambda m, s: True

async def main():
    try:
        api_key = os.getenv("SARVAM_API_KEY", "")
        if not api_key:
            print("❌ SARVAM_API_KEY not set in .env — Sarvam TTS will fail!")
            return
        # User wants to use a custom voice ID or old name like 'meera'
        tts = TTS(speaker="meera", target_language_code="hi-IN", model="bulbul:v2", api_key=api_key)
        print(f"✅ TTS initialized successfully with monkey-patch (key={api_key[:8]}...)")
    except Exception as e:
        print(f"❌ Error initializing: {e}")

asyncio.run(main())
