import asyncio
import os
from dotenv import load_dotenv
from livekit.plugins.sarvam import TTS

load_dotenv(".env")

async def main():
    try:
        api_key = os.getenv("SARVAM_API_KEY", "")
        if not api_key:
            print("❌ SARVAM_API_KEY not set in .env — Sarvam TTS will fail!")
            return
        tts = TTS(speaker="ritu", target_language_code="hi-IN", model="bulbul:v3", api_key=api_key)
        print(f"✅ TTS initialized successfully (key={api_key[:8]}...)")
    except Exception as e:
        print(f"❌ Error initializing: {e}")

asyncio.run(main())
