"""
Text-to-Speech service.
- Default: gTTS (free, Google TTS, no API key needed)
- Optional: ElevenLabs (premium voice, set ELEVENLABS_API_KEY to enable)
"""
import os
import io
from gtts import gTTS


def _tts_gtts(text: str) -> bytes:
    """Free TTS using Google Text-to-Speech."""
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _tts_elevenlabs(text: str) -> bytes:
    """Premium TTS using ElevenLabs (better voice quality)."""
    import requests

    api_key = os.environ["ELEVENLABS_API_KEY"]
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel

    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
    )
    response.raise_for_status()
    return response.content


def text_to_speech(text: str) -> bytes:
    """
    Returns MP3 audio bytes for the given text.
    Uses ElevenLabs if ELEVENLABS_API_KEY is set, else falls back to gTTS.
    """
    if os.getenv("ELEVENLABS_API_KEY"):
        print("[TTS] Using ElevenLabs")
        return _tts_elevenlabs(text)
    else:
        print("[TTS] Using gTTS (free)")
        return _tts_gtts(text)
