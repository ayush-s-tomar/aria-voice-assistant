"""
Text-to-Speech service.
- Default: gTTS (free, Google TTS, no API key needed)
- Optional: ElevenLabs (premium voice, set ELEVENLABS_API_KEY to enable)

Cluster B additions:
- In-memory cache: identical (text + lang) pairs skip regeneration entirely
- Language passthrough: detected STT language fed into gTTS accent + ElevenLabs multilingual
- ElevenLabs upgraded to eleven_multilingual_v2 (supports 29 languages)
"""

import hashlib
import io
import os

from gtts import gTTS

# ── In-memory TTS cache ───────────────────────────────────────────────────────
# Key: (text_hash, lang) → Value: MP3 bytes
# Lives for the lifetime of the server process — survives across requests,
# resets on redeploy (intentional: no stale audio from old model versions).
_cache: dict[tuple[str, str], bytes] = {}
MAX_CACHE_ENTRIES = int(os.getenv("TTS_CACHE_SIZE", "200"))


def _cache_key(text: str, lang: str) -> tuple[str, str]:
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return (text_hash, lang)


def _get_cached(text: str, lang: str) -> bytes | None:
    return _cache.get(_cache_key(text, lang))


def _set_cached(text: str, lang: str, audio: bytes) -> None:
    # Simple eviction: drop oldest half when limit hit
    if len(_cache) >= MAX_CACHE_ENTRIES:
        keys = list(_cache.keys())
        for k in keys[: MAX_CACHE_ENTRIES // 2]:
            del _cache[k]
    _cache[_cache_key(text, lang)] = audio


# ── gTTS ──────────────────────────────────────────────────────────────────────

# Map Whisper language codes → gTTS-compatible codes where they differ
_GTTS_LANG_MAP: dict[str, str] = {
    "zh":    "zh-CN",
    "zh-tw": "zh-TW",
    "jw":    "jv",     # Javanese
}

def _normalize_lang_gtts(lang: str) -> str:
    lang = lang.lower().split("-")[0] if "-" not in _GTTS_LANG_MAP.get(lang.lower(), "") else lang.lower()
    return _GTTS_LANG_MAP.get(lang, lang)


def _tts_gtts(text: str, lang: str = "en") -> bytes:
    """Free TTS using Google Text-to-Speech, with language matching."""
    gtts_lang = _normalize_lang_gtts(lang)
    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
    except ValueError:
        # gTTS doesn't support the language — fall back to English
        print(f"[TTS] gTTS doesn't support lang='{gtts_lang}', falling back to en")
        tts = gTTS(text=text, lang="en", slow=False)

    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ── ElevenLabs ────────────────────────────────────────────────────────────────

def _tts_elevenlabs(text: str, lang: str = "en") -> bytes:
    """
    Premium TTS using ElevenLabs.
    Uses eleven_multilingual_v2 (29 languages) instead of the old monolingual model.
    """
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
            "model_id": "eleven_multilingual_v2",   # upgraded: supports 29 languages
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            "language_code": lang,                  # hint to the model
        },
    )
    response.raise_for_status()
    return response.content


# ── Public interface ──────────────────────────────────────────────────────────

def text_to_speech(text: str, lang: str = "en") -> bytes:
    """
    Returns MP3 audio bytes for the given text.
    - lang: ISO 639-1 language code from Whisper (e.g. 'en', 'hi', 'es')
    - Uses ElevenLabs if ELEVENLABS_API_KEY is set, else gTTS
    - Caches result: identical (text, lang) pairs never hit the API twice
    """
    cached = _get_cached(text, lang)
    if cached:
        print(f"[TTS] Cache hit (lang={lang})")
        return cached

    if os.getenv("ELEVENLABS_API_KEY"):
        print(f"[TTS] ElevenLabs | lang={lang}")
        audio = _tts_elevenlabs(text, lang)
    else:
        print(f"[TTS] gTTS | lang={lang}")
        audio = _tts_gtts(text, lang)

    _set_cached(text, lang, audio)
    return audio