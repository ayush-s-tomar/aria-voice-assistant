import hashlib
import io
import os

from gtts import gTTS

_cache = {}
MAX_CACHE_ENTRIES = int(os.getenv("TTS_CACHE_SIZE", "200"))


def _cache_key(text, lang, engine):
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return (text_hash, lang, engine)


def _get_cached(text, lang, engine):
    return _cache.get(_cache_key(text, lang, engine))


def _set_cached(text, lang, engine, audio):
    if len(_cache) >= MAX_CACHE_ENTRIES:
        keys = list(_cache.keys())
        for k in keys[: MAX_CACHE_ENTRIES // 2]:
            del _cache[k]
    _cache[_cache_key(text, lang, engine)] = audio


_GTTS_LANG_MAP = {"zh": "zh-CN", "zh-tw": "zh-TW", "jw": "jv"}


def _normalize_lang_gtts(lang):
    lang = lang.lower()
    return _GTTS_LANG_MAP.get(lang, lang)


def _tts_gtts(text, lang="en"):
    gtts_lang = _normalize_lang_gtts(lang)
    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
    except ValueError:
        print(f"[TTS] gTTS doesn't support lang='{gtts_lang}', falling back to en")
        tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _tts_elevenlabs(text, lang="en"):
    import requests
    api_key = os.environ["ELEVENLABS_API_KEY"]
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            "language_code": lang,
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.content


def text_to_speech(text, lang="en", use_elevenlabs=False):
    engine = "elevenlabs" if (use_elevenlabs and os.getenv("ELEVENLABS_API_KEY")) else "gtts"
    cached = _get_cached(text, lang, engine)
    if cached:
        print(f"[TTS] Cache hit (lang={lang}, engine={engine})")
        return cached
    if engine == "elevenlabs":
        print(f"[TTS] ElevenLabs | lang={lang}")
        audio = _tts_elevenlabs(text, lang)
    else:
        print(f"[TTS] gTTS | lang={lang}")
        audio = _tts_gtts(text, lang)
    _set_cached(text, lang, engine, audio)
    return audio
