"""
Speech-to-text via Groq Whisper large-v3.
(Streamlit port — same logic as the FastAPI version, minus the
FastAPI-specific ErrorCode plumbing. Raises plain TranscriptionError.)
"""

import os
from groq import Groq

_client = None

MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
MIN_AUDIO_BYTES = 1024

SUPPORTED_LANGUAGE_LOCKS = {
    "en", "hi", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar",
}


class TranscriptionError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _validate_file(file_path: str) -> int:
    if not os.path.exists(file_path):
        raise TranscriptionError("Audio file not found")

    size = os.path.getsize(file_path)
    if size < MIN_AUDIO_BYTES:
        raise TranscriptionError("Audio clip is empty or too short to transcribe")
    if size > MAX_AUDIO_BYTES:
        raise TranscriptionError(
            f"Audio file exceeds the {MAX_AUDIO_BYTES // (1024 * 1024)}MB limit"
        )
    return size


def transcribe_audio(file_path: str, language: str | None = None) -> tuple[str, str]:
    """
    Transcribe audio via Groq Whisper large-v3.
    Returns (text, detected_language).
    """
    size = _validate_file(file_path)
    lock_lang = language if language in SUPPORTED_LANGUAGE_LOCKS else None

    client = _get_client()
    try:
        with open(file_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language=lock_lang,
                response_format="verbose_json",
            )
    except Exception as e:
        print(f"[Groq Whisper] API error on {size}-byte file: {e}")
        raise TranscriptionError("Speech-to-text service failed") from e

    text = result.text.strip()
    detected = getattr(result, "language", None) or lock_lang or "en"
    print(f"[Groq Whisper] lang={detected} locked={lock_lang or 'no'} | {text[:80]!r}")
    return text, detected