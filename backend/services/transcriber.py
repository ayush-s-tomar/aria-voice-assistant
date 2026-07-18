"""
Speech-to-text via Groq Whisper large-v3.
Raises TranscriptionError with a structured error_code so callers in
main.py can branch on it (e.g. return 413 for AUDIO_TOO_LARGE) instead
of string-matching free-text messages.
"""

import os
from groq import Groq
from services.errors import ErrorCode

_client = None

MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
MIN_AUDIO_BYTES = 1024

SUPPORTED_LANGUAGE_LOCKS = {
    "en", "hi", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar",
}


class TranscriptionError(Exception):
    def __init__(self, message: str, error_code: str = ErrorCode.TRANSCRIPTION_FAILED):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _validate_file(file_path: str) -> int:
    if not os.path.exists(file_path):
        raise TranscriptionError("Audio file not found", ErrorCode.AUDIO_INVALID_FORMAT)

    size = os.path.getsize(file_path)
    if size < MIN_AUDIO_BYTES:
        raise TranscriptionError(
            "Audio clip is empty or too short to transcribe", ErrorCode.AUDIO_EMPTY
        )
    if size > MAX_AUDIO_BYTES:
        raise TranscriptionError(
            f"Audio file exceeds the {MAX_AUDIO_BYTES // (1024 * 1024)}MB limit",
            ErrorCode.AUDIO_TOO_LARGE,
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
        raise TranscriptionError(
            "Speech-to-text service failed", ErrorCode.TRANSCRIPTION_FAILED
        ) from e

    text = result.text.strip()
    detected = getattr(result, "language", None) or lock_lang or "en"
    print(f"[Groq Whisper] lang={detected} locked={lock_lang or 'no'} | {text[:80]!r}")
    return text, detected