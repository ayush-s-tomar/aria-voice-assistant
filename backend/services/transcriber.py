"""
Speech-to-text via Groq Whisper large-v3.

Cluster F additions:
- Validates file existence/size BEFORE hitting the Groq API — fails fast
  on empty or oversized clips instead of paying for a doomed API call.
- Optional language locking: pass `language` to skip Whisper's
  auto-detect pass (faster, avoids mis-detection on short/noisy clips).
- Wraps Groq/network failures into a typed TranscriptionError carrying
  an error_code, so main.py can turn it into a structured response
  instead of a raw 500.
"""

import os
from groq import Groq
from services.errors import ErrorCode

_client = None

# Groq's audio endpoint caps uploads around 25MB on the free tier.
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
# Anything under ~1KB is essentially guaranteed to be a blank/corrupt clip.
MIN_AUDIO_BYTES = 1024

# Locking to one of these skips Whisper's language-detection pass.
# Anything outside this set is treated as "no lock" (auto-detect).
SUPPORTED_LANGUAGE_LOCKS = {
    "en", "hi", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar",
}


class TranscriptionError(Exception):
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(message)


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _validate_file(file_path: str) -> int:
    if not os.path.exists(file_path):
        raise TranscriptionError(ErrorCode.AUDIO_INVALID_FORMAT, "Audio file not found")

    size = os.path.getsize(file_path)
    if size < MIN_AUDIO_BYTES:
        raise TranscriptionError(
            ErrorCode.AUDIO_EMPTY,
            "Audio clip is empty or too short to transcribe"
        )
    if size > MAX_AUDIO_BYTES:
        raise TranscriptionError(
            ErrorCode.AUDIO_TOO_LARGE,
            f"Audio file exceeds the {MAX_AUDIO_BYTES // (1024 * 1024)}MB limit"
        )
    return size


def transcribe_audio(file_path: str, language: str | None = None) -> tuple[str, str]:
    """
    Transcribe audio via Groq Whisper large-v3.
    Returns (text, detected_language).

    language: optional ISO 639-1 code to lock recognition to (e.g. 'en', 'hi').
              Falls back to auto-detect if not in SUPPORTED_LANGUAGE_LOCKS.

    Raises TranscriptionError (with .error_code) on invalid/oversized
    audio or a Groq API failure — callers should catch this specifically
    rather than a bare Exception.
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
            ErrorCode.TRANSCRIPTION_FAILED, "Speech-to-text service failed"
        ) from e

    text = result.text.strip()
    detected = getattr(result, "language", None) or lock_lang or "en"
    print(f"[Groq Whisper] lang={detected} locked={lock_lang or 'no'} | {text[:80]!r}")
    return text, detected