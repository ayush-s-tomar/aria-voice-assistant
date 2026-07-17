"""
Structured error codes.

Cluster F: every user-facing error now carries a machine-readable
`error_code` so the frontend can branch on it (show a retry banner,
redirect to auth, back off and retry) instead of string-matching
free-text error messages, which breaks the moment wording changes.
"""

from fastapi import HTTPException


class ErrorCode:
    AUDIO_EMPTY = "AUDIO_EMPTY"
    AUDIO_TOO_LARGE = "AUDIO_TOO_LARGE"
    AUDIO_INVALID_FORMAT = "AUDIO_INVALID_FORMAT"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    EMPTY_TRANSCRIPT = "EMPTY_TRANSCRIPT"
    RATE_LIMITED = "RATE_LIMITED"
    SESSION_BUSY = "SESSION_BUSY"
    LLM_FAILED = "LLM_FAILED"
    TTS_FAILED = "TTS_FAILED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    AUTH_INVALID = "AUTH_INVALID"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def http_error(status_code: int, code: str, message: str, **extra) -> HTTPException:
    """Structured HTTPException — detail is a dict {error_code, message, ...} not a bare string."""
    detail = {"error_code": code, "message": message, **extra}
    return HTTPException(status_code=status_code, detail=detail)


def ws_error(code: str, message: str, **extra) -> dict:
    """Structured WS error payload, same shape as the HTTP one for frontend consistency."""
    return {"type": "error", "error_code": code, "text": message, **extra}