@'
"""Unit tests for TranscriptionError structure — no Groq API calls needed."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.transcriber import TranscriptionError, _validate_file
from services.errors import ErrorCode


def test_transcription_error_has_error_code():
    err = TranscriptionError("test message", ErrorCode.AUDIO_TOO_LARGE)
    assert err.error_code == ErrorCode.AUDIO_TOO_LARGE
    assert err.message == "test message"


def test_transcription_error_defaults_to_transcription_failed():
    err = TranscriptionError("generic failure")
    assert err.error_code == ErrorCode.TRANSCRIPTION_FAILED


def test_validate_file_missing_raises_invalid_format():
    try:
        _validate_file("/nonexistent/path/audio.webm")
        assert False, "should have raised"
    except TranscriptionError as e:
        assert e.error_code == ErrorCode.AUDIO_INVALID_FORMAT
'@ | Out-File -Encoding utf8 backend\tests\test_transcriber.py