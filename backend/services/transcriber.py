from faster_whisper import WhisperModel
import os

_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
_model = None

def _get_model():
    global _model
    if _model is None:
        print(f"[Whisper] Loading model: {_MODEL_SIZE}")
        _model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model

def transcribe_audio(file_path: str, language: str = None) -> tuple[str, str]:
    """
    Transcribe audio. Returns (text, detected_language).
    Pass language="en" to force English, or leave None for auto-detect.
    Supports: Hindi, English, Spanish, French, German, Japanese, etc.
    """
    model = _get_model()
    segments, info = model.transcribe(file_path, language=language)
    text = " ".join(segment.text for segment in segments).strip()
    detected = info.language
    print(f"[Whisper] Language: {detected} | Text: {text}")
    return text, detected