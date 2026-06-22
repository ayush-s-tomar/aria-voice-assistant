import os
from groq import Groq

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

def transcribe_audio(file_path: str, language: str = None) -> tuple[str, str]:
    """
    Transcribe audio via Groq Whisper large-v3.
    Returns (text, detected_language).
    """
    client = _get_client()
    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            language=language,
            response_format="verbose_json"
        )
    text = result.text.strip()
    detected = result.language if hasattr(result, "language") else (language or "en")
    print(f"[Groq Whisper] Language: {detected} | Text: {text}")
    return text, detected