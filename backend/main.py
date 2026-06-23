from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile, os, io, base64
from services.transcriber import transcribe_audio
from services.llm import chat_with_memory
from services.tts import text_to_speech
from services.memory import get_history, append_messages, clear_history, get_session_metadata

app = FastAPI(
    title="ARIA – Voice AI Assistant",
    description="Speech-to-speech AI pipeline: Groq Whisper → LLaMA 3.3-70B → gTTS",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ARIA is running", "version": "2.0.0"}


# ── Voice pipeline ────────────────────────────────────────────────────────────

@app.post("/chat/voice", tags=["Chat"])
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = "default",
):
    """Full pipeline: audio → STT → LLM (with Redis memory) → TTS → audio."""
    suffix = os.path.splitext(audio.filename or "")[-1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        user_text, detected_lang = transcribe_audio(tmp_path)
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        history = get_history(session_id)
        assistant_text, updated_history = chat_with_memory(user_text, history)
        append_messages(session_id, user_text, assistant_text)

        audio_bytes = text_to_speech(assistant_text)

        user_header      = base64.b64encode(user_text.encode()).decode("ascii")
        assistant_header = base64.b64encode(assistant_text.encode()).decode("ascii")

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "X-User-Text":       user_header,
                "X-Assistant-Text":  assistant_header,
                "X-Language":        detected_lang,
                "Access-Control-Expose-Headers": (
                    "X-User-Text, X-Assistant-Text, X-Language"
                ),
            },
        )
    finally:
        os.unlink(tmp_path)


# ── Text pipeline ─────────────────────────────────────────────────────────────

@app.post("/chat/text", tags=["Chat"])
async def text_chat(payload: dict, session_id: str = "default"):
    """Text-only: message → LLM (with Redis memory) → response."""
    user_text = payload.get("message", "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="'message' field is required")

    history = get_history(session_id)
    assistant_text, updated_history = chat_with_memory(user_text, history)
    append_messages(session_id, user_text, assistant_text)

    return {"user": user_text, "assistant": assistant_text}


# ── Session management ────────────────────────────────────────────────────────

@app.get("/session/{session_id}", tags=["Session"])
def session_info(session_id: str):
    """Return message count and TTL for a session."""
    return get_session_metadata(session_id)


@app.delete("/session/{session_id}", tags=["Session"])
def clear_session(session_id: str):
    """Clear a session's conversation history from Redis."""
    clear_history(session_id)
    return {"cleared": session_id}