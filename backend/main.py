from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile, os, io, base64
from services.transcriber import transcribe_audio
from services.llm import chat_with_memory
from services.tts import text_to_speech

app = FastAPI(title="Voice AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_store: dict[str, list] = {}


@app.get("/")
def root():
    return {"status": "Voice AI Assistant running"}


@app.post("/chat/voice")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = "default",
):
    suffix = os.path.splitext(audio.filename)[-1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        user_text, detected_lang = transcribe_audio(tmp_path)
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        history = conversation_store.setdefault(session_id, [])
        assistant_text, updated_history = chat_with_memory(user_text, history)
        conversation_store[session_id] = updated_history

        audio_bytes = text_to_speech(assistant_text)

        user_header = base64.b64encode(user_text.encode("utf-8")).decode("ascii")
        assistant_header = base64.b64encode(assistant_text.encode("utf-8")).decode("ascii")

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "X-User-Text": user_header,
                "X-Assistant-Text": assistant_header,
                "X-Language": detected_lang,
                "Access-Control-Expose-Headers": "X-User-Text, X-Assistant-Text, X-Language",
            },
        )
    finally:
        os.unlink(tmp_path)


@app.post("/chat/text")
async def text_chat(payload: dict, session_id: str = "default"):
    user_text = payload.get("message", "")
    if not user_text:
        raise HTTPException(status_code=400, detail="message is required")

    history = conversation_store.setdefault(session_id, [])
    assistant_text, updated_history = chat_with_memory(user_text, history)
    conversation_store[session_id] = updated_history

    return {"user": user_text, "assistant": assistant_text}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    conversation_store.pop(session_id, None)
    return {"cleared": session_id}