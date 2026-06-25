from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile, os, io, base64, json
from services.transcriber import transcribe_audio
from services.llm import chat_with_memory, stream_llm_response
from services.tts import text_to_speech
from services.memory import get_history, append_messages, clear_history, get_session_metadata

app = FastAPI(
    title="ARIA – Voice AI Assistant",
    description="Speech-to-speech AI pipeline: Groq Whisper → LLaMA 3.3-70B → gTTS",
    version="2.2.0",
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
    return {"status": "ARIA is running", "version": "2.2.0"}


# ── WebSocket streaming pipeline ──────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_voice(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                # Step 1: STT
                await websocket.send_json({"type": "status", "text": "Transcribing…"})
                user_text, detected_lang = transcribe_audio(tmp_path)

                if not user_text.strip():
                    await websocket.send_json({"type": "error", "text": "Could not transcribe audio"})
                    continue

                await websocket.send_json({
                    "type": "transcript",
                    "text": user_text,
                    "lang": detected_lang
                })

                # Step 2: Stream LLM (with tool use inside stream_llm_response)
                await websocket.send_json({"type": "status", "text": "Thinking…"})
                history = get_history(session_id)

                full_response = ""
                sentence_buffer = ""

                async for token in stream_llm_response(user_text, history):
                    # Tool status tokens — show in UI but skip TTS and response storage
                    is_tool_status = token.startswith("\n[") and "…]" in token
                    if is_tool_status:
                        await websocket.send_json({"type": "token", "text": token})
                        continue

                    full_response += token
                    sentence_buffer += token

                    await websocket.send_json({"type": "token", "text": token})

                    # Sentence complete → TTS immediately
                    if any(sentence_buffer.rstrip().endswith(p) for p in [".", "!", "?", "…"]):
                        sentence = sentence_buffer.strip()
                        if sentence:
                            await websocket.send_json({"type": "status", "text": "Speaking…"})
                            audio_bytes = text_to_speech(sentence)
                            await websocket.send_json({
                                "type": "audio_chunk",
                                "data": base64.b64encode(audio_bytes).decode(),
                            })
                        sentence_buffer = ""

                # Flush remaining buffer
                if sentence_buffer.strip():
                    audio_bytes = text_to_speech(sentence_buffer.strip())
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": base64.b64encode(audio_bytes).decode(),
                    })

                # Persist to Redis
                append_messages(session_id, user_text, full_response)
                await websocket.send_json({"type": "done", "text": full_response})

            finally:
                os.unlink(tmp_path)

    except WebSocketDisconnect:
        pass


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.post("/chat/voice", tags=["Chat"])
async def voice_chat(audio: UploadFile = File(...), session_id: str = "default"):
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
            io.BytesIO(audio_bytes), media_type="audio/mpeg",
            headers={
                "X-User-Text":      user_header,
                "X-Assistant-Text": assistant_header,
                "X-Language":       detected_lang,
                "Access-Control-Expose-Headers": "X-User-Text, X-Assistant-Text, X-Language",
            },
        )
    finally:
        os.unlink(tmp_path)


@app.post("/chat/text", tags=["Chat"])
async def text_chat(payload: dict, session_id: str = "default"):
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
    return get_session_metadata(session_id)


@app.delete("/session/{session_id}", tags=["Session"])
def clear_session(session_id: str):
    clear_history(session_id)
    return {"cleared": session_id}