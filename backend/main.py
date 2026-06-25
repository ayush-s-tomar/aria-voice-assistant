from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile, os, io, base64, json
from services.transcriber import transcribe_audio
from services.llm import chat_with_memory, stream_llm_response
from services.tts import text_to_speech, tts_chunk
from services.memory import get_history, append_messages, clear_history, get_session_metadata

app = FastAPI(
    title="ARIA – Voice AI Assistant",
    description="Speech-to-speech AI pipeline: Groq Whisper → LLaMA 3.3-70B → gTTS",
    version="2.1.0",
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
    return {"status": "ARIA is running", "version": "2.1.0"}


# ── WebSocket streaming pipeline ──────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_voice(websocket: WebSocket, session_id: str):
    """
    Streaming pipeline:
    1. Receive audio bytes from client
    2. STT → send transcript back immediately
    3. Stream LLM tokens → accumulate sentences → TTS each sentence
    4. Send audio chunks as base64 as they're ready
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            # Save audio to temp file
            suffix = ".webm"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                # Step 1: STT
                await websocket.send_json({"type": "status", "text": "Transcribing…"})
                user_text, detected_lang = transcribe_audio(tmp_path)

                if not user_text.strip():
                    await websocket.send_json({"type": "error", "text": "Could not transcribe audio"})
                    continue

                # Send transcript immediately — user sees their words right away
                await websocket.send_json({
                    "type": "transcript",
                    "text": user_text,
                    "lang": detected_lang
                })

                # Step 2: Stream LLM
                await websocket.send_json({"type": "status", "text": "Thinking…"})
                history = get_history(session_id)

                full_response = ""
                sentence_buffer = ""

                async for token in stream_llm_response(user_text, history):
                    full_response += token
                    sentence_buffer += token

                    # Send each token to frontend for live text display
                    await websocket.send_json({"type": "token", "text": token})

                    # When we have a complete sentence, convert to speech immediately
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

                # Flush any remaining text as final TTS chunk
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


# ── Original HTTP endpoints (keep for backwards compatibility) ─────────────────

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
                "X-User-Text": user_header,
                "X-Assistant-Text": assistant_header,
                "X-Language": detected_lang,
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


@app.get("/session/{session_id}", tags=["Session"])
def session_info(session_id: str):
    return get_session_metadata(session_id)


@app.delete("/session/{session_id}", tags=["Session"])
def clear_session(session_id: str):
    clear_history(session_id)
    return {"cleared": session_id}