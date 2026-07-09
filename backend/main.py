from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
import tempfile, os, io, base64, json
from services.transcriber import transcribe_audio
from services.llm import chat_with_memory, stream_llm_response
from services.tts import text_to_speech
from services.memory import (
    get_history,
    append_messages,
    clear_history,
    get_session_metadata,
    get_session_persona,
    set_session_persona,
)
from services.auth import get_github_auth_url, exchange_code_for_token, create_jwt, verify_jwt

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8000")

app = FastAPI(
    title="ARIA – Voice AI Assistant",
    description="Speech-to-speech AI pipeline: Groq Whisper → LLaMA 3.3-70B → gTTS",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def get_user_from_token(authorization: str = None) -> dict | None:
    """Extract user from Bearer token. Returns None if missing/invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        return verify_jwt(authorization.split(" ", 1)[1])
    except Exception:
        return None


def user_session_id(user: dict, session_id: str) -> str:
    """Namespace session by user so histories never mix."""
    return f"user:{user['github_id']}:session:{session_id}"


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ARIA is running", "version": "3.1.0"}


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.get("/auth/login", tags=["Auth"])
def github_login():
    """Redirect user to GitHub OAuth page."""
    return RedirectResponse(get_github_auth_url())


@app.get("/auth/callback", tags=["Auth"])
async def github_callback(code: str):
    """GitHub redirects here with a code. Exchange it for a JWT."""
    try:
        user = await exchange_code_for_token(code)
        token = create_jwt(user)
        # Redirect to frontend with token in URL fragment
        return RedirectResponse(
            f"{FRONTEND_URL}?token={token}"
        )
    except Exception as e:
        return RedirectResponse(
            f"{FRONTEND_URL}?error=auth_failed"
        )


@app.get("/auth/me", tags=["Auth"])
def get_me(authorization: str = Header(default=None)):
    """Return current user info from JWT."""
    user = get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "github_id": user["github_id"],
        "username":  user["username"],
        "name":      user["name"],
        "avatar":    user["avatar"],
    }


# ── WebSocket streaming pipeline ──────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_voice(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Auth via first message (token handshake)
    user = None
    try:
        auth_msg = await websocket.receive_json()
        if auth_msg.get("type") == "auth":
            try:
                user = verify_jwt(auth_msg.get("token", ""))
                await websocket.send_json({
                    "type": "auth_ok",
                    "name": user["name"],
                    "avatar": user["avatar"],
                })
            except Exception:
                await websocket.send_json({"type": "auth_error", "text": "Invalid token"})
    except Exception:
        pass

    # Resolve session ID — namespaced if logged in, anonymous if not
    resolved_session = (
        user_session_id(user, session_id) if user else session_id
    )

    try:
        while True:
            data = await websocket.receive_bytes()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
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

                await websocket.send_json({"type": "status", "text": "Thinking…"})
                history = get_history(resolved_session)
                persona = get_session_persona(resolved_session)

                full_response = ""
                sentence_buffer = ""

                async for token in stream_llm_response(user_text, history, persona=persona):
                    is_tool_status = token.startswith("\n[") and "…]" in token
                    if is_tool_status:
                        await websocket.send_json({"type": "token", "text": token})
                        continue

                    full_response += token
                    sentence_buffer += token
                    await websocket.send_json({"type": "token", "text": token})

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

                if sentence_buffer.strip():
                    audio_bytes = text_to_speech(sentence_buffer.strip())
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": base64.b64encode(audio_bytes).decode(),
                    })

                append_messages(resolved_session, user_text, full_response)
                await websocket.send_json({"type": "done", "text": full_response})

            finally:
                os.unlink(tmp_path)

    except WebSocketDisconnect:
        pass


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.post("/chat/voice", tags=["Chat"])
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = "default",
    authorization: str = Header(default=None),
):
    user = get_user_from_token(authorization)
    resolved_session = user_session_id(user, session_id) if user else session_id

    suffix = os.path.splitext(audio.filename or "")[-1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        user_text, detected_lang = transcribe_audio(tmp_path)
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        history = get_history(resolved_session)
        persona = get_session_persona(resolved_session)
        assistant_text, _ = chat_with_memory(user_text, history, persona=persona)
        append_messages(resolved_session, user_text, assistant_text)
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
async def text_chat(
    payload: dict,
    session_id: str = "default",
    authorization: str = Header(default=None),
):
    user = get_user_from_token(authorization)
    resolved_session = user_session_id(user, session_id) if user else session_id

    user_text = payload.get("message", "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="'message' field is required")
    history = get_history(resolved_session)
    persona = get_session_persona(resolved_session)
    assistant_text, _ = chat_with_memory(user_text, history, persona=persona)
    append_messages(resolved_session, user_text, assistant_text)
    return {"user": user_text, "assistant": assistant_text}


# ── Session management ────────────────────────────────────────────────────────

@app.get("/session/{session_id}", tags=["Session"])
def session_info(
    session_id: str,
    authorization: str = Header(default=None),
):
    user = get_user_from_token(authorization)
    resolved_session = user_session_id(user, session_id) if user else session_id
    return get_session_metadata(resolved_session)


@app.delete("/session/{session_id}", tags=["Session"])
def clear_session(
    session_id: str,
    authorization: str = Header(default=None),
):
    user = get_user_from_token(authorization)
    resolved_session = user_session_id(user, session_id) if user else session_id
    clear_history(resolved_session)
    return {"cleared": resolved_session}


@app.put("/session/{session_id}/persona", tags=["Session"])
def update_persona(
    session_id: str,
    payload: dict,
    authorization: str = Header(default=None),
):
    """Let a user customize ARIA's tone/persona for this session."""
    user = get_user_from_token(authorization)
    resolved_session = user_session_id(user, session_id) if user else session_id
    persona = payload.get("persona", "").strip()
    if not persona:
        raise HTTPException(status_code=400, detail="'persona' field is required")
    set_session_persona(resolved_session, persona)
    return {"session_id": resolved_session, "persona": persona}