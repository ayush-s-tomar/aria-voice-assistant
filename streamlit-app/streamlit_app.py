"""
ARIA – Voice AI Assistant (Streamlit edition)

Single-file deployment target: Streamlit Community Cloud (free).
No FastAPI, no WebSocket server, no Render. Everything runs in one process.

Pipeline: mic recording -> Groq Whisper STT -> LLaMA 3.3-70B (+ tools) -> gTTS/ElevenLabs -> spoken reply
Memory: Upstash Redis (same DB as before) if configured, else in-session only.
"""

import hashlib
import os
import tempfile
import uuid

import streamlit as st

# ── Secrets → env (MUST run before importing services.*) ─────────────────────
for k in [
    "GROQ_API_KEY", "TAVILY_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID",
    "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN",
    "ARIA_NAME", "ARIA_PERSONA_EXTRA", "SESSION_TTL_HOURS",
]:
    if k in st.secrets and not os.getenv(k):
        os.environ[k] = str(st.secrets[k])

from services.transcriber import transcribe_audio, TranscriptionError
from services.llm import chat_with_memory
from services.tts import text_to_speech
from services.memory import (
    get_history,
    append_messages,
    clear_history,
    get_session_persona,
    set_session_persona,
    get_session_metadata,
)

ARIA_NAME = os.getenv("ARIA_NAME", "ARIA")

st.set_page_config(page_title=ARIA_NAME, page_icon="🎙️", layout="centered")

# ── Visual polish ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Audio player: default browser controls look washed-out on dark themes.
       This re-skins the <audio> element to match the app's palette. */
    audio {
        width: 100%;
        height: 40px;
        border-radius: 10px;
        background: #1a1c22;
        filter: invert(1) hue-rotate(180deg);
        margin-top: 6px;
    }

    /* Chat bubbles: give user vs assistant messages distinct backgrounds
       and a bit of breathing room instead of flat, same-toned blocks. */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #2a1414, #241010);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #1c1f2b, #171a24);
    }

    /* Sidebar: tighten label spacing, add breathing room between sections. */
    section[data-testid="stSidebar"] {
        padding-top: 0.5rem;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] hr {
        margin: 1.1rem 0;
        border-color: rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] label p {
        font-size: 0.85rem;
        opacity: 0.85;
    }

    /* Buttons: consistent rounded style instead of Streamlit's flat default. */
    section[data-testid="stSidebar"] button {
        border-radius: 8px !important;
        transition: transform 0.1s ease;
    }
    section[data-testid="stSidebar"] button:hover {
        transform: translateY(-1px);
    }

    /* Recorder + chat input: round corners so they match the rest of the UI. */
    [data-testid="stAudioInput"], [data-testid="stChatInput"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session identity ──────────────────────────────────────────────────────────
# session_id is PINNED in session_state. It is only ever changed explicitly,
# via the on_change callback below — never recomputed as a bare local variable
# on every rerun. This is what previously caused "Apply persona" and the chat
# pipeline to sometimes read/write under different session keys.
if "anon_session_id" not in st.session_state:
    st.session_state.anon_session_id = str(uuid.uuid4())[:8]

if "session_id" not in st.session_state:
    st.session_state.session_id = st.session_state.anon_session_id

if "messages" not in st.session_state:
    st.session_state.messages = []

if "_last_audio_hash" not in st.session_state:
    st.session_state._last_audio_hash = None


def _sync_session_id():
    """Called on every keystroke change of the name field (on_change),
    so session_id updates immediately and deterministically — not lazily
    on the next unrelated rerun (e.g. a button click)."""
    name = st.session_state.get("display_name", "").strip().lower()
    st.session_state.session_id = name if name else st.session_state.anon_session_id


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🎙️ {ARIA_NAME}")
    st.caption("AI Real-Time Intelligent Assistant")

    st.text_input(
        "Your name (optional — enables cross-device memory)",
        key="display_name",
        placeholder="e.g. ayush",
        on_change=_sync_session_id,
    )

    # Always resolve from the pinned value — never recompute inline here.
    session_id = st.session_state.session_id
    st.caption(f"Session: `{session_id}`")

    st.divider()
    st.markdown("**Persona**")
    preset = st.selectbox(
        "Tone preset",
        ["Default", "Concise", "Casual", "Formal", "Tutor", "Witty", "Hindi", "Custom"],
    )
    persona_text = None
    if preset == "Custom":
        persona_text = st.text_area("Custom instruction", placeholder="Speak like a pirate.")
    elif preset != "Default":
        _presets = {
            "Concise": "Keep every answer to one short sentence.",
            "Casual": "Speak casually, like a friend texting.",
            "Formal": "Speak formally and precisely.",
            "Tutor": "Explain things step by step, like a patient tutor.",
            "Witty": "Add light humor and wit to your answers.",
            "Hindi": "Respond in Hindi.",
        }
        persona_text = _presets[preset]

    if st.button("Apply persona", use_container_width=True):
        # Uses the pinned session_id — guaranteed in sync with the name field
        # because on_change fired before this rerun even started.
        if persona_text:
            set_session_persona(session_id, persona_text)
            st.success(f"Persona updated for session `{session_id}`.")
        else:
            set_session_persona(session_id, "")
            st.success(f"Reset to default persona for session `{session_id}`.")

    st.divider()
    voice_choice = st.radio("Voice engine", ["Free (gTTS)", "ElevenLabs (needs key)"], index=0)
    use_elevenlabs = voice_choice == "ElevenLabs (needs key)" and bool(os.getenv("ELEVENLABS_API_KEY"))
    if voice_choice == "ElevenLabs (needs key)" and not os.getenv("ELEVENLABS_API_KEY"):
        st.warning("No ELEVENLABS_API_KEY set — falling back to gTTS.")
    autoplay = st.checkbox("Auto-play replies", value=True)

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        clear_history(session_id)
        st.session_state.messages = []
        st.session_state._last_audio_hash = None
        st.rerun()

    meta = get_session_metadata(session_id)
    persistence_note = "persistent (Upstash)" if meta["persistent"] else "this browser tab only"
    st.caption(f"Memory: {persistence_note} · {meta['message_count']} msgs stored")

# ── Load history from Redis on first run of this session ─────────────────────
if not st.session_state.messages:
    st.session_state.messages = get_history(session_id)

# ── Render existing conversation ──────────────────────────────────────────────
st.title(f"🎙️ {ARIA_NAME}")
st.caption("Speech-to-speech assistant · Groq Whisper + LLaMA 3.3-70B + TTS")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3", autoplay=False)

st.divider()

# ── Input: voice or text ───────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("**🎤 Speak**")
    audio_value = st.audio_input("Record a message", label_visibility="collapsed")
with col2:
    st.markdown("**⌨️ Or type**")
    typed_text = st.chat_input("Type a message to ARIA...")

user_text = None
detected_lang = "en"

if audio_value is not None:
    audio_bytes_raw = audio_value.getvalue()
    audio_hash = hashlib.sha256(audio_bytes_raw).hexdigest()
    if audio_hash != st.session_state._last_audio_hash:
        st.session_state._last_audio_hash = audio_hash
        with st.spinner("Transcribing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes_raw)
                tmp_path = tmp.name
            try:
                user_text, detected_lang = transcribe_audio(tmp_path)
            except TranscriptionError as e:
                st.error(f"Transcription failed: {e.message}")
            finally:
                os.unlink(tmp_path)

if typed_text:
    user_text = typed_text

# ── Run the pipeline on new input ─────────────────────────────────────────────
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    history = get_history(session_id)

    # get_session_persona can legitimately return None (nothing set yet).
    # Guard it here so "None" never gets concatenated into the prompt string.
    persona = get_session_persona(session_id) or ""

    # Inject the user's name as a fact the model can actually see.
    # The sidebar text field previously only affected the Redis session key —
    # it was never passed into the system prompt, so ARIA had history but no name.
    display_name = st.session_state.get("display_name", "")
    if display_name.strip():
        # Phrased as background context with an explicit usage rule —
        # otherwise LLaMA treats a bare "The user's name is X" fact as an
        # instruction to open every reply with the name.
        name_fact = (
            f"(Background only: the user's name is {display_name.strip()}. "
            "You may know this, but do not greet them by name or mention "
            "their name in every reply — only use it when it's naturally "
            "relevant, e.g. if they ask you to.)"
        )
        persona = f"{name_fact} {persona}".strip()

    with st.chat_message("assistant"):
        assistant_text = None
        tools_used = None
        with st.spinner("Thinking..."):
            try:
                assistant_text, _, tools_used = chat_with_memory(user_text, history, persona=persona)
            except Exception as e:
                st.error(f"Assistant failed to respond: {e}")

        if assistant_text:
            if tools_used:
                st.caption(f"🛠️ used: {', '.join(tools_used)}")
            st.write(assistant_text)

            audio_bytes = None
            with st.spinner("Generating voice..."):
                try:
                    audio_bytes = text_to_speech(
                        assistant_text,
                        lang=detected_lang,
                        use_elevenlabs=use_elevenlabs,
                    )
                    st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay)
                except TypeError:
                    try:
                        audio_bytes = text_to_speech(assistant_text, lang=detected_lang)
                        st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay)
                    except Exception as e:
                        st.warning(f"Voice playback failed: {e}")
                except Exception as e:
                    st.warning(f"Voice playback failed: {e}")

            append_messages(session_id, user_text, assistant_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_text,
                "audio": audio_bytes,
            })