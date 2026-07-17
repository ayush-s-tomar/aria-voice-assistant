# backend/services/memory.py
import json
import os
from upstash_redis import Redis

redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
)

SESSION_TTL = int(os.getenv("SESSION_TTL_HOURS", "24")) * 3600
MAX_MESSAGES = 20  # rolling window
PREFIX = "aria:session:"
PERSONA_PREFIX = "aria:persona:"


def _key(session_id: str) -> str:
    return f"{PREFIX}{session_id}"


def _persona_key(session_id: str) -> str:
    return f"{PERSONA_PREFIX}{session_id}"


# ── History ───────────────────────────────────────────────────────────────────

def get_history(session_id: str) -> list[dict]:
    """Return message history for a session, or [] if not found."""
    raw = redis.get(_key(session_id))
    if raw is None:
        return []
    return json.loads(raw)


def append_messages(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Append a user/assistant turn and enforce rolling window."""
    history = get_history(session_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})

    if len(history) > MAX_MESSAGES:
        history = history[-MAX_MESSAGES:]

    redis.set(_key(session_id), json.dumps(history), ex=SESSION_TTL)


def clear_history(session_id: str) -> None:
    """Delete a session's history and persona."""
    redis.delete(_key(session_id))
    redis.delete(_persona_key(session_id))


def get_session_metadata(session_id: str) -> dict:
    """Return TTL, message count, and persona — useful for the frontend."""
    key = _key(session_id)
    ttl = redis.ttl(key)
    history = get_history(session_id)
    persona = get_session_persona(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "expires_in_seconds": ttl,
        "persona": persona,
    }


# ── Persona ───────────────────────────────────────────────────────────────────

def set_session_persona(session_id: str, persona: str) -> None:
    """Store a custom persona/tone override for this session."""
    redis.set(_persona_key(session_id), persona, ex=SESSION_TTL)


def get_session_persona(session_id: str) -> str | None:
    """Return the persona override for this session, or None if not set."""
    raw = redis.get(_persona_key(session_id))
    return raw if raw else None