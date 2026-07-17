"""
Persistent memory via Upstash Redis (same free-tier DB you already have).
If Upstash env vars aren't set, falls back to a process-local dict so the
app still runs (memory just won't survive a restart) — handy for first
local test before you've added secrets.
"""

import json
import os

SESSION_TTL = int(os.getenv("SESSION_TTL_HOURS", "24")) * 3600
MAX_MESSAGES = 20
PREFIX = "aria:session:"
PERSONA_PREFIX = "aria:persona:"

_redis = None
_local_store: dict[str, str] = {}  # fallback if Upstash isn't configured


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if url and token:
        from upstash_redis import Redis
        _redis = Redis(url=url, token=token)
    else:
        _redis = False  # sentinel: "not configured"
    return _redis


def _key(session_id: str) -> str:
    return f"{PREFIX}{session_id}"


def _persona_key(session_id: str) -> str:
    return f"{PERSONA_PREFIX}{session_id}"


def _store_get(key: str):
    redis = _get_redis()
    if redis:
        return redis.get(key)
    return _local_store.get(key)


def _store_set(key: str, value: str):
    redis = _get_redis()
    if redis:
        redis.set(key, value, ex=SESSION_TTL)
    else:
        _local_store[key] = value


def _store_delete(key: str):
    redis = _get_redis()
    if redis:
        redis.delete(key)
    else:
        _local_store.pop(key, None)


# ── History ───────────────────────────────────────────────────────────────────

def get_history(session_id: str) -> list[dict]:
    raw = _store_get(_key(session_id))
    if raw is None:
        return []
    return json.loads(raw)


def append_messages(session_id: str, user_msg: str, assistant_msg: str) -> None:
    history = get_history(session_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})

    if len(history) > MAX_MESSAGES:
        history = history[-MAX_MESSAGES:]

    _store_set(_key(session_id), json.dumps(history))


def clear_history(session_id: str) -> None:
    _store_delete(_key(session_id))
    _store_delete(_persona_key(session_id))


def get_session_metadata(session_id: str) -> dict:
    history = get_history(session_id)
    persona = get_session_persona(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "persona": persona,
        "persistent": bool(_get_redis()),
    }


# ── Persona ───────────────────────────────────────────────────────────────────

def set_session_persona(session_id: str, persona: str) -> None:
    _store_set(_persona_key(session_id), persona)


def get_session_persona(session_id: str):
    raw = _store_get(_persona_key(session_id))
    return raw if raw else None