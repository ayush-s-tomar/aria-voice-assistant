import json
import os

SESSION_TTL = int(os.getenv("SESSION_TTL_HOURS", "24")) * 3600
MAX_MESSAGES = 20
PREFIX = "aria:session:"
PERSONA_PREFIX = "aria:persona:"

_REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL")
_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

redis = None
if _REDIS_URL and _REDIS_TOKEN:
    try:
        from upstash_redis import Redis
        redis = Redis(url=_REDIS_URL, token=_REDIS_TOKEN)
    except Exception as e:
        print(f"[memory] Upstash Redis init failed, falling back to session-only: {e}")
        redis = None


def _key(session_id):
    return f"{PREFIX}{session_id}"


def _persona_key(session_id):
    return f"{PERSONA_PREFIX}{session_id}"


def get_history(session_id):
    if redis is None:
        return []
    raw = redis.get(_key(session_id))
    if raw is None:
        return []
    return json.loads(raw)


def append_messages(session_id, user_msg, assistant_msg):
    if redis is None:
        return
    history = get_history(session_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    if len(history) > MAX_MESSAGES:
        history = history[-MAX_MESSAGES:]
    redis.set(_key(session_id), json.dumps(history), ex=SESSION_TTL)


def clear_history(session_id):
    if redis is None:
        return
    redis.delete(_key(session_id))
    redis.delete(_persona_key(session_id))


def get_session_metadata(session_id):
    history = get_history(session_id)
    persona = get_session_persona(session_id)
    ttl = None
    if redis is not None:
        try:
            ttl = redis.ttl(_key(session_id))
        except Exception:
            ttl = None
    return {
        "session_id": session_id,
        "persistent": redis is not None,
        "message_count": len(history),
        "expires_in_seconds": ttl,
        "persona": persona,
    }


def set_session_persona(session_id, persona):
    if redis is None:
        return
    redis.set(_persona_key(session_id), persona, ex=SESSION_TTL)


def get_session_persona(session_id):
    if redis is None:
        return None
    raw = redis.get(_persona_key(session_id))
    return raw if raw else None
