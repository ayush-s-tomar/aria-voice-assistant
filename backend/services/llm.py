import os
from groq import Groq
from typing import AsyncGenerator

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are ARIA, a helpful, concise, and friendly voice AI assistant.
Keep responses conversational and brief — you're speaking aloud, not writing text.
Avoid markdown, bullet points, or lists in your responses."""


def chat_with_memory(user_message: str, history: list) -> tuple[str, list]:
    """Non-streaming — used by HTTP endpoints."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    assistant_reply = response.choices[0].message.content
    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_reply},
    ]
    return assistant_reply, updated_history


async def stream_llm_response(user_message: str, history: list) -> AsyncGenerator[str, None]:
    """Streaming — used by WebSocket endpoint. Yields tokens as they arrive."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        stream=True,   # ← the key change
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token