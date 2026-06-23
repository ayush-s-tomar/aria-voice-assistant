# backend/services/llm.py
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are ARIA, a helpful, concise, and friendly voice AI assistant.
Keep responses conversational and brief — you're speaking aloud, not writing text.
Avoid markdown, bullet points, or lists in your responses."""


def chat_with_memory(user_message: str, history: list) -> tuple[str, list]:
    """Takes message + history, returns (assistant_reply, updated_history).
    Redis persistence is handled by main.py via memory.py — not here."""

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