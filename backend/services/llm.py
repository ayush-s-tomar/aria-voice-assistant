"""
LLM service using Groq (same API you used in StartupScope & AgentLoop).
Maintains rolling conversation history for memory across turns.
"""
from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are ARIA (AI Real-time Intelligent Assistant) — a friendly, concise voice assistant.

Rules:
- Keep responses SHORT (2-4 sentences max) since they will be spoken aloud.
- Be conversational and natural, like a real voice assistant.
- Avoid bullet points, markdown, or lists — plain sentences only.
- If asked your name, say you are ARIA.
- Remember context from earlier in the conversation.
"""

MAX_HISTORY = 20  # keep last 10 exchanges (20 messages)


def chat_with_memory(
    user_message: str,
    history: list[dict],
    model: str = "llama-3.3-70b-versatile",
) -> tuple[str, list[dict]]:
    """
    Send a message with history and return (assistant_reply, updated_history).
    """
    # Append user message
    history = history + [{"role": "user", "content": user_message}]

    # Trim history to avoid token overflow
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        temperature=0.7,
        max_tokens=300,
    )

    assistant_reply = response.choices[0].message.content.strip()
    print(f"[LLM] Response: {assistant_reply}")

    # Append assistant reply to history
    history = history + [{"role": "assistant", "content": assistant_reply}]

    return assistant_reply, history
