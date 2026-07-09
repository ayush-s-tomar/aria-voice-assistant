import os
import json
from groq import Groq
from typing import AsyncGenerator, Optional
from services.tools import TOOLS, run_tool

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DEFAULT_SYSTEM_PROMPT = """You are {name}, a helpful, concise, and friendly voice AI assistant.
Keep responses conversational and brief — you're speaking aloud, not writing text.
Avoid markdown, bullet points, or lists in your responses.
You have access to web search and a calculator. Use them when the user asks about
current events, news, weather, prices, or math calculations."""

# Set ARIA_NAME / ARIA_PERSONA_EXTRA in env to rebrand without touching code,
# or pass a per-session override (see services/memory.set_session_persona).
ARIA_NAME = os.getenv("ARIA_NAME", "ARIA")
ARIA_PERSONA_EXTRA = os.getenv("ARIA_PERSONA_EXTRA", "")


def build_system_prompt(persona_override: Optional[str] = None) -> str:
    """Compose the system prompt, layering: override > env extras > default."""
    base = DEFAULT_SYSTEM_PROMPT.format(name=ARIA_NAME)
    if ARIA_PERSONA_EXTRA:
        base += f"\n{ARIA_PERSONA_EXTRA}"
    if persona_override:
        base += f"\n{persona_override}"
    return base


def chat_with_memory(user_message: str, history: list, persona: Optional[str] = None) -> tuple[str, list]:
    """Non-streaming with tool use — used by HTTP endpoints."""
    messages = [{"role": "system", "content": build_system_prompt(persona)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # First call — LLM decides if it needs a tool
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=500,
        temperature=0.7,
    )

    msg = response.choices[0].message

    # If LLM called a tool, run it and call LLM again with result
    if msg.tool_calls:
        messages.append(msg)  # append assistant's tool call message

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_result = run_tool(tool_name, tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # Second call — LLM now has tool result, generates final response
        response2 = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        assistant_reply = response2.choices[0].message.content
    else:
        assistant_reply = msg.content

    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_reply},
    ]
    return assistant_reply, updated_history


async def stream_llm_response(user_message: str, history: list, persona: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Streaming with tool use — used by WebSocket endpoint."""
    messages = [{"role": "system", "content": build_system_prompt(persona)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # First call — check for tool use (non-streaming, fast)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=100,   # small — just enough to decide on tool
        temperature=0.7,
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        # Run tools
        messages.append(msg)
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_result = run_tool(tool_name, tool_args)
            # Yield a status so frontend shows what's happening
            yield f"\n[Using {tool_name.replace('_', ' ')}…]\n"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # Second call — stream the final response with tool results injected
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            stream=True,
        )
    else:
        # No tool needed — stream directly
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            stream=True,
        )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token