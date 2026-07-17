import os
import json
from groq import Groq
from typing import Optional
from services.tools import TOOLS, run_tool

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DEFAULT_SYSTEM_PROMPT = """You are {name}, a helpful, concise, and friendly voice AI assistant.
Keep responses conversational and brief — you're speaking aloud, not writing text.
Avoid markdown, bullet points, or lists in your responses.
You have access to web search and a calculator. Use them when the user asks about
current events, news, weather, prices, or math calculations."""

ARIA_NAME = os.getenv("ARIA_NAME", "ARIA")
ARIA_PERSONA_EXTRA = os.getenv("ARIA_PERSONA_EXTRA", "")


def build_system_prompt(persona_override: Optional[str] = None) -> str:
    base = DEFAULT_SYSTEM_PROMPT.format(name=ARIA_NAME)
    if ARIA_PERSONA_EXTRA:
        base += f"\n{ARIA_PERSONA_EXTRA}"
    if persona_override:
        base += f"\n{persona_override}"
    return base


def chat_with_memory(user_message: str, history: list, persona: Optional[str] = None) -> tuple[str, list]:
    """Non-streaming with tool use. Used by the Streamlit app.

    Groq/LLaMA occasionally malforms its tool-call syntax (raises a 400
    'tool_use_failed' error). This is a transient generation issue, not a
    bug in our request — so we retry once with tools enabled, and if that
    still fails, fall back to answering without tools rather than crashing
    the whole response.
    """
    messages = [{"role": "system", "content": build_system_prompt(persona)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    def _first_call(use_tools: bool):
        kwargs = dict(model="llama-3.3-70b-versatile", messages=messages, max_tokens=500, temperature=0.7)
        if use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"
        return client.chat.completions.create(**kwargs)

    response = None
    last_error = None
    for attempt in range(2):  # try with tools twice (Groq's tool-call syntax is occasionally malformed)
        try:
            response = _first_call(use_tools=True)
            break
        except Exception as e:
            last_error = e

    if response is None:
        # Both attempts failed — fall back to a plain answer with no tools
        response = _first_call(use_tools=False)

    msg = response.choices[0].message
    tool_calls_used = []

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_result = run_tool(tool_name, tool_args)
            tool_calls_used.append(tool_name)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

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
    return assistant_reply, updated_history, tool_calls_used