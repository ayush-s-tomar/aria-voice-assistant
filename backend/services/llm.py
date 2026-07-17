import os
import re
import json
from groq import Groq
from typing import Optional
from services.tools import TOOLS, run_tool

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DEFAULT_SYSTEM_PROMPT = """You are {name}, a helpful, concise, and friendly voice AI assistant.
Keep responses conversational and brief — you're speaking aloud, not writing text.
Avoid markdown, bullet points, or lists in your responses.

You have access to tools: web_search, calculator, get_weather, wikipedia, get_datetime, unit_converter.

CRITICAL RULE: your training data has a cutoff and does NOT know current prices, scores,
news, or anything happening today. For ANY question about prices (crypto, stocks, currency
rates), current events, news, weather, or "what's happening now" — you MUST call web_search
or get_weather rather than answering from memory, even if you think you know the answer.
Confidently answering a stale number is worse than saying you checked and got the current one.
Only skip tools for timeless facts (history, science, definitions, general knowledge).

If a tool result begins with "TOOL_UNAVAILABLE", do not guess or make up an answer —
tell the user plainly that you're unable to check that right now."""

ARIA_NAME = os.getenv("ARIA_NAME", "ARIA")
ARIA_PERSONA_EXTRA = os.getenv("ARIA_PERSONA_EXTRA", "")

# Keyword patterns that should force a specific tool call rather than trusting
# the model's "auto" judgment — LLaMA under Groq frequently skips tool_choice=auto
# for things it feels falsely confident about (e.g. crypto/stock prices).
#
# Tightened to require more specific phrases rather than bare words like
# "today", "market", or "score" on their own, which over-triggered on
# unrelated messages (e.g. "market research", "what's today's agenda").
_FORCE_WEB_SEARCH_PATTERNS = re.compile(
    r"\b(current price|latest price|stock price|crypto price|bitcoin price|"
    r"price of [a-z0-9 ]+|exchange rate|"
    r"latest news|breaking news|today'?s news|"
    r"today'?s score|live score|"
    r"who won (the|last)|election results?|current market)\b",
    re.IGNORECASE,
)
_FORCE_WEATHER_PATTERNS = re.compile(r"\bweather\b", re.IGNORECASE)
_FORCE_DATETIME_PATTERNS = re.compile(
    r"\bwhat(?:'s| is) (?:the )?(?:date|time)\b|\bwhat day is it\b", re.IGNORECASE
)


def build_system_prompt(persona_override: Optional[str] = None) -> str:
    base = DEFAULT_SYSTEM_PROMPT.format(name=ARIA_NAME)
    if ARIA_PERSONA_EXTRA:
        base += f"\n{ARIA_PERSONA_EXTRA}"
    if persona_override:
        base += f"\n{persona_override}"
    return base


def _pick_forced_tool_choice(user_message: str) -> Optional[dict]:
    """Return a forced tool_choice dict if the message clearly needs a specific
    tool, else None to let the model decide normally (tool_choice='auto')."""
    if _FORCE_WEATHER_PATTERNS.search(user_message):
        return {"type": "function", "function": {"name": "get_weather"}}
    if _FORCE_DATETIME_PATTERNS.search(user_message):
        return {"type": "function", "function": {"name": "get_datetime"}}
    if _FORCE_WEB_SEARCH_PATTERNS.search(user_message):
        return {"type": "function", "function": {"name": "web_search"}}
    return None


def _serialize_assistant_msg(msg) -> dict:
    """Convert the Groq SDK message object into a plain dict the API will
    reliably accept on the follow-up call. Passing the raw SDK object
    directly is fragile across SDK versions."""
    out = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def chat_with_memory(user_message: str, history: list, persona: Optional[str] = None) -> tuple[str, list, list]:
    """Non-streaming with tool use. Used by the Streamlit app."""
    messages = [{"role": "system", "content": build_system_prompt(persona)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    forced_choice = _pick_forced_tool_choice(user_message)
    tool_choice = forced_choice if forced_choice else "auto"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=TOOLS,
        tool_choice=tool_choice,
        max_tokens=500,
        temperature=0.7,
    )

    msg = response.choices[0].message
    tool_calls_used = []

    if msg.tool_calls:
        messages.append(_serialize_assistant_msg(msg))
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