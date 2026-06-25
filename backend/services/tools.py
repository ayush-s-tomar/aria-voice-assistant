# backend/services/tools.py
import os
import math
import re
from tavily import TavilyClient

_tavily = None

def _get_tavily():
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily


def web_search(query: str) -> str:
    """Search the web and return a concise summary."""
    try:
        client = _get_tavily()
        result = client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=True,
        )
        # Tavily returns a direct answer + sources
        answer = result.get("answer", "")
        if answer:
            return f"Web search result: {answer}"
        # Fallback to first result snippet
        results = result.get("results", [])
        if results:
            return f"Web search result: {results[0].get('content', 'No result found')[:400]}"
        return "Web search returned no results."
    except Exception as e:
        return f"Web search failed: {str(e)}"


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Only allow safe characters
        safe = re.sub(r"[^0-9+\-*/()., %sqrtalogepicosintanbfx^]", "", expression.lower())
        # Replace common math words
        safe = safe.replace("^", "**").replace("√", "math.sqrt").replace("pi", "str(math.pi)")
        result = eval(safe, {"__builtins__": {}, "math": math})
        return f"Calculator result: {expression} = {result}"
    except Exception:
        return f"Could not calculate: {expression}"


# Tool definitions for LLM function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use when asked about "
                "news, weather, prices, recent events, sports scores, or anything "
                "that requires up-to-date information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression. Use for arithmetic, "
                "percentages, conversions, or any calculation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '15% of 8500' or '(23 * 4) / 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


def run_tool(name: str, args: dict) -> str:
    """Dispatch tool call by name."""
    if name == "web_search":
        return web_search(args.get("query", ""))
    elif name == "calculate":
        return calculate(args.get("expression", ""))
    return f"Unknown tool: {name}"