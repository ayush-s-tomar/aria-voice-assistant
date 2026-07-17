"""
Tool definitions and dispatch for ARIA's LLM tool-use pipeline.

Tools available:
  web_search      — Tavily web search (requires TAVILY_API_KEY)
  calculator      — Safe math expression evaluator (no API key)
  get_weather     — Live weather via wttr.in (free, no API key)
  wikipedia       — Wikipedia article summary (free, no API key)
  get_datetime    — Current date, time, day, timezone (no API key)
  unit_converter  — Convert between common units (no API key)
"""

import ast
import math
import operator
import os
import re
from datetime import datetime, timezone

import requests

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current events, news, prices, or any real-time information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use for arithmetic, percentages, or any numeric calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression to evaluate, e.g. '15% of 8500' or 'sqrt(144) + 20'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for any city or location. Returns temperature, conditions, humidity, and wind.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location, e.g. 'Mumbai' or 'New York'",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia",
            "description": "Get a concise summary of any topic from Wikipedia. Use for factual questions about people, places, events, or concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to look up on Wikipedia",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date, time, and day of the week. Use when the user asks what time or date it is.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Optional timezone name, e.g. 'Asia/Kolkata'. Defaults to UTC.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Convert between units of length, weight, temperature, speed, volume, or area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The numeric value to convert"},
                    "from_unit": {"type": "string", "description": "The unit to convert from, e.g. 'km', 'kg', 'celsius'"},
                    "to_unit": {"type": "string", "description": "The unit to convert to, e.g. 'miles', 'pounds', 'fahrenheit'"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]


def _web_search(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search is unavailable — TAVILY_API_KEY is not set."
    try:
        res = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 5},
            timeout=8,
        )
        res.raise_for_status()
        results = res.json().get("results", [])
        if not results:
            return f"No results found for '{query}'."
        lines = [f"- {r['title']}: {r['content'][:200]}" for r in results[:4]]
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def _calculator(expression: str) -> str:
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
                        r"(\1/100)*\2", expression, flags=re.IGNORECASE)
    expression = expression.replace("%", "/100")

    _safe_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    _safe_fns = {
        "sqrt": math.sqrt, "abs": abs, "round": round,
        "floor": math.floor, "ceil": math.ceil,
        "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in _safe_fns:
            return _safe_fns[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _safe_ops:
            return _safe_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _safe_ops:
            return _safe_ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call):
            fn = _eval(node.func)
            args = [_eval(a) for a in node.args]
            if callable(fn):
                return fn(*args)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


def _get_weather(location: str) -> str:
    try:
        res = requests.get(
            f"https://wttr.in/{requests.utils.quote(location)}?format=j1",
            timeout=6,
            headers={"User-Agent": "ARIA-VoiceAssistant/3.0"},
        )
        res.raise_for_status()
        data = res.json()
        current = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        city = area.get("areaName", [{}])[0].get("value", location)
        country = area.get("country", [{}])[0].get("value", "")

        temp_c = current["temp_C"]
        temp_f = current["temp_F"]
        feels_c = current["FeelsLikeC"]
        desc = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind_kmph = current["windspeedKmph"]
        visibility = current["visibility"]

        return (
            f"Weather in {city}, {country}: {desc}. "
            f"Temperature: {temp_c}°C ({temp_f}°F), feels like {feels_c}°C. "
            f"Humidity: {humidity}%, wind: {wind_kmph} km/h, visibility: {visibility} km."
        )
    except Exception as e:
        return f"Could not fetch weather for '{location}': {e}"


def _wikipedia(topic: str) -> str:
    try:
        res = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(topic)}",
            timeout=6,
            headers={"User-Agent": "ARIA-VoiceAssistant/3.0"},
        )
        if res.status_code == 404:
            search = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
                timeout=5,
            ).json()
            if search[1]:
                return _wikipedia(search[1][0])
            return f"No Wikipedia article found for '{topic}'."
        res.raise_for_status()
        data = res.json()
        extract = data.get("extract", "")
        title = data.get("title", topic)
        if len(extract) > 400:
            extract = extract[:400].rsplit(".", 1)[0] + "."
        return f"{title}: {extract}"
    except Exception as e:
        return f"Wikipedia lookup failed for '{topic}': {e}"


def _get_datetime(timezone_name: str = "UTC") -> str:
    try:
        import pytz
        tz = pytz.timezone(timezone_name)
        now = datetime.now(tz)
        tz_label = timezone_name
    except Exception:
        now = datetime.now(timezone.utc)
        tz_label = "UTC"

    return (
        f"Current date and time: {now.strftime('%A, %B %d, %Y')} "
        f"at {now.strftime('%I:%M %p')} ({tz_label})."
    )


_CONVERSIONS: dict[str, float] = {
    "m": 1, "metres": 1, "meter": 1, "meters": 1,
    "km": 1000, "kilometres": 1000, "kilometers": 1000,
    "cm": 0.01, "centimetres": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimetres": 0.001, "millimeters": 0.001,
    "miles": 1609.344, "mile": 1609.344,
    "yards": 0.9144, "yard": 0.9144, "yd": 0.9144,
    "feet": 0.3048, "foot": 0.3048, "ft": 0.3048,
    "inches": 0.0254, "inch": 0.0254, "in": 0.0254,
    "kg": 1, "kilograms": 1, "kilogram": 1,
    "g": 0.001, "grams": 0.001, "gram": 0.001,
    "mg": 0.000001, "milligrams": 0.000001,
    "lbs": 0.453592, "lb": 0.453592, "pounds": 0.453592, "pound": 0.453592,
    "oz": 0.0283495, "ounces": 0.0283495, "ounce": 0.0283495,
    "tonnes": 1000, "tonne": 1000, "ton": 907.185, "tons": 907.185,
    "m/s": 1, "ms": 1,
    "km/h": 0.277778, "kmh": 0.277778, "kph": 0.277778,
    "mph": 0.44704,
    "knots": 0.514444, "knot": 0.514444,
    "l": 1, "litre": 1, "litres": 1, "liter": 1, "liters": 1,
    "ml": 0.001, "millilitre": 0.001, "millilitres": 0.001,
    "gallon": 3.78541, "gallons": 3.78541,
    "pint": 0.473176, "pints": 0.473176,
    "cup": 0.236588, "cups": 0.236588,
    "fl oz": 0.0295735, "floz": 0.0295735,
    "m2": 1, "sqm": 1,
    "km2": 1e6, "sqkm": 1e6,
    "cm2": 0.0001, "sqcm": 0.0001,
    "ft2": 0.092903, "sqft": 0.092903,
    "acres": 4046.86, "acre": 4046.86,
    "hectares": 10000, "hectare": 10000, "ha": 10000,
}


def _unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()

    temp_aliases = {"celsius": "c", "centigrade": "c", "fahrenheit": "f", "kelvin": "k"}
    f = temp_aliases.get(f, f)
    t = temp_aliases.get(t, t)

    if f in ("c", "f", "k") or t in ("c", "f", "k"):
        if f == "c":   c = value
        elif f == "f": c = (value - 32) * 5 / 9
        elif f == "k": c = value - 273.15
        else:
            return f"Unknown temperature unit '{from_unit}'."
        if t == "c":   result = c
        elif t == "f": result = c * 9 / 5 + 32
        elif t == "k": result = c + 273.15
        else:
            return f"Unknown temperature unit '{to_unit}'."
        return f"{value} {from_unit} = {round(result, 4)} {to_unit}"

    if f not in _CONVERSIONS:
        return f"Unknown unit '{from_unit}'."
    if t not in _CONVERSIONS:
        return f"Unknown unit '{to_unit}'."

    base_value = value * _CONVERSIONS[f]
    result = base_value / _CONVERSIONS[t]
    return f"{value} {from_unit} = {round(result, 6)} {to_unit}"


def run_tool(name: str, args: dict) -> str:
    try:
        if name == "web_search":
            return _web_search(args["query"])
        elif name == "calculator":
            return _calculator(args["expression"])
        elif name == "get_weather":
            return _get_weather(args["location"])
        elif name == "wikipedia":
            return _wikipedia(args["topic"])
        elif name == "get_datetime":
            return _get_datetime(args.get("timezone", "UTC"))
        elif name == "unit_converter":
            return _unit_converter(args["value"], args["from_unit"], args["to_unit"])
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool '{name}' failed: {e}"