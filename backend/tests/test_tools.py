"""Unit tests for pure-logic tool functions — no API keys or network needed."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.tools import _calculator, _unit_converter


def test_calculator_basic_arithmetic():
    assert "20" in _calculator("15 + 5")


def test_calculator_percentage():
    result = _calculator("15% of 8500")
    assert "1275" in result


def test_calculator_sqrt():
    result = _calculator("sqrt(144) + 20")
    assert "32" in result


def test_calculator_invalid_expression_does_not_crash():
    result = _calculator("this is not math")
    assert "Could not evaluate" in result


def test_unit_converter_km_to_miles():
    result = _unit_converter(100, "km", "miles")
    assert "62" in result


def test_unit_converter_celsius_to_fahrenheit():
    result = _unit_converter(37, "celsius", "fahrenheit")
    assert "98" in result


def test_unit_converter_unknown_unit():
    result = _unit_converter(5, "bananas", "km")
    assert "Unknown unit" in result