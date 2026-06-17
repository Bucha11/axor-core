"""Unit tests for budget meter + usage extraction (no model)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import eval_budget as eb  # noqa: E402


def test_token_meter_accumulates_and_resets():
    m = eb.TokenMeter()
    m.add(100)
    m.add(50)
    assert m.total == 150
    m.reset()
    assert m.total == 0
    m.add(None)  # tolerate missing usage
    assert m.total == 0


def test_usage_openai_total_tokens():
    assert eb.usage_openai({"usage": {"total_tokens": 1234}}) == 1234


def test_usage_openai_sums_prompt_completion():
    assert eb.usage_openai({"usage": {"prompt_tokens": 800, "completion_tokens": 200}}) == 1000


def test_usage_openai_missing():
    assert eb.usage_openai({}) == 0


def test_usage_anthropic_sums_input_output():
    assert eb.usage_anthropic({"usage": {"input_tokens": 700, "output_tokens": 300}}) == 1000


def test_get_usage_reads_meter():
    m = eb.TokenMeter(total=4242)
    assert eb.make_get_usage(m)() == "4242"
