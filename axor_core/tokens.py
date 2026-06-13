"""Token estimation — the single heuristic used across the kernel.

Several subsystems need a rough token count for text they do not tokenize (budget
projections, context fragments, export sizing). They previously each inlined
``len(...) // 4``; centralising it keeps the estimate consistent and makes the
heuristic replaceable in one place if a real tokenizer is ever wired in.
"""

from __future__ import annotations

# Average characters per token for English-ish text — the standard rough ratio.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: object) -> int:
    """Rough token estimate for `text` (~4 chars/token). Accepts any object;
    non-strings are stringified. Empty / falsy → 0."""
    if not text:
        return 0
    return len(str(text)) // _CHARS_PER_TOKEN
