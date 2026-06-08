"""Per-value taint ledger (TM2) — content-derivation provenance for an opaque
LLM-tool loop.

A sound CaMeL-style per-value tracker needs an interpreter of the agent's data
flow. We do not have one (the model emits tool calls directly), so we track
provenance by **content derivation**: register the content a tainted/sensitive
read produced, and at a sink decide the driving argument's causal_root by whether
it *contains* that content.

This is **sound in the deny direction** (if a registered tainted/sensitive
fragment appears in a sink argument, that argument really does carry it), so it
is wired as an *additional* deny layer on top of the session-sticky floor — it
never loosens the floor. It is **incomplete**: a value the model paraphrases or
re-encodes will not match (the X1 in-process-LLM over-taint gap). Sound per-value
enforcement that could *loosen* the session floor needs the interpreter we cede
to CaMeL; we do not claim it.
"""

from __future__ import annotations

import re

from axor_core.taint.causal_root import CausalRoot

# Minimum length of a distinctive fragment to track. Shorter → catches more
# (safe direction: over-deny) but more coincidental matches; this is a heuristic.
_MIN_SEGMENT = 12
# Bounds so a huge read cannot blow up memory / match cost.
_MAX_SEGMENTS_PER_REGISTER = 256
_MAX_TOTAL_SEGMENTS = 20000


class ValueTaintLedger:
    """Maps distinctive content fragments → the CausalRoot that produced them."""

    def __init__(self) -> None:
        self._segments: dict[str, CausalRoot] = {}

    def register(self, content: object, root: CausalRoot) -> None:
        """Record that `content` came from a value with the given causal_root."""
        if not root.is_tainted and not root.sensitive:
            return
        if len(self._segments) >= _MAX_TOTAL_SEGMENTS:
            return
        for seg in self._segmentize(content)[:_MAX_SEGMENTS_PER_REGISTER]:
            existing = self._segments.get(seg)
            self._segments[seg] = CausalRoot.mint(existing, root) if existing else root

    def derive(self, value: object) -> CausalRoot:
        """Return the joined causal_root of every registered fragment that the
        given value contains. Constant (clean) if none — the per-value win:
        a clean argument carries no taint even in a tainted session.
        """
        text = self._flatten(value)
        if not text or not self._segments:
            return CausalRoot.constant()
        root = CausalRoot.constant()
        for seg, seg_root in self._segments.items():
            if seg in text:
                root = CausalRoot.mint(root, seg_root)
        return root

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _segmentize(content: object) -> list[str]:
        s = ValueTaintLedger._flatten(content)
        segs: set[str] = set()
        for line in s.splitlines():
            line = line.strip()
            if len(line) >= _MIN_SEGMENT:
                segs.add(line)
        for tok in re.split(r"\s+", s):
            if len(tok) >= _MIN_SEGMENT:
                segs.add(tok)
        return list(segs)

    @staticmethod
    def _flatten(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            return " ".join(
                ValueTaintLedger._flatten(v) for v in list(value.keys()) + list(value.values())
            )
        if isinstance(value, (list, tuple, set)):
            return " ".join(ValueTaintLedger._flatten(v) for v in value)
        return str(value)
