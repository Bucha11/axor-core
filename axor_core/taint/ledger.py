"""Per-value taint ledger — content-derivation provenance for an opaque
LLM-tool loop.

A fully sound per-value tracker would need an interpreter of the agent's data
flow. We do not have one (the model emits tool calls directly), so we track
provenance by **content derivation**: register the content a tainted/sensitive
read produced, and at a sink decide the driving argument's causal root by whether
it *contains* that content.

This is **sound in the deny direction** (if a registered tainted/sensitive
fragment appears in a sink argument, that argument really does carry it), so it
is wired as an *additional* deny layer on top of the coarse session floor — it
never loosens the floor. It is **incomplete**: a value the model paraphrases or
re-encodes will not match. Sound per-value enforcement that could *loosen* the
session floor would need a sound per-value interpreter backend; we do not claim
it.
"""

from __future__ import annotations

import re

from axor_core.taint.causal_root import CausalRoot

# Minimum length of a distinctive fragment to track. Shorter → catches more
# (safe direction: over-deny) but more coincidental matches; this is a heuristic.
_MIN_SEGMENT = 12
# Punctuation stripped from the ENDS of a whitespace token (never the middle) so a
# source-side "x@y.z." or "'x@y.z'" still matches the clean "x@y.z" a sink extracts.
_EDGE_PUNCT = ".,;:!?'\"`()[]{}<>«»…|*"
# Bounds so a huge read cannot blow up memory / match cost.
_MAX_SEGMENTS_PER_REGISTER = 256
_MAX_TOTAL_SEGMENTS = 20000


class ValueTaintLedger:
    """Maps distinctive content fragments → [CausalRoot, refcount].

    Refcounted: a fragment shared by two registered values is held until BOTH
    release it, so endorsing one value cannot under-taint the other. Saturation is
    fail-closed: when the segment cap is reached the ledger flips to a coarse
    over-taint mode — derive() returns an untrusted root for everything — so an
    attacker cannot flood the ledger to evict a real secret's fragment and launder
    it.
    """

    def __init__(self) -> None:
        # seg -> [joined CausalRoot, refcount]
        self._segments: dict[str, list] = {}
        self._saturated = False

    def register(self, content: object, root: CausalRoot) -> None:
        """Record that `content` came from a value with the given causal_root."""
        if not root.is_tainted and not root.sensitive:
            return
        for seg in self._segmentize(content)[:_MAX_SEGMENTS_PER_REGISTER]:
            entry = self._segments.get(seg)
            if entry is not None:
                entry[0] = CausalRoot.mint(entry[0], root)
                entry[1] += 1
            elif len(self._segments) < _MAX_TOTAL_SEGMENTS:
                self._segments[seg] = [root, 1]
            else:
                self._saturated = True  # fail-closed, not silent drop

    def merge(self, other: "ValueTaintLedger") -> None:
        """Fold another ledger's fragments into this one (parent → child spawn) —
        the per-value analog of taint-source inheritance. Deterministic order so a
        near-cap merge drops the same fragments regardless of dict/hash seed."""
        self._saturated = self._saturated or other._saturated
        for seg, entry in sorted(other._segments.items()):
            existing = self._segments.get(seg)
            if existing is not None:
                existing[0] = CausalRoot.mint(existing[0], entry[0])
                existing[1] += entry[1]
            elif len(self._segments) < _MAX_TOTAL_SEGMENTS:
                self._segments[seg] = [entry[0], entry[1]]
            else:
                self._saturated = True

    def unregister(self, content: object) -> int:
        """Release one reference to the fragments of `content` (endorsement).
        A fragment is removed only when its refcount reaches zero, so a fragment
        shared with another live value survives. The retained joined root keeps the
        other value at least as tainted (safe direction). Returns fragments removed."""
        removed = 0
        for seg in self._segmentize(content):
            entry = self._segments.get(seg)
            if entry is not None:
                entry[1] -= 1
                if entry[1] <= 0:
                    del self._segments[seg]
                    removed += 1
        return removed

    def derive(self, value: object) -> CausalRoot:
        """Return the joined causal_root of every registered fragment that the
        given value contains. Constant (clean) if none — the per-value win:
        a clean argument carries no taint even in a tainted session.

        When saturated (ledger flooded past the cap) this fails closed: every value
        derives as untrusted (the same maximal taint applied at a process boundary),
        so a flood cannot launder a tracked value into looking clean.
        """
        if self._saturated:
            return CausalRoot.cross_process_in()
        text = self._flatten(value)
        if not text or not self._segments:
            return CausalRoot.constant()
        root = CausalRoot.constant()
        for seg, entry in self._segments.items():
            if seg in text:
                root = CausalRoot.mint(root, entry[0])
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
            # Also emit the token with surrounding punctuation stripped. A source
            # writes an attacker identifier adjacent to punctuation ("Relay: x@y.z."
            # or "'x@y.z'"), but the model extracts the clean token into the sink
            # argument ("x@y.z"). Whitespace-only tokenisation would keep the
            # trailing "." / quote and miss the substring match. Stripping only the
            # ENDS preserves internal punctuation (an email's dots, an IBAN, a URL).
            stripped = tok.strip(_EDGE_PUNCT)
            if stripped != tok and len(stripped) >= _MIN_SEGMENT:
                segs.add(stripped)
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
