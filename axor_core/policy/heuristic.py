from __future__ import annotations

import json
import re
from pathlib import Path

from axor_core.contracts.policy import (
    SignalClassifier,
    TaskSignal,
    TaskComplexity,
    TaskNature,
)


# ── Signal patterns ────────────────────────────────────────────────────────────
#
# Patterns and weights live in heuristic_coefficients.json so that telemetry-
# driven updates can ship new coefficients without a code release. Multiple
# matches accumulate weight — avoids single-keyword false positives.

_COEFFICIENTS_PATH = Path(__file__).parent / "heuristic_coefficients.json"

_REGEX_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def _compile_flags(spec: str) -> int:
    flags = 0
    for ch in spec or "":
        f = _REGEX_FLAGS.get(ch)
        if f is None:
            raise ValueError(f"Unknown regex flag {ch!r} in heuristic coefficients")
        flags |= f
    return flags


def _load_patterns() -> dict[str, list[tuple[re.Pattern[str], float]]]:
    with _COEFFICIENTS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[str, list[tuple[re.Pattern[str], float]]] = {}
    for group, entries in data["patterns"].items():
        out[group] = [
            (re.compile(entry["regex"], _compile_flags(entry.get("flags", ""))), float(entry["weight"]))
            for entry in entries
        ]
    return out


_PATTERNS = _load_patterns()

_EXPANSIVE_PATTERNS:  list[tuple[re.Pattern[str], float]] = _PATTERNS["complexity.expansive"]
_MODERATE_PATTERNS:   list[tuple[re.Pattern[str], float]] = _PATTERNS["complexity.moderate"]
_READONLY_PATTERNS:   list[tuple[re.Pattern[str], float]] = _PATTERNS["nature.readonly"]
_MUTATIVE_PATTERNS:   list[tuple[re.Pattern[str], float]] = _PATTERNS["nature.mutative"]
_GENERATIVE_PATTERNS: list[tuple[re.Pattern[str], float]] = _PATTERNS["nature.generative"]

# Contradiction signals — presence of both reduces confidence
_CONTRADICTION_PAIRS = [
    (_EXPANSIVE_PATTERNS, _READONLY_PATTERNS),
    (_MUTATIVE_PATTERNS, _READONLY_PATTERNS),
]

_CONFIDENCE_THRESHOLD = 0.75


def _score(text: str, patterns: list[tuple[re.Pattern, float]]) -> float:
    return sum(weight for pattern, weight in patterns if pattern.search(text))


def _marginal_complexity_scores(text: str) -> dict[str, float]:
    """
    Return a normalized marginal distribution over TaskComplexity.

    Focused has no patterns — it absorbs the residual mass when expansive
    and moderate do not fire strongly. Keys are namespaced `complexity.*`.
    """
    expansive = _score(text, _EXPANSIVE_PATTERNS)
    moderate  = _score(text, _MODERATE_PATTERNS)
    # focused residual: full mass when neither other class fires,
    # shrinks as stronger classes accumulate score
    focused = max(0.0, 1.0 - expansive - moderate)
    total = expansive + moderate + focused
    if total == 0:
        return {"complexity.focused": 1.0, "complexity.moderate": 0.0, "complexity.expansive": 0.0}
    return {
        "complexity.focused":   focused   / total,
        "complexity.moderate":  moderate  / total,
        "complexity.expansive": expansive / total,
    }


def _marginal_nature_scores(text: str) -> dict[str, float]:
    """
    Return a normalized marginal distribution over TaskNature.
    Keys are namespaced `nature.*`. Uniform when no patterns fire.
    """
    mutative   = _score(text, _MUTATIVE_PATTERNS)
    generative = _score(text, _GENERATIVE_PATTERNS)
    readonly   = _score(text, _READONLY_PATTERNS)
    total = mutative + generative + readonly
    if total == 0:
        return {"nature.readonly": 1/3, "nature.generative": 1/3, "nature.mutative": 1/3}
    return {
        "nature.readonly":   readonly   / total,
        "nature.generative": generative / total,
        "nature.mutative":   mutative   / total,
    }


class HeuristicClassifier(SignalClassifier):
    """
    Rule-based classifier. Always available in core. Zero tokens. Zero latency.

    Accuracy: ~65-70% on typical inputs.
    Falls back gracefully — when confidence is low, TaskAnalyzer
    escalates to an injected SignalClassifier (local or cloud ML model).

    This is the only classifier that ships with axor-core.
    Everything else is a separate package implementing SignalClassifier.
    """

    async def classify(self, raw_input: str) -> tuple[TaskSignal, float]:
        signal, confidence, _ = await self.classify_with_scores(raw_input)
        return signal, confidence

    async def classify_with_scores(
        self, raw_input: str
    ) -> tuple[TaskSignal, float, dict[str, float]]:
        text = raw_input.strip()

        complexity, complexity_conf = self._classify_complexity(text)
        nature, nature_conf = self._classify_nature(text)
        contradiction_penalty = self._contradiction_penalty(text)

        # overall confidence is weakest link, penalised by contradictions
        confidence = min(complexity_conf, nature_conf) - contradiction_penalty
        confidence = max(0.0, min(1.0, confidence))

        signal = TaskSignal(
            raw_input=raw_input,
            complexity=complexity,
            nature=nature,
            estimated_scope=self._estimate_scope(complexity),
            requires_children=complexity == TaskComplexity.EXPANSIVE,
            requires_mutation=nature == TaskNature.MUTATIVE,
        )

        scores = {
            **_marginal_complexity_scores(text),
            **_marginal_nature_scores(text),
        }
        return signal, confidence, scores

    def _classify_complexity(self, text: str) -> tuple[TaskComplexity, float]:
        expansive_score = _score(text, _EXPANSIVE_PATTERNS)
        moderate_score  = _score(text, _MODERATE_PATTERNS)

        if expansive_score >= 0.8:
            return TaskComplexity.EXPANSIVE, min(1.0, expansive_score / 1.0)
        if moderate_score >= 0.7:
            return TaskComplexity.MODERATE, min(1.0, moderate_score / 0.8)

        # default to FOCUSED — safest assumption for unknown inputs
        focused_conf = 0.6 if expansive_score == 0 and moderate_score == 0 else 0.5
        return TaskComplexity.FOCUSED, focused_conf

    def _classify_nature(self, text: str) -> tuple[TaskNature, float]:
        mutative_score   = _score(text, _MUTATIVE_PATTERNS)
        generative_score = _score(text, _GENERATIVE_PATTERNS)
        readonly_score   = _score(text, _READONLY_PATTERNS)

        scores = {
            TaskNature.MUTATIVE:   mutative_score,
            TaskNature.GENERATIVE: generative_score,
            TaskNature.READONLY:   readonly_score,
        }
        best = max(scores, key=lambda k: scores[k])
        best_score = scores[best]

        if best_score == 0:
            return TaskNature.GENERATIVE, 0.5  # safest default

        # normalize confidence against best possible score
        confidence = min(1.0, best_score / 1.5)
        return best, confidence

    def _contradiction_penalty(self, text: str) -> float:
        """
        When a text scores on contradicting dimensions (e.g. both
        readonly and mutative), confidence should drop.
        """
        penalty = 0.0
        for group_a, group_b in _CONTRADICTION_PAIRS:
            if _score(text, group_a) > 0 and _score(text, group_b) > 0:
                penalty += 0.2
        return penalty

    def _estimate_scope(self, complexity: TaskComplexity) -> int:
        return {
            TaskComplexity.FOCUSED:   1,
            TaskComplexity.MODERATE:  5,
            TaskComplexity.EXPANSIVE: 999,
        }[complexity]
