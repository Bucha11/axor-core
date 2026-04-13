from __future__ import annotations

import re
from axor_core.contracts.policy import (
    SignalClassifier,
    TaskSignal,
    TaskComplexity,
    TaskNature,
)


# ── Signal patterns ────────────────────────────────────────────────────────────
#
# Each pattern is (regex, weight).
# Multiple matches accumulate weight — avoids single-keyword false positives.

_EXPANSIVE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\brewrite\s+(entire|whole|full|the)\b", re.I), 1.0),
    (re.compile(r"\bmigrate\s+(entire|whole|full|the|all)\b", re.I), 1.0),
    (re.compile(r"\bport\s+(entire|whole|full|the|all)\b", re.I), 1.0),
    (re.compile(r"\brewrite\s+\w+\s+(to|in|into)\s+\w+", re.I), 0.8),  # rewrite X to Y
    (re.compile(r"\bmigrate\s+\w+\s+(to|from)\b", re.I), 0.8),
    (re.compile(r"\brefactor\s+(entire|whole|all|the\s+entire)\b", re.I), 0.9),
    (re.compile(r"\barchitecture\s+(overhaul|redesign|rework)\b", re.I), 1.0),
    (re.compile(r"\bconvert\s+(the\s+)?(entire|whole|full)\b", re.I), 0.9),
]

_MODERATE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\badd\s+(a\s+)?feature\b", re.I), 0.8),
    (re.compile(r"\bimplement\s+\w+", re.I), 0.6),
    (re.compile(r"\brefactor\s+\w+", re.I), 0.7),
    (re.compile(r"\bupdate\s+\w+", re.I), 0.5),
    (re.compile(r"\bextend\s+\w+", re.I), 0.6),
    (re.compile(r"\bintegrate\s+\w+", re.I), 0.7),
    (re.compile(r"\breplace\s+\w+\s+with\b", re.I), 0.7),
]

_READONLY_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bexplain\b", re.I), 0.9),
    (re.compile(r"\banalyze\b", re.I), 0.8),
    (re.compile(r"\bwhat\s+(is|are|does)\b", re.I), 0.7),
    (re.compile(r"\bhow\s+does\b", re.I), 0.7),
    (re.compile(r"\bwhy\s+(is|does|did)\b", re.I), 0.7),
    (re.compile(r"\bsummarize\b", re.I), 0.9),
    (re.compile(r"\breview\b", re.I), 0.6),
    (re.compile(r"\bcheck\s+(if|whether|that)\b", re.I), 0.6),
    (re.compile(r"\bfind\s+(all|the|where|any)\b", re.I), 0.5),
]

_MUTATIVE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bfix\b", re.I), 0.7),
    (re.compile(r"\bwrite\s+(a\s+)?(test|spec|function|class|module)\b", re.I), 0.8),
    (re.compile(r"\bcreate\b", re.I), 0.6),
    (re.compile(r"\bmodify\b", re.I), 0.8),
    (re.compile(r"\bdelete\b", re.I), 0.8),
    (re.compile(r"\bremove\b", re.I), 0.7),
    (re.compile(r"\brename\b", re.I), 0.8),
    (re.compile(r"\bmove\b", re.I), 0.5),
    (re.compile(r"\brefactor\b", re.I), 0.7),
    (re.compile(r"\brewrite\b", re.I), 0.9),
]

_GENERATIVE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bwrite\s+(a\s+)?(test|spec|mock|stub)\b", re.I), 0.9),
    (re.compile(r"\bgenerate\b", re.I), 0.8),
    (re.compile(r"\bcreate\s+(a\s+)?(new\s+)?(file|module|class|function)\b", re.I), 0.8),
    (re.compile(r"\bscaffold\b", re.I), 0.9),
    (re.compile(r"\bboilerplate\b", re.I), 0.9),
    (re.compile(r"\btemplate\b", re.I), 0.6),
]

# Contradiction signals — presence of both reduces confidence
_CONTRADICTION_PAIRS = [
    (_EXPANSIVE_PATTERNS, _READONLY_PATTERNS),
    (_MUTATIVE_PATTERNS, _READONLY_PATTERNS),
]

_CONFIDENCE_THRESHOLD = 0.75


def _score(text: str, patterns: list[tuple[re.Pattern, float]]) -> float:
    return sum(weight for pattern, weight in patterns if pattern.search(text))


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
        return signal, confidence

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
