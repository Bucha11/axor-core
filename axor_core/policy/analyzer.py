from __future__ import annotations

import logging

from axor_core.contracts.policy import SignalClassifier, TaskSignal
from axor_core.contracts.trace import (
    SignalChosenEvent,
    TraceEventKind,
)
from axor_core.policy.heuristic import HeuristicClassifier

_log = logging.getLogger("axor.policy.analyzer")

# If confidence is below this threshold and an external classifier
# is available — escalate. Otherwise use heuristic result as-is.
_ESCALATION_THRESHOLD = 0.75

# Domain detection keyword sets
_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "research": [
        "research", "literature", "papers", "survey", "summarize", "findings",
        "compare studies", "review", "bibliography", "citations", "academic",
        "analyze data", "dataset", "hypothesis", "evidence", "sources",
    ],
    "support": [
        "help me", "how do i", "why is", "what is", "explain", "i'm getting",
        "error:", "traceback", "doesn't work", "broken", "fix my", "issue with",
        "question", "confused", "understand", "guide me",
    ],
    "analysis": [
        "analyze", "metrics", "performance", "benchmark", "profil", "measure",
        "report", "dashboard", "visuali", "statistics", "trend", "pattern",
        "compare", "evaluate", "assess", "audit", "review the code",
    ],
    "coding": [
        "write", "implement", "refactor", "add", "create", "build", "test",
        "fix", "debug", "migrate", "update", "edit", "change", "modify",
    ],
}


def _raw_domain_scores(raw_input: str) -> dict[str, int]:
    """Keyword-match counts per domain. Used by both detection and telemetry."""
    text = raw_input.lower()
    counts: dict[str, int] = {d: 0 for d in _DOMAIN_SIGNALS}
    for domain, keywords in _DOMAIN_SIGNALS.items():
        for kw in keywords:
            if kw in text:
                counts[domain] += 1
    return counts


def _detect_domain(raw_input: str, agent_domain: str = "general") -> str:
    """
    Heuristic domain detection from raw input text.

    If agent has a declared domain (from AgentDefinition), that takes priority
    over task-level detection. Task detection is a fallback for GENERAL agents.

    Returns AgentDomain.value string.
    """
    if agent_domain != "general":
        return agent_domain

    counts = _raw_domain_scores(raw_input)
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else "coding"


def _detect_language(text: str) -> str:
    """
    Lightweight Unicode-block heuristic — no external deps, no tokenization.

    Counts codepoints in major script blocks and returns the dominant tag.
    Ties and unknowns fall back to "en".
    """
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if 0x0400 <= cp <= 0x04FF:
            counts["ru"] = counts.get("ru", 0) + 1
        elif 0x4E00 <= cp <= 0x9FFF:
            counts["zh"] = counts.get("zh", 0) + 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["ar"] = counts.get("ar", 0) + 1
        elif 0x0900 <= cp <= 0x097F:
            counts["hi"] = counts.get("hi", 0) + 1
        elif 0xAC00 <= cp <= 0xD7FF:
            counts["ko"] = counts.get("ko", 0) + 1
        elif 0x0590 <= cp <= 0x05FF:
            counts["he"] = counts.get("he", 0) + 1
    if not counts:
        return "en"
    return max(counts, key=lambda k: counts[k])


def _marginal_domain_scores(raw_input: str) -> dict[str, float]:
    """
    Normalized marginal distribution over domains, namespaced `domain.*`.
    Uniform when no keywords match.
    """
    counts = _raw_domain_scores(raw_input)
    total = sum(counts.values())
    if total == 0:
        uniform = 1.0 / len(counts)
        return {f"domain.{d}": uniform for d in counts}
    return {f"domain.{d}": c / total for d, c in counts.items()}


class TaskAnalyzer:
    """
    Orchestrates signal classification with domain detection.

    Pipeline:
        raw input
          → HeuristicClassifier       (always, 0ms, 0 tokens)
               confidence >= threshold → TaskSignal
               confidence <  threshold → escalate to external classifier
          → domain detection          (keyword-based, 0ms, 0 tokens)
          → TaskSignal (with domain)

    The external classifier is optional and injected at session
    construction time. Core never imports external classifiers.

        axor-classifier-local  — ML trained on user's own traces
        axor-classifier-cloud  — ML trained on anonymized traces from all users

    Both implement SignalClassifier. Core treats them identically.
    """

    def __init__(
        self,
        external_classifier: SignalClassifier | None = None,
        escalation_threshold: float = _ESCALATION_THRESHOLD,
        agent_domain: str = "general",
    ) -> None:
        self._heuristic    = HeuristicClassifier()
        self._external     = external_classifier
        self._threshold    = escalation_threshold
        self._agent_domain = agent_domain   # from AgentDefinition.domain

    @property
    def ambiguity_threshold(self) -> float:
        """The single source of the ambiguity decision: classifications below
        this confidence are ambiguous. Consumers (e.g. the session's adaptive
        narrowing) must read this property instead of hard-coding a number —
        the analyzer's configured threshold and the session's interpretation
        of confidence must never diverge."""
        return self._threshold

    async def analyze(self, raw_input: str) -> tuple[TaskSignal, SignalChosenEvent]:
        """
        Classify raw input into a TaskSignal with domain detection.
        Returns the signal and a trace event recording the decision.
        """
        signal, confidence, scores = await self._heuristic.classify_with_scores(raw_input)
        classifier_used = "heuristic"
        external_domain = ""

        if confidence < self._threshold and self._external is not None:
            # Fail-closed (classifier-threat-model.md): a broken external
            # classifier degrades to the heuristic result — it never crashes
            # governed execution.
            try:
                ext_signal, ext_conf, ext_scores = await self._external.classify_with_scores(raw_input)
            except Exception:
                _log.warning(
                    "external classifier %s raised; falling back to heuristic result",
                    type(self._external).__name__,
                    exc_info=True,
                )
            else:
                if ext_conf > confidence:
                    signal = ext_signal
                    confidence = ext_conf
                    classifier_used = type(self._external).__name__
                    external_domain = ext_signal.domain
                    # external scores replace heuristic scores only if provided;
                    # empty dict means external did not expose a distribution
                    if ext_scores:
                        scores = ext_scores

        # domain resolution priority: agent domain (operator-declared) →
        # external classifier's domain head (when it won the escalation and
        # produced a non-default domain) → keyword detection fallback.
        if self._agent_domain != "general":
            domain = self._agent_domain
        elif external_domain and external_domain != "general":
            domain = external_domain
        else:
            domain = _detect_domain(raw_input, self._agent_domain)
        # domain distribution always comes from the text heuristic; agent-domain
        # assignment is a policy override, not a probabilistic statement.
        scores = {**scores, **_marginal_domain_scores(raw_input)}

        language = _detect_language(raw_input)

        # rebuild signal with domain and language
        signal = TaskSignal(
            raw_input=signal.raw_input,
            complexity=signal.complexity,
            nature=signal.nature,
            estimated_scope=signal.estimated_scope,
            requires_children=signal.requires_children,
            requires_mutation=signal.requires_mutation,
            domain=domain,
            language=language,
        )

        event = SignalChosenEvent(
            kind=TraceEventKind.SIGNAL_CHOSEN,
            node_id="",
            sequence=0,
            raw_input=raw_input,
            signal=signal,
            confidence=confidence,
            classifier=classifier_used,
            scores=scores,
        )

        return signal, event
