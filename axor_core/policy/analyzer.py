from __future__ import annotations

from axor_core.contracts.policy import SignalClassifier, TaskSignal
from axor_core.contracts.trace import (
    SignalChosenEvent,
    TraceEventKind,
)
from axor_core.policy.heuristic import HeuristicClassifier

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


def _detect_domain(raw_input: str, agent_domain: str = "general") -> str:
    """
    Heuristic domain detection from raw input text.

    If agent has a declared domain (from AgentDefinition), that takes priority
    over task-level detection. Task detection is a fallback for GENERAL agents.

    Returns AgentDomain.value string.
    """
    if agent_domain != "general":
        return agent_domain

    text = raw_input.lower()
    scores: dict[str, int] = {d: 0 for d in _DOMAIN_SIGNALS}

    for domain, keywords in _DOMAIN_SIGNALS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1

    # return highest scoring domain, default to "coding"
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "coding"


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

    async def analyze(self, raw_input: str) -> tuple[TaskSignal, SignalChosenEvent]:
        """
        Classify raw input into a TaskSignal with domain detection.
        Returns the signal and a trace event recording the decision.
        """
        signal, confidence = await self._heuristic.classify(raw_input)
        classifier_used = "heuristic"

        if confidence < self._threshold and self._external is not None:
            external_signal, external_confidence = await self._external.classify(raw_input)
            if external_confidence > confidence:
                signal = external_signal
                confidence = external_confidence
                classifier_used = type(self._external).__name__

        # domain detection — agent domain takes priority over task-level
        domain = _detect_domain(raw_input, self._agent_domain)

        # rebuild signal with domain
        signal = TaskSignal(
            raw_input=signal.raw_input,
            complexity=signal.complexity,
            nature=signal.nature,
            estimated_scope=signal.estimated_scope,
            requires_children=signal.requires_children,
            requires_mutation=signal.requires_mutation,
            domain=domain,
        )

        event = SignalChosenEvent(
            kind=TraceEventKind.SIGNAL_CHOSEN,
            node_id="",
            sequence=0,
            raw_input=raw_input,
            signal=signal,
            confidence=confidence,
            classifier=classifier_used,
        )

        return signal, event
