"""
TaskAnalyzer resilience around the external (ML) classifier.

Invariants:
  1. A broken external classifier degrades to the heuristic result —
     it never crashes governed execution (classifier-threat-model.md).
  2. The external classifier's domain head is consumed, not discarded:
     agent domain > external domain > keyword detection.
"""
from __future__ import annotations

import pytest

from axor_core.contracts.policy import (
    SignalClassifier,
    TaskComplexity,
    TaskNature,
    TaskSignal,
)
from axor_core.policy.analyzer import TaskAnalyzer


def _signal(raw_input: str, domain: str = "general") -> TaskSignal:
    return TaskSignal(
        raw_input=raw_input,
        complexity=TaskComplexity.MODERATE,
        nature=TaskNature.MUTATIVE,
        estimated_scope=5,
        requires_children=False,
        requires_mutation=True,
        domain=domain,
    )


class _RaisingClassifier(SignalClassifier):
    async def classify(self, raw_input: str) -> tuple[TaskSignal, float]:
        raise RuntimeError("model file corrupted")


class _ConfidentClassifier(SignalClassifier):
    def __init__(self, domain: str = "general", confidence: float = 0.9) -> None:
        self._domain = domain
        self._confidence = confidence

    async def classify(self, raw_input: str) -> tuple[TaskSignal, float]:
        return _signal(raw_input, domain=self._domain), self._confidence


# Low heuristic signal on purpose: no complexity/nature keywords, so the
# analyzer always escalates to the external classifier.
_AMBIGUOUS_TASK = "sdrujghs wpoit nrvz qqle"


@pytest.mark.asyncio
async def test_external_classifier_exception_falls_back_to_heuristic():
    analyzer = TaskAnalyzer(external_classifier=_RaisingClassifier())

    signal, event = await analyzer.analyze(_AMBIGUOUS_TASK)

    assert signal is not None
    assert event.classifier == "heuristic"


@pytest.mark.asyncio
async def test_external_domain_head_is_used():
    analyzer = TaskAnalyzer(external_classifier=_ConfidentClassifier(domain="research"))

    signal, event = await analyzer.analyze(_AMBIGUOUS_TASK)

    assert event.classifier == "_ConfidentClassifier"
    assert signal.domain == "research"


@pytest.mark.asyncio
async def test_external_general_domain_falls_back_to_keywords():
    analyzer = TaskAnalyzer(external_classifier=_ConfidentClassifier(domain="general"))

    signal, _ = await analyzer.analyze("fix the bug in the parser and debug the test")

    # external won the escalation but declared no domain — keyword detection decides
    assert signal.domain == "coding"


@pytest.mark.asyncio
async def test_agent_domain_wins_over_external_domain():
    analyzer = TaskAnalyzer(
        external_classifier=_ConfidentClassifier(domain="research"),
        agent_domain="support",
    )

    signal, _ = await analyzer.analyze(_AMBIGUOUS_TASK)

    assert signal.domain == "support"


@pytest.mark.asyncio
async def test_low_confidence_external_does_not_override_heuristic():
    analyzer = TaskAnalyzer(external_classifier=_ConfidentClassifier(confidence=0.1))

    _, event = await analyzer.analyze(_AMBIGUOUS_TASK)

    assert event.classifier == "heuristic"
