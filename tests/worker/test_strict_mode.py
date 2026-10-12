"""
Tests for ExecutionMode STRICT enforcement.

STRICT mode is a superset of PRODUCTION mode.  These tests confirm that each
STRICT-only restriction is active and that PRODUCTION mode does NOT apply them.

STRICT-only restrictions:
  1. Classifier disabled — ignored even if provided.
  2. audit_required=True by default in TraceConfig.
  3. deny_on_ambiguity=True posture.
  4. strict_escalation=True — human escalation requires non-empty reason.
  5. Executor wrapped in LockedExecutor (same as PRODUCTION).
"""
from __future__ import annotations

from unittest.mock import MagicMock


from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.trace import TraceConfig
from axor_core.worker.session import GovernedSession


def _session(mode: ExecutionMode, **kwargs) -> GovernedSession:
    executor = MagicMock()
    cap_executor = CapabilityExecutor()
    return GovernedSession(
        executor=executor,
        capability_executor=cap_executor,
        mode=mode,
        **kwargs,
    )


# ── 1. Classifier disabled in STRICT mode ─────────────────────────────────────

def test_strict_classifier_ignored():
    """
    In STRICT mode, a provided classifier is discarded — policy is rule-based only.
    """
    classifier = MagicMock()
    session = _session(ExecutionMode.STRICT, classifier=classifier)
    # The analyzer's external_classifier must be None
    assert session._analyzer._external is None, (
        "STRICT mode must ignore the provided classifier"
    )


def test_production_classifier_accepted():
    """In PRODUCTION mode, a provided classifier is accepted."""
    classifier = MagicMock()
    session = _session(ExecutionMode.PRODUCTION, classifier=classifier)
    assert session._analyzer._external is classifier, (
        "PRODUCTION mode must accept the provided classifier"
    )


def test_library_classifier_accepted():
    """In LIBRARY mode, a provided classifier is accepted."""
    classifier = MagicMock()
    session = _session(ExecutionMode.LIBRARY, classifier=classifier)
    assert session._analyzer._external is classifier


# ── 2. audit_required=True in STRICT mode ─────────────────────────────────────

def test_strict_audit_required_default():
    """STRICT mode sets audit_required=True on TraceConfig by default."""
    session = _session(ExecutionMode.STRICT)
    assert session._trace_config.audit_required is True


def test_strict_audit_required_overrides_false():
    """
    STRICT mode forces audit_required=True even if caller passes audit_required=False.
    """
    cfg = TraceConfig(audit_required=False)
    session = _session(ExecutionMode.STRICT, trace_config=cfg)
    assert session._trace_config.audit_required is True


def test_production_audit_required_not_forced():
    """PRODUCTION mode does not force audit_required=True."""
    cfg = TraceConfig(audit_required=False)
    session = _session(ExecutionMode.PRODUCTION, trace_config=cfg)
    assert session._trace_config.audit_required is False


# ── 3. deny_on_ambiguity posture ──────────────────────────────────────────────

def test_strict_deny_on_ambiguity():
    """STRICT mode sets deny_on_ambiguity=True."""
    session = _session(ExecutionMode.STRICT)
    assert session._deny_on_ambiguity is True


def test_production_deny_on_ambiguity_not_set():
    """PRODUCTION mode does not set deny_on_ambiguity=True."""
    session = _session(ExecutionMode.PRODUCTION)
    assert session._deny_on_ambiguity is False


def test_library_deny_on_ambiguity_not_set():
    """LIBRARY mode does not set deny_on_ambiguity=True."""
    session = _session(ExecutionMode.LIBRARY)
    assert session._deny_on_ambiguity is False


# ── 4. strict_escalation: human escalation requires non-empty reason ──────────

def test_strict_escalation_flag_set():
    """STRICT mode sets strict_escalation=True."""
    session = _session(ExecutionMode.STRICT)
    assert session._strict_escalation is True


def test_production_strict_escalation_not_set():
    """PRODUCTION mode does not require strict escalation reason."""
    session = _session(ExecutionMode.PRODUCTION)
    assert session._strict_escalation is False


# ── 5. Executor is LockedExecutor in STRICT and PRODUCTION modes ───────────────

def test_strict_executor_is_locked():
    """STRICT mode wraps executor in LockedExecutor."""
    from axor_core.capability.locked import LockedExecutor
    session = _session(ExecutionMode.STRICT)
    assert isinstance(session._executor, LockedExecutor)


def test_production_executor_is_locked():
    """PRODUCTION mode also wraps executor in LockedExecutor."""
    from axor_core.capability.locked import LockedExecutor
    session = _session(ExecutionMode.PRODUCTION)
    assert isinstance(session._executor, LockedExecutor)


def test_library_executor_not_locked():
    """LIBRARY mode does NOT wrap executor in LockedExecutor."""
    from axor_core.capability.locked import LockedExecutor
    session = _session(ExecutionMode.LIBRARY)
    assert not isinstance(session._executor, LockedExecutor)


# ── 6. STRICT is superset of PRODUCTION, not a separate path ─────────────────

def test_strict_is_superset_of_production():
    """
    STRICT mode satisfies all PRODUCTION mode requirements plus its own.

    Verified by checking that all PRODUCTION-mode properties hold in STRICT.
    """
    from axor_core.capability.locked import LockedExecutor

    session = _session(ExecutionMode.STRICT)

    # PRODUCTION-level properties
    assert isinstance(session._executor, LockedExecutor), "STRICT must lock executor"
    assert session._mode == ExecutionMode.STRICT

    # STRICT-only properties
    assert session._deny_on_ambiguity is True
    assert session._strict_escalation is True
    assert session._trace_config.audit_required is True
    assert session._analyzer._external is None


# ── 6. STRICT will not take authority from task classification ────────────────
#
# STRICT is the mode that CLAIMS the guarantee, and the guarantee is that the
# model sits outside the trusted base of the decision. A policy derived from the
# task text does not satisfy it: the classifier reads text that in an agentic
# session is not reliably the principal's, and what the selector returns carries
# AUTHORITY (tool_policy, allowed_paths, the consequence ceiling), not just a
# plan. Disabling the external classifier (tests 1-3 above) removes the MODEL
# from that path; it does not make the remaining keyword heuristic an author of
# authority. So the configuration is refused rather than silently claimed.

import pytest

from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy
from axor_core.errors.exceptions import PolicyError


async def test_strict_refuses_to_derive_authority_from_classification():
    session = _session(ExecutionMode.STRICT)
    with pytest.raises(PolicyError, match="will not derive authority"):
        await session.run("summarise the quarterly numbers")


async def test_strict_with_a_default_policy_passes_the_guard():
    session = _session(
        ExecutionMode.STRICT,
        default_policy=ExecutionPolicy(
            name="operator", tool_policy=ToolPolicy(allow_read=True)
        ),
    )
    try:
        await session.run("summarise the quarterly numbers")
    except PolicyError as exc:  # pragma: no cover - the assertion is the point
        pytest.fail(f"guard fired although the operator supplied authority: {exc}")
    except Exception:
        pass  # the MagicMock executor fails further down; irrelevant here


async def test_strict_with_a_per_call_policy_passes_the_guard():
    session = _session(ExecutionMode.STRICT)
    try:
        await session.run(
            "summarise the quarterly numbers",
            policy=ExecutionPolicy(
                name="per_call", tool_policy=ToolPolicy(allow_read=True)
            ),
        )
    except PolicyError as exc:  # pragma: no cover
        pytest.fail(f"guard fired although the caller supplied authority: {exc}")
    except Exception:
        pass


async def test_production_still_classifies_and_only_warns():
    """The guard is scoped to the mode that claims the guarantee. PRODUCTION keeps
    the legacy behaviour (classification selects the policy) and warns once —
    which is also the honest statement that such a configuration is not the one
    the theorem covers."""
    session = _session(ExecutionMode.PRODUCTION)
    try:
        await session.run("summarise the quarterly numbers")
    except PolicyError as exc:  # pragma: no cover
        pytest.fail(f"PRODUCTION must not be refused by the STRICT guard: {exc}")
    except Exception:
        pass
