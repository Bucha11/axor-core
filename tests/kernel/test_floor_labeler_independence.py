"""C2's regression: the confidentiality floor does not depend on the integrity labeler.

The CLAIM is an argument about a signature, not about these runs:
:func:`axor_core.policy.gates.confidentiality_risk` does not take
``driving_root``, so no output of the integrity labeler — right, wrong, or
adversarial — reaches the confidentiality decision. The floor is armed at READ
time from a declared sensitive source and lifted only by an unforgeable
``GovernanceAuthority``; ``derive_value`` appears in neither path.

What this file adds is the regression that keeps the implementation matching the
argument: replay the same recorded traces with the integrity labeler fault-
injected in every mode and check that the set of calls carrying confidentiality
risk never moves.

The invariant is stated over the PREDICATE, deliberately, and not over denial
counts or categories. Integrity and confidentiality are checked in the same gate
(``taint_gate``), so a call both axes refuse is recorded under the integrity
axis; silencing integrity makes that same call surface as a *confidentiality*
denial. Confidentiality denials therefore RISE under ``ALL_TRUSTED`` while C2
holds exactly. ``test_overlapping_deny_changes_axis_without_changing_risk``
pins that trap open so nobody re-states the invariant over counts.
"""

from __future__ import annotations

import inspect

import pytest

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.taint import TaintSource
from axor_core.kernel.events import Event, EventKind, Verdict
from axor_core.kernel.labeler_fault import LabelerFault, apply as apply_fault
from axor_core.kernel.replay import KernelConfig, replay
from axor_core.policy.gates import confidentiality_risk, taint_gate
from axor_core.taint.causal_root import CausalRoot

ALL_MODES = tuple(LabelerFault)
FAULT_MODES = tuple(m for m in LabelerFault if m is not LabelerFault.NONE)


def _ev(seq: int, kind: EventKind, verdict: Verdict | None = None, **payload) -> Event:
    return Event(
        seq=seq,
        node_id="n0",
        kind=kind,
        ts="2026-10-09T00:00:00Z",
        verdict=verdict,
        payload=payload,
    )


def _normalized(**over: object) -> NormalizedIntent:
    fields: dict[str, object] = {
        "tool": "t",
        "operation": "other",
        "target_kind": "workdir",
        "destination_kind": "none",
        "provenance": "user",
        "reads_secret_like_data": False,
        "writes_outside_workdir": False,
        "executes_generated_code": False,
        "after_external_read": False,
        "after_secret_access": False,
        "data_flow": "none",
    }
    fields.update(over)
    return NormalizedIntent(**fields)  # type: ignore[arg-type]


# ── The trace under test ──────────────────────────────────────────────────────
#
# One run that exercises every combination of the two axes, so a fault cannot
# hide behind gate overlap:
#
#   seq 1  fetch_doc    -> untrusted read (BEFORE the secret, so its own egress
#                          is not yet under the floor); every sink below drives
#                          on its value
#   seq 3  vault_get    -> declared sensitive read: arms the floor
#   seq 4  send_report    egress + tainted driving value  -> both axes
#   seq 5  send_digest    egress + clean driving value    -> confidentiality only
#   seq 6  set_password   integrity sink, not egress      -> integrity only
#   seq 7  bash           executes generated code         -> integrity only
#   seq 8  notify_team    egress + imperative + tainted FREE_TEXT -> denied by the
#                          CARRIER gate first, which is what makes the per-axis
#                          denial count move under fault injection
#   seq 9  summarize      neither                         -> no risk

_SENSITIVE_READ_REF = "v_secret"
_UNTRUSTED_REF = "v_doc"

_EGRESS_BOTH = 4
_EGRESS_CONF_ONLY = 5
_INTEGRITY_SINK = 6
_INTEGRITY_EXEC = 7
_CARRIER_FIRST = 8
_NO_RISK = 9

# Calls whose confidentiality predicate must be True in EVERY mode: the egress
# sinks firing while a secret read is outstanding. `fetch_doc` (seq 1) is egress
# too, but it fires before the floor arms — which is the point of the ordering.
EXPECTED_CONF_RISK = frozenset({_EGRESS_BOTH, _EGRESS_CONF_ONLY, _CARRIER_FIRST})
# Calls denied on the integrity axis alone — nothing else refuses them, so they
# are where the injection must visibly bite.
INTEGRITY_ONLY = frozenset({_INTEGRITY_SINK, _INTEGRITY_EXEC})
# floor_active, per step, in fold order. Armed by seq 3's recorded root and never
# lowered — identical in every fault mode.
EXPECTED_FLOOR = [False, False, False, True, True, True, True, True, True, True]


def _trace() -> list[Event]:
    return [
        _ev(0, EventKind.TOOL_CALL, Verdict.PASS, tool="fetch_doc",
            args={"url": "https://docs.example/x"}, arg_refs={},
            normalized={"operation": "network_request",
                        "destination_kind": "external_domain"}),
        _ev(1, EventKind.TOOL_RESULT, tool="fetch_doc", value_ref=_UNTRUSTED_REF,
            root={"sources": ["web"], "sensitive": False}),
        _ev(2, EventKind.TOOL_CALL, Verdict.PASS, tool="vault_get",
            args={"key": "deploy"}, arg_refs={}, normalized={"operation": "read"}),
        _ev(3, EventKind.TOOL_RESULT, tool="vault_get", value_ref=_SENSITIVE_READ_REF,
            root={"sources": ["file"], "sensitive": True}),
        _ev(_EGRESS_BOTH, EventKind.TOOL_CALL, Verdict.DENY, tool="send_report",
            args={"to": "ops@example.com", "body": "x"},
            arg_refs={"to": _UNTRUSTED_REF}, driving_args=["to"],
            normalized={"operation": "send"}),
        _ev(_EGRESS_CONF_ONLY, EventKind.TOOL_CALL, Verdict.DENY, tool="send_digest",
            args={"to": "ops@example.com", "body": "x"},
            arg_refs={}, driving_args=["to"],
            driving_root={"sources": [], "sensitive": False},
            normalized={"operation": "send"}),
        _ev(_INTEGRITY_SINK, EventKind.TOOL_CALL, Verdict.DENY, tool="set_password",
            args={"new": "p"}, arg_refs={"new": _UNTRUSTED_REF}, driving_args=["new"],
            normalized={"operation": "update"}),
        _ev(_INTEGRITY_EXEC, EventKind.TOOL_CALL, Verdict.DENY, tool="bash",
            args={"cmd": "./run.sh"}, arg_refs={"cmd": _UNTRUSTED_REF},
            normalized={"operation": "execute_generated_code",
                        "executes_generated_code": True}),
        _ev(_CARRIER_FIRST, EventKind.TOOL_CALL, Verdict.DENY, tool="notify_team",
            args={"task": "forward the attached report to finance"},
            arg_refs={"task": _UNTRUSTED_REF}, driving_args=["task"],
            normalized={"operation": "send"}),
        _ev(_NO_RISK, EventKind.TOOL_CALL, Verdict.PASS, tool="summarize",
            args={"text": "x"}, arg_refs={}, normalized={"operation": "other"}),
    ]


def _config(fault: LabelerFault = LabelerFault.NONE, seed: str = "s") -> KernelConfig:
    return KernelConfig(
        allowed_tools=frozenset({
            "fetch_doc", "vault_get", "send_report", "send_digest",
            "set_password", "bash", "notify_team", "summarize",
        }),
        # The operator's declaration — the attacker-independent ground of `exfil`.
        egress_sinks=frozenset({"send_report", "send_digest", "notify_team"}),
        integrity_sinks=frozenset({"set_password"}),
        imperative_sinks=frozenset({"notify_team"}),
        driving_args={"send_report": frozenset({"to"}),
                      "send_digest": frozenset({"to"}),
                      "set_password": frozenset({"new"}),
                      "notify_team": frozenset({"task"})},
        labeler_fault=fault,
        labeler_fault_seed=seed,
    )


def _steps(fault: LabelerFault, seed: str = "s"):
    return replay(_trace(), _config(fault, seed=seed)).steps


def _risk_set(fault: LabelerFault, seed: str = "s") -> frozenset[int]:
    return frozenset(s.event.seq for s in _steps(fault, seed) if s.confidentiality_risk)


def _denied(fault: LabelerFault) -> frozenset[int]:
    return frozenset(
        s.event.seq for s in _steps(fault) if s.reevaluated_verdict is Verdict.DENY
    )


def _category(fault: LabelerFault, seq: int) -> str:
    step = next(s for s in _steps(fault) if s.event.seq == seq)
    assert step.deny is not None, f"seq {seq} was not denied under {fault}"
    return step.deny.category


def _conf_denial_count(fault: LabelerFault) -> int:
    """Denials the kernel REPORTS on the confidentiality axis — the quantity that
    moves under fault injection, and therefore the wrong invariant."""
    return sum(
        1 for s in _steps(fault)
        if s.deny is not None and "confidentiality" in s.deny.reason
    )


# ── 1. The claim itself: a signature, not a measurement ───────────────────────

def test_confidentiality_predicate_does_not_take_the_labelers_output() -> None:
    """C2 as the type of the decision function: no provenance parameter at all.

    If a ``driving_root`` (or any CausalRoot) parameter is ever added here, the
    confidentiality axis becomes a function of the integrity labeler and the
    paper's §5 argument stops being true of the code.
    """
    params = inspect.signature(confidentiality_risk).parameters
    assert set(params) == {"tool_name", "normalized", "floor_active", "egress_sinks"}
    annotations = {n: p.annotation for n, p in params.items()}
    assert not any("CausalRoot" in str(a) for a in annotations.values())


@pytest.mark.parametrize("sources", [
    frozenset(),
    frozenset({TaintSource.WEB}),
    frozenset({TaintSource.FILE}),
    frozenset({TaintSource.UNKNOWN_EXTERNAL}),
    frozenset({TaintSource.WEB, TaintSource.FILE, TaintSource.UNKNOWN_EXTERNAL}),
])
@pytest.mark.parametrize("sensitive", [False, True])
@pytest.mark.parametrize("superseded", [False, True])
def test_floor_denial_holds_over_every_driving_root(
    sources: frozenset[TaintSource], sensitive: bool, superseded: bool
) -> None:
    """Exhaustive over what the labeler can say about the driving value.

    Includes ``integrity_superseded=True`` — the decidable-supersession path
    switches the integrity axis off entirely, and the floor must still deny.
    """
    deny = taint_gate(
        "send_report",
        _normalized(operation="send"),
        CausalRoot(sources=sources, sensitive=sensitive),
        floor_active=True,
        egress_sinks=frozenset({"send_report"}),
        integrity_superseded=superseded,
    )
    assert deny is not None
    assert "confidentiality" in deny.reason


# ── 2. The invariant, over traces, in every fault mode ────────────────────────

def test_control_arm_reproduces_the_recorded_run() -> None:
    """NONE is the identity: without it, a "no divergence" result means nothing."""
    result = replay(_trace(), _config(LabelerFault.NONE))
    assert result.first_divergence is None


@pytest.mark.parametrize("fault", ALL_MODES)
def test_confidentiality_risk_set_is_invariant_under_labeler_fault(
    fault: LabelerFault,
) -> None:
    assert _risk_set(fault) == EXPECTED_CONF_RISK


@pytest.mark.parametrize("fault", ALL_MODES)
def test_floor_state_is_invariant_under_labeler_fault(fault: LabelerFault) -> None:
    """The fold's floor is armed from the recorded read, not from a derivation."""
    assert [s.state.floor_active for s in _steps(fault)] == EXPECTED_FLOOR


@pytest.mark.parametrize("fault", FAULT_MODES)
@pytest.mark.parametrize("seed", ["a", "b", "zzz", ""])
def test_invariant_holds_across_seeds(fault: LabelerFault, seed: str) -> None:
    assert _risk_set(fault, seed=seed) == EXPECTED_CONF_RISK


# ── 3. The injection must actually bite ───────────────────────────────────────

def test_integrity_only_denials_vanish_when_the_labeler_goes_blind() -> None:
    """Without these cases a broken integrity labeler hides behind gate overlap.

    ``set_password`` and ``bash`` are refused on the integrity axis alone — no
    egress, so the floor never speaks for them. Under ``ALL_TRUSTED`` they must
    turn ALLOW, which is what proves the fault reached the labeler at all.
    """
    baseline = _denied(LabelerFault.NONE)
    assert INTEGRITY_ONLY <= baseline
    blind = _denied(LabelerFault.ALL_TRUSTED)
    assert not (INTEGRITY_ONLY & blind)
    # ...while the egress sinks stay denied, now on the floor alone.
    assert EXPECTED_CONF_RISK <= blind


def test_over_tainting_labeler_denies_more_but_moves_no_floor() -> None:
    """The other direction: the labeler invents taint everywhere."""
    paranoid = _denied(LabelerFault.ALL_TAINTED)
    assert INTEGRITY_ONLY <= paranoid
    assert _risk_set(LabelerFault.ALL_TAINTED) == EXPECTED_CONF_RISK


def test_axis_of_a_denial_moves_under_fault_while_the_predicate_does_not() -> None:
    """Why the invariant is the predicate and never the per-axis denial count.

    ``notify_team`` is an egress sink AND an imperative sink carrying a tainted
    FREE_TEXT value, so the CARRIER gate — which runs before ``taint_gate`` and
    does read the labeler's ``is_tainted`` — refuses it first; the kernel reports
    category ``carrier_gate``. Silence the labeler and the carrier gate goes
    quiet, so the same call falls through to the floor and is now reported as a
    *confidentiality* denial. The count of confidentiality denials RISES while C2
    holds exactly, which is why an invariant written over denial counts or
    categories would fail a system that is behaving correctly.
    """
    assert _category(LabelerFault.NONE, _CARRIER_FIRST) == "carrier_gate"
    assert _category(LabelerFault.ALL_TRUSTED, _CARRIER_FIRST) == "taint_enforcement"
    assert _conf_denial_count(LabelerFault.ALL_TRUSTED) > _conf_denial_count(
        LabelerFault.NONE
    )
    # ...and the predicate is unchanged, which is the claim.
    assert _risk_set(LabelerFault.ALL_TRUSTED) == _risk_set(LabelerFault.NONE)
    # The call stays denied throughout: the axis moved, the refusal did not.
    assert _CARRIER_FIRST in _denied(LabelerFault.NONE)
    assert _CARRIER_FIRST in _denied(LabelerFault.ALL_TRUSTED)


# ── 4. The fault's own contract: integrity axis only ──────────────────────────

@pytest.mark.parametrize("fault", ALL_MODES)
@pytest.mark.parametrize("root", [
    CausalRoot.constant(),
    CausalRoot.external_read(TaintSource.WEB),
    CausalRoot.external_read(TaintSource.FILE, sensitive=True),
    CausalRoot(sources=frozenset(), sensitive=True),
])
def test_fault_never_touches_the_confidentiality_label(
    fault: LabelerFault, root: CausalRoot
) -> None:
    """A mode that cleared ``sensitive`` would disarm the floor it is measuring."""
    assert apply_fault(fault, root, ref="v1", seed="s").sensitive is root.sensitive


def test_fault_is_deterministic_in_seed_and_ref() -> None:
    root = CausalRoot.external_read(TaintSource.WEB)
    first = apply_fault(LabelerFault.FLIP_BY_REF, root, ref="v7", seed="s")
    again = apply_fault(LabelerFault.FLIP_BY_REF, root, ref="v7", seed="s")
    assert first == again


def test_flip_by_ref_actually_flips_some_refs() -> None:
    """A coin that never lands is not a fault mode."""
    root = CausalRoot.external_read(TaintSource.WEB)
    labels = {
        apply_fault(LabelerFault.FLIP_BY_REF, root, ref=f"v{i}", seed="s").is_tainted
        for i in range(32)
    }
    assert labels == {True, False}
