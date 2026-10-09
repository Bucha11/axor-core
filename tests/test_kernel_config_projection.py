"""`GovernanceConfig.as_kernel_config` — one declaration, two consumers.

The replay kernel re-gates a recorded run through the same pure predicates the
runtime used (architecture rule 0). That only holds if the config it re-gates
with is the deployment's own declaration rather than a hand-written echo of it —
so this pins the projection field by field, and pins the omissions as deliberate
rather than forgotten.
"""

from __future__ import annotations

import dataclasses

from axor_core.config import ExecutionMode, GovernanceConfig
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.kernel.labeler_fault import LabelerFault
from axor_core.kernel.replay import KernelConfig
from axor_core.policy.value_policy import ValuePredicate

# Fields a KernelConfig has that the projection deliberately leaves at default,
# each with the reason it is not part of an operator's declaration.
_INTENTIONAL_OMISSIONS = {
    "allowed_tools": "a session's capability table, not a declaration",
    "budget_cap_calls": "a session's spend",
    "budget_cap_cost": "a session's spend",
    "tool_weights": "a session's cost model",
    "default_tool_weight": "a session's cost model",
    "max_unattended_consequence": "rides on the execution envelope's policy",
    "synthetic_taint_refs": "a counterfactual knob, named per replay",
    "labeler_fault": "a counterfactual knob, named per replay",
    "labeler_fault_seed": "a counterfactual knob, named per replay",
}


def _full_config() -> GovernanceConfig:
    return GovernanceConfig(
        mode=ExecutionMode.STRICT,
        egress_sinks=frozenset({"send_email"}),
        imperative_sinks=frozenset({"notify"}),
        integrity_sinks=frozenset({"set_password"}),
        positional_sinks=frozenset({"transfer"}),
        value_policies={"send_email": [ValuePredicate(
            arg="to", kind="enum", allowed=frozenset({"ops@example.com"}),
        )]},
        driving_args={"send_email": ["to"]},
        consequence_overrides={"delete": ConsequenceClass.CATASTROPHIC},
    )


def test_every_gate_field_is_carried() -> None:
    governance = _full_config()
    kernel = governance.as_kernel_config()
    assert kernel.egress_sinks == frozenset({"send_email"})
    assert kernel.imperative_sinks == frozenset({"notify"})
    assert kernel.integrity_sinks == frozenset({"set_password"})
    assert kernel.positional_sinks == frozenset({"transfer"})
    assert kernel.value_policies == governance.value_policies
    # driving_args are frozensets on the kernel side (the gates do membership tests)
    assert kernel.driving_args == {"send_email": frozenset({"to"})}
    assert kernel.consequence_overrides == governance.consequence_overrides
    assert kernel.strict_consequence is True


def test_non_strict_mode_does_not_turn_on_strict_consequence() -> None:
    assert GovernanceConfig().as_kernel_config().strict_consequence is False


def test_omissions_are_enumerated_so_a_new_field_cannot_be_dropped_silently() -> None:
    """A new KernelConfig field must be either projected or listed as omitted.

    Without this, adding a gate input to KernelConfig would silently make every
    replay re-gate on its default — which is how a counterfactual starts quietly
    disagreeing with the run it claims to reproduce.
    """
    projected = {
        "egress_sinks", "imperative_sinks", "integrity_sinks", "positional_sinks",
        "value_policies", "driving_args", "consequence_overrides",
        "strict_consequence",
    }
    declared = {f.name for f in dataclasses.fields(KernelConfig)}
    assert declared == projected | set(_INTENTIONAL_OMISSIONS)


def test_counterfactual_knobs_pass_through() -> None:
    kernel = GovernanceConfig().as_kernel_config(
        labeler_fault=LabelerFault.ALL_TRUSTED,
        labeler_fault_seed="arm-1",
        allowed_tools=frozenset({"read"}),
    )
    assert kernel.labeler_fault is LabelerFault.ALL_TRUSTED
    assert kernel.labeler_fault_seed == "arm-1"
    assert kernel.allowed_tools == frozenset({"read"})
