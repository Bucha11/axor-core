"""Ceiling preservation across the authority-composition path.

The composition path must never hand back an authority wider than the trusted
operand it was composed against: for every composition step the result is the
MEET of the two inputs, per field. `tests/invariants/test_regime_lattice_laws.py`
pins the order-theoretic laws for the fields it exercises; this module pins the
field that escaped them — ``max_unattended_consequence``, the consequence gate's
unattended ceiling (`node/intent_loop.py::_check_consequence`).

Both regressions here failed before the fix:

  * the deployment overlay ASSIGNED its profile ceiling, so a looser profile
    ("balanced" = CONSEQUENTIAL) widened a stricter per-task policy (REVERSIBLE);
  * ``apply_parent_restrictions`` did not carry the field at all, so a
    classifier-selected child — which always carries the dataclass default,
    CONSEQUENTIAL, since no preset sets it — out-ranked a parent pinned to
    REVERSIBLE and skipped the governance gate the parent requires.
    ``_validate_child_policy`` did not check it either, so nothing caught it.
"""
from __future__ import annotations

import itertools

import pytest

from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy
from axor_core.errors.exceptions import SpawnValidationError
from axor_core.node.spawn import _validate_child_policy
from axor_core.policy import PolicyComposer

_CLASSES = tuple(ConsequenceClass)


def _policy(name: str, ceiling: ConsequenceClass, **kw) -> ExecutionPolicy:
    return ExecutionPolicy(
        name=name,
        max_unattended_consequence=ceiling,
        tool_policy=kw.pop("tool_policy", ToolPolicy(allow_read=True)),
        **kw,
    )


class TestDeploymentOverlayCeiling:
    """The overlay is a ceiling, not an assignment (composer docstring)."""

    @pytest.mark.parametrize("policy_c,overlay_c", list(itertools.product(_CLASSES, _CLASSES)))
    def test_overlay_never_widens(self, policy_c, overlay_c):
        composed = PolicyComposer(consequence_ceiling=overlay_c).compose(
            _policy("task", policy_c), []
        )
        assert composed.max_unattended_consequence == min(policy_c, overlay_c)
        assert composed.max_unattended_consequence <= policy_c
        assert composed.max_unattended_consequence <= overlay_c

    def test_loose_profile_does_not_widen_strict_task_policy(self):
        """The shipped default ("balanced" = CONSEQUENTIAL) over an operator's own
        REVERSIBLE policy: the stricter per-task value must survive."""
        composed = PolicyComposer(
            consequence_ceiling=ConsequenceClass.CONSEQUENTIAL
        ).compose(_policy("task", ConsequenceClass.REVERSIBLE), [])
        assert composed.max_unattended_consequence == ConsequenceClass.REVERSIBLE

    def test_strict_profile_still_narrows(self):
        composed = PolicyComposer(
            consequence_ceiling=ConsequenceClass.REVERSIBLE
        ).compose(_policy("task", ConsequenceClass.CATASTROPHIC), [])
        assert composed.max_unattended_consequence == ConsequenceClass.REVERSIBLE

    def test_no_overlay_leaves_the_field_untouched(self):
        base = _policy("task", ConsequenceClass.BENIGN)
        assert (
            PolicyComposer().compose(base, []).max_unattended_consequence
            == ConsequenceClass.BENIGN
        )


class TestParentCeiling:
    """A child never out-ranks its parent on the consequence axis."""

    @pytest.mark.parametrize("child_c,parent_c", list(itertools.product(_CLASSES, _CLASSES)))
    def test_parent_restriction_is_a_meet(self, child_c, parent_c):
        narrowed = PolicyComposer().apply_parent_restrictions(
            _policy("child", child_c), _policy("parent", parent_c)
        )
        assert narrowed.max_unattended_consequence == min(child_c, parent_c)

    def test_classifier_default_child_under_strict_parent(self):
        """A selector-produced policy sets no ceiling, so it carries the dataclass
        default (CONSEQUENTIAL). Composed under a REVERSIBLE parent it must come
        back REVERSIBLE — this is the spawn path (`ChildSpawner` → child's own
        `GovernedNode.run` → `compose(parent_policy=...)`)."""
        child = ExecutionPolicy(name="focused_generative")
        assert child.max_unattended_consequence == ConsequenceClass.CONSEQUENTIAL
        composed = PolicyComposer().compose(
            child, [], parent_policy=_policy("parent", ConsequenceClass.REVERSIBLE)
        )
        assert composed.max_unattended_consequence == ConsequenceClass.REVERSIBLE
        # and the defensive validator accepts what compose() produced
        _validate_child_policy(
            composed, _policy("parent", ConsequenceClass.REVERSIBLE), child_depth=0
        )

    def test_validator_rejects_an_over_ceiling_child(self):
        with pytest.raises(SpawnValidationError, match="max_unattended_consequence"):
            _validate_child_policy(
                _policy("child", ConsequenceClass.CATASTROPHIC),
                _policy("parent", ConsequenceClass.REVERSIBLE),
                child_depth=0,
            )

    def test_full_compose_preserves_the_parent_ceiling_under_an_overlay(self):
        """Overlay then parent restriction: the result is the meet of all three."""
        composed = PolicyComposer(
            consequence_ceiling=ConsequenceClass.CATASTROPHIC
        ).compose(
            ExecutionPolicy(name="child"),
            [],
            parent_policy=_policy("parent", ConsequenceClass.BENIGN),
        )
        assert composed.max_unattended_consequence == ConsequenceClass.BENIGN
