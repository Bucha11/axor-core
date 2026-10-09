"""Fault injection on the INTEGRITY labeler, for counterfactual replay.

What this is for
----------------
The paper's claim C2 says that a defined part of the enforcement surface
survives an *arbitrary* error of the integrity labeler — not a measured
degradation under noise, but an invariant under any output the labeler could
produce. The argument is one of signature: :func:`axor_core.policy.gates.
confidentiality_risk` does not take ``driving_root``, so nothing the labeler
says reaches it. This module is the regression that keeps the argument honest
against the implementation: it replaces the labeler's verdict on every driving
value in a recorded trace and re-gates the run.

What a fault may and may not touch
----------------------------------
``CausalRoot`` carries BOTH axes in one frozen value — ``sources`` (integrity)
and ``sensitive`` (confidentiality) — and ``TaintEngine.register_value``
branches on each separately: ``is_tainted`` grows the context root,
``sensitive`` arms the confidentiality floor. An injection written as "return a
clean root" therefore silently erases the floor as well, and the experiment then
measures a system with no confidentiality enforcement rather than a system with
a broken integrity labeler.

So every mode here rewrites ``sources`` ONLY and carries ``sensitive`` through
untouched (:func:`apply`, asserted by ``_preserve``). The floor is not reachable
from here by construction for a second reason too: it is armed at READ time, in
the engine's outstanding-secrets registry, from the root
:func:`axor_core.policy.provenance.output_root` returns for a declared sensitive
source — never from ``derive_value``, which is the function these modes stand in
for. The replay fold mirrors that split: ``state.floor_active`` is set from the
recorded TOOL_RESULT root and is not derived from ``arg_refs``.

Determinism
-----------
The kernel must stay bit-reproducible (``tests/kernel/test_kernel_purity.py``
forbids ``random``), so :attr:`LabelerFault.FLIP_BY_REF` derives its per-value
coin from an FNV-1a hash of the seed and the value ref. Same trace, same seed,
same labels — a counterfactual anyone can re-run.
"""

from __future__ import annotations

from enum import Enum

from axor_core.taint.causal_root import CausalRoot, TaintSource

# FNV-1a 32-bit, written out because the kernel may not import hashlib-grade
# machinery and does not need it: this picks a side of a coin, it is not a MAC.
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF

# The source a fault attributes when it invents integrity taint. UNKNOWN_EXTERNAL
# is the same source the cross-process re-mint uses: maximal, unattributable.
_INVENTED_SOURCE = TaintSource.UNKNOWN_EXTERNAL


class LabelerFault(Enum):
    """How the integrity labeler misbehaves during a counterfactual replay.

    ``NONE`` is the control arm, not a fault: it is the identity, so a replay
    under it must reproduce the recorded run exactly.
    """

    NONE = "none"
    # Every value reads back trusted — the labeler is blind to taint. This is the
    # mode C2 is about: all integrity gating goes silent at once.
    ALL_TRUSTED = "all_trusted"
    # Every value reads back maximally tainted — the labeler over-taints. Catches
    # anything that (wrongly) relies on a value deriving CLEAN to stay permitted.
    ALL_TAINTED = "all_tainted"
    # The label is inverted per value: clean becomes tainted and vice versa.
    INVERTED = "inverted"
    # The label is replaced by a deterministic per-ref coin flip — a labeler that
    # is neither blind nor paranoid but simply wrong at a bounded rate.
    FLIP_BY_REF = "flip_by_ref"


def _fnv1a(text: str) -> int:
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * _FNV_PRIME) & _MASK32
    return h


def _coin(seed: str, ref: str) -> bool:
    """Deterministic per-(seed, ref) coin. No clock, no RNG state."""
    return bool(_fnv1a(f"{seed}\x1f{ref}") & 1)


def _tainted(root: CausalRoot) -> CausalRoot:
    """Maximal integrity taint, confidentiality carried through."""
    return CausalRoot(
        sources=frozenset({_INVENTED_SOURCE}), sensitive=root.sensitive
    )


def _trusted(root: CausalRoot) -> CausalRoot:
    """No integrity sources, confidentiality carried through.

    Note the asymmetry with ``CausalRoot.constant()``: a constant is clean on
    both axes, and using one here would clear the secret label the trace
    recorded. ``is_tainted`` is ``bool(sources)``, so a root that is untainted
    but sensitive is representable and is exactly what this mode needs.
    """
    return CausalRoot(sources=frozenset(), sensitive=root.sensitive)


def apply(
    fault: LabelerFault, root: CausalRoot, *, ref: str = "", seed: str = ""
) -> CausalRoot:
    """The root the gates see instead of what the labeler derived.

    ``ref`` identifies the value for the per-ref modes (a value ref, or the
    recorded call's tool name when no ref resolved); ``seed`` names the
    counterfactual arm. Confidentiality is invariant: ``apply(...).sensitive ==
    root.sensitive`` for every mode, which is what keeps a fault injection on the
    integrity axis from quietly disarming the floor.
    """
    if fault is LabelerFault.NONE:
        return root
    if fault is LabelerFault.ALL_TRUSTED:
        return _preserve(root, _trusted(root))
    if fault is LabelerFault.ALL_TAINTED:
        return _preserve(root, _tainted(root))
    if fault is LabelerFault.INVERTED:
        return _preserve(
            root, _trusted(root) if root.is_tainted else _tainted(root)
        )
    if fault is LabelerFault.FLIP_BY_REF:
        flipped = _trusted(root) if root.is_tainted else _tainted(root)
        return _preserve(root, flipped if _coin(seed, ref) else root)
    raise AssertionError(f"unhandled labeler fault {fault!r}")


def _preserve(original: CausalRoot, faulted: CausalRoot) -> CausalRoot:
    """Guard the one invariant every mode owes the experiment.

    A mode that dropped ``sensitive`` would disarm the confidentiality floor
    along with the integrity label, and the run would then show C2 holding for a
    system that has no confidentiality enforcement left to hold.
    """
    if faulted.sensitive != original.sensitive:
        raise AssertionError(
            "labeler fault changed the confidentiality label "
            f"({original.sensitive} -> {faulted.sensitive}); faults are "
            "integrity-only by contract"
        )
    return faulted
