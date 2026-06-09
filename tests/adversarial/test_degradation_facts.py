"""TM8 fact-driven degradation contract (v4.12 Phase 3).

Pins the counters-out / facts-in change: transitions are driven by decidable
facts (parameter-free Booleans), never by accumulation against a threshold.
"""

from __future__ import annotations

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.degradation import DegradationLevel, DegradationPolicy
from axor_core.contracts.denial import DenialResponse
from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot
from axor_core.degradation.engine import DegradationEngine


def _ni(tool="read", operation="file_read", provenance="repo", **kw):
    base = dict(
        tool=tool, operation=operation, target_kind="workdir", destination_kind="none",
        provenance=provenance, reads_secret_like_data=False, writes_outside_workdir=False,
        executes_generated_code=False, after_external_read=False, after_secret_access=False,
        data_flow="none",
    )
    base.update(kw)
    return NormalizedIntent(**base)


def _deny():
    return DenialResponse(status="denied", coarse_category="tool_denied")


def test_accumulation_of_benign_denials_does_not_escalate_past_cautious():
    """Many benign, clean, non-dangerous denials never reach RESTRICTED/LOCKED —
    a count is not a fact. (Old engine escalated to LOCKED at session_deny>=5.)
    """
    engine = DegradationEngine(DegradationPolicy())
    for _ in range(10):
        engine.record_signal(_ni(), _deny())  # clean driving value, benign tool
    assert engine.state.level == DegradationLevel.CAUTIOUS
    assert engine.state.session_deny_count == 10  # counter recorded as telemetry only


def test_single_untrusted_dangerous_deny_restricts_immediately():
    """One deny on a tainted/dangerous root → RESTRICTED on the FIRST occurrence
    (no pressure-count threshold of 2)."""
    engine = DegradationEngine(DegradationPolicy())
    driving = CausalRoot.external_read(TaintSource.WEB)  # the driving value is tainted
    t = engine.record_signal(_ni(tool="bash", operation="execute"), _deny(), driving_root=driving)
    assert engine.state.level == DegradationLevel.RESTRICTED
    assert t is not None and t.new_level == DegradationLevel.RESTRICTED


def test_tool_pressure_counter_kept_as_telemetry_not_driver():
    """The counter still increments (telemetry) but is not what caused the
    transition — the fact (untrusted dangerous deny) is."""
    engine = DegradationEngine(DegradationPolicy())
    driving = CausalRoot.external_read(TaintSource.WEB)
    engine.record_signal(_ni(tool="bash", operation="execute"), _deny(), driving_root=driving)
    src = list(engine.state.sources.values())[0]
    assert src.tool_pressure_count >= 1
