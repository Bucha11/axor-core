"""
Replays the OpenClaw trace through a REAL DegradationEngine and records, per
step, exactly what the engine did. No engine behavior is faked — every row is
read back from engine state and drained trace events after a real
`record_signal` call.

Two configurations:

  Config A — baseline (generic only). The engine as it ships.

  Config B — generic + a domain predicate (`privileged_shutdown`). The predicate
             is an ILLUSTRATION layered *on top* of an unmodified engine; it is
             not a core feature. It marks restart/shutdown as high-stakes and
             raises an *effective* level independent of pressure.

The "signal fired" column is derived from real engine state deltas (session
deny-count and per-source pressure counters), not re-implemented heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from axor_core.contracts.degradation import DegradationLevel, DegradationPolicy
from axor_core.contracts.denial import DenialResponse
from axor_core.contracts.policy import (
    ExecutionPolicy,
    ExportMode,
    ToolPolicy,
)
from axor_core.contracts.taint import TaintState
from axor_core.degradation.engine import DegradationEngine

from .scenario import Step, openclaw_trace

# A deliberately permissive base policy so that any narrowing the engine applies
# is visible in the report (bash/write allowed, full export).
BASE_POLICY = ExecutionPolicy(
    name="openclaw-base",
    tool_policy=ToolPolicy(allow_read=True, allow_write=True, allow_bash=True,
                           allow_search=True),
    export_mode=ExportMode.FULL,
)

# Tools the domain layer treats as privileged-shutdown class. ILLUSTRATION ONLY.
_PRIVILEGED_SHUTDOWN_TOOLS = frozenset({"restart_gateway", "shutdown"})


def privileged_shutdown(step: Step) -> bool:
    """Domain predicate (Config B). Illustration, not a shipped feature.

    A privileged shutdown/restart is high-stakes regardless of whether it was
    denied or generated any tool/instruction pressure.
    """
    return step.intent.tool.lower() in _PRIVILEGED_SHUTDOWN_TOOLS


@dataclass
class TraceRow:
    step: str
    intent_flags: str
    signal_fired: str
    deny_count: int
    level: str
    transition: str
    policy_narrowing: str


def _intent_flags(step: Step) -> str:
    flags = []
    if step.intent.executes_generated_code:
        flags.append("executes_generated_code")
    if step.intent.after_external_read:
        flags.append("after_external_read")
    flags.append("denied" if step.denied else "not-denied")
    return ", ".join(flags)


def _policy_narrowing(engine: DegradationEngine) -> str:
    """Diff the engine-narrowed policy against BASE_POLICY for the report."""
    narrowed = engine.apply_to_policy(BASE_POLICY, source_id=None)
    changes = []
    if narrowed.tool_policy.allow_bash != BASE_POLICY.tool_policy.allow_bash:
        changes.append("allow_bash→False")
    if narrowed.tool_policy.allow_write != BASE_POLICY.tool_policy.allow_write:
        changes.append("allow_write→False")
    if narrowed.export_mode != BASE_POLICY.export_mode:
        changes.append(f"export_mode→{narrowed.export_mode.value}")
    return ", ".join(changes) if changes else "—"


def _signal_fired(engine: DegradationEngine, step: Step, taint: TaintState,
                  deny_before: int) -> str:
    """Report which signal fired, read from real engine state deltas."""
    if engine.state.session_deny_count == deny_before:
        # record_signal returned at `if denial is None` — engine saw nothing.
        return "none"
    # A denial was processed. Inspect the source the engine attributed it to.
    source_id = engine.derive_source_id(step.intent, taint)
    src = engine.state.sources.get(source_id)
    sigs = []
    if src is not None:
        if src.tool_pressure_count > 0:
            sigs.append("tool-pressure")
        if src.instruction_pressure_count > 0:
            sigs.append("instruction-pressure")
    if not sigs:
        sigs.append("deny-count only")
    return " + ".join(sigs)


def run_config_a() -> list[TraceRow]:
    """Baseline: real engine, generic signals only."""
    engine = DegradationEngine(DegradationPolicy())
    taint = TaintState()  # untainted session — provenance drives source attribution
    rows: list[TraceRow] = []
    for step in openclaw_trace():
        deny_before = engine.state.session_deny_count
        denial = (DenialResponse(status="denied", coarse_category=step.denial_category)
                  if step.denied else None)
        engine.record_signal(step.intent, denial, taint)
        events = engine.drain_events()
        transition_str = "—"
        for ev in events:
            if getattr(ev, "previous_level", None) is not None:
                transition_str = f"{ev.previous_level}→{ev.new_level}"
        rows.append(TraceRow(
            step=step.label,
            intent_flags=_intent_flags(step),
            signal_fired=_signal_fired(engine, step, taint, deny_before),
            deny_count=engine.state.session_deny_count,
            level=engine.state.level.name,
            transition=transition_str,
            policy_narrowing=_policy_narrowing(engine),
        ))
    return rows


def run_config_b() -> list[TraceRow]:
    """Generic engine + the `privileged_shutdown` domain predicate (illustration).

    The engine is driven identically to Config A. On top, the domain predicate
    computes an *effective* level: when it fires, the effective level is forced
    to LOCKED, independent of generic pressure. The engine itself is untouched.
    """
    engine = DegradationEngine(DegradationPolicy())
    taint = TaintState()
    rows: list[TraceRow] = []
    domain_effective = DegradationLevel.NORMAL  # monotonic domain overlay
    for step in openclaw_trace():
        deny_before = engine.state.session_deny_count
        denial = (DenialResponse(status="denied", coarse_category=step.denial_category)
                  if step.denied else None)
        engine.record_signal(step.intent, denial, taint)
        events = engine.drain_events()
        generic_signal = _signal_fired(engine, step, taint, deny_before)
        generic_level = engine.state.level

        # Domain overlay — illustration only.
        domain_signal = ""
        if privileged_shutdown(step):
            domain_signal = "privileged_shutdown"
            if DegradationLevel.LOCKED > domain_effective:
                domain_effective = DegradationLevel.LOCKED

        effective_level = max(generic_level, domain_effective)

        # Transition string: prefer the generic engine event; otherwise note the
        # domain-driven jump.
        transition_str = "—"
        for ev in events:
            if getattr(ev, "previous_level", None) is not None:
                transition_str = f"{ev.previous_level}→{ev.new_level}"
        if domain_signal and effective_level > generic_level:
            transition_str = f"{generic_level.name}→{effective_level.name} (domain)"

        signal_str = generic_signal
        if domain_signal:
            signal_str = f"{generic_signal} (generic) / {domain_signal} (domain)"

        # Policy narrowing reflects the effective level. At LOCKED the engine
        # freezes tools to read+escalate and forces export RESTRICTED.
        if effective_level >= DegradationLevel.LOCKED > engine.state.level:
            narrowing = "tools frozen (read+escalate only), export_mode→restricted"
        else:
            narrowing = _policy_narrowing(engine)

        rows.append(TraceRow(
            step=step.label,
            intent_flags=_intent_flags(step),
            signal_fired=signal_str,
            deny_count=engine.state.session_deny_count,
            level=effective_level.name,
            transition=transition_str,
            policy_narrowing=narrowing,
        ))
    return rows


# ── Rendering ───────────────────────────────────────────────────────────────

_COLUMNS = [
    ("step", "step"),
    ("intent_flags", "intent flags"),
    ("signal_fired", "signal fired"),
    ("deny_count", "deny-count"),
    ("level", "DegradationLevel"),
    ("transition", "transition?"),
    ("policy_narrowing", "policy narrowing"),
]


def render_table(rows: list[TraceRow]) -> str:
    header = "| " + " | ".join(h for _, h in _COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    lines = [header, sep]
    for r in rows:
        cells = [str(getattr(r, attr)) for attr, _ in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
