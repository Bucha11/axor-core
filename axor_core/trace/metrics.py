"""Governance metrics — aggregate a DecisionTrace into operator-facing counters.

The trace already records every governance decision; this turns that stream into
counts an operator can scrape (denials, degradation transitions by level, taint
propagations by source, quarantines). Dependency-free: emits Prometheus text
exposition format without requiring prometheus_client.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from axor_core.contracts.trace import TraceEvent, TraceEventKind


@dataclass
class GovernanceMetrics:
    by_kind: Counter[str] = field(default_factory=Counter)
    denials: int = 0
    anomalies_flagged: int = 0
    taint_propagations_by_source: Counter[str] = field(default_factory=Counter)
    degradation_transitions_by_level: Counter[str] = field(default_factory=Counter)
    sources_quarantined: int = 0
    escalations_granted: int = 0
    escalations_denied: int = 0
    # Density (TM3.3): high-stakes sink firings and how many carried a tainted
    # driving value, per operation. per-value density = tainted / firings.
    sink_firings_by_op: Counter[str] = field(default_factory=Counter)
    sink_tainted_by_op: Counter[str] = field(default_factory=Counter)

    @property
    def density(self) -> float:
        """Overall per-value density: tainted high-stakes firings / all firings."""
        total = sum(self.sink_firings_by_op.values())
        return (sum(self.sink_tainted_by_op.values()) / total) if total else 0.0

    @classmethod
    def from_events(cls, events: Iterable[TraceEvent]) -> "GovernanceMetrics":
        m = cls()
        for ev in events:
            kind = getattr(ev, "kind", None)
            kind_name = kind.value if isinstance(kind, TraceEventKind) else str(kind)
            m.by_kind[kind_name] += 1

            if kind == TraceEventKind.INTENT_DENIED:
                m.denials += 1
            elif kind == TraceEventKind.ANOMALY_FLAGGED:
                m.anomalies_flagged += 1
            elif kind == TraceEventKind.TAINT_PROPAGATED:
                src = getattr(ev, "taint_source", "") or "unknown"
                m.taint_propagations_by_source[src] += 1
            elif kind == TraceEventKind.DEGRADATION_TRANSITION:
                lvl = getattr(ev, "new_level", "") or "unknown"
                m.degradation_transitions_by_level[lvl] += 1
            elif kind == TraceEventKind.SOURCE_QUARANTINED:
                m.sources_quarantined += 1
            elif kind == TraceEventKind.ESCALATION_GRANTED:
                m.escalations_granted += 1
            elif kind == TraceEventKind.ESCALATION_DENIED:
                m.escalations_denied += 1
            elif kind == TraceEventKind.SINK_DENSITY:
                op = getattr(ev, "operation", "") or "unknown"
                m.sink_firings_by_op[op] += 1
                if getattr(ev, "tainted", False):
                    m.sink_tainted_by_op[op] += 1
        return m

    def to_prometheus(self, prefix: str = "axor_governance") -> str:
        """Render counters in Prometheus text exposition format."""
        lines: list[str] = []

        def counter(name: str, value: int, help_text: str, labels: str = "") -> None:
            metric = f"{prefix}_{name}"
            lines.append(f"# HELP {metric} {help_text}")
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric}{labels} {value}")

        counter("denials_total", self.denials, "Intents denied by governance.")
        counter("anomalies_flagged_total", self.anomalies_flagged, "Suspicious intents flagged.")
        counter("sources_quarantined_total", self.sources_quarantined, "Sources quarantined.")
        counter("escalations_granted_total", self.escalations_granted, "Escalations granted.")
        counter("escalations_denied_total", self.escalations_denied, "Escalations denied.")

        for src, n in sorted(self.taint_propagations_by_source.items()):
            counter("taint_propagations_total", n,
                    "Taint propagations by source.", labels=f'{{source="{src}"}}')
        for lvl, n in sorted(self.degradation_transitions_by_level.items()):
            counter("degradation_transitions_total", n,
                    "Degradation transitions by target level.", labels=f'{{level="{lvl}"}}')
        return "\n".join(lines) + "\n"
