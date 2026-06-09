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
    # driving value, per operation, split by axis. per-value density = tainted /
    # firings; session-sticky density = session-shadow tainted / firings.
    sink_firings_by_op: Counter[str] = field(default_factory=Counter)
    sink_tainted_by_op: Counter[str] = field(default_factory=Counter)          # per-value integrity
    sink_sensitive_by_op: Counter[str] = field(default_factory=Counter)        # per-value confidentiality
    sink_session_tainted_by_op: Counter[str] = field(default_factory=Counter)  # session-sticky integrity
    sink_session_sensitive_by_op: Counter[str] = field(default_factory=Counter)  # session-sticky confidentiality

    @staticmethod
    def _density(num: Counter[str], den: Counter[str]) -> float:
        total = sum(den.values())
        return (sum(num.values()) / total) if total else 0.0

    @property
    def density(self) -> float:
        """Overall per-value integrity density: tainted high-stakes firings / all."""
        return self._density(self.sink_tainted_by_op, self.sink_firings_by_op)

    @property
    def integrity_density(self) -> float:
        return self._density(self.sink_tainted_by_op, self.sink_firings_by_op)

    @property
    def confidentiality_density(self) -> float:
        return self._density(self.sink_sensitive_by_op, self.sink_firings_by_op)

    @property
    def session_integrity_density(self) -> float:
        return self._density(self.sink_session_tainted_by_op, self.sink_firings_by_op)

    @property
    def session_confidentiality_density(self) -> float:
        return self._density(self.sink_session_sensitive_by_op, self.sink_firings_by_op)

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
                if getattr(ev, "sensitive", False):
                    m.sink_sensitive_by_op[op] += 1
                if getattr(ev, "session_tainted", False):
                    m.sink_session_tainted_by_op[op] += 1
                if getattr(ev, "session_sensitive", False):
                    m.sink_session_sensitive_by_op[op] += 1
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

        # Density (TM3.3) — the make-or-break number, scrapeable, split by axis and
        # by model (per-value vs session-sticky shadow).
        def gauge(name: str, value: float, help_text: str) -> None:
            metric = f"{prefix}_{name}"
            lines.append(f"# HELP {metric} {help_text}")
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f"{metric} {value:.6f}")

        gauge("sink_firings_total", float(sum(self.sink_firings_by_op.values())),
              "High-stakes sink firings observed (density denominator).")
        gauge("density_integrity_per_value", self.integrity_density,
              "Per-value integrity density (TM3.3).")
        gauge("density_integrity_session_sticky", self.session_integrity_density,
              "Session-sticky integrity density shadow (TM3.3).")
        gauge("density_confidentiality_per_value", self.confidentiality_density,
              "Per-value confidentiality density (TM3.3).")
        gauge("density_confidentiality_session_sticky", self.session_confidentiality_density,
              "Session-sticky confidentiality density shadow (TM3.3).")
        return "\n".join(lines) + "\n"
