"""Density meter (TM3.3) — the make-or-break measurement for per-value taint.

Density = fraction of high-stakes sink firings that receive a *tainted projection*
when they fire. The spec stakes half of Part II on this number:

    ~5%  → per-value buys real separation over session-sticky taint.
    ~80% → "session-sticky renamed"; per-value degenerates and the dual-label /
           causal_root machinery is not earning its place (TM3.3).

This meter records, per high-stakes firing, two booleans and nothing else:

    session_tainted — would the *session-scoped* model (current code,
                      ``TaintEngine``) consider the session tainted here?
    value_tainted   — does the *per-value* model (``CausalRoot``) consider the
                      driving value of THIS call tainted?

``per_value_density`` ≤ ``session_sticky_density`` by construction (a tainted
value implies the session is tainted). The **gap** between them is the entire
benefit of per-value over session-sticky — the "прибыль" Phase 0 measures.

Observe-only: this aggregates booleans handed to it. It makes no decision.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DensityReport:
    high_stakes_firings: int
    session_sticky_tainted: int
    per_value_tainted: int
    # operation -> (firings, session_sticky_tainted, per_value_tainted)
    by_operation: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    @staticmethod
    def _ratio(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    @property
    def session_sticky_density(self) -> float:
        return self._ratio(self.session_sticky_tainted, self.high_stakes_firings)

    @property
    def per_value_density(self) -> float:
        return self._ratio(self.per_value_tainted, self.high_stakes_firings)

    @property
    def gap(self) -> float:
        """session_sticky_density − per_value_density. The per-value benefit."""
        return self.session_sticky_density - self.per_value_density

    def render(self) -> str:
        lines = [
            "=== Density (TM3.3) — per-value vs session-sticky ===",
            f"high-stakes firings        : {self.high_stakes_firings}",
            f"session-sticky density     : {self.session_sticky_density:.1%} "
            f"({self.session_sticky_tainted}/{self.high_stakes_firings})",
            f"per-value density          : {self.per_value_density:.1%} "
            f"({self.per_value_tainted}/{self.high_stakes_firings})",
            f"GAP (per-value benefit)    : {self.gap:.1%}",
            "",
            "by operation (firings | sticky | per-value):",
        ]
        for op in sorted(self.by_operation):
            f, s, v = self.by_operation[op]
            lines.append(
                f"  {op:24s} {f:7d} | {self._ratio(s, f):6.1%} | {self._ratio(v, f):6.1%}"
            )
        return "\n".join(lines)


class DensityMeter:
    """Accumulates high-stakes sink firings and their two taint booleans."""

    def __init__(self) -> None:
        self._firings = 0
        self._sticky = 0
        self._value = 0
        self._by_op: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    def record(
        self,
        operation: str,
        *,
        session_tainted: bool,
        value_tainted: bool,
    ) -> None:
        """Record one high-stakes sink firing.

        Enforces the structural invariant per_value ⊆ session_sticky: a tainted
        driving value implies the session is tainted (the value's source is an
        external read that already happened). This keeps gap ≥ 0 by construction.
        """
        session_tainted = session_tainted or value_tainted
        self._firings += 1
        row = self._by_op[operation]
        row[0] += 1
        if session_tainted:
            self._sticky += 1
            row[1] += 1
        if value_tainted:
            self._value += 1
            row[2] += 1

    def report(self) -> DensityReport:
        return DensityReport(
            high_stakes_firings=self._firings,
            session_sticky_tainted=self._sticky,
            per_value_tainted=self._value,
            by_operation={op: (r[0], r[1], r[2]) for op, r in self._by_op.items()},
        )
