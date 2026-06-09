"""Density meter (TM3.3) — the make-or-break measurement for per-value taint.

Density = fraction of high-stakes sink firings that receive a *tainted projection*
when they fire. The spec stakes half of Part II on this number:

    ~5%  → per-value buys real separation over session-sticky taint.
    ~80% → "session-sticky renamed"; per-value degenerates and the dual-label /
           causal_root machinery is not earning its place (TM3.3).

The meter measures TWO axes independently (the P-T / P-F split, TM2/TM3.1), because
the taint-explosion risk is asymmetric:

    integrity      — set by ANY external read; the data-dependent-action workload
                     taints most values after the first read, so a session-sticky
                     model is expected to explode here. This is the axis the
                     per-value rewrite must justify.
    confidentiality — set only by a *sensitive* read (a secret); sparse by nature,
                     so a coarse session floor is cheap here and density is low.

For each axis it records two measured booleans per high-stakes firing and **does
not rewrite them**:

    session_*  — would the session-scoped model consider the session tainted here?
    value_*    — does the per-value model consider THIS call's driving value tainted?

The structural expectation is per_value ⊆ session_sticky (a tainted driving value
implies a prior external read that already tainted the session). The meter does not
*enforce* this by masking the input — it records what it is given and counts any
violation separately, so a measurement bug (or a session-shadow that under-reports)
is visible instead of silently producing gap = 0.

Observe-only: this aggregates booleans handed to it. It makes no decision.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AxisDensity:
    """Session-sticky vs per-value density for one axis (integrity or sensitivity)."""

    firings: int
    session_sticky_tainted: int
    per_value_tainted: int
    # Firings where the per-value model said tainted but the session model did not.
    # Structurally expected to be 0; non-zero means a measurement/shadow bug.
    invariant_violations: int = 0

    @staticmethod
    def _ratio(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    @property
    def session_sticky_density(self) -> float:
        return self._ratio(self.session_sticky_tainted, self.firings)

    @property
    def per_value_density(self) -> float:
        return self._ratio(self.per_value_tainted, self.firings)

    @property
    def gap(self) -> float:
        """session_sticky_density − per_value_density. The per-value benefit.

        Computed from the raw measured counts; may be negative if the session
        shadow under-reports (which `invariant_violations` flags) — that is a
        signal to fix the measurement, not something to hide by clamping.
        """
        return self.session_sticky_density - self.per_value_density


@dataclass(frozen=True)
class DensityReport:
    integrity: AxisDensity
    sensitivity: AxisDensity
    # operation -> (firings, sess_integ, val_integ, sess_sens, val_sens)
    by_operation: dict[str, tuple[int, int, int, int, int]] = field(default_factory=dict)

    @property
    def high_stakes_firings(self) -> int:
        return self.integrity.firings

    @staticmethod
    def _ratio(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    def render(self) -> str:
        def axis_lines(name: str, a: AxisDensity) -> list[str]:
            out = [
                f"{name} session-sticky density : {a.session_sticky_density:.1%} "
                f"({a.session_sticky_tainted}/{a.firings})",
                f"{name} per-value density      : {a.per_value_density:.1%} "
                f"({a.per_value_tainted}/{a.firings})",
                f"{name} GAP (per-value benefit): {a.gap:.1%}",
            ]
            if a.invariant_violations:
                out.append(
                    f"{name} INVARIANT VIOLATIONS  : {a.invariant_violations} "
                    "(per-value tainted while session was not — measurement bug)"
                )
            return out

        lines = [
            "=== Density (TM3.3) — per-value vs session-sticky ===",
            f"high-stakes firings        : {self.high_stakes_firings}",
            "",
            *axis_lines("integrity     ", self.integrity),
            "",
            *axis_lines("confidentiality", self.sensitivity),
            "",
            "by operation (firings | integ sticky/value | sens sticky/value):",
        ]
        for op in sorted(self.by_operation):
            f, si, vi, ss, vs = self.by_operation[op]
            lines.append(
                f"  {op:24s} {f:6d} | {self._ratio(si, f):6.1%}/{self._ratio(vi, f):6.1%}"
                f" | {self._ratio(ss, f):6.1%}/{self._ratio(vs, f):6.1%}"
            )
        return "\n".join(lines)


class DensityMeter:
    """Accumulates high-stakes sink firings and their measured taint booleans,
    on the integrity and confidentiality axes independently."""

    def __init__(self) -> None:
        self._firings = 0
        # [session_tainted, value_tainted, invariant_violations] per axis
        self._integ = [0, 0, 0]
        self._sens = [0, 0, 0]
        # op -> [firings, sess_integ, val_integ, sess_sens, val_sens]
        self._by_op: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])

    def record(
        self,
        operation: str,
        *,
        session_tainted: bool,
        value_tainted: bool,
        session_sensitive: bool = False,
        value_sensitive: bool = False,
    ) -> None:
        """Record one high-stakes sink firing with its measured booleans.

        The booleans are stored as given. The structural expectation
        per_value ⊆ session_sticky is *checked* (violations counted), never
        imposed by rewriting the session boolean.
        """
        self._firings += 1
        row = self._by_op[operation]
        row[0] += 1

        for axis, sess, val, off in (
            (self._integ, session_tainted, value_tainted, 1),
            (self._sens, session_sensitive, value_sensitive, 3),
        ):
            if sess:
                axis[0] += 1
                row[off] += 1
            if val:
                axis[1] += 1
                row[off + 1] += 1
            if val and not sess:
                axis[2] += 1

    def report(self) -> DensityReport:
        return DensityReport(
            integrity=AxisDensity(
                firings=self._firings,
                session_sticky_tainted=self._integ[0],
                per_value_tainted=self._integ[1],
                invariant_violations=self._integ[2],
            ),
            sensitivity=AxisDensity(
                firings=self._firings,
                session_sticky_tainted=self._sens[0],
                per_value_tainted=self._sens[1],
                invariant_violations=self._sens[2],
            ),
            by_operation={op: tuple(r) for op, r in self._by_op.items()},
        )
