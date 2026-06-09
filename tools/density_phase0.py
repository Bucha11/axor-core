#!/usr/bin/env python3
"""Phase 0 density experiment (TM3.3) — observe-only.

Measures per-value vs session-sticky taint density over a realistic corpus of
behavioural trajectories, WITHOUT touching the enforcement path. This is the
make-or-break number the spec (§0.0, TM3.3, X4) says to learn first: if per-value
density is no lower than session-sticky density, per-value buys nothing on this
workload and half of Part II's rationale weakens.

Corpus: the team's `axor_classifier_simple.data.anomaly_data.generate()` —
~51k labelled NormalizedIntent trajectories authored independently of this
experiment (so the number is not hand-picked). A tiny built-in fallback is used
only if that package is not importable.

Two signals are computed per high-stakes firing, replayed in trajectory order:

  session_tainted — running "any prior external read happened" flag (this is what
                    the current session-scoped TaintEngine effectively gives).
  value_tainted   — does THIS call's *driving value* come from external data?
                    (data_flow == external_to_shell, or provenance == external_web)

session_tainted is computed by us from the trajectory; value_tainted is read from
each intent's own structural fields. The two are independent by construction —
that is what lets the gap be non-zero.

Run:  python tools/density_phase0.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make axor_core importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axor_core.taint.density import DensityMeter  # noqa: E402

# World-effecting sinks. Reported per-operation below for full transparency.
HIGH_STAKES_OPS = frozenset(
    {"file_write", "execute_generated_code", "package_install", "network_request"}
)

_EXTERNAL_TARGETS = frozenset(
    {"external_url", "cloud_metadata", "docker_socket", "private_network"}
)


def is_external_read(i: dict) -> bool:
    """Does this intent introduce external/sensitive data into the session?

    This is the session-sticky taint trigger — it mirrors the propagate() rules in
    IntentLoop (web/network -> WEB; secret/system -> FILE).
    """
    op = i.get("operation", "")
    tk = i.get("target_kind", "")
    prov = i.get("provenance", "")
    if op == "network_request" and tk in _EXTERNAL_TARGETS:
        return True
    if tk in {"external_url", "cloud_metadata", "docker_socket"}:
        return True
    if prov == "external_web":
        return True
    if i.get("reads_secret_like_data") or tk == "secret":
        return True
    return False


def is_value_tainted(i: dict) -> bool:
    """Is the *driving value* of this sink call derived from external data?

    Per-value INTEGRITY signal, read from the intent's own structural fields:
      - data_flow == external_to_shell : external data is being fed into a shell.
      - provenance == external_web     : the action itself is driven by external content.
    local_to_external (local data leaving) and local_to_local/none keep the driving
    value local => not integrity-tainted.
    """
    return i.get("data_flow") == "external_to_shell" or i.get("provenance") == "external_web"


def is_sensitive_read(i: dict) -> bool:
    """Does this intent introduce a *sensitive* (secret) source — the confidentiality
    session-sticky trigger? Far sparser than external reads (that asymmetry is the
    whole point of measuring the two axes separately)."""
    return bool(i.get("reads_secret_like_data") or i.get("target_kind") == "secret")


def is_value_sensitive(i: dict) -> bool:
    """Proxy for 'this sink's driving value carries a secret' (per-value
    confidentiality). The corpus has no per-value secret-lineage field, so this is
    a conservative proxy: a sink reading/forwarding a secret-typed source."""
    return bool(i.get("carries_secret") or i.get("provenance") == "secret")


def replay(trajectories: list[tuple[list[dict], str]]) -> tuple[DensityMeter, dict[str, DensityMeter]]:
    overall = DensityMeter()
    by_label: dict[str, DensityMeter] = {}
    for intents, label in trajectories:
        meter_for_label = by_label.setdefault(label, DensityMeter())
        sticky = False
        sticky_sensitive = False
        for i in intents:
            if i.get("operation") in HIGH_STAKES_OPS:
                vt = is_value_tainted(i)
                vs = is_value_sensitive(i)
                for meter in (overall, meter_for_label):
                    meter.record(
                        i["operation"],
                        session_tainted=sticky,
                        value_tainted=vt,
                        session_sensitive=sticky_sensitive,
                        value_sensitive=vs,
                    )
            # taint takes effect AFTER the read executes -> "prior reads" semantics
            if is_external_read(i):
                sticky = True
            if is_sensitive_read(i):
                sticky_sensitive = True
    return overall, by_label


def load_corpus(limit: int | None) -> tuple[list[tuple[list[dict], str]], str]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "axor-classifier-simple"))
        from axor_classifier_simple.data.anomaly_data import generate  # type: ignore

        windows = generate()
        if limit is not None:
            windows = windows[:limit]
        return windows, f"axor_classifier_simple.anomaly_data.generate() [{len(windows)} trajectories]"
    except Exception as exc:  # pragma: no cover - fallback only
        print(f"[warn] corpus import failed ({exc!r}); using built-in fallback", file=sys.stderr)
        fallback = [
            ([
                {"operation": "network_request", "target_kind": "external_url", "provenance": "external_web", "data_flow": "local_to_external"},
                {"operation": "file_write", "target_kind": "workdir", "provenance": "repo", "data_flow": "local_to_local"},
                {"operation": "execute_generated_code", "target_kind": "workdir", "provenance": "external_web", "data_flow": "external_to_shell"},
            ], "normal"),
        ]
        return fallback, "built-in fallback (3 intents)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 density experiment (observe-only)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of trajectories")
    args = ap.parse_args()

    corpus, source = load_corpus(args.limit)
    print(f"corpus: {source}\n")
    overall, by_label = replay(corpus)

    print(overall.report().render())
    print("\n--- by label (integrity axis) ---")
    for label in sorted(by_label):
        r = by_label[label].report()
        a = r.integrity
        print(
            f"{label:11s} firings={r.high_stakes_firings:7d}  "
            f"sticky={a.session_sticky_density:6.1%}  "
            f"per-value={a.per_value_density:6.1%}  "
            f"gap={a.gap:6.1%}"
        )


if __name__ == "__main__":
    main()
