"""Density pipeline regression tests.

The density number is a key measurement, so it must survive being written to disk
and re-read. These tests pin two failure modes:

  - SinkDensityEvent fields could be dropped on persistence (not whitelisted by the
    serializer), so the number could not be reconstructed from a durable trace.
  - The meter could mask the measured session boolean, fabricating a gap >= 0.

They also confirm the two axes (integrity / confidentiality) survive to disk and to
the Prometheus exposition.
"""

from __future__ import annotations

import json

import pytest

from axor_core.contracts.trace import (
    SinkDensityEvent,
    TraceConfig,
    TraceEventKind,
)
from axor_core.trace.collector import TraceCollector
from axor_core.trace.metrics import GovernanceMetrics

pytestmark = pytest.mark.adversarial


def _density_event(node_id="n1", *, op="bash", tainted=False, sensitive=False,
                   session_tainted=False, session_sensitive=False):
    return SinkDensityEvent(
        kind=TraceEventKind.SINK_DENSITY,
        node_id=node_id,
        sequence=0,
        operation=op,
        tainted=tainted,
        sensitive=sensitive,
        session_tainted=session_tainted,
        session_sensitive=session_sensitive,
    )


def test_density_event_survives_persistence(tmp_path):
    # Every axis field must reach disk, not be silently dropped by the
    # serialization whitelist.
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True)
    c = TraceCollector(config=cfg, session_id="dens")
    c.record(_density_event(op="send", tainted=True, sensitive=True,
                            session_tainted=True, session_sensitive=True))
    c.close()
    body = json.loads((tmp_path / "dens.jsonl").read_text().splitlines()[0])["event"]
    assert body["operation"] == "send"
    assert body["tainted"] is True
    assert body["sensitive"] is True
    assert body["session_tainted"] is True
    assert body["session_sensitive"] is True


def test_density_reconstructable_from_persisted_trace(tmp_path):
    # The whole point: an operator can scrape the number back from durable JSONL.
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True)
    c = TraceCollector(config=cfg, session_id="recon")
    # 4 firings: 1 per-value integrity-tainted, 3 session-tainted, 1 sensitive.
    c.record(_density_event(tainted=True, session_tainted=True))
    c.record(_density_event(session_tainted=True))
    c.record(_density_event(session_tainted=True, sensitive=True, session_sensitive=True))
    c.record(_density_event())
    c.close()

    events = []
    for line in (tmp_path / "recon.jsonl").read_text().splitlines():
        body = json.loads(line)["event"]
        events.append(_density_event(
            tainted=body["tainted"], sensitive=body["sensitive"],
            session_tainted=body["session_tainted"],
            session_sensitive=body["session_sensitive"],
        ))
    m = GovernanceMetrics.from_events(events)
    assert m.integrity_density == 0.25           # 1/4
    assert m.session_integrity_density == 0.75   # 3/4
    assert m.confidentiality_density == 0.25     # 1/4


def test_prometheus_exposes_density_split_by_axis(tmp_path):
    m = GovernanceMetrics.from_events([
        _density_event(tainted=True, session_tainted=True),
        _density_event(session_tainted=True, sensitive=True, session_sensitive=True),
    ])
    text = m.to_prometheus()
    assert "axor_governance_density_integrity_per_value" in text
    assert "axor_governance_density_integrity_session_sticky" in text
    assert "axor_governance_density_confidentiality_per_value" in text
    assert "axor_governance_density_confidentiality_session_sticky" in text
    assert "axor_governance_sink_firings_total 2" in text
