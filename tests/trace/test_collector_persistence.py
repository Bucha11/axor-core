"""Persistence/retention/whitelist tests for TraceCollector."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from axor_core.contracts.policy import (
    TaskComplexity,
    TaskNature,
    TaskSignal,
)
from axor_core.contracts.trace import (
    AnonymizedTraceRecord,
    SignalChosenEvent,
    TokensSpentEvent,
    TraceConfig,
    TraceEventKind,
)
from axor_core.trace.collector import TraceCollector


def _make_signal() -> TaskSignal:
    return TaskSignal(
        raw_input="write a test",
        complexity=TaskComplexity.FOCUSED,
        nature=TaskNature.GENERATIVE,
        estimated_scope=1,
        requires_children=False,
        requires_mutation=False,
        domain="coding",
    )


def _signal_event(node_id: str = "node_1") -> SignalChosenEvent:
    return SignalChosenEvent(
        kind=TraceEventKind.SIGNAL_CHOSEN,
        node_id=node_id,
        sequence=0,
        raw_input="write a test",
        signal=_make_signal(),
        confidence=0.85,
        classifier="heuristic",
        scores={"complexity.focused": 1.0},
    )


def test_trace_file_path_none_when_persistence_disabled(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=False)
    c = TraceCollector(config=cfg, session_id="test")
    assert c.trace_file_path() is None


def test_trace_file_path_returns_expected_location(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True)
    c = TraceCollector(config=cfg, session_id="sess_abc")
    assert c.trace_file_path() == tmp_path / "sess_abc.jsonl"


def test_persist_disabled_skips_io(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=False)
    c = TraceCollector(config=cfg, session_id="off")
    c.record(_signal_event())
    c.close()
    assert not (tmp_path / "off.jsonl").exists()


def test_cleanup_removes_old_files(tmp_path):
    # Drop two old JSONL files and one fresh file
    old1 = tmp_path / "old1.jsonl"
    old2 = tmp_path / "old2.jsonl"
    fresh = tmp_path / "fresh.jsonl"
    for p in (old1, old2, fresh):
        p.write_text("{}\n", encoding="utf-8")
    stale = time.time() - 40 * 86_400
    os.utime(old1, (stale, stale))
    os.utime(old2, (stale, stale))

    # Instantiate with retention_days=30 — old1/old2 should go, fresh stays.
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True, retention_days=30)
    TraceCollector(config=cfg, session_id="new")

    assert not old1.exists()
    assert not old2.exists()
    assert fresh.exists()


def test_retention_zero_disables_cleanup(tmp_path):
    ancient = tmp_path / "ancient.jsonl"
    ancient.write_text("{}\n", encoding="utf-8")
    stale = time.time() - 1000 * 86_400
    os.utime(ancient, (stale, stale))

    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True, retention_days=0)
    TraceCollector(config=cfg, session_id="keep")
    assert ancient.exists()


def test_close_is_idempotent(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True)
    c = TraceCollector(config=cfg, session_id="close")
    c.record(_signal_event())
    c.close()
    # Second close must not raise and must keep file content intact.
    c.close()
    content = (tmp_path / "close.jsonl").read_text()
    assert len(content.splitlines()) == 1


def test_record_after_close_does_not_write(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True)
    c = TraceCollector(config=cfg, session_id="closed")
    c.record(_signal_event())
    c.close()
    c.record(_signal_event(node_id="node_2"))  # must not raise, must not append
    lines = (tmp_path / "closed.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_persist_inputs_true_serializes_raw_input(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True, persist_inputs=True)
    c = TraceCollector(config=cfg, session_id="persist")
    c.record(_signal_event())
    c.close()
    line = json.loads((tmp_path / "persist.jsonl").read_text().splitlines()[0])
    assert line["event"]["raw_input"] == "write a test"
    assert line["event"]["signal"]["raw_input"] == "write a test"


def test_persist_inputs_false_strips_raw_input(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True, persist_inputs=False)
    c = TraceCollector(config=cfg, session_id="noraw")
    c.record(_signal_event())
    c.close()
    line = json.loads((tmp_path / "noraw.jsonl").read_text().splitlines()[0])
    assert "raw_input" not in line["event"]
    assert "raw_input" not in line["event"]["signal"]


def test_unknown_event_kind_serializes_to_empty_body(tmp_path):
    """Fail-closed: an event kind not in whitelist drops all fields."""
    from axor_core.contracts.trace import TraceEvent
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True)
    c = TraceCollector(config=cfg, session_id="unk")
    c.record(TraceEvent(
        kind=TraceEventKind.INTENT_APPROVED,  # not in allowlist
        node_id="n",
        sequence=0,
        payload={"dangerous": "value"},
    ))
    c.close()
    line = json.loads((tmp_path / "unk.jsonl").read_text().splitlines()[0])
    assert line["event"] == {}
    assert line["kind"] == "intent_approved"


def test_to_anonymized_record_returns_none_when_not_opted_in(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=False, training_opt_in=False)
    c = TraceCollector(config=cfg, session_id="s")
    c.record(_signal_event())
    out = c.to_anonymized_record(
        node_id="node_1",
        signal=_make_signal(),
        classifier_used="heuristic",
        confidence=0.9,
        input_embedding=[1.0] * 128,
    )
    assert out is None


def test_to_anonymized_record_returns_record_when_opted_in(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=False, training_opt_in=True)
    c = TraceCollector(config=cfg, session_id="s2")
    c.record(_signal_event())
    c.record(TokensSpentEvent(
        kind=TraceEventKind.TOKENS_SPENT,
        node_id="node_1",
        sequence=0,
        input_tokens=120,
        output_tokens=30,
    ))
    rec = c.to_anonymized_record(
        node_id="node_1",
        signal=_make_signal(),
        classifier_used="heuristic",
        confidence=0.9,
        input_embedding=[1.0] * 128,
        fingerprint_kind="minhash_v1",
    )
    assert isinstance(rec, AnonymizedTraceRecord)
    assert rec.fingerprint_kind == "minhash_v1"
    assert rec.tokens_spent == 150
    assert rec.policy_adjusted is False


def test_to_anonymized_record_missing_node_returns_none(tmp_path):
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=False, training_opt_in=True)
    c = TraceCollector(config=cfg, session_id="s3")
    rec = c.to_anonymized_record(
        node_id="nonexistent",
        signal=_make_signal(),
        classifier_used="heuristic",
        confidence=0.9,
        input_embedding=None,
    )
    assert rec is None


def test_concurrent_record_is_thread_safe(tmp_path):
    """Two threads hammer record() — resulting JSONL must have N well-formed lines."""
    import threading
    cfg = TraceConfig(trace_dir=str(tmp_path), persist_to_disk=True)
    c = TraceCollector(config=cfg, session_id="concurrent")

    def worker(tag: str):
        for i in range(50):
            c.record(_signal_event(node_id=f"node_{tag}_{i}"))

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    c.close()

    lines = (tmp_path / "concurrent.jsonl").read_text().splitlines()
    assert len(lines) == 100
    # Every line must parse as valid JSON — no partial writes.
    for line in lines:
        json.loads(line)
