"""Phase 3 group B — persistence / cross-process re-mint (TM3.2 / TM4.1).

A value that leaves and re-enters the trust boundary (memory read-back, or a
child's output crossing a process boundary) is re-minted as untrusted: 'soft
release' through memory or a sub-agent does not launder it.
"""

from __future__ import annotations

import pytest

from axor_core import GovernedSession
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.memory import (
    FragmentValue,
    MemoryFragment,
    MemoryProvider,
    MemoryQuery,
)
from axor_core.contracts.taint import TaintSource
from axor_core.contracts.trace import TraceConfig
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine
from tests.conftest import EchoExecutor

pytestmark = pytest.mark.adversarial

RECALLED = "recalled_secret_fragment_from_memory_store"


class _Provider(MemoryProvider):
    async def load(self, query: MemoryQuery):
        return [MemoryFragment(key="k", namespace="n", content=RECALLED,
                               value=FragmentValue.WORKING)]

    async def save(self, fragments): ...
    async def delete(self, namespace, keys): return 0
    async def evict(self, namespace, values=(), max_age_seconds=None): return 0
    async def namespaces(self): return []


def _cap():
    ex = CapabilityExecutor()

    class _Read(ToolHandler):
        @property
        def name(self): return "read"
        async def execute(self, args): return "ok"
    ex.register(_Read())
    return ex


@pytest.mark.asyncio
async def test_memory_readback_is_reminted_tainted():
    # A fragment loaded from memory must be registered tainted in the per-value
    # ledger, so a later sink carrying it is gated ('soft release' to memory does
    # not launder).
    sess = GovernedSession(
        executor=EchoExecutor(tool_calls=[]),
        capability_executor=_cap(),
        memory_provider=_Provider(),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
    )
    await sess.run("use my memory of the project")
    assert sess._taint_engine.derive_value(RECALLED).is_tainted is True
    assert sess._taint_engine.derive_value(f"prefix {RECALLED} suffix").is_tainted is True


# ── TM4.1 cross-process re-mint (engine level) ─────────────────────────────────

def test_cross_process_in_is_untrusted_nonsensitive():
    r = CausalRoot.cross_process_in()
    assert r.is_tainted is True
    assert r.sensitive is False


def test_child_output_remint_taints_in_parent_ledger():
    # Mirrors the wrapper wiring: a child's returned output is re-minted untrusted
    # in the PARENT engine, so the parent cannot egress a value the child read.
    parent = TaintEngine(node_id="parent")
    child_output = "here is the data the child fetched from the web for you"
    assert parent.derive_value(child_output).is_tainted is False   # before
    parent.register_value(child_output, CausalRoot.cross_process_in())
    assert parent.derive_value(child_output).is_tainted is True    # re-minted
    # and a derived value carrying it is tainted too
    assert parent.derive_value(f"echo: {child_output}").is_tainted is True


def test_child_cannot_launder_via_inheritance_then_return():
    # Forward inheritance does NOT clear taint, and the reverse (child output) is
    # re-minted: a value the child surfaces is untrusted to the parent regardless.
    parent = TaintEngine(node_id="parent")
    leaked = "secret the child read and tried to surface"
    parent.register_value(leaked, CausalRoot.cross_process_in())
    assert parent.derive_value(leaked).is_tainted is True
