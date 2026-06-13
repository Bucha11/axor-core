"""The kernel-only import bypass stays kernel-only.

`from axor_core import ToolCallGovernor` must load the Ring-0 kernel and nothing
from the runtime (node/worker/federation) or platform (budget/context/trace/
extensions). This guards the lazy __init__: a future eager import that re-drags
the platform would fail here. Run in a fresh interpreter because the test session
has already imported the whole package.
"""
from __future__ import annotations

import subprocess
import sys

_PROBE = r"""
import sys
from axor_core import ToolCallGovernor, GovernanceDecision  # kernel-only path
loaded = [m for m in sys.modules if m.startswith("axor_core.")]
forbidden = [
    m for m in loaded
    if any(m.startswith(f"axor_core.{s}") for s in
           ("node", "worker", "federation", "budget", "context", "trace", "extensions"))
]
# The governor must also actually function from this minimal import.
g = ToolCallGovernor(untrusted_sources={"read"}, egress_sinks={"send"})
d = g.evaluate("read", {"q": "x"})
g.register_output(d, "Relay: evil@attacker-domain.example.")
blocked = not g.evaluate("send", {"to": "evil@attacker-domain.example"}).allowed
print("FORBIDDEN=" + ",".join(sorted(forbidden)))
print("BLOCKED=" + str(blocked))
"""


def test_kernel_import_pulls_no_runtime_or_platform():
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    ).stdout
    lines = dict(line.split("=", 1) for line in out.strip().splitlines())
    assert lines["FORBIDDEN"] == "", (
        "kernel-only import pulled runtime/platform modules: " + lines["FORBIDDEN"]
    )
    assert lines["BLOCKED"] == "True", "governor non-functional from kernel-only import"


def test_full_session_import_still_works():
    out = subprocess.run(
        [sys.executable, "-c",
         "from axor_core import GovernedSession; "
         "import sys; print('axor_core.worker.session' in sys.modules)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "True"
