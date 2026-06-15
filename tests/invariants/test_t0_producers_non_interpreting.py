"""T0 — the projection-producing process must be non-interpreting.

K4 (docs/kernel-theorem.md) holds only if every trusted-path projection is produced
by a deterministic, structural function — never a model reading the governed content,
and never an I/O / network / subprocess call whose result could steer the projection.
A steerable producer can emit a valid-but-unfaithful projection even when its output
type-checks (the Firewalls LLM-projector case).

The structural half of this obligation (a producer must not import the in-core
advisory/model surface) is pinned by the ``t0-producers-non-interpreting`` contract
in ``.importlinter``. import-linter graphs only the root package, so it cannot see an
*external* model SDK or a network import. This module closes that half: it statically
scans each producer's own imports for an interpreting surface, and asserts the two
pure classifiers are deterministic (same input → same output, no hidden state).

These two checks together are the named CI gate for T0 — the gap docs/kernel-theorem.md
§3 flagged as held-by-construction-only.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# The functions that produce a trusted-path projection the gates read.
_PRODUCER_MODULES = [
    "axor_core.security.carrier",       # carrier classification (imperative channel)
    "axor_core.policy.consequence",     # consequence (action-class) projection
    "axor_core.policy.normalizer",      # structural NormalizedIntent
    "axor_core.node.canonicalizer",     # CanonicalizedIntent (raw-stripped projection)
]

# Importing any of these from a producer would make the projection depend on a model
# inference, a network round-trip, or a shelled-out process — i.e. on something other
# than the structural form of the value. Matched by exact module or dotted prefix.
# NOTE: ``urllib.parse`` is a pure string parser and is deliberately NOT banned;
# ``urllib.request`` / ``urllib.error`` (the network surface) are.
_FORBIDDEN_PREFIXES = (
    # model / inference SDKs
    "anthropic", "openai", "cohere", "mistralai", "google.generativeai",
    "transformers", "torch", "tensorflow", "sentence_transformers", "vllm",
    "llama_cpp", "litellm", "langchain",
    # network
    "requests", "httpx", "aiohttp", "socket", "http.client",
    "urllib.request", "urllib.error", "ftplib", "smtplib", "telnetlib",
    # process / dynamic execution
    "subprocess", "multiprocessing",
    # the in-core advisory/model surface (also pinned structurally in .importlinter)
    "axor_core.kernel.adjudicator",
)


def _imported_modules(module_name: str) -> set[str]:
    """Every module name a producer's source imports (top-level statements)."""
    mod = importlib.import_module(module_name)
    src = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _is_forbidden(imported: str) -> bool:
    return any(
        imported == prefix or imported.startswith(prefix + ".")
        for prefix in _FORBIDDEN_PREFIXES
    )


@pytest.mark.parametrize("producer", _PRODUCER_MODULES)
def test_producer_imports_no_interpreting_surface(producer: str):
    """A projection producer must not import a model / network / subprocess surface."""
    offending = sorted(i for i in _imported_modules(producer) if _is_forbidden(i))
    assert not offending, (
        f"T0 violated: producer {producer!r} imports an interpreting surface "
        f"{offending} — a trusted-path projection must be produced structurally, "
        f"not by a model/network/subprocess call. See docs/kernel-theorem.md §3."
    )


def test_carrier_classifier_is_deterministic():
    """Same input → same carrier, repeatedly (no hidden state / I/O)."""
    from axor_core.security.carrier import classify_carrier

    samples = [
        "rm -rf / # free text with an instruction",
        "ENDORSED_TOKEN",
        42, 3.14, True, None,
        {"k": "v"}, [1, 2, 3],
        '{"closed": "schema"}',
        "../etc/passwd",
        "line one\nline two",
    ]
    for value in samples:
        results = {classify_carrier(value) for _ in range(5)}
        assert len(results) == 1, f"carrier classification not deterministic for {value!r}: {results}"


def test_consequence_classifier_is_deterministic():
    """Same sink/operation → same consequence class, repeatedly."""
    from axor_core.policy.consequence import consequence_class

    cases = [
        ("shutdown", None),
        ("restart_gateway", None),
        ("write", "file_write"),
        ("read", "file_read"),
        ("send", "network_request"),
        ("bash", None),
    ]
    for sink, op in cases:
        results = {consequence_class(sink, op) for _ in range(5)}
        assert len(results) == 1, f"consequence not deterministic for ({sink!r},{op!r}): {results}"
