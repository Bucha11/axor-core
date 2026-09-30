"""Contract: axor_core.kernel stays pure and stdlib-only.

Complements the .importlinter Ring-0 contract (which forbids runtime/platform
imports inside the package graph) with the external half: no I/O libraries, no
schema/validation frameworks, no clock/randomness. The kernel is imported by
both the runtime adapter and the platform replay engine (architecture rule 0);
anything nondeterministic or environment-touching here breaks bit-reproducible
replay.
"""
from __future__ import annotations

import ast
import pathlib

# I/O and framework imports that must never appear in kernel modules.
FORBIDDEN_MODULES = {
    "httpx", "fastapi", "starlette", "kuzu", "asyncpg", "sqlalchemy",
    "requests", "aiohttp", "socket", "anyio", "uvicorn", "pydantic",
    "yaml", "asyncio", "threading", "subprocess", "os", "pathlib", "io",
    "random", "secrets", "uuid",
}
# time is allowed nowhere; datetime only for typing — forbid both outright.
FORBIDDEN_MODULES |= {"time", "datetime"}

SRC = pathlib.Path(__file__).parent.parent.parent / "axor_core" / "kernel"


def test_kernel_modules_import_no_io_and_no_clock() -> None:
    for f in sorted(SRC.glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            bad = FORBIDDEN_MODULES.intersection(names)
            assert not bad, f"{f.name}: forbidden import {sorted(bad)}"


def test_kernel_imports_stay_inside_ring0() -> None:
    # Kernel may import stdlib + Ring 0 modules only (mirrors .importlinter).
    ring0 = {"contracts", "errors", "taint", "security", "policy", "kernel",
             "degradation", "governor"}
    for f in sorted(SRC.glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[0] == "axor_core":
                    assert len(parts) >= 2 and parts[1] in ring0, (
                        f"{f.name}: kernel imports outside Ring 0: {node.module}"
                    )
