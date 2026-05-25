from __future__ import annotations

from pathlib import Path
import tomllib

import axor_core


def test_runtime_version_matches_pyproject() -> None:
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert axor_core.__version__ == data["project"]["version"]
