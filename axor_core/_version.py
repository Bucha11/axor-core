from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as metadata_version
from pathlib import Path
import tomllib


def get_version(distribution_name: str) -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    try:
        return metadata_version(distribution_name)
    except PackageNotFoundError:
        return "0.0.0"
