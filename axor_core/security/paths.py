"""Canonical path resolution and allowlist containment.

Two resolution modes:
  resolve_path     — follows symlinks and resolves .. against the real filesystem
                     (authoritative; use for allowlist enforcement)
  lexical_normalize — pure-lexical .. resolution, no filesystem access
                     (use for classification before a file is opened)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


def resolve_path(path: str) -> Path:
    """Resolve a path to an absolute, symlink-free form (filesystem-aware)."""
    return Path(os.path.expanduser(path)).resolve(strict=False)


def lexical_normalize(path: str) -> str:
    """Resolve .. and ~ lexically without touching the filesystem."""
    try:
        return os.path.normpath(os.path.expanduser(path)).replace("\\", "/")
    except Exception:
        return path


def path_within(path: str, root: str) -> bool:
    """True when `path` equals or is contained by `root` after resolution."""
    candidate = resolve_path(path)
    base = resolve_path(root)
    if candidate == base:
        return True
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False


def path_matches_allowlist(path: str, allowed_paths: Sequence[str]) -> bool:
    """True only when `path` is equal to or contained by an allowed root."""
    if not path:
        return False
    return any(path_within(path, root) for root in allowed_paths)


def paths_within(candidate_paths: Sequence[str], allowed_paths: Sequence[str]) -> bool:
    """True only when every candidate path is contained by an allowed root."""
    return all(path_matches_allowlist(p, allowed_paths) for p in candidate_paths)
