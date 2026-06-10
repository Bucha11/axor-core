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
    """True when `path` equals or is contained by `root` after resolution.

    Fails CLOSED on any resolution error (e.g. an embedded null byte, which makes
    Path.resolve raise): an unresolvable path is never inside the allowlist, and a
    governance check must deny it, not crash."""
    try:
        candidate = resolve_path(path)
        base = resolve_path(root)
    except (ValueError, OSError):
        return False
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


def intersect_allowlist(
    policy_paths: Sequence[str], ceiling_paths: Sequence[str]
) -> tuple[str, ...]:
    """Narrowing intersection of two path allowlists.

    For each (policy root p, ceiling root c) the deeper of the two is kept only
    when they overlap; disjoint pairs contribute nothing. Every returned root is
    therefore contained by BOTH a policy root and a ceiling root — the result can
    only narrow, never widen past either side. An empty result means the two
    allowlists are disjoint (confine to nothing — fail closed)."""
    result: list[str] = []
    for p in policy_paths:
        for c in ceiling_paths:
            if path_within(p, c):       # p is inside the ceiling — p is narrower
                result.append(p)
            elif path_within(c, p):     # ceiling is inside p — c is narrower
                result.append(c)
            # disjoint: neither contains the other — grants nothing
    return tuple(dict.fromkeys(result))  # dedupe, preserve order
