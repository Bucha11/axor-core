#!/usr/bin/env python3
"""Fail if documentation references a module path that does not exist.

Scans docs/, README.md, CHANGELOG.md and examples/ markdown for
`axor_core/....py` and `axor_core/...` directory references and verifies each
one exists on disk. Catches the kind of drift found in review (docs naming
`capability/executor_lock.py` when the file is `capability/locked.py`, or an
example pinning a module path that has since moved).

Run from the axor-core package root:  python tools/check_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_GLOBS = ["docs/**/*.md", "README.md", "CHANGELOG.md", "examples/**/*.md"]

# Match axor_core/foo/bar.py or axor_core/foo/  (module file or package dir).
_REF = re.compile(r"\baxor_core/[A-Za-z0-9_./]+")


def _iter_doc_files():
    for pattern in DOC_GLOBS:
        yield from ROOT.glob(pattern)


def main() -> int:
    failures: list[str] = []
    for doc in _iter_doc_files():
        text = doc.read_text(encoding="utf-8")
        for m in _REF.finditer(text):
            ref = m.group(0).rstrip("./")
            target = ROOT / ref
            # Accept either an exact file, a package dir, or a dir given without
            # trailing slash. Skip references that are clearly globs/wildcards.
            if "*" in ref:
                continue
            if target.exists():
                continue
            # Tolerate a directory reference written without ".py".
            if (ROOT / ref).with_suffix(".py").exists():
                continue
            failures.append(f"{doc.relative_to(ROOT)}: missing reference '{ref}'")

    if failures:
        print("Documentation references nonexistent modules:")
        for f in sorted(set(failures)):
            print(f"  - {f}")
        return 1
    print("✓ all documented module paths exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
