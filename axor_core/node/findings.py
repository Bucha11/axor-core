"""Findings: a child's structured answer, checked against what it actually did.

A child spawned under a role whose ``output`` is ``"findings"`` is expected to
answer with JSON::

    {"findings": [{"claim": "...", "file": "src/auth.py", "line": 42}, ...]}

Before that answer reaches the parent, each finding is checked against the
child's own APPROVED tool calls — the ones the gates let through, not the ones
it asked for. A finding is ``verified`` when the file it names is a path the
child read (or a directory it searched or listed that contains it). A finding
naming no file, or a file the child never touched, is kept but marked
unverified, with the reason: the parent — and the context that flows on from it
— can tell a reported observation from an assertion.

This is structural, not semantic. It cannot tell whether line 42 says what the
claim says; it can tell that the child never opened the file it is describing.
That is the commonest way a reader fabricates, and it is decidable.
"""
from __future__ import annotations

import json
import posixpath
import shlex
from typing import Any

# argument names that carry a path a call read from
_PATH_ARGS = ("path", "file", "file_path", "filepath", "filename", "dir", "directory",
              "root", "cwd", "target", "pattern_path", "paths", "files")
# a call that answers for a whole directory (its results name files inside it)
_DIR_TOOLS = ("search", "grep", "glob", "list", "ls", "find", "rg")
# coding agents read through the shell as often as through a read tool
_SHELL_TOOLS = ("bash", "shell", "exec", "run", "command", "terminal")
_SHELL_READERS = frozenset({"cat", "sed", "head", "tail", "less", "more", "nl", "awk",
                            "bat", "wc", "view", "file"})
_SHELL_PATTERN_FIRST = frozenset({"grep", "rg", "ag", "egrep", "fgrep"})
_SHELL_LISTERS = frozenset({"ls", "tree", "fd", "find"})


def _shell_paths(command: str) -> tuple[set[str], set[str]]:
    """Paths a shell command line read: operands of reader commands are files,
    operands of search/list commands are directories (no operand = the working
    directory). Pipelines and ``&&`` chains are split; anything unparseable
    reads nothing."""
    files: set[str] = set()
    dirs: set[str] = set()
    for chunk in command.replace("&&", ";").replace("||", ";").replace("|", ";").split(";"):
        try:
            words = shlex.split(chunk)
        except ValueError:
            continue
        if not words:
            continue
        head = posixpath.basename(words[0])
        rest = words[1:]
        if head in _SHELL_READERS:
            operands = [w for w in rest if not w.startswith("-") and ("/" in w or "." in w)]
            files.update(p for p in (_norm(w) for w in operands) if p)
            continue
        if head == "find":
            # find PATH... EXPRESSION — the paths end at the first option
            operands = []
            for w in rest:
                if w.startswith("-") or w in ("(", "!"):
                    break
                operands.append(w)
        elif head in _SHELL_PATTERN_FIRST:
            operands = [w for w in rest if not w.startswith("-")][1:]  # [0] is the pattern
        elif head in _SHELL_LISTERS:
            operands = [w for w in rest if not w.startswith("-")]
        else:
            continue
        if not operands:
            dirs.add("")  # searched the working directory
        for w in operands:
            p = _norm(w)
            if posixpath.splitext(p)[1]:
                files.add(p)
            else:
                dirs.add(p)
    return files, dirs


def _norm(path: str) -> str:
    path = path.strip().replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    path = posixpath.normpath(path) if path else ""
    return "" if path == "." else path


def touched_paths(approved_calls: list[tuple[str, dict]]) -> tuple[set[str], set[str]]:
    """(files read, directories searched) from approved calls. A directory tool
    given no path searched the working directory, recorded as ``""``."""
    files: set[str] = set()
    dirs: set[str] = set()
    for tool, args in approved_calls:
        command = (args or {}).get("command") or (args or {}).get("cmd")
        if any(t in tool.lower() for t in _SHELL_TOOLS) and isinstance(command, str):
            f, d = _shell_paths(command)
            files |= f
            dirs |= d
            continue
        values: list[str] = []
        for key in _PATH_ARGS:
            v = (args or {}).get(key)
            if isinstance(v, str):
                values.append(v)
            elif isinstance(v, (list, tuple)):
                values += [x for x in v if isinstance(x, str)]
        if any(t in tool.lower() for t in _DIR_TOOLS):
            dirs.update(_norm(v) for v in values) if values else dirs.add("")
        else:
            files.update(p for p in (_norm(v) for v in values) if p)
    return files, dirs


def parse_findings(output: str) -> list[dict] | None:
    """The findings list in a child's answer, or None if it gave none. The
    answer may wrap the JSON in prose or a fence, or follow other text (tool
    output an executor echoed): every object in it is tried, and the LAST one
    carrying a findings list wins — the child's final word."""
    decoder = json.JSONDecoder()
    found: list[dict] | None = None
    pos = output.find("{")
    while pos != -1:
        try:
            data, end = decoder.raw_decode(output, pos)
        except ValueError:
            pos = output.find("{", pos + 1)
            continue
        if isinstance(data, dict) and isinstance(data.get("findings"), list):
            found = [f for f in data["findings"] if isinstance(f, dict)]
        pos = output.find("{", end)
    return found


def verify_findings(output: str, approved_calls: list[tuple[str, dict]]) -> dict[str, Any]:
    """Each finding marked ``verified`` (and ``why`` when not) against the
    child's approved calls. A child that answered without a findings list comes
    back as ``{"unstructured": <output>, "findings": []}`` — nothing in it is
    vouched for."""
    findings = parse_findings(output)
    if findings is None:
        return {"findings": [], "unstructured": output, "verified": 0, "unverified": 0}
    files, dirs = touched_paths(approved_calls)
    out = []
    for f in findings:
        item = {k: f[k] for k in ("claim", "file", "line", "evidence") if k in f}
        name = f.get("file")
        if not isinstance(name, str) or not _norm(name):
            item.update(verified=False, why="names no file, so nothing it read can back it")
        else:
            p = _norm(name)
            inside = any(d == "" or p == d or p.startswith(d + "/") for d in dirs)
            if p in files or inside or any(p.endswith("/" + x) or x.endswith("/" + p)
                                           for x in files):
                item["verified"] = True
            else:
                item.update(verified=False, why=f"the child never read {p}")
        out.append(item)
    ok = sum(1 for i in out if i["verified"])
    return {"findings": out, "verified": ok, "unverified": len(out) - ok}
