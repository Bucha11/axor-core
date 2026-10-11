"""Where the alternatives come from.

An alternative has to be a value the agent could actually have emitted at that
point: something that exists in the environment the reference plan is running
against. So the candidate pool is harvested from the live AgentDojo environment
state (every scalar in the environment model, plus every mapping key — record
ids live there), grouped by a syntactic *shape* so a substitution is type-
plausible for the argument it replaces.

Two exclusions are applied before anything is counted, both deliberate:

* the original value itself — a substitution has to substitute something;
* every value the *rest of the reference plan* needs. A second recipient the
  plan itself addresses is a required co-addressee, not an excess admission, and
  counting it would inflate the result. (This is the exclusion the earlier
  census added; it is kept here by construction.)

Shapes are syntactic and deliberately coarse. A shape is *not* part of the
monitor's observation language — it is this experiment's way of proposing
plausible alternatives. Whether the monitor can tell two of them apart is
measured, never assumed.
"""
from __future__ import annotations

import re
from typing import Any

_IBAN = re.compile(r"^[A-Z]{2}[0-9A-Z]{10,32}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_URL = re.compile(r"^(https?://|www\.)\S+$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")
_FILENAME = re.compile(r"^[\w\-. ]+\.(txt|pdf|docx?|xlsx?|csv|png|jpe?g|md|yaml|json)$")
_INT_ID = re.compile(r"^\d{1,8}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_SHORT_TOKEN = re.compile(r"^[\w\-.']{1,40}$")


def shape(value: Any) -> str:
    """The syntactic class of a value — how a candidate is matched to an argument."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if not isinstance(value, str):
        return "other"
    v = value.strip()
    if not v:
        return "empty"
    if _EMAIL.match(v):
        return "email"
    if _URL.match(v):
        return "url"
    if _IBAN.match(v):
        return "iban"
    if _DATETIME.match(v):
        return "datetime"
    if _DATE.match(v):
        return "date"
    if _FILENAME.match(v):
        return "filename"
    if _UUID.match(v):
        return "uuid"
    if _INT_ID.match(v):
        return "int_id"
    if " " not in v and _SHORT_TOKEN.match(v):
        return "token"
    return "free_text"


def _walk(node: Any, out: list[tuple[str, Any]], path: str = "", depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(node, dict):
        # Mapping keys carry record identity in AgentDojo environments (file ids,
        # channel names, inbox owners), so they are values too — but only in a
        # *record map*, where every value is itself a record. In a dumped model the
        # keys are schema field names ("amount", "subject"), which are not values
        # the agent could sensibly pass, so they are skipped.
        record_map = bool(node) and all(
            isinstance(v, (dict, list)) for v in node.values()
        )
        for k, v in node.items():
            if record_map and isinstance(k, str):
                out.append((f"{path}.<key>", k))
            _walk(v, out, f"{path}.{k}", depth + 1)
    elif isinstance(node, (list, tuple, set)):
        for i, v in enumerate(node):
            _walk(v, out, f"{path}[{i}]", depth + 1)
    elif isinstance(node, (str, int, float)) and not isinstance(node, bool):
        out.append((path, node))


def env_values(env: Any) -> list[tuple[str, Any]]:
    """Every scalar and mapping key in the environment, with its path."""
    try:
        dumped = env.model_dump(mode="python")
    except Exception:
        dumped = getattr(env, "__dict__", {})
    out: list[tuple[str, Any]] = []
    _walk(dumped, out)
    return out


def slot(path: str) -> str:
    """The structural slot a path names, with list positions erased:
    ``.cloud_drive.initial_files[3].filename`` -> ``.cloud_drive.initial_files[].filename``.

    Two values in the same slot are two records of the same kind — which is what
    makes a substitution between them a *referential* substitution ("act on a
    different record of the same class") rather than a change of payload text.
    """
    collapsed = re.sub(r"\[\d+\]", "[]", path)
    # AgentDojo keys record maps by id ("calendar.events.9.title"), so a numeric
    # path segment is a record position too, not part of the slot.
    return re.sub(r"\.\d+(?=\.|$)", ".<n>", collapsed)


def pools(env: Any) -> tuple[dict[str, list[tuple[str, Any]]], dict[Any, set[str]]]:
    """Environment values grouped by shape, plus the slots each value occupies.

    De-duplicated and deterministically ordered, so a run is reproducible.
    """
    by_shape: dict[str, dict[Any, str]] = {}
    slots: dict[Any, set[str]] = {}
    for path, value in env_values(env):
        s = shape(value)
        by_shape.setdefault(s, {}).setdefault(value, path)
        slots.setdefault(value, set()).add(slot(path))
    ordered = {
        s: sorted(((p, v) for v, p in seen.items()), key=lambda pv: (str(pv[1]), pv[0]))
        for s, seen in by_shape.items()
    }
    return ordered, slots


def prompt_values(prompt: str) -> list[tuple[str, Any]]:
    """Shape-classified tokens of the user prompt.

    The prompt is the trusted origin under context-default integrity, so a value
    lifted from it is *proven trusted* — which makes two prompt tokens of the same
    shape indistinguishable to the monitor. Those are the substitutions that test
    whether "the user named this one" survives into the verdict.
    """
    out: list[tuple[str, Any]] = []
    seen: set[Any] = set()
    for raw in re.split(r"[\s,;()\[\]{}<>\"]+", prompt or ""):
        tok = raw.strip(".:!?'")
        if not tok or tok in seen:
            continue
        seen.add(tok)
        out.append(("<prompt>", tok))
    return out


def plan_values(plan: list[Any]) -> set[Any]:
    """Every value the reference plan itself uses, anywhere in any call.

    Candidates equal to one of these are dropped: the plan needs them, so
    admitting them is not excess admission.
    """
    out: set[Any] = set()

    def add(v: Any) -> None:
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            out.add(v)
        elif isinstance(v, (list, tuple, set)):
            for x in v:
                add(x)
        elif isinstance(v, dict):
            for x in v.values():
                add(x)

    for call in plan:
        for v in dict(call.args).values():
            add(v)
    return out
