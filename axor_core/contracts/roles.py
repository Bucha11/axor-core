"""Child roles — which agent and model a child runs on, decided by the operator.

A parent's model may ask for a child by ROLE NAME (``spawn_child(task, role=
"reader")``); what the role means — the executor (an agent CLI, a model), the
policy ceiling it runs under — is operator configuration handed to the session,
never something the spawn arguments can describe. The model picks among the
roles it was given; it cannot invent one, and it cannot widen one:

* an unknown role is denied (fail closed, like an unclassified sink);
* a role on a different executor is a model switch, and needs
  ``allow_model_switch`` on the parent's policy (``AuthorityPolicy``) — the
  right to change models is authority, not planning;
* a role's ``policy`` is the child's starting policy, and it is composed under
  the parent's ceiling like any child policy: a role can narrow (a read-only
  reader), never exceed the node that spawned it.

The role's name and ``label`` ride into the trace on ``ChildSpawnedEvent``, so
a tree shows which agent and model each node was — derived from the spawn, not
self-reported by the child.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axor_core.contracts.invokable import Invokable
    from axor_core.contracts.policy import ExecutionPolicy


@dataclass(frozen=True, eq=False)
class ChildRole:
    """One kind of child a parent may spawn.

    ``executor``  what the child runs on. None = the session's child executor
                  (the parent's own model) — no model switch.
    ``label``     how the trace names that executor, e.g. ``"codex/luna"``.
    ``policy``    the child's starting policy (e.g. no write, no bash for a
                  reader). None = selected from the task, as for any child.
                  Either way the parent's policy is the ceiling.
    ``output``    ``"text"`` (default) — the child's answer as it is; or
                  ``"findings"`` — a JSON findings list, each finding checked
                  against the files the child actually read before the parent
                  sees it (axor_core.node.findings).
    """

    name: str
    executor: "Invokable | None" = None
    label: str = ""
    policy: "ExecutionPolicy | None" = None
    description: str = ""
    output: str = "text"

    def __post_init__(self) -> None:
        if self.output not in ("text", "findings"):
            raise ValueError(f"child role output must be 'text' or 'findings', "
                             f"got {self.output!r}")
        if not self.name or not self.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"child role name must be a plain identifier, got {self.name!r}")


def role_roster(roles: "list[ChildRole] | tuple[ChildRole, ...] | dict | None"
                ) -> dict[str, ChildRole]:
    """Normalise what a caller passes (a list of roles or a name→role map) into
    a name→role map, refusing duplicates — two roles under one name would make
    the choice the spawn arguments make ambiguous."""
    if not roles:
        return {}
    if isinstance(roles, dict):
        for name, role in roles.items():
            if role.name != name:
                raise ValueError(f"child role keyed {name!r} is named {role.name!r}")
        return dict(roles)
    out: dict[str, ChildRole] = {}
    for role in roles:
        if role.name in out:
            raise ValueError(f"two child roles named {role.name!r}")
        out[role.name] = role
    return out
