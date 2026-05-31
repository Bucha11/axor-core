from __future__ import annotations

from enum import Enum


class ExecutionMode(str, Enum):
    """
    Declares the runtime isolation guarantee of a GovernedSession.

    LIBRARY    — same-process, no strong isolation. Useful for development.
    PRODUCTION — executor owned by GovernedSession, ToolInterceptor mandatory,
                 direct executor bypass blocked.
    STRICT     — superset of PRODUCTION with additional containment restrictions
                 for high-risk or regulated deployments.
    OBSERVE    — evaluation / measurement mode. Full pipeline runs and all trace
                 events are emitted, but deny/lock actions are not applied — the
                 agent proceeds unblocked so measurement is not contaminated by
                 enforcement. Used exclusively by axor-eval.
    """

    LIBRARY = "library"
    PRODUCTION = "production"
    STRICT = "strict"
    OBSERVE = "observe"
