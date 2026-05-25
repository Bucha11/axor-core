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
    """

    LIBRARY = "library"
    PRODUCTION = "production"
    STRICT = "strict"
