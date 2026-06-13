"""
axor_core
─────────
Provider-agnostic governance kernel for agent systems.

Quick start:

    from axor_core import GovernedSession
    from axor_core.capability import CapabilityExecutor, ToolHandler

    class MyReadHandler(ToolHandler):
        @property
        def name(self): return "read"
        async def execute(self, args): ...

    cap_executor = CapabilityExecutor()
    cap_executor.register(MyReadHandler())

    session = GovernedSession(
        executor=MyExecutor(),
        capability_executor=cap_executor,
    )
    result = await session.run("write a test for the auth endpoint")

With agent definition:

    from axor_core import AgentDefinition, AgentDomain, TrustLevel

    agent = AgentDefinition(
        name="research-assistant",
        domain=AgentDomain.RESEARCH,
        trust_level=TrustLevel.STANDARD,
        personality="You are a meticulous research assistant...",
    )
    session = GovernedSession(
        executor=..., capability_executor=..., agent_def=agent,
    )

With memory provider:

    from axor_memory_sqlite import SQLiteMemoryProvider
    session = GovernedSession(
        executor=..., capability_executor=...,
        agent_def=AgentDefinition(name="bot", memory_namespaces=("bot",)),
        memory_provider=SQLiteMemoryProvider("memory.db"),
    )
"""

# Public entry point.
#
# Imports are LAZY (PEP 562). Importing `axor_core` — or any kernel symbol such as
# `ToolCallGovernor` — does not drag in the runtime (node/worker/capability/
# federation) or the platform (budget/context/trace/extensions). You pay only for
# the rings you actually use: `from axor_core import ToolCallGovernor` loads just
# the Ring-0 kernel; `from axor_core import GovernedSession` pulls the full stack.
# This is the kernel-only bypass for callers that own their agent loop and want
# the gate engine without the orchestration.
from typing import TYPE_CHECKING
import importlib

from axor_core._version import get_version

__version__ = get_version("axor-core")

# name → module that defines it. Submodules (returned as-is) are in _SUBMODULES.
_LAZY = {
    # session (Ring 1/2 — pulls the full stack)
    "GovernedSession": "axor_core.worker.session",
    "TokenCostRates": "axor_core.budget",
    # capability (Ring 1, light)
    "CapabilityExecutor": "axor_core.capability.executor",
    "ToolHandler": "axor_core.capability.executor",
    # governor (Ring 0 kernel only)
    "ToolCallGovernor": "axor_core.governor",
    "GovernanceDecision": "axor_core.governor",
    # declarative config
    "GovernanceConfig": "axor_core.config",
    # contracts (Ring 0)
    "Invokable": "axor_core.contracts.invokable",
    "CancelToken": "axor_core.contracts.cancel",
    "CancelReason": "axor_core.contracts.cancel",
    "make_token": "axor_core.contracts.cancel",
    "SignalClassifier": "axor_core.contracts.policy",
    "ExecutionPolicy": "axor_core.contracts.policy",
    "TaskSignal": "axor_core.contracts.policy",
    "TaskComplexity": "axor_core.contracts.policy",
    "TaskNature": "axor_core.contracts.policy",
    "ExecutionResult": "axor_core.contracts.result",
    "TokenUsage": "axor_core.contracts.result",
    "ExtensionLoader": "axor_core.contracts.extension",
    "ExtensionBundle": "axor_core.contracts.extension",
    "TraceConfig": "axor_core.contracts.trace",
    "AgentDefinition": "axor_core.contracts.agent",
    "AgentDomain": "axor_core.contracts.agent",
    "TrustLevel": "axor_core.contracts.agent",
    "MemoryFragment": "axor_core.contracts.memory",
    "MemoryProvider": "axor_core.contracts.memory",
    "MemoryQuery": "axor_core.contracts.memory",
    "FragmentValue": "axor_core.contracts.memory",
    "NullMemoryProvider": "axor_core.contracts.memory",
    # policy presets (Ring 0)
    "presets": "axor_core.policy.presets",
    # errors (Ring 0)
    "AxorError": "axor_core.errors.exceptions",
    "IntentDeniedError": "axor_core.errors.exceptions",
    "ToolNotAllowedError": "axor_core.errors.exceptions",
    "ChildNotAllowedError": "axor_core.errors.exceptions",
}
_SUBMODULES = {"presets"}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'axor_core' has no attribute {name!r}")
    module = importlib.import_module(target)
    obj = module if name in _SUBMODULES else getattr(module, name)
    globals()[name] = obj  # cache so subsequent access skips __getattr__
    return obj


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # static visibility for type checkers / IDEs (not loaded at runtime)
    from axor_core.worker.session import GovernedSession
    from axor_core.budget import TokenCostRates
    from axor_core.capability.executor import CapabilityExecutor, ToolHandler
    from axor_core.governor import ToolCallGovernor, GovernanceDecision
    from axor_core.config import GovernanceConfig
    from axor_core.contracts.invokable import Invokable
    from axor_core.contracts.cancel import CancelToken, CancelReason, make_token
    from axor_core.contracts.policy import (
        SignalClassifier, ExecutionPolicy, TaskSignal, TaskComplexity, TaskNature,
    )
    from axor_core.contracts.result import ExecutionResult, TokenUsage
    from axor_core.contracts.extension import ExtensionLoader, ExtensionBundle
    from axor_core.contracts.trace import TraceConfig
    from axor_core.contracts.agent import AgentDefinition, AgentDomain, TrustLevel
    from axor_core.contracts.memory import (
        MemoryFragment, MemoryProvider, MemoryQuery, FragmentValue, NullMemoryProvider,
    )
    from axor_core.policy import presets
    from axor_core.errors.exceptions import (
        AxorError, IntentDeniedError, ToolNotAllowedError, ChildNotAllowedError,
    )

__all__ = [
    # session
    "GovernedSession",
    # contracts
    "Invokable",
    "CancelToken", "CancelReason", "make_token",
    "SignalClassifier",
    "ExecutionPolicy",
    "TaskSignal", "TaskComplexity", "TaskNature",
    "ExecutionResult", "TokenUsage",
    "ExtensionLoader", "ExtensionBundle",
    "TraceConfig",
    # agent
    "AgentDefinition", "AgentDomain", "TrustLevel",
    # memory
    "MemoryFragment", "MemoryProvider", "MemoryQuery", "FragmentValue",
    "NullMemoryProvider",
    # capability
    "CapabilityExecutor", "ToolHandler",
    # governor
    "ToolCallGovernor", "GovernanceDecision",
    "GovernanceConfig",
    # budget
    "TokenCostRates",
    # policy
    "presets",
    # errors
    "AxorError", "IntentDeniedError", "ToolNotAllowedError", "ChildNotAllowedError",
    # version
    "__version__",
]
