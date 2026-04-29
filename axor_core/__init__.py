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

# Public entry point
from axor_core.worker.session import GovernedSession
from axor_core.budget import TokenCostRates

# Core contracts — what adapters and users need to implement
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.cancel import CancelToken, CancelReason, make_token
from axor_core.contracts.policy import (
    SignalClassifier,
    ExecutionPolicy,
    TaskSignal,
    TaskComplexity,
    TaskNature,
)
from axor_core.contracts.result import ExecutionResult, TokenUsage
from axor_core.contracts.extension import ExtensionLoader, ExtensionBundle
from axor_core.contracts.trace import TraceConfig

# Agent definition
from axor_core.contracts.agent import AgentDefinition, AgentDomain, TrustLevel

# Memory contracts
from axor_core.contracts.memory import (
    MemoryFragment,
    MemoryProvider,
    MemoryQuery,
    FragmentValue,
    NullMemoryProvider,
)

# Capability
from axor_core.capability.executor import CapabilityExecutor, ToolHandler

# Policy presets — for override_policy usage
from axor_core.policy import presets

# Errors
from axor_core.errors.exceptions import (
    AxorError,
    IntentDeniedError,
    ToolNotAllowedError,
    ChildNotAllowedError,
)

__version__ = "0.1.0"

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
    # policy
    "presets",
    # errors
    "AxorError", "IntentDeniedError", "ToolNotAllowedError", "ChildNotAllowedError",
    # version
    "__version__",
]
