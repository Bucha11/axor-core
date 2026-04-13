"""
axor_core.capability
────────────────────
Capability derivation and tool execution.

    CapabilityResolver  — ExecutionPolicy → Capabilities (what goes into envelope)
    CapabilityExecutor  — executes approved tool intents via registered handlers
    ToolHandler         — contract for adapter-provided tool implementations
"""

from axor_core.capability.resolver import CapabilityResolver, TOOL_READ, TOOL_WRITE, TOOL_BASH, TOOL_SEARCH, TOOL_SPAWN
from axor_core.capability.executor import CapabilityExecutor, ToolHandler

__all__ = [
    "CapabilityResolver",
    "CapabilityExecutor",
    "ToolHandler",
    "TOOL_READ",
    "TOOL_WRITE",
    "TOOL_BASH",
    "TOOL_SEARCH",
    "TOOL_SPAWN",
]
