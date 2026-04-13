from __future__ import annotations

"""
Agent contracts — AgentDefinition, AgentDomain, TrustLevel.

AgentDefinition is a first-class entity that describes what an agent is,
what domain it operates in, what it remembers, and how much it is trusted.

It is not a running object. It is a declaration that governance uses to:
  - select appropriate policy defaults for the domain
  - derive capabilities appropriate to trust level
  - determine what memory to load at session start
  - label child spawns with the agent's identity in lineage

Design rules:
  - AgentDefinition is immutable (frozen dataclass)
  - Core never forces a domain — it is a hint, not a constraint
  - TrustLevel affects capability derivation, not policy semantics
  - Personality is a plain string injected into context — never interpreted by governance
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Domain ─────────────────────────────────────────────────────────────────────

class AgentDomain(str, Enum):
    """
    What kind of work this agent primarily does.

    Domain is used by:
      - PolicySelector  → adjust policy defaults (e.g. research gets broader context)
      - TaskAnalyzer    → domain-aware complexity estimation
      - presets.py      → domain-specific presets (research, support, analysis)
      - ContextManager  → domain-aware fragment selection

    Domain is NOT used to restrict what an agent can do.
    It is a hint for optimization — not a hard boundary.
    """
    CODING     = "coding"      # default — write/refactor/test code
    RESEARCH   = "research"    # read-heavy, synthesis, long context
    SUPPORT    = "support"     # Q&A, diagnosis, short turns
    ANALYSIS   = "analysis"    # data interpretation, structured output
    GENERAL    = "general"     # no domain hint — use task signal only


# ── Trust Level ────────────────────────────────────────────────────────────────

class TrustLevel(str, Enum):
    """
    How much an agent is trusted by the governance kernel.

    TrustLevel affects capability derivation:
      - RESTRICTED: read-only tools, no bash, no write, no spawn
      - STANDARD:   default capabilities from policy
      - ELEVATED:   may request additional tools beyond default policy
      - FULL:       no capability restrictions from trust (policy still applies)

    TrustLevel does NOT bypass policy — it adjusts what CapabilityResolver
    is willing to grant when the policy allows a range.

    Example:
      policy allows_bash=True but requires TrustLevel >= ELEVATED
      → STANDARD agent: bash denied
      → ELEVATED agent: bash allowed

    Child agents inherit at most parent's trust level, never higher.
    """
    RESTRICTED = "restricted"  # minimal trust — read only, no spawn
    STANDARD   = "standard"    # default — capabilities from policy as-is
    ELEVATED   = "elevated"    # may use extended tools if policy permits
    FULL       = "full"        # no trust-based restrictions (policy still governs)


# ── Agent Definition ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentDefinition:
    """
    Immutable declaration of an agent's identity and governance preferences.

    Passed to GovernedSession at construction time.
    Governance reads it — executors never see it directly.

    Usage:

        from axor_core.contracts.agent import AgentDefinition, AgentDomain, TrustLevel

        agent = AgentDefinition(
            name="research-assistant",
            domain=AgentDomain.RESEARCH,
            trust_level=TrustLevel.STANDARD,
            personality="You are a meticulous research assistant...",
            default_tools=("read", "search", "glob"),
            memory_namespaces=("research", "shared"),
        )

        session = GovernedSession(
            executor=ClaudeCodeExecutor(),
            capability_executor=cap,
            agent_def=agent,
        )
    """

    name: str                              # human-readable identifier
    domain: AgentDomain = AgentDomain.GENERAL

    # trust
    trust_level: TrustLevel = TrustLevel.STANDARD

    # personality — injected as pinned context fragment
    # governance never interprets this; it is context for the executor only
    personality: str = ""

    # default tools for this agent — may be overridden by policy
    # empty tuple = use policy defaults
    default_tools: tuple[str, ...] = field(default_factory=tuple)

    # memory namespaces this agent reads from at session start
    # format: ("namespace1", "namespace2", ...)
    # loaded by MemoryProvider if one is configured
    memory_namespaces: tuple[str, ...] = field(default_factory=tuple)

    # metadata — arbitrary, never interpreted by governance
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_trust(self, level: TrustLevel) -> AgentDefinition:
        """Return a copy with a different trust level."""
        return AgentDefinition(
            name=self.name,
            domain=self.domain,
            trust_level=level,
            personality=self.personality,
            default_tools=self.default_tools,
            memory_namespaces=self.memory_namespaces,
            metadata=self.metadata,
        )

    def child_def(self, name: str | None = None) -> AgentDefinition:
        """
        Derive a child agent definition.
        Child inherits domain and personality but NOT trust level.
        Child trust is always capped at STANDARD unless explicitly elevated.
        """
        child_trust = (
            TrustLevel.STANDARD
            if self.trust_level == TrustLevel.FULL
            else self.trust_level
        )
        return AgentDefinition(
            name=name or f"{self.name}:child",
            domain=self.domain,
            trust_level=child_trust,
            personality=self.personality,
            default_tools=self.default_tools,
            memory_namespaces=(),        # children don't inherit memory namespaces
            metadata={},
        )
