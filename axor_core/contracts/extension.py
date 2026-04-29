from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_mapping(value: Any) -> Mapping[str, Any]:
    """Coerce a dict to a read-only MappingProxyType view.

    Frozen dataclasses with `dict` fields otherwise leak mutability —
    `frag.policy_overrides["allow_bash"] = True` would silently rewrite a
    "frozen" extension fragment. The proxy keeps dict-style access (`get`,
    `[]`, `in`, `.items()`) while blocking writes.
    """
    if isinstance(value, MappingProxyType):
        return value
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


# ── What an extension contributes ─────────────────────────────────────────────

@dataclass(frozen=True)
class ExtensionFragment:
    """
    What an extension wants to add to ContextView.

    Core does not care where this came from —
    CLAUDE.md, a skill file, a plugin, a database.
    That is the adapter's concern.

    Core only cares about:
    - what context fragment to inject
    - what tools are needed
    - what policy changes are requested
    """
    name: str
    context_fragment: str              # text injected into ContextView
    required_tools: tuple[str, ...]    # tool names this extension needs
    policy_overrides: Mapping[str, Any]  # requested policy changes — may be denied
    source: str                        # human-readable origin, for trace

    def __post_init__(self) -> None:
        if not isinstance(self.policy_overrides, MappingProxyType):
            object.__setattr__(self, "policy_overrides", _freeze_mapping(self.policy_overrides))


@dataclass(frozen=True)
class ExtensionTool:
    """
    A tool an extension wants to register via capability/resolver.py.

    Never registered directly into a provider SDK.
    Always goes through capability resolution — policy decides if it appears.
    """
    name: str
    description: str
    parameters: Mapping[str, Any]
    source: str                        # which extension registered this

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, MappingProxyType):
            object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True)
class ExtensionCommand:
    """
    A slash command an extension wants to register via SlashCommandRouter.
    """
    name: str
    description: str
    source: str


@dataclass(frozen=True)
class ExtensionHook:
    """
    A subscription to a trace event kind.
    Handler is called after the event is recorded — never before.
    Extensions cannot intercept or block trace events.
    """
    event_kind: str    # matches TraceEventKind value
    handler: str       # dotted path to async callable
    source: str


# ── Extension loader contract ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ExtensionBundle:
    """
    Everything an extension loader contributes for one session.
    Produced by adapters, consumed by extensions/sanitizer.py in core.
    """
    fragments: tuple[ExtensionFragment, ...] = field(default_factory=tuple)
    tools: tuple[ExtensionTool, ...]         = field(default_factory=tuple)
    commands: tuple[ExtensionCommand, ...]   = field(default_factory=tuple)
    hooks: tuple[ExtensionHook, ...]         = field(default_factory=tuple)


class ExtensionLoader(ABC):
    """
    Contract for anything that loads extensions into a session.

    Implemented by adapters — not by core.

    Examples:
        axor-claude:  ClaudeSkillLoader   (reads CLAUDE.md, skills/)
                      ClaudePluginLoader  (reads .claude/plugins/)
        axor-openai:  OpenAIPluginLoader
        custom:       MyInternalToolLoader
    """

    @abstractmethod
    async def load(self) -> ExtensionBundle:
        """
        Load and return everything this loader contributes.
        Core will sanitize the bundle against policy before use.
        """
        ...
