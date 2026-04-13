from __future__ import annotations

from axor_core.contracts.extension import (
    ExtensionBundle,
    ExtensionFragment,
    ExtensionTool,
    ExtensionCommand,
    ExtensionHook,
)
from axor_core.contracts.trace import TraceEventKind


# Tool name prefixes that are always denied regardless of policy
_ALWAYS_DENIED_PREFIXES = ("__", "axor_internal_")

# Max context fragment size — prevents extensions injecting huge prompts
_MAX_FRAGMENT_TOKENS = 2000


class ExtensionSanitizer:
    """
    Filters an ExtensionBundle against safety rules before it reaches core.

    Called by GovernedSession.start() after each loader returns a bundle.
    The sanitized bundle goes into ExtensionRegistry.

    Sanitizer does not know about policy — it applies baseline safety rules.
    Policy-level filtering happens in capability/resolver.py and
    policy/composer.py when the bundle is applied to a specific execution.

    Rules:
        fragments   — max token size enforced, source recorded
        tools       — denied if name starts with reserved prefix
        commands    — denied if name conflicts with governance commands
        hooks       — accepted as-is (handlers are dotted paths, not code)
    """

    # Governance command names that extensions cannot override
    _RESERVED_COMMANDS = {
        "tools", "policy", "cost", "export", "status",
        "compact", "clear", "memory",
    }

    def sanitize(self, bundle: ExtensionBundle) -> ExtensionBundle:
        return ExtensionBundle(
            fragments=tuple(self._sanitize_fragments(bundle.fragments)),
            tools=tuple(self._sanitize_tools(bundle.tools)),
            commands=tuple(self._sanitize_commands(bundle.commands)),
            hooks=bundle.hooks,   # hooks pass through — no code execution at load time
        )

    def _sanitize_fragments(
        self, fragments: tuple[ExtensionFragment, ...]
    ) -> list[ExtensionFragment]:
        result = []
        for fragment in fragments:
            token_estimate = len(fragment.context_fragment) // 4
            if token_estimate > _MAX_FRAGMENT_TOKENS:
                # truncate rather than reject — partial context is better than none
                truncated = fragment.context_fragment[: _MAX_FRAGMENT_TOKENS * 4]
                fragment = ExtensionFragment(
                    name=fragment.name,
                    context_fragment=truncated + "\n[truncated by sanitizer]",
                    required_tools=fragment.required_tools,
                    policy_overrides=fragment.policy_overrides,
                    source=fragment.source,
                )
            result.append(fragment)
        return result

    def _sanitize_tools(
        self, tools: tuple[ExtensionTool, ...]
    ) -> list[ExtensionTool]:
        result = []
        for tool in tools:
            if any(tool.name.startswith(p) for p in _ALWAYS_DENIED_PREFIXES):
                continue   # silently drop reserved names
            result.append(tool)
        return result

    def _sanitize_commands(
        self, commands: tuple[ExtensionCommand, ...]
    ) -> list[ExtensionCommand]:
        result = []
        for cmd in commands:
            if cmd.name.lower() in self._RESERVED_COMMANDS:
                continue   # cannot override governance commands
            result.append(cmd)
        return result
