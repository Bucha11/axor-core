from __future__ import annotations

from axor_core.contracts.extension import (
    ExtensionBundle,
    ExtensionFragment,
    ExtensionTool,
    ExtensionCommand,
    ExtensionHook,
)


class ExtensionRegistry:
    """
    Holds all sanitized extensions active for a session.

    Populated during GovernedSession.start() from loader bundles.
    Read during node execution to build the ExtensionBundle
    that goes into GovernedNode.run().

    One registry per session — not shared across sessions.
    """

    def __init__(self) -> None:
        self._fragments: list[ExtensionFragment] = []
        self._tools: list[ExtensionTool]         = []
        self._commands: list[ExtensionCommand]   = []
        self._hooks: list[ExtensionHook]         = []

    def register(self, bundle: ExtensionBundle) -> None:
        """Add a sanitized bundle to the registry."""
        self._fragments.extend(bundle.fragments)
        self._tools.extend(bundle.tools)
        self._commands.extend(bundle.commands)
        self._hooks.extend(bundle.hooks)
        # track skill names for SKILL_ACTIVATED events
        if not hasattr(self, '_registered_skills'):
            self._registered_skills: list[str] = []
        for f in bundle.fragments:
            self._registered_skills.append(f.name)

    def current_bundle(self) -> ExtensionBundle:
        """Return current registry contents as a single bundle."""
        return ExtensionBundle(
            fragments=tuple(self._fragments),
            tools=tuple(self._tools),
            commands=tuple(self._commands),
            hooks=tuple(self._hooks),
        )

    def fragment_count(self) -> int:
        return len(self._fragments)

    def tool_count(self) -> int:
        return len(self._tools)

    def clear(self) -> None:
        """Reset registry — used on session clear."""
        self._fragments.clear()
        self._tools.clear()
        self._commands.clear()
        self._hooks.clear()
