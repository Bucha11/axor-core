from __future__ import annotations

from axor_core.contracts.envelope import Capabilities
from axor_core.contracts.extension import ExtensionTool
from axor_core.contracts.policy import ExecutionPolicy, ChildMode

# ── Built-in tool names ────────────────────────────────────────────────────────
#
# These are the canonical tool names core knows about.
# Adapters map these to provider-native tool definitions.
# Extensions may add to this set via ExtensionTool.

TOOL_READ   = "read"
TOOL_WRITE  = "write"
TOOL_BASH   = "bash"
TOOL_SEARCH = "search"
TOOL_SPAWN  = "spawn_child"


class CapabilityResolver:
    """
    Derives Capabilities from ExecutionPolicy + registered extension tools.

    This is the single place where policy becomes operational permissions.

    Rules:
    - Capabilities are always derived — never self-assigned by executors
    - Extension tools are included only if base policy allows the tool class
    - extra_denied always wins over extra_allowed (explicit deny > allow)
    - spawn capability requires both allow_spawn in ToolPolicy AND
      child_mode != DENIED in policy

    The resulting Capabilities object is what goes into ExecutionEnvelope.
    The executor sees only what Capabilities contains — nothing more.
    """

    def resolve(
        self,
        policy: ExecutionPolicy,
        extension_tools: list[ExtensionTool] | None = None,
    ) -> Capabilities:
        allowed = self._resolve_builtin_tools(policy)
        allowed = self._apply_extension_tools(allowed, policy, extension_tools or [])
        allowed = self._apply_deny_list(allowed, policy)

        allow_children = (
            policy.child_mode != ChildMode.DENIED
            and policy.tool_policy.allow_spawn
        )
        allow_nested = (
            allow_children
            and policy.child_mode == ChildMode.ALLOWED
            and policy.max_child_depth > 1
        )

        return Capabilities(
            allowed_tools=frozenset(allowed),
            allow_children=allow_children,
            allow_nested_children=allow_nested,
            allow_context_expansion=self._allow_context_expansion(policy),
            allow_export=policy.export_mode != "restricted",
            allow_mutation=policy.tool_policy.allow_write or policy.tool_policy.allow_bash,
            max_child_depth=policy.max_child_depth,
        )

    # ── private ────────────────────────────────────────────────────────────────

    def _resolve_builtin_tools(self, policy: ExecutionPolicy) -> set[str]:
        tp = policy.tool_policy
        tools: set[str] = set()

        if tp.allow_read:
            tools.add(TOOL_READ)
        if tp.allow_write:
            tools.add(TOOL_WRITE)
        if tp.allow_bash:
            tools.add(TOOL_BASH)
        if tp.allow_search:
            tools.add(TOOL_SEARCH)
        if tp.allow_spawn and policy.child_mode != ChildMode.DENIED:
            tools.add(TOOL_SPAWN)

        # extra_allowed — additional named tools requested by policy
        tools.update(policy.tool_policy.extra_allowed)

        return tools

    def _apply_extension_tools(
        self,
        allowed: set[str],
        policy: ExecutionPolicy,
        extension_tools: list[ExtensionTool],
    ) -> set[str]:
        """
        Add extension tools only if they pass the policy gate.

        Extension tools are accepted when:
        - they are already in extra_allowed (explicitly granted), OR
        - they map to a builtin class that policy allows

        Extensions never add tools that policy has not enabled at the class level.
        """
        for ext_tool in extension_tools:
            name = ext_tool.name

            if name in policy.tool_policy.extra_denied:
                continue  # explicit deny — skip immediately

            if name in policy.tool_policy.extra_allowed:
                allowed.add(name)  # explicitly granted
                continue

            # infer tool class from name prefix and check policy
            if name.startswith("read_") and policy.tool_policy.allow_read:
                allowed.add(name)
            elif name.startswith("write_") and policy.tool_policy.allow_write:
                allowed.add(name)
            elif name.startswith("search_") and policy.tool_policy.allow_search:
                allowed.add(name)
            elif name.startswith("bash_") and policy.tool_policy.allow_bash:
                allowed.add(name)
            # unknown prefix — not added unless explicitly in extra_allowed

        return allowed

    def _apply_deny_list(self, allowed: set[str], policy: ExecutionPolicy) -> set[str]:
        """extra_denied always wins — explicit deny overrides everything."""
        return allowed - set(policy.tool_policy.extra_denied)

    def _allow_context_expansion(self, policy: ExecutionPolicy) -> bool:
        """
        Context expansion (executor requests more context visibility) is
        allowed only for moderate and broad context modes.
        Minimal mode means we deliberately scoped context — don't expand.
        """
        from axor_core.contracts.policy import ContextMode
        return policy.context_mode != ContextMode.MINIMAL
