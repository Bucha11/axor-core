from __future__ import annotations

from axor_core.contracts.extension import ExtensionFragment
from axor_core.contracts.policy import (
    ExecutionPolicy,
    ToolPolicy,
    ChildMode,
    CompressionMode,
    ContextMode,
    ExportMode,
)


class PolicyComposer:
    """
    Merges an ExecutionPolicy with external inputs.

    Three merge operations:

    1. apply_extension_overrides()
       Skills and plugins may request policy changes.
       Core decides what to grant — not the extension.

    2. apply_parent_restrictions()
       Child nodes inherit restrictions from parents.
       A parent can never grant a child more than it has itself.

    3. compose()
       Full pipeline: base → extensions → parent restrictions.
    """

    def compose(
        self,
        base: ExecutionPolicy,
        extensions: list[ExtensionFragment],
        parent_policy: ExecutionPolicy | None = None,
    ) -> ExecutionPolicy:
        policy = self.apply_extension_overrides(base, extensions)
        if parent_policy is not None:
            policy = self.apply_parent_restrictions(policy, parent_policy)
        return policy

    def apply_extension_overrides(
        self,
        base: ExecutionPolicy,
        extensions: list[ExtensionFragment],
    ) -> ExecutionPolicy:
        """
        Apply requested policy overrides from extensions.

        Extensions request — policy grants or denies.
        An extension can never escalate beyond what base policy allows.

        Supported overrides:
            allow_bash: bool
            allow_write: bool
            allow_search: bool
            export_mode: str    (only to more restrictive mode)
        """
        if not extensions:
            return base

        tool_policy = base.tool_policy

        for fragment in extensions:
            overrides = fragment.policy_overrides

            # tools — extensions can request additional tools
            # but never beyond what base allows for this complexity level
            if overrides.get("allow_bash") and base.tool_policy.allow_bash:
                tool_policy = _with_tool(tool_policy, allow_bash=True)
            if overrides.get("allow_write") and base.tool_policy.allow_write:
                tool_policy = _with_tool(tool_policy, allow_write=True)
            if overrides.get("allow_search") and base.tool_policy.allow_search:
                tool_policy = _with_tool(tool_policy, allow_search=True)

            # extra_allowed — extension-specific tool names
            if extra := overrides.get("extra_allowed_tools", []):
                tool_policy = _with_tool(
                    tool_policy,
                    extra_allowed=tuple(set(tool_policy.extra_allowed) | set(extra)),
                )

        return _with_policy(base, tool_policy=tool_policy)

    def apply_parent_restrictions(
        self,
        child_policy: ExecutionPolicy,
        parent_policy: ExecutionPolicy,
    ) -> ExecutionPolicy:
        """
        Enforce that a child never exceeds parent governance.

        A parent cannot grant a child more than it has itself.
        This is the fundamental federation invariant.
        """
        parent_tools = parent_policy.tool_policy
        child_tools  = child_policy.tool_policy

        restricted_tools = ToolPolicy(
            allow_read=child_tools.allow_read and parent_tools.allow_read,
            allow_write=child_tools.allow_write and parent_tools.allow_write,
            allow_bash=child_tools.allow_bash and parent_tools.allow_bash,
            allow_search=child_tools.allow_search and parent_tools.allow_search,
            allow_spawn=child_tools.allow_spawn and parent_tools.allow_spawn,
            # child keeps its own extra_allowed only if parent also allows those tools
            extra_allowed=tuple(
                t for t in child_tools.extra_allowed
                if t in parent_tools.extra_allowed
            ),
            extra_denied=tuple(
                set(child_tools.extra_denied) | set(parent_tools.extra_denied)
            ),
        )

        # child depth cannot exceed parent's remaining depth budget
        max_depth = min(
            child_policy.max_child_depth,
            max(0, parent_policy.max_child_depth - 1),
        )

        # export cannot be more permissive than parent
        export_mode = _most_restrictive_export(
            child_policy.export_mode,
            parent_policy.export_mode,
        )

        # child mode — if parent denies children, so does child;
        # if parent only allows shallow, child cannot be ALLOWED.
        child_mode = _most_restrictive_child_mode(
            child_policy.child_mode, parent_policy.child_mode
        )

        # context_mode and compression_mode: child cannot be more permissive
        context_mode = _most_restrictive_context(
            child_policy.context_mode, parent_policy.context_mode
        )
        compression_mode = _most_restrictive_compression(
            child_policy.compression_mode, parent_policy.compression_mode
        )

        # child_context_fraction: less inheritance is more restrictive
        child_context_fraction = min(
            child_policy.child_context_fraction,
            parent_policy.child_context_fraction,
        )

        # passthrough commands: child keeps only what parent also allows
        passthrough = tuple(
            c for c in child_policy.allowed_passthrough_commands
            if c in parent_policy.allowed_passthrough_commands
        )

        # allow_model_switch: AND of both
        allow_model_switch = (
            child_policy.allow_model_switch and parent_policy.allow_model_switch
        )

        return _with_policy(
            child_policy,
            tool_policy=restricted_tools,
            max_child_depth=max_depth,
            child_mode=child_mode,
            export_mode=export_mode,
            context_mode=context_mode,
            compression_mode=compression_mode,
            child_context_fraction=child_context_fraction,
            allowed_passthrough_commands=passthrough,
            allow_model_switch=allow_model_switch,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _with_tool(base: ToolPolicy, **kwargs) -> ToolPolicy:
    """Return a new ToolPolicy with selected fields overridden."""
    return ToolPolicy(
        allow_read=kwargs.get("allow_read", base.allow_read),
        allow_write=kwargs.get("allow_write", base.allow_write),
        allow_bash=kwargs.get("allow_bash", base.allow_bash),
        allow_search=kwargs.get("allow_search", base.allow_search),
        allow_spawn=kwargs.get("allow_spawn", base.allow_spawn),
        extra_allowed=kwargs.get("extra_allowed", base.extra_allowed),
        extra_denied=kwargs.get("extra_denied", base.extra_denied),
    )


def _with_policy(base: ExecutionPolicy, **kwargs) -> ExecutionPolicy:
    """Return a new ExecutionPolicy with selected fields overridden."""
    return ExecutionPolicy(
        name=kwargs.get("name", base.name),
        derived_from=kwargs.get("derived_from", base.derived_from),
        context_mode=kwargs.get("context_mode", base.context_mode),
        compression_mode=kwargs.get("compression_mode", base.compression_mode),
        child_mode=kwargs.get("child_mode", base.child_mode),
        max_child_depth=kwargs.get("max_child_depth", base.max_child_depth),
        tool_policy=kwargs.get("tool_policy", base.tool_policy),
        export_mode=kwargs.get("export_mode", base.export_mode),
        child_context_fraction=kwargs.get("child_context_fraction", base.child_context_fraction),
        allowed_passthrough_commands=kwargs.get(
            "allowed_passthrough_commands", base.allowed_passthrough_commands
        ),
        allow_model_switch=kwargs.get("allow_model_switch", base.allow_model_switch),
    )


_EXPORT_RESTRICTIVENESS = {
    ExportMode.FULL:       0,
    ExportMode.FILTERED:   1,
    ExportMode.SUMMARY:    2,
    ExportMode.RESTRICTED: 3,
}


def _most_restrictive_export(a: ExportMode, b: ExportMode) -> ExportMode:
    return a if _EXPORT_RESTRICTIVENESS[a] >= _EXPORT_RESTRICTIVENESS[b] else b


# Higher rank == more restrictive.
_CONTEXT_RESTRICTIVENESS = {
    ContextMode.BROAD:    0,
    ContextMode.MODERATE: 1,
    ContextMode.MINIMAL:  2,
}


def _most_restrictive_context(a: ContextMode, b: ContextMode) -> ContextMode:
    return a if _CONTEXT_RESTRICTIVENESS[a] >= _CONTEXT_RESTRICTIVENESS[b] else b


_COMPRESSION_RESTRICTIVENESS = {
    CompressionMode.LIGHT:      0,
    CompressionMode.BALANCED:   1,
    CompressionMode.AGGRESSIVE: 2,
}


def _most_restrictive_compression(
    a: CompressionMode, b: CompressionMode
) -> CompressionMode:
    return a if _COMPRESSION_RESTRICTIVENESS[a] >= _COMPRESSION_RESTRICTIVENESS[b] else b


_CHILD_MODE_RESTRICTIVENESS = {
    ChildMode.ALLOWED: 0,
    ChildMode.SHALLOW: 1,
    ChildMode.DENIED:  2,
}


def _most_restrictive_child_mode(a: ChildMode, b: ChildMode) -> ChildMode:
    return a if _CHILD_MODE_RESTRICTIVENESS[a] >= _CHILD_MODE_RESTRICTIVENESS[b] else b
