from __future__ import annotations

import dataclasses

from axor_core.contracts.extension import ExtensionFragment
from axor_core.contracts.policy import (
    EscalationPolicy,
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

    def __init__(
        self,
        *,
        consequence_ceiling=None,
        escalation_policy: EscalationPolicy | None = None,
        allowed_paths: tuple[str, ...] | None = None,
    ) -> None:
        # Deployment overlay from a profile (operator-wide knobs applied to every
        # per-task policy). None → no overlay.
        self._consequence_ceiling = consequence_ceiling
        self._overlay_escalation = escalation_policy
        self._overlay_allowed_paths = allowed_paths

    def compose(
        self,
        base: ExecutionPolicy,
        extensions: list[ExtensionFragment],
        parent_policy: ExecutionPolicy | None = None,
    ) -> ExecutionPolicy:
        policy = self.apply_extension_overrides(base, extensions)
        policy = self._apply_deployment_overlay(policy)
        if parent_policy is not None:
            policy = self.apply_parent_restrictions(policy, parent_policy)
        return policy

    def _apply_deployment_overlay(self, policy: ExecutionPolicy) -> ExecutionPolicy:
        """Impose operator-wide profile knobs (ceiling / escalation / workspace),
        applied before parent restrictions so a child can still be narrowed
        further, never widened past the operator's choice."""
        changes: dict = {}
        if self._consequence_ceiling is not None:
            changes["max_unattended_consequence"] = self._consequence_ceiling
        if self._overlay_escalation is not None:
            changes["escalation_policy"] = self._overlay_escalation
        if self._overlay_allowed_paths and not (policy.allowed_paths or ()):
            changes["allowed_paths"] = tuple(self._overlay_allowed_paths)
        return dataclasses.replace(policy, **changes) if changes else policy

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
            allow_read: bool
            allow_bash: bool
            allow_write: bool
            allow_search: bool
            allow_spawn: bool
            export_mode: str    (only to more restrictive mode)
        """
        if not extensions:
            return base

        tool_policy = base.tool_policy

        for fragment in extensions:
            overrides = fragment.policy_overrides

            # tools — extensions can request additional tools
            # but never beyond what base allows for this complexity level
            if overrides.get("allow_read") and base.tool_policy.allow_read:
                tool_policy = _with_tool(tool_policy, allow_read=True)
            if overrides.get("allow_bash") and base.tool_policy.allow_bash:
                tool_policy = _with_tool(tool_policy, allow_bash=True)
            if overrides.get("allow_write") and base.tool_policy.allow_write:
                tool_policy = _with_tool(tool_policy, allow_write=True)
            if overrides.get("allow_search") and base.tool_policy.allow_search:
                tool_policy = _with_tool(tool_policy, allow_search=True)
            if overrides.get("allow_spawn") and base.tool_policy.allow_spawn:
                tool_policy = _with_tool(tool_policy, allow_spawn=True)

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

        # allowed_paths: child cannot widen the parent's filesystem ceiling.
        allowed_paths = _restrict_allowed_paths(
            child_policy.allowed_paths, parent_policy.allowed_paths
        )

        # escalation_policy: child cannot escalate if parent forbids it;
        # grantable_tools is capped to parent's ceiling; numeric limits take the min.
        escalation_policy = _restrict_escalation(
            child_policy.escalation_policy, parent_policy.escalation_policy
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
            allowed_paths=allowed_paths,
            escalation_policy=escalation_policy,
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
    """Return a new ExecutionPolicy with selected fields overridden.

    Uses dataclasses.replace so fields not listed here (e.g. escalation_policy,
    allowed_paths) are preserved rather than silently reset to defaults.
    """
    return dataclasses.replace(base, **kwargs)


def _restrict_allowed_paths(
    child_paths: tuple[str, ...],
    parent_paths: tuple[str, ...],
) -> tuple[str, ...]:
    """Narrow a child's allowed_paths so it can never widen the parent ceiling.

    - parent unrestricted (empty) → child keeps its own restriction
    - child unrestricted but parent restricted → child inherits parent's ceiling
    - both restricted → keep only child roots contained by the parent ceiling
    """
    from axor_core.capability.lease_validator import path_matches_allowlist

    if not parent_paths:
        return child_paths
    if not child_paths:
        return parent_paths
    contained = tuple(p for p in child_paths if path_matches_allowlist(p, parent_paths))
    # If the child asked for roots entirely outside the parent ceiling, fall back
    # to the parent ceiling rather than granting an empty (deny-all) set.
    return contained or parent_paths


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


def _restrict_escalation(
    child: EscalationPolicy,
    parent: EscalationPolicy,
) -> EscalationPolicy:
    if not parent.allow_escalation:
        return EscalationPolicy(allow_escalation=False)
    parent_tools = frozenset(parent.grantable_tools)
    return EscalationPolicy(
        allow_escalation=child.allow_escalation and parent.allow_escalation,
        grantable_tools=tuple(t for t in child.grantable_tools if t in parent_tools),
        max_escalations=min(child.max_escalations, parent.max_escalations),
        max_ops_per_grant=min(child.max_ops_per_grant, parent.max_ops_per_grant),
        require_human=child.require_human or parent.require_human,
    )
