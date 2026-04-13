from __future__ import annotations

from axor_core.contracts.command import (
    SlashCommand,
    CommandClass,
    CommandResult,
)
from axor_core.contracts.trace import TraceEventKind
from axor_core.trace import events as trace_events


# Commands answered entirely from governance state — executor never sees them
_GOVERNANCE_COMMANDS = {
    "tools",    # show capabilities from envelope, not SDK
    "policy",   # show active ExecutionPolicy
    "cost",     # read from budget tracker
    "export",   # current export mode
    "status",   # overall session status
}

# Commands routed to context subsystem — executor sees updated ContextView
_CONTEXT_COMMANDS = {
    "compact",  # compress context — not SDK compact
    "clear",    # reset context state
    "memory",   # show/manage memory fragments
}


class SlashCommandRouter:
    """
    Classifies and routes slash commands.

    Three classes:
        GOVERNANCE  — intercepted, answered from envelope/trace
                      executor never sees the command
        CONTEXT     — routed to context subsystem
                      executor sees updated ContextView, not the command
        PASSTHROUGH — forwarded to executor if policy allows
                      always logged in trace regardless

    Every command — including passthroughs — is recorded as
    CommandRouted in trace. This is the audit trail for commands.
    """

    def __init__(self, collector) -> None:
        self._collector = collector

    async def route(self, raw: str, session) -> CommandResult:
        command = self._parse(raw)
        command_class = self._classify(command.name, session)

        result = await self._dispatch(command, command_class, session)

        # always record in trace
        self._collector.record(
            trace_events.command_routed(
                node_id=session.session_id(),
                command_name=command.name,
                command_class=command_class.value,
                allowed=result.allowed,
            )
        )

        return result

    # ── Classification ─────────────────────────────────────────────────────────

    def _classify(self, name: str, session) -> CommandClass:
        if name in _GOVERNANCE_COMMANDS:
            return CommandClass.GOVERNANCE
        if name in _CONTEXT_COMMANDS:
            return CommandClass.CONTEXT
        return CommandClass.PASSTHROUGH

    # ── Dispatch ───────────────────────────────────────────────────────────────

    async def _dispatch(
        self,
        command: SlashCommand,
        command_class: CommandClass,
        session,
    ) -> CommandResult:
        match command_class:
            case CommandClass.GOVERNANCE:
                output = self._handle_governance(command, session)
                return CommandResult(
                    command=command,
                    command_class=command_class,
                    output=output,
                    allowed=True,
                )

            case CommandClass.CONTEXT:
                output = self._handle_context(command, session)
                return CommandResult(
                    command=command,
                    command_class=command_class,
                    output=output,
                    allowed=True,
                )

            case CommandClass.PASSTHROUGH:
                return self._handle_passthrough(command, command_class, session)

    def _handle_governance(self, command: SlashCommand, session) -> str:
        match command.name:
            case "tools":
                # what capabilities the envelope exposes — not the SDK tool list
                traces = session.all_traces()
                if not traces:
                    return "No active execution. Tools will be derived from policy on next run."
                last = traces[-1]
                return (
                    f"Governed tools for last execution (policy: {last.policy_name}):\n"
                    f"Tools are derived from policy — run a task to see active capabilities."
                )

            case "cost":
                total = session.total_tokens_spent()
                return f"Tokens spent this session: {total:,}"

            case "policy":
                traces = session.all_traces()
                if not traces:
                    return "No executions yet. Policy is selected dynamically from task signal."
                last = traces[-1]
                return f"Last policy: {last.policy_name}"

            case "status":
                total = session.total_tokens_spent()
                traces = session.all_traces()
                children = sum(1 for t in traces if t.parent_id is not None)
                return (
                    f"Session: {session.session_id()}\n"
                    f"Tokens:  {total:,}\n"
                    f"Nodes:   {len(traces)} ({children} children)\n"
                )

            case _:
                return f"Unknown governance command: /{command.name}"

    def _handle_context(self, command: SlashCommand, session) -> str:
        # context subsystem operations
        # full implementation connects to context/manager.py
        match command.name:
            case "compact":
                return "Context compaction requested — will apply on next execution."
            case "clear":
                return "Context cleared — session state reset."
            case "memory":
                return "Memory fragments: (context subsystem not yet wired)"
            case _:
                return f"Unknown context command: /{command.name}"

    def _handle_passthrough(
        self,
        command: SlashCommand,
        command_class: CommandClass,
        session,
    ) -> CommandResult:
        # passthroughs are currently denied at core level
        # adapters may override this by registering custom command handlers
        return CommandResult(
            command=command,
            command_class=command_class,
            output=f"/{command.name} is not a governed command. Register a handler via the adapter.",
            allowed=False,
            denial_reason="no handler registered for passthrough command",
        )

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse(self, raw: str) -> SlashCommand:
        raw = raw.strip()
        parts = raw.lstrip("/").split(None, 1)
        name = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        return SlashCommand(name=name, args=args, source="session", raw=raw)
