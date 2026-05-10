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
                output = await self._handle_context_async(command, session)
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
                summary = session.cache_summary()
                inp = summary["input_tokens"]
                out = summary["output_tokens"]
                cw  = summary["cache_creation_input_tokens"]
                cr  = summary["cache_read_input_tokens"]
                hit = summary["hit_rate"]
                has_cache = bool(cw or cr)

                lines = [f"Total billable tokens: {total:,}"]
                if has_cache:
                    lines.extend([
                        f"  input (incl. cache): {summary['total_input_tokens']:,}",
                        f"    uncached input:   {inp:,}",
                        f"    cache writes:     {cw:,} (billed at 1.25x input)",
                        f"    cache reads:      {cr:,} (billed at 0.1x input)",
                        f"    cache hit rate:   {hit:.1%}",
                        f"  output:              {out:,}",
                        "",
                        f"Tokens (uncached input + output): {inp + out:,}",
                    ])
                else:
                    lines.extend([
                        f"  input:  {inp:,}",
                        f"  output: {out:,}",
                    ])

                cost = session.cost_summary()
                if cost is not None:
                    currency = cost["currency"]
                    lines.append("")
                    lines.append(
                        f"estimated cost: {cost['total_cost']:.6f} {currency}"
                    )
                    if has_cache:
                        lines.extend([
                            f"  uncached input: {cost['input_cost']:.6f}",
                            f"  cache writes:   {cost['cache_creation_cost']:.6f}",
                            f"  cache reads:    {cost['cache_read_cost']:.6f}",
                            f"  output:         {cost['output_cost']:.6f}",
                        ])
                    else:
                        lines.extend([
                            f"  input:  {cost['input_cost']:.6f}",
                            f"  output: {cost['output_cost']:.6f}",
                        ])
                return "\n".join(lines)

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

    async def _handle_context_async(self, command: SlashCommand, session) -> str:
        match command.name:
            case "compact":
                before, after = session.compact_context()
                saved = max(0, before - after)
                pct = int(100 * saved / before) if before > 0 else 0
                return (
                    f"Context compacted: {before:,} → {after:,} tokens "
                    f"({saved:,} saved, {pct}% reduction)"
                )
            case "clear":
                removed = session.clear_context()
                noun = "fragment" if removed == 1 else "fragments"
                return f"Context cleared: {removed} {noun} removed."
            case "memory":
                return await self._handle_memory(command, session)
            case _:
                return f"Unknown context command: /{command.name}"

    async def _handle_memory(self, command: SlashCommand, session) -> str:
        """
        /memory              — list fragments in active namespace
        /memory add <text>   — save a new fragment
        /memory forget <key> — delete a fragment by key
        /memory search <q>   — full-text search
        /memory clear        — delete all fragments in namespace
        """
        if session._memory_provider is None:
            return (
                "Memory provider not configured.\n"
                "Add memory_provider=SQLiteMemoryProvider() to your session."
            )

        args = command.args.strip()
        ns = session.memory_namespace()

        if not args or args == "list":
            fragments = await session.list_memories(ns)
            if not fragments:
                return f"No memories in namespace '{ns}'."
            lines = [f"Memories ({len(fragments)}) in '{ns}':"]
            for f in fragments:
                preview = f.content[:80].replace("\n", " ")
                ellipsis = "…" if len(f.content) > 80 else ""
                lines.append(f"  [{f.key}] ({f.value.value})  {preview}{ellipsis}")
            return "\n".join(lines)

        sub, _, rest = args.partition(" ")
        sub = sub.lower()

        if sub == "add":
            if not rest.strip():
                return "Usage: /memory add <text>"
            await session.save_memory(rest.strip(), namespace=ns)
            return f"Saved to memory (namespace: {ns})."

        if sub == "forget":
            key = rest.strip()
            if not key:
                return "Usage: /memory forget <key>"
            deleted = await session.forget_memory(key, namespace=ns)
            return f"Deleted {deleted} fragment(s)." if deleted else f"Key '{key}' not found."

        if sub == "search":
            if not rest.strip():
                return "Usage: /memory search <query>"
            results = await session.search_memories(rest.strip(), namespace=ns)
            if not results:
                return "No matches."
            lines = [f"Search results ({len(results)}):"]
            for f in results:
                preview = f.content[:80].replace("\n", " ")
                ellipsis = "…" if len(f.content) > 80 else ""
                lines.append(f"  [{f.key}]  {preview}{ellipsis}")
            return "\n".join(lines)

        if sub == "clear":
            if session._memory_provider is not None:
                deleted = await session._memory_provider.delete(
                    ns,
                    [f.key for f in await session.list_memories(ns)],
                )
                return f"Cleared {deleted} fragment(s) from '{ns}'."
            return "No memory provider."

        # unknown subcommand — treat entire args as text to save
        await session.save_memory(args, namespace=ns)
        return f"Saved to memory (namespace: {ns})."

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
