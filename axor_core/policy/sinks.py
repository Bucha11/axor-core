"""Sink classification sets shared across the gate sequence.

These name the two structural sink categories the carrier and positional gates
key on. They live here (not in the IntentLoop) so any per-call governor —
streaming or synchronous — resolves the same sink semantics from one source.
"""
from __future__ import annotations

# Instruction-following sinks: a sink that would interpret its argument as a
# directive (spawn a sub-agent, send a message, post, notify). An untrusted
# free-text value reaching one of these is the imperative channel.
IMPERATIVE_SINKS = frozenset({
    "spawn_child", "send", "message", "prompt", "ask", "delegate",
    "reply", "email", "slack", "post", "notify",
})

# Sinks whose input space is instruction-COMPLETE by definition: they interpret
# their argument as a program / directive (a shell command, a child-agent task).
# These can NEVER be declared positional — a positional gate would either deny
# every legitimate call (their legit input IS free text) or, worse, admit a
# closed-schema string the sink still executes. They stay on the
# content-derivation path with the paraphrase residual acknowledged. Declaring
# one positional is a configuration error.
INSTRUCTION_COMPLETE_SINKS = frozenset({
    "bash", "shell", "execute", "run", "exec", "execute_generated_code",
    "spawn_child", "eval", "python", "sh", "command", "system",
})


def is_imperative_sink(tool_name: str, normalized) -> bool:
    """Instruction-following sink: it would interpret its argument as a directive
    (spawn a sub-agent, send a message, execute generated code). The imperative
    channel — distinct from the risky-op list, which misses free-text-as-directive
    (e.g. spawn_child(task=<free text>))."""
    return (
        tool_name.lower() in IMPERATIVE_SINKS
        or getattr(normalized, "executes_generated_code", False)
        or getattr(normalized, "operation", "") == "execute_generated_code"
    )
