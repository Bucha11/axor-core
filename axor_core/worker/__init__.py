"""
axor_core.worker
────────────────
Entry layer — accepts input, routes to governed node flow.

    GovernedSession     — public interface, what users instantiate
    SlashCommandRouter  — classifies and routes slash commands

Worker starts execution.
Node governs execution.
"""

from axor_core.worker.session import GovernedSession
from axor_core.worker.commands import SlashCommandRouter

__all__ = [
    "GovernedSession",
    "SlashCommandRouter",
]
