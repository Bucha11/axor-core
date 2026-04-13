"""
axor_core.worker
────────────────
Entry layer — accepts input, routes to governed node flow.

    GovernedSession     — public interface, what users instantiate
    SlashCommandRouter  — classifies and routes slash commands
    Dispatcher          — handoff from worker to node flow

Worker starts execution.
Node governs execution.
"""

from axor_core.worker.session import GovernedSession
from axor_core.worker.commands import SlashCommandRouter
from axor_core.worker.dispatcher import Dispatcher

__all__ = [
    "GovernedSession",
    "SlashCommandRouter",
    "Dispatcher",
]
