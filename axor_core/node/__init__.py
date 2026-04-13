"""
axor_core.node
──────────────
The governance boundary around an executor.

    GovernedNode    — central primitive, ties all subsystems together
    EnvelopeBuilder — constructs ExecutionEnvelope
    IntentLoop      — intercepts stream, resolves intents
    ExportFilter    — applies export contract to output
    ChildSpawner    — governed child-node creation
"""

from axor_core.node.wrapper import GovernedNode
from axor_core.node.envelope import EnvelopeBuilder
from axor_core.node.intent_loop import IntentLoop
from axor_core.node.export import ExportFilter
from axor_core.node.spawn import ChildSpawner

__all__ = [
    "GovernedNode",
    "EnvelopeBuilder",
    "IntentLoop",
    "ExportFilter",
    "ChildSpawner",
]
