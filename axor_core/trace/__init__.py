"""
axor_core.trace
───────────────
Governance decision visibility and lineage continuity.

    TraceCollector  — collects events across all nodes in the lineage tree
    events          — typed event constructors

Trace is not flat logging.
Every event belongs to a node which belongs to a lineage.
PolicyAdjusted events are the most valuable — they show where
classification was wrong and what it cost in tokens.
"""

from axor_core.trace.collector import TraceCollector
from axor_core.trace import events

__all__ = [
    "TraceCollector",
    "events",
]
