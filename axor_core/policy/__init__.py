"""
axor_core.policy
────────────────
Policy selection and composition.

Public surface:

    TaskAnalyzer    — raw input → TaskSignal (heuristic + optional external classifier)
    PolicySelector  — TaskSignal → ExecutionPolicy
    PolicyComposer  — merge base policy with extension overrides and parent restrictions
    presets         — named ready-to-use policies
"""

from axor_core.policy.analyzer import TaskAnalyzer
from axor_core.policy.heuristic import HeuristicClassifier
from axor_core.policy.selector import PolicySelector
from axor_core.policy.composer import PolicyComposer
from axor_core.policy import presets

__all__ = [
    "TaskAnalyzer",
    "HeuristicClassifier",
    "PolicySelector",
    "PolicyComposer",
    "presets",
]
