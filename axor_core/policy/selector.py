from __future__ import annotations

from axor_core.contracts.policy import (
    TaskSignal,
    TaskComplexity,
    TaskNature,
    ExecutionPolicy,
    ContextMode,
    CompressionMode,
    ExportMode,
    ChildMode,
    ToolPolicy,
)


class PolicySelector:
    """
    Maps a TaskSignal to an ExecutionPolicy.

    Principle: minimum sufficient for quality.
    Not a hard cap — exactly what the task needs.

    A focused task gets minimal context not because we restrict it —
    but because that is sufficient for quality output.

    An expansive task gets broad context and full tools not because
    we override limits — but because the task genuinely requires it.

    Matrix:
        FOCUSED  + READONLY   → focused_readonly
        FOCUSED  + GENERATIVE → focused_generative
        FOCUSED  + MUTATIVE   → focused_mutative
        MODERATE + READONLY   → moderate_readonly
        MODERATE + GENERATIVE → moderate_generative
        MODERATE + MUTATIVE   → moderate_mutative
        EXPANSIVE + *         → expansive  (nature matters less at this scale)
    """

    def select(self, signal: TaskSignal) -> ExecutionPolicy:
        match (signal.complexity, signal.nature):

            case (TaskComplexity.FOCUSED, TaskNature.READONLY):
                # explain function, analyze code, find bug — no writes needed
                return ExecutionPolicy(
                    name="focused_readonly",
                    derived_from=signal.complexity,
                    context_mode=ContextMode.MINIMAL,
                    compression_mode=CompressionMode.AGGRESSIVE,
                    child_mode=ChildMode.DENIED,
                    max_child_depth=0,
                    tool_policy=ToolPolicy(
                        allow_read=True,
                        allow_write=False,
                        allow_bash=False,
                        allow_search=True,
                    ),
                    export_mode=ExportMode.SUMMARY,
                    child_context_fraction=0.0,
                )

            case (TaskComplexity.FOCUSED, TaskNature.GENERATIVE):
                # write a test, create a module — needs write, no bash
                return ExecutionPolicy(
                    name="focused_generative",
                    derived_from=signal.complexity,
                    context_mode=ContextMode.MINIMAL,
                    compression_mode=CompressionMode.BALANCED,
                    child_mode=ChildMode.DENIED,
                    max_child_depth=0,
                    tool_policy=ToolPolicy(
                        allow_read=True,
                        allow_write=True,
                        allow_bash=False,
                        allow_search=True,
                    ),
                    export_mode=ExportMode.SUMMARY,
                    child_context_fraction=0.0,
                )

            case (TaskComplexity.FOCUSED, TaskNature.MUTATIVE):
                # fix a bug, rename a function — write + bash for tests
                return ExecutionPolicy(
                    name="focused_mutative",
                    derived_from=signal.complexity,
                    context_mode=ContextMode.MINIMAL,
                    compression_mode=CompressionMode.BALANCED,
                    child_mode=ChildMode.DENIED,
                    max_child_depth=0,
                    tool_policy=ToolPolicy(
                        allow_read=True,
                        allow_write=True,
                        allow_bash=True,
                        allow_search=True,
                    ),
                    export_mode=ExportMode.SUMMARY,
                    child_context_fraction=0.0,
                )

            case (TaskComplexity.MODERATE, TaskNature.READONLY):
                # review a module, analyze architecture — broader context, no writes
                return ExecutionPolicy(
                    name="moderate_readonly",
                    derived_from=signal.complexity,
                    context_mode=ContextMode.MODERATE,
                    compression_mode=CompressionMode.BALANCED,
                    child_mode=ChildMode.DENIED,
                    max_child_depth=0,
                    tool_policy=ToolPolicy(
                        allow_read=True,
                        allow_write=False,
                        allow_bash=False,
                        allow_search=True,
                    ),
                    export_mode=ExportMode.FILTERED,
                    child_context_fraction=0.0,
                )

            case (TaskComplexity.MODERATE, TaskNature.GENERATIVE):
                # add a feature, implement an endpoint
                return ExecutionPolicy(
                    name="moderate_generative",
                    derived_from=signal.complexity,
                    context_mode=ContextMode.MODERATE,
                    compression_mode=CompressionMode.BALANCED,
                    child_mode=ChildMode.SHALLOW,
                    max_child_depth=1,
                    tool_policy=ToolPolicy(
                        allow_read=True,
                        allow_write=True,
                        allow_bash=True,
                        allow_search=True,
                    ),
                    export_mode=ExportMode.FILTERED,
                    child_context_fraction=0.3,
                )

            case (TaskComplexity.MODERATE, TaskNature.MUTATIVE):
                # refactor a module, replace a dependency
                return ExecutionPolicy(
                    name="moderate_mutative",
                    derived_from=signal.complexity,
                    context_mode=ContextMode.MODERATE,
                    compression_mode=CompressionMode.BALANCED,
                    child_mode=ChildMode.SHALLOW,
                    max_child_depth=1,
                    tool_policy=ToolPolicy(
                        allow_read=True,
                        allow_write=True,
                        allow_bash=True,
                        allow_search=True,
                    ),
                    export_mode=ExportMode.FILTERED,
                    child_context_fraction=0.4,
                )

            case (TaskComplexity.EXPANSIVE, _):
                # rewrite repo, migrate stack, architectural overhaul
                # agent needs everything — core governs each step, not the whole
                return ExecutionPolicy(
                    name="expansive",
                    derived_from=signal.complexity,
                    context_mode=ContextMode.BROAD,
                    compression_mode=CompressionMode.LIGHT,
                    child_mode=ChildMode.ALLOWED,
                    max_child_depth=3,
                    tool_policy=ToolPolicy(
                        allow_read=True,
                        allow_write=True,
                        allow_bash=True,
                        allow_search=True,
                        allow_spawn=True,
                    ),
                    export_mode=ExportMode.FULL,
                    child_context_fraction=0.6,
                )

            case _:
                return self._default()

    def _default(self) -> ExecutionPolicy:
        """
        Safest fallback. Used when signal is ambiguous and
        no external classifier is available to resolve it.
        Minimal context, no children, read-only.
        """
        return ExecutionPolicy(
            name="default",
            derived_from=TaskComplexity.FOCUSED,
            context_mode=ContextMode.MINIMAL,
            compression_mode=CompressionMode.AGGRESSIVE,
            child_mode=ChildMode.DENIED,
            max_child_depth=0,
            tool_policy=ToolPolicy(
                allow_read=True,
                allow_write=False,
                allow_bash=False,
                allow_search=False,
            ),
            export_mode=ExportMode.SUMMARY,
            child_context_fraction=0.0,
        )
