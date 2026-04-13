from __future__ import annotations

from axor_core.contracts.policy import (
    ExecutionPolicy,
    TaskComplexity,
    ContextMode,
    CompressionMode,
    ExportMode,
    ChildMode,
    ToolPolicy,
)


def readonly() -> ExecutionPolicy:
    """
    Strict read-only. No writes, no bash, no children.
    Use for analysis, review, explanation tasks.
    """
    return ExecutionPolicy(
        name="preset:readonly",
        derived_from=TaskComplexity.FOCUSED,
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


def sandboxed() -> ExecutionPolicy:
    """
    Maximum restriction. Read-only, no search, no children, restricted export.
    Use when you want to observe what an agent would do, not let it act.
    """
    return ExecutionPolicy(
        name="preset:sandboxed",
        derived_from=TaskComplexity.FOCUSED,
        context_mode=ContextMode.MINIMAL,
        compression_mode=CompressionMode.AGGRESSIVE,
        child_mode=ChildMode.DENIED,
        max_child_depth=0,
        tool_policy=ToolPolicy(
            allow_read=False,
            allow_write=False,
            allow_bash=False,
            allow_search=False,
        ),
        export_mode=ExportMode.RESTRICTED,
        child_context_fraction=0.0,
    )


def standard() -> ExecutionPolicy:
    """
    Balanced default for typical coding tasks.
    Read + write + bash. No spawn. Filtered export.
    """
    return ExecutionPolicy(
        name="preset:standard",
        derived_from=TaskComplexity.MODERATE,
        context_mode=ContextMode.MODERATE,
        compression_mode=CompressionMode.BALANCED,
        child_mode=ChildMode.DENIED,
        max_child_depth=0,
        tool_policy=ToolPolicy(
            allow_read=True,
            allow_write=True,
            allow_bash=True,
            allow_search=True,
        ),
        export_mode=ExportMode.FILTERED,
        child_context_fraction=0.0,
    )


def federated() -> ExecutionPolicy:
    """
    Full federation. All tools, children allowed up to depth 3.
    Use for expansive tasks where agent-driven decomposition is expected.
    """
    return ExecutionPolicy(
        name="preset:federated",
        derived_from=TaskComplexity.EXPANSIVE,
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


def research() -> ExecutionPolicy:
    """
    Research domain. Read-heavy, broad context, light compression.
    No writes. Search + read. Children shallow (for parallel document analysis).

    Designed for: literature review, document synthesis, knowledge gathering.
    Context: BROAD — research tasks need wide context to synthesize correctly.
    Compression: LIGHT — knowledge fragments must not be over-compressed.
    """
    return ExecutionPolicy(
        name="preset:research",
        derived_from=TaskComplexity.MODERATE,
        context_mode=ContextMode.BROAD,
        compression_mode=CompressionMode.LIGHT,
        child_mode=ChildMode.SHALLOW,
        max_child_depth=1,
        tool_policy=ToolPolicy(
            allow_read=True,
            allow_write=False,
            allow_bash=False,
            allow_search=True,
            allow_spawn=True,
        ),
        export_mode=ExportMode.FULL,
        child_context_fraction=0.4,
    )


def support() -> ExecutionPolicy:
    """
    Support domain. Short turns, minimal context, fast responses.
    Read + search only. No writes, no children.

    Designed for: Q&A, diagnosis, quick explanations, help requests.
    Context: MINIMAL — support tasks are usually self-contained.
    Compression: AGGRESSIVE — keep turns lean, avoid context accumulation.
    """
    return ExecutionPolicy(
        name="preset:support",
        derived_from=TaskComplexity.FOCUSED,
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


def analysis() -> ExecutionPolicy:
    """
    Analysis domain. Structured reading and synthesis, moderate context.
    Read + search + bash (for data scripts). No writes. No children.

    Designed for: code review, performance audit, metrics analysis, reporting.
    Context: MODERATE — needs enough context to draw meaningful conclusions.
    Compression: BALANCED — preserve analytical findings, discard noise.
    """
    return ExecutionPolicy(
        name="preset:analysis",
        derived_from=TaskComplexity.MODERATE,
        context_mode=ContextMode.MODERATE,
        compression_mode=CompressionMode.BALANCED,
        child_mode=ChildMode.DENIED,
        max_child_depth=0,
        tool_policy=ToolPolicy(
            allow_read=True,
            allow_write=False,
            allow_bash=True,    # for running analysis scripts
            allow_search=True,
        ),
        export_mode=ExportMode.FILTERED,
        child_context_fraction=0.0,
    )


# Named preset registry — for lookup by string name
PRESETS: dict[str, ExecutionPolicy] = {
    "readonly":  readonly(),
    "sandboxed": sandboxed(),
    "standard":  standard(),
    "federated": federated(),
    "research":  research(),
    "support":   support(),
    "analysis":  analysis(),
}


def get(name: str) -> ExecutionPolicy:
    """
    Retrieve a named preset.
    Raises KeyError for unknown names — fail explicitly, not silently.
    """
    if name not in PRESETS:
        raise KeyError(
            f"Unknown policy preset: '{name}'. "
            f"Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name]
