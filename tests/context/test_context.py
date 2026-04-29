"""Tests for context subsystem — cache, compressor, selector, symbol_table."""
from __future__ import annotations

import pytest
from axor_core.context.cache import ContextCache
from axor_core.context.compressor import ContextCompressor
from axor_core.context.selector import ContextSelector
from axor_core.context.symbol_table import SymbolTable, SymbolStatus
from axor_core.context.manager import ContextManager
from axor_core.contracts.context import ContextFragment, RawExecutionState, LineageSummary
from axor_core.contracts.policy import (
    ExecutionPolicy, TaskComplexity, ContextMode, CompressionMode,
    ChildMode, ExportMode, ToolPolicy,
)


def make_fragment(content: str, kind: str = "fact", relevance: float = 1.0) -> ContextFragment:
    return ContextFragment(
        kind=kind, content=content,
        token_estimate=len(content) // 4,
        source="test", relevance=relevance,
    )


def make_policy(context_mode=ContextMode.MODERATE, compression_mode=CompressionMode.BALANCED):
    return ExecutionPolicy(
        name="test", derived_from=TaskComplexity.MODERATE,
        context_mode=context_mode, compression_mode=compression_mode,
        child_mode=ChildMode.DENIED, max_child_depth=0,
        tool_policy=ToolPolicy(allow_read=True),
        export_mode=ExportMode.SUMMARY,
    )


class TestContextCache:

    def test_file_put_and_get(self):
        cache = ContextCache()
        cache.put_file("auth.py", "def authenticate(): pass")
        cached = cache.get_file("auth.py")
        assert cached is not None
        assert cached.content == "def authenticate(): pass"

    def test_file_changed_detects_modification(self):
        cache = ContextCache()
        cache.put_file("auth.py", "version 1")
        assert cache.file_changed("auth.py", "version 2") is True
        assert cache.file_changed("auth.py", "version 1") is False

    def test_unknown_file_always_changed(self):
        cache = ContextCache()
        assert cache.file_changed("new.py", "anything") is True

    def test_tool_result_memoization(self):
        cache = ContextCache()
        cache.put_tool_result("bash", {"cmd": "git log"}, "abc123")
        result = cache.get_tool_result("bash", {"cmd": "git log"})
        assert result == "abc123"

    def test_tool_result_cache_miss_returns_none(self):
        cache = ContextCache()
        result = cache.get_tool_result("bash", {"cmd": "unknown"})
        assert result is None

    def test_invalidate_file(self):
        cache = ContextCache()
        cache.put_file("x.py", "content")
        cache.invalidate_file("x.py")
        assert cache.get_file("x.py") is None

    def test_snapshot_files(self):
        cache = ContextCache()
        cache.put_file("a.py", "aaa")
        cache.put_file("b.py", "bbb")
        snap = cache.snapshot_files()
        assert set(snap.keys()) == {"a.py", "b.py"}

    def test_file_cache_evicts_lru_over_cap(self):
        """Regression: long sessions used to grow _files unboundedly."""
        cache = ContextCache(max_files=3)
        cache.put_file("a.py", "1")
        cache.put_file("b.py", "2")
        cache.put_file("c.py", "3")
        # Touch 'a' so it becomes most recently used.
        cache.get_file("a.py")
        # Fourth entry — 'b' is now LRU and must be evicted.
        cache.put_file("d.py", "4")

        assert cache.get_file("b.py") is None
        assert cache.get_file("a.py") is not None
        assert cache.get_file("c.py") is not None
        assert cache.get_file("d.py") is not None

    def test_tool_result_cache_evicts_over_cap(self):
        cache = ContextCache(max_tool_results=2)
        cache.put_tool_result("bash", {"cmd": "a"}, "ra")
        cache.put_tool_result("bash", {"cmd": "b"}, "rb")
        cache.put_tool_result("bash", {"cmd": "c"}, "rc")
        # Oldest ('a') should have been evicted.
        assert cache.get_tool_result("bash", {"cmd": "a"}) is None
        assert cache.get_tool_result("bash", {"cmd": "b"}) == "rb"
        assert cache.get_tool_result("bash", {"cmd": "c"}) == "rc"


class TestContextCompressor:

    @pytest.fixture
    def compressor(self):
        return ContextCompressor()

    def test_truncates_oversized_tool_output_aggressive(self, compressor):
        big = make_fragment("line\n" * 500, kind="tool_result")
        big = ContextFragment(kind="tool_result", content="line\n" * 500,
                              token_estimate=2000, source="bash", relevance=1.0)
        result = compressor.compress([big], CompressionMode.AGGRESSIVE, current_turn=1)
        assert result.after_tokens < result.before_tokens
        assert "truncate_tool_outputs" in result.strategies_applied

    def test_deduplication_removes_exact_duplicates(self, compressor):
        frag = make_fragment("same content here")
        fragments = [frag, frag, frag]
        result = compressor.compress(fragments, CompressionMode.BALANCED, current_turn=1)
        assert "deduplicate" in result.strategies_applied
        contents = [f.content for f in result.fragments]
        assert contents.count("same content here") == 1

    def test_collapse_repeated_errors(self, compressor):
        error_line = "Error: connection refused to database"
        fragments = [
            make_fragment(f"Step 1\n{error_line}", kind="tool_result"),
            make_fragment(f"Step 2\n{error_line}", kind="tool_result"),
            make_fragment(f"Step 3\n{error_line}", kind="tool_result"),
        ]
        result = compressor.compress(fragments, CompressionMode.BALANCED, current_turn=1)
        assert "collapse_errors" in result.strategies_applied

    def test_path_normalization(self, compressor):
        # normalization applies to source field, not content
        frag = ContextFragment(
            kind="fact",
            content="file content here",
            token_estimate=10,
            source="/home/user/project/src/auth.py",
            relevance=1.0,
        )
        result = compressor.compress([frag], CompressionMode.LIGHT, current_turn=1)
        source = result.fragments[0].source
        assert "/home/user" not in source
        assert "./project/src/auth.py" in source
        # content should remain untouched
        assert result.fragments[0].content == "file content here"

    def test_smart_truncation_keeps_head_and_tail(self, compressor):
        lines = [f"line {i}" for i in range(200)]
        content = "\n".join(lines)
        truncated = compressor._smart_truncate(content, max_tokens=50)
        assert "line 0" in truncated      # head preserved
        assert "line 199" in truncated    # tail preserved
        assert "omitted" in truncated     # middle marker


class TestSymbolTable:

    def test_ingest_file_extracts_functions(self):
        table = SymbolTable()
        new = table.ingest_file("auth.py", "def authenticate(token):\n    pass\ndef verify_jwt(t):\n    pass")
        assert "authenticate" in new
        assert "verify_jwt" in new

    def test_ingest_file_extracts_classes(self):
        table = SymbolTable()
        new = table.ingest_file("models.py", "class UserModel:\n    pass")
        assert "UserModel" in new

    def test_ingest_file_extracts_todos(self):
        table = SymbolTable()
        table.ingest_file("auth.py", "def auth():\n    # TODO: add rate limiting\n    pass")
        pending = table.unresolved_intents()
        assert any("rate limiting" in p.description for p in pending)

    def test_mark_renamed_deprecates_old(self):
        table = SymbolTable()
        table.ingest_file("auth.py", "def authenticate(token): pass")
        table.mark_renamed("authenticate", "verify_token")
        assert "authenticate" in table.deprecated_names()

    def test_relevance_penalty_for_deprecated(self):
        table = SymbolTable()
        table.ingest_file("auth.py", "def authenticate(token): pass")
        table.mark_renamed("authenticate", "verify_token")
        penalty = table.relevance_penalty("call authenticate() here")
        assert penalty > 0.0

    def test_no_penalty_for_active_symbols(self):
        table = SymbolTable()
        table.ingest_file("auth.py", "def authenticate(token): pass")
        penalty = table.relevance_penalty("call authenticate() here")
        assert penalty == 0.0

    def test_pending_resolved(self):
        table = SymbolTable()
        table.ingest_file("auth.py", "# TODO: add rate limiting")
        table.mark_pending_resolved("rate limiting")
        assert all(p.resolved for p in table._pending_intents if "rate limiting" in p.description)


class TestContextManager:

    @pytest.fixture
    def manager(self):
        return ContextManager()

    @pytest.fixture
    def lineage(self):
        return LineageSummary(
            node_id="n_test", parent_id=None, depth=0,
            ancestry_ids=[], inherited_restrictions=[],
        )

    def test_build_returns_context_view(self, manager, lineage):
        state = RawExecutionState(
            task="refactor auth", session_id="s1",
            parent_export=None, session_state={},
            memory_fragments=[], lineage=None,
        )
        ctx = manager.build(state, lineage, policy=make_policy())
        assert ctx.node_id == lineage.node_id
        assert ctx.token_count > 0
        assert len(ctx.visible_fragments) > 0

    def test_context_view_fields_are_immutable(self, manager, lineage):
        """Regression: frozen=True with list-typed fields used to leak
        mutability — `view.visible_fragments.append(...)` would silently
        mutate a "frozen" view. Tuples close that hole.
        """
        state = RawExecutionState(
            task="refactor auth", session_id="s1",
            parent_export=None, session_state={},
            memory_fragments=[], lineage=None,
        )
        ctx = manager.build(state, lineage, policy=make_policy())
        assert isinstance(ctx.visible_fragments, tuple)
        assert isinstance(ctx.active_constraints, tuple)
        with pytest.raises(AttributeError):
            ctx.visible_fragments.append(None)  # type: ignore[attr-defined]
        # LineageSummary likewise
        assert isinstance(lineage.ancestry_ids, tuple)
        with pytest.raises(AttributeError):
            lineage.ancestry_ids.append("x")  # type: ignore[attr-defined]

    def test_record_file_read_caches_content(self, manager, lineage):
        manager.record_file_read("auth.py", "def authenticate(): pass")
        assert manager.is_cached("auth.py")
        assert not manager.is_cached("missing.py")

    def test_record_file_read_ingests_symbols(self, manager, lineage):
        manager.record_file_read("auth.py", "def authenticate(token):\n    pass")
        symbols = manager._symbols.active_symbols()
        names = [s.name for s in symbols]
        assert "authenticate" in names

    def test_tool_result_memoization(self, manager, lineage):
        manager.cache_tool_result("bash", {"cmd": "git log"}, "abc123")
        result = manager.get_cached_tool_result("bash", {"cmd": "git log"})
        assert result == "abc123"

    def test_update_adds_assistant_prose(self, manager, lineage):
        state = RawExecutionState(
            task="task", session_id="s1",
            parent_export=None, session_state={},
            memory_fragments=[], lineage=None,
        )
        manager.build(state, lineage, policy=make_policy())
        manager.update("I decided to use JWT tokens for auth.", "n_test")
        # next build should include the prose
        ctx2 = manager.build(state, lineage, policy=make_policy())
        sources = [f.source for f in ctx2.visible_fragments]
        assert any("node:" in s for s in sources)
