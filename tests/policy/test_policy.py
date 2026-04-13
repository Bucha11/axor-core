"""Tests for policy/analyzer, selector, composer, heuristic."""
from __future__ import annotations

import pytest
from axor_core.policy import TaskAnalyzer, PolicySelector, PolicyComposer, presets
from axor_core.policy.heuristic import HeuristicClassifier
from axor_core.contracts.policy import (
    TaskComplexity, TaskNature, ChildMode, ContextMode,
    ExportMode, ExecutionPolicy, ToolPolicy,
)
from axor_core.contracts.extension import ExtensionFragment


# ── HeuristicClassifier ────────────────────────────────────────────────────────

class TestHeuristicClassifier:

    @pytest.fixture
    def clf(self):
        return HeuristicClassifier()

    @pytest.mark.asyncio
    async def test_focused_readonly(self, clf):
        signal, conf = await clf.classify("explain why this function is slow")
        assert signal.complexity == TaskComplexity.FOCUSED
        assert signal.nature == TaskNature.READONLY

    @pytest.mark.asyncio
    async def test_focused_generative(self, clf):
        signal, conf = await clf.classify("write a test for the payment endpoint")
        assert signal.complexity == TaskComplexity.FOCUSED
        assert signal.nature == TaskNature.GENERATIVE

    @pytest.mark.asyncio
    async def test_expansive(self, clf):
        signal, conf = await clf.classify("rewrite the entire repo from Python to Go")
        assert signal.complexity == TaskComplexity.EXPANSIVE
        assert signal.requires_children is True

    @pytest.mark.asyncio
    async def test_contradiction_lowers_confidence(self, clf):
        # "find" (readonly) + "rewrite" (mutative) → contradiction
        _, conf_normal = await clf.classify("explain the auth flow")
        _, conf_contradiction = await clf.classify(
            "find all places where we mutate state and rewrite them"
        )
        assert conf_contradiction < conf_normal

    @pytest.mark.asyncio
    async def test_unknown_input_returns_default(self, clf):
        signal, conf = await clf.classify("xyz abc 123")
        # should not raise — returns some default
        assert signal.complexity in TaskComplexity.__members__.values()


# ── TaskAnalyzer ───────────────────────────────────────────────────────────────

class TestTaskAnalyzer:

    @pytest.mark.asyncio
    async def test_returns_signal_and_event(self):
        analyzer = TaskAnalyzer()
        signal, event = await analyzer.analyze("write a test for the auth endpoint")
        assert signal is not None
        assert event.classifier == "heuristic"

    @pytest.mark.asyncio
    async def test_uses_external_classifier_on_low_confidence(self):
        from axor_core.contracts.policy import SignalClassifier, TaskSignal

        class HighConfidenceClassifier(SignalClassifier):
            async def classify(self, raw_input: str):
                signal = TaskSignal(
                    raw_input=raw_input,
                    complexity=TaskComplexity.EXPANSIVE,
                    nature=TaskNature.MUTATIVE,
                    estimated_scope=99,
                    requires_children=True,
                    requires_mutation=True,
                )
                return signal, 0.99

        analyzer = TaskAnalyzer(
            external_classifier=HighConfidenceClassifier(),
            escalation_threshold=0.99,  # always escalate
        )
        signal, event = await analyzer.analyze("do something")
        assert signal.complexity == TaskComplexity.EXPANSIVE
        assert event.classifier == "HighConfidenceClassifier"


# ── PolicySelector ─────────────────────────────────────────────────────────────

class TestPolicySelector:

    @pytest.fixture
    def selector(self):
        return PolicySelector()

    @pytest.mark.parametrize("task,expected_policy", [
        ("write a test for the endpoint", "focused_generative"),
        ("explain why this is slow",      "focused_readonly"),
        ("fix the auth bug",              "focused_mutative"),
        ("add a feature to the API",      "moderate_generative"),
        ("refactor the auth module",      "moderate_mutative"),
        ("rewrite the entire repo to Go", "expansive"),
    ])
    @pytest.mark.asyncio
    async def test_policy_selection(self, task, expected_policy):
        analyzer = TaskAnalyzer()
        selector = PolicySelector()
        signal, _ = await analyzer.analyze(task)
        policy = selector.select(signal)
        assert policy.name == expected_policy

    def test_expansive_allows_children(self, selector):
        from axor_core.contracts.policy import TaskSignal
        signal = TaskSignal(
            raw_input="x", complexity=TaskComplexity.EXPANSIVE,
            nature=TaskNature.MUTATIVE, estimated_scope=999,
            requires_children=True, requires_mutation=True,
        )
        policy = selector.select(signal)
        assert policy.child_mode == ChildMode.ALLOWED
        assert policy.max_child_depth == 3

    def test_focused_denies_children(self, selector):
        from axor_core.contracts.policy import TaskSignal
        signal = TaskSignal(
            raw_input="x", complexity=TaskComplexity.FOCUSED,
            nature=TaskNature.GENERATIVE, estimated_scope=1,
            requires_children=False, requires_mutation=False,
        )
        policy = selector.select(signal)
        assert policy.child_mode == ChildMode.DENIED


# ── PolicyComposer ─────────────────────────────────────────────────────────────

class TestPolicyComposer:

    @pytest.fixture
    def composer(self):
        return PolicyComposer()

    def test_parent_restriction_blocks_write(self, composer):
        parent = presets.get("readonly")  # no writes
        child  = presets.get("federated") # writes allowed
        restricted = composer.apply_parent_restrictions(child, parent)
        assert restricted.tool_policy.allow_write is False

    def test_parent_restriction_blocks_bash(self, composer):
        parent = presets.get("readonly")
        child  = presets.get("federated")
        restricted = composer.apply_parent_restrictions(child, parent)
        assert restricted.tool_policy.allow_bash is False

    def test_parent_denied_children_propagates(self, composer):
        parent = presets.get("readonly")
        child  = presets.get("federated")
        restricted = composer.apply_parent_restrictions(child, parent)
        assert restricted.child_mode == ChildMode.DENIED

    def test_export_most_restrictive_wins(self, composer):
        parent = ExecutionPolicy(
            name="p", derived_from=TaskComplexity.FOCUSED,
            export_mode=ExportMode.SUMMARY,
            tool_policy=ToolPolicy(),
        )
        child = ExecutionPolicy(
            name="c", derived_from=TaskComplexity.FOCUSED,
            export_mode=ExportMode.FULL,
            tool_policy=ToolPolicy(),
        )
        restricted = composer.apply_parent_restrictions(child, parent)
        assert restricted.export_mode == ExportMode.SUMMARY

    def test_extension_cannot_escalate_beyond_base(self, composer):
        base = presets.get("readonly")  # allow_bash=False
        fragment = ExtensionFragment(
            name="ext",
            context_fragment="",
            required_tools=(),
            policy_overrides={"allow_bash": True},  # requests bash
            source="test",
        )
        result = composer.apply_extension_overrides(base, [fragment])
        # bash still denied — base policy doesn't allow it
        assert result.tool_policy.allow_bash is False


# ── Presets ────────────────────────────────────────────────────────────────────

class TestPresets:

    def test_readonly_no_writes(self):
        p = presets.get("readonly")
        assert p.tool_policy.allow_write is False
        assert p.tool_policy.allow_bash is False
        assert p.child_mode == ChildMode.DENIED

    def test_sandboxed_no_tools(self):
        p = presets.get("sandboxed")
        assert p.tool_policy.allow_read is False
        assert p.export_mode == ExportMode.RESTRICTED

    def test_federated_allows_spawn(self):
        p = presets.get("federated")
        assert p.child_mode == ChildMode.ALLOWED
        assert p.tool_policy.allow_spawn is True

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown policy preset"):
            presets.get("does_not_exist")
