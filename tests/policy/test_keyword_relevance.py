"""
Core-level tests for `axor_core.policy.keyword_relevance` and the
underlying `topics` data.

Adapter-specific orchestration (anchor extraction, top-K + floor +
READONLY filter) lives in adapters and is tested there. These tests
cover only the provider-agnostic primitives.
"""
from __future__ import annotations

import pytest

from axor_core.policy import (
    compute_topic_strength,
    expand_with_synonyms,
    extract_query_keywords,
    name_has_destructive_token,
    score_tool_relevance,
    tool_topics,
    topics,
)


# ── Tokenization ──────────────────────────────────────────────────────────────

class TestExtractQueryKeywords:

    def test_drops_stopwords_and_short_tokens(self):
        kws = extract_query_keywords(
            "Investigate the checkout latency incident from the metrics."
        )
        assert "checkout" in kws
        assert "latency" in kws
        assert "incident" in kws
        assert "metrics" in kws
        assert "the" not in kws
        assert "from" not in kws

    def test_empty_input(self):
        assert extract_query_keywords("") == set()
        assert extract_query_keywords(None) == set()

    def test_lowercase(self):
        assert "checkout" in extract_query_keywords("CHECKOUT incident")

    def test_short_domain_terms_are_kept(self):
        kws = extract_query_keywords("check the API for SQL injection in DB logs")
        assert {"api", "sql", "db", "logs", "injection"} <= kws


# ── Synonym expansion ─────────────────────────────────────────────────────────

class TestExpandWithSynonyms:

    def test_security_terms_pull_each_other(self):
        expanded = expand_with_synonyms({"vulnerable"})
        assert "security" in expanded
        assert "audit" in expanded
        assert "auth" in expanded

    def test_observability_terms_pull_each_other(self):
        expanded = expand_with_synonyms({"p95"})
        assert "trace" in expanded
        assert "metric" in expanded
        assert "log" in expanded

    def test_unknown_word_no_expansion(self):
        assert expand_with_synonyms({"asdfqwerty"}) == set()

    def test_excludes_input_keywords(self):
        # An input keyword should not appear in its own expansion.
        expanded = expand_with_synonyms({"latency"})
        assert "latency" not in expanded


# ── Topic strength ────────────────────────────────────────────────────────────

class TestComputeTopicStrength:

    def test_direct_keyword_membership(self):
        # "incident" lives in `incident` topic alone (1 topic).
        strength = compute_topic_strength({"incident"})
        # Direct contribution = 1.0 to incident.
        assert strength["incident"] >= 1.0
        # Tickets receives only the propagated 0.3 share from incident impl.
        assert 0.0 < strength.get("tickets", 0.0) < 1.0

    def test_propagation_lifts_implied_topics(self):
        # `incident` topic implies `code` (RCA forensics need source
        # files even when the prompt doesn't say "code").
        strength = compute_topic_strength({"incident"})
        # Without propagation, code strength would be 0; with it,
        # code receives 0.3 × incident_strength.
        assert strength.get("code", 0.0) > 0.0

    def test_unknown_word_zero_strength(self):
        assert compute_topic_strength({"asdfqwerty"}) == {}

    def test_multi_topic_word_splits_evenly(self):
        # "package" lives in `code` and `dependencies` — each gets 1/2.
        strength = compute_topic_strength({"package"})
        assert "code" in strength
        assert "dependencies" in strength
        for t in ("code", "dependencies"):
            assert strength[t] >= 0.5


# ── tool_topics ───────────────────────────────────────────────────────────────

class TestToolTopics:

    def test_token_split_by_underscore_and_punctuation(self):
        # `query_metrics` → tokens {query, metrics} → topics from each.
        result = tool_topics("query_metrics")
        # "metrics" lives in observability and quality; "query" in data.
        assert "observability" in result or "quality" in result
        assert "data" in result

    def test_empty_name(self):
        assert tool_topics("") == set()

    def test_unknown_tokens(self):
        assert tool_topics("asdfqwerty_zzz") == set()


# ── Destructive token detection ───────────────────────────────────────────────

class TestNameHasDestructiveToken:

    def test_detects_underscored_names(self):
        # The whole reason this is token-based: `\bdeploy\b` regex
        # would NOT match `deploy_service` because underscore is a
        # word char in Python's regex engine.
        assert name_has_destructive_token("deploy_service")
        assert name_has_destructive_token("delete_file")
        assert name_has_destructive_token("write_decision_matrix")

    def test_safe_names_pass_through(self):
        assert not name_has_destructive_token("read_source_file")
        assert not name_has_destructive_token("query_metrics")
        assert not name_has_destructive_token("inspect_runbook")

    def test_empty_name(self):
        assert not name_has_destructive_token("")
        assert not name_has_destructive_token(None)


# ── score_tool_relevance ──────────────────────────────────────────────────────

class TestScoreToolRelevance:

    def test_empty_keywords_zero_score(self):
        assert score_tool_relevance(
            name="anything", description="anything", keywords=set(),
        ) == 0.0

    @pytest.mark.parametrize(
        ("keyword", "name"),
        [
            ("code", "decoder"),
            ("api", "rapid_search"),
            ("log", "prologue"),
        ],
    )
    def test_name_matching_is_token_based_not_substring(self, keyword, name):
        assert score_tool_relevance(
            name=name,
            description="generic helper",
            keywords={keyword},
            use_synonyms=False,
        ) == 0.0

    @pytest.mark.parametrize(
        ("keyword", "description"),
        [
            ("code", "Decode a payload."),
            ("api", "Perform rapid lookup."),
            ("log", "Write the prologue."),
        ],
    )
    def test_description_matching_is_token_based_not_substring(
        self, keyword, description
    ):
        assert score_tool_relevance(
            name="generic_helper",
            description=description,
            keywords={keyword},
            use_synonyms=False,
        ) == 0.0

    def test_direct_name_hit_beats_synonym(self):
        kws = {"latency"}
        direct = score_tool_relevance(
            name="latency_probe", description="x", keywords=kws,
        )
        synonym_only = score_tool_relevance(
            name="trace_collector", description="x", keywords=kws,
        )
        assert direct > synonym_only

    def test_synonym_lifts_relevant_tool_from_zero(self):
        kws = {"vulnerable", "oauth", "tokens"}  # security-flavored task
        no_syn = score_tool_relevance(
            name="security_scan", description="Return security findings.",
            keywords=kws, use_synonyms=False,
        )
        with_syn = score_tool_relevance(
            name="security_scan", description="Return security findings.",
            keywords=kws, use_synonyms=True,
        )
        assert no_syn == 0.0
        assert with_syn > 0.0

    def test_topic_alignment_promotes_aligned_tool(self):
        kws = {"vulnerable", "oauth", "tokens", "tenant"}
        ts = compute_topic_strength(kws)
        # security_scan name lands in security topic, which is the
        # task's strongest topic → alignment bonus pushes it up.
        sec = score_tool_relevance(
            name="security_scan", description="Return security findings.",
            keywords=kws, topic_strength=ts,
        )
        # write_decision_matrix has no topic alignment with security.
        wdm = score_tool_relevance(
            name="write_decision_matrix",
            description="Create a decision matrix.",
            keywords=kws, topic_strength=ts,
        )
        assert sec > wdm

    def test_alignment_optional(self):
        # Calling without topic_strength must still produce a real score.
        kws = {"latency"}
        score = score_tool_relevance(
            name="query_metrics", description="Query latency metrics.",
            keywords=kws,
        )
        assert score > 0.0


# ── Topic data sanity ─────────────────────────────────────────────────────────

class TestTopicsData:
    """Light sanity checks so accidental edits to `topics.py` that delete
    or rename load-bearing entries get caught at test time."""

    def test_critical_topics_present(self):
        for t in ("observability", "incident", "security", "code",
                  "testing", "cost", "migration", "ops"):
            assert t in topics.DOMAIN_TOPICS, f"missing topic: {t}"

    def test_implication_targets_exist(self):
        # Every implied topic name must point to a real topic.
        for src, implied_list in topics.TOPIC_IMPLICATIONS.items():
            assert src in topics.DOMAIN_TOPICS, f"unknown source topic: {src}"
            for impl in implied_list:
                assert impl in topics.DOMAIN_TOPICS, (
                    f"{src} implies unknown topic: {impl}"
                )

    def test_word_topics_built(self):
        # Inverted index covers every word in every topic.
        all_words = set().union(*topics.DOMAIN_TOPICS.values())
        assert set(topics.WORD_TOPICS) == all_words

    def test_destructive_tokens_disjoint_from_read_actions(self):
        # A token can't be both destructive and a read-action — that
        # would defeat the READONLY filter.
        assert (topics.DESTRUCTIVE_TOKENS
                & topics.DOMAIN_TOPICS["read_actions"]) == set()
