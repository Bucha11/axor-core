"""
axor_core.policy.keyword_relevance
──────────────────────────────────

Provider-agnostic relevance scoring for tool selection. Used by adapters
(axor-langchain, axor-claude, …) to gate tools by domain relevance to
the latest user task — without any hand-curated per-task allowlist.

Pipeline:
    extract_query_keywords(text)        → set[str]   light tokenization
    compute_topic_strength(keywords)    → {topic: w} direct + propagated
    expand_with_synonyms(keywords)      → set[str]   for fuzzy matching
    score_tool_relevance(name, desc, …) → float      per-tool score

The scorer combines four signals:
    1. Direct keyword hits in tool name (×1.5) and description (×0.5).
    2. Synonym hits in name (×0.5) and description (×0.15) via the
       precomputed `topics.SYNONYM_MAP`.
    3. Topic-strength alignment: tools whose name tokens land in the
       task's strong topics get a proportional bonus, with a focus
       penalty for diffuse name coverage.

Direct keyword hits dominate; synonyms and alignment break ties and
recover the cases where task language and tool language diverge.

This module never imports a provider SDK, never reads message lists,
and never knows the difference between LangChain `BaseTool` and a dict
with `{name, description}`. All adapter-specific bridging (anchor
extraction from messages, orchestration, READONLY filter on destructive
names) belongs in the adapters.
"""
from __future__ import annotations

import re

from axor_core.policy.topics import (
    DESTRUCTIVE_TOKENS,
    STOPWORDS,
    SYNONYM_MAP,
    TOPIC_IMPLICATIONS,
    WORD_TOPICS,
)


_TOKEN_SPLIT = re.compile(r"[\W_]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def extract_query_keywords(text: str | None) -> set[str]:
    """Lowercase tokenization with stopword filter and length>=2."""
    if not text:
        return set()
    tokens = _WORD_RE.findall(text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) >= 2}


def expand_with_synonyms(keywords: set[str]) -> set[str]:
    """Return only the *additional* synonym terms (originals excluded)."""
    expanded: set[str] = set()
    for kw in keywords:
        expanded |= SYNONYM_MAP.get(kw, set())
    return expanded - keywords


def compute_topic_strength(keywords: set[str]) -> dict[str, float]:
    """
    Score each topic by how many of the task's keywords belong to it.
    A keyword that maps to N topics contributes 1/N to each (so an
    ambiguous word like "package" doesn't unfairly dominate any single
    topic).

    After direct counting, strength is *propagated* along
    `TOPIC_IMPLICATIONS` edges at 30%. The intent: when a task is
    strongly about `incident`, code/test/observability tools should
    receive partial alignment even if those words never appear in the
    prompt — exactly the domain knowledge a hand-curated allowlist
    encodes.

    One-hop only — no transitive propagation.

    Returns {topic_name: weight}; absent keys mean strength 0.
    """
    direct: dict[str, float] = {}
    for kw in keywords:
        topics = WORD_TOPICS.get(kw)
        if not topics:
            continue
        share = 1.0 / len(topics)
        for t in topics:
            direct[t] = direct.get(t, 0.0) + share

    propagated = dict(direct)
    for topic, strength in direct.items():
        for implied in TOPIC_IMPLICATIONS.get(topic, []):
            propagated[implied] = propagated.get(implied, 0.0) + strength * 0.3
    return propagated


def tool_topics(name: str) -> set[str]:
    """
    Topics implied by a tool's name tokens. Description excluded —
    name-topic alignment is the strongest signal we can extract from
    metadata without parsing user-provided descriptions, and the
    description weight already runs through the synonym path.
    """
    tokens = set(_TOKEN_SPLIT.split((name or "").lower()))
    topics: set[str] = set()
    for tok in tokens:
        if tok:
            topics |= WORD_TOPICS.get(tok, set())
    return topics


def name_has_destructive_token(name: str) -> bool:
    """True if any token in the tool name is in `DESTRUCTIVE_TOKENS`.
    Token-based — `\\b...\\b` regex would miss `deploy_service` since
    underscore is a word character to Python's regex engine."""
    if not name:
        return False
    tokens = _TOKEN_SPLIT.split(name.lower())
    return any(t in DESTRUCTIVE_TOKENS for t in tokens if t)


def score_tool_relevance(
    *,
    name: str,
    description: str = "",
    keywords: set[str],
    use_synonyms: bool = True,
    topic_strength: dict[str, float] | None = None,
) -> float:
    """
    Score a tool against task keywords.

    Hits in the *name* matter more than hits in description, because
    description text drifts ("Search the codebase" has zero overlap with
    "checkout latency incident" but the tool is `grep_code`).

    With `use_synonyms=True`, the keyword set is expanded via
    `topics.SYNONYM_MAP` so domain-language tasks ("vulnerable code
    paths") still score against admin-language tools ("security
    findings"). Direct hits weigh ~3x synonym hits, so over-broad
    expansion can't dethrone an actually-named tool.

    `topic_strength` (output of `compute_topic_strength`) adds an
    alignment bonus weighted by `sum × 0.25 + focus × 0.5` where
    `focus = sum / len(tool_topics)`. The focus term penalizes tools
    with diffuse name coverage where most topic memberships are off-task.

    Returns a non-negative score; higher = more relevant.
    """
    if not keywords:
        return 0.0
    name_l = (name or "").lower()
    desc_l = (description or "").lower()
    name_tokens = set(_TOKEN_SPLIT.split(name_l))
    desc_tokens = set(_TOKEN_SPLIT.split(desc_l))

    name_hits = sum(1 for kw in keywords if kw in name_tokens)
    desc_hits = sum(1 for kw in keywords if kw in desc_tokens)
    score = name_hits * 1.5 + desc_hits * 0.5

    if use_synonyms:
        synonyms = expand_with_synonyms(keywords)
        if synonyms:
            syn_name_hits = sum(1 for s in synonyms if s in name_tokens)
            syn_desc_hits = sum(1 for s in synonyms if s in desc_tokens)
            score += syn_name_hits * 0.5 + syn_desc_hits * 0.15

    if topic_strength:
        # Bonus for tools whose name aligns with the task's dominant topics.
        # Two-part formula:
        #   sum_part: total topic-strength accumulated across tool topics —
        #             rewards tools that hit multiple strong task topics.
        #   focus_part: average strength per topic the tool belongs to —
        #             penalizes tools with diffuse name coverage where
        #             most topics carry zero task strength.
        # Multiplier keeps alignment competitive with but not dominant
        # over direct keyword hits.
        tool_topics_set = tool_topics(name)
        if tool_topics_set:
            sum_part = sum(
                topic_strength.get(t, 0.0) for t in tool_topics_set
            )
            focus_part = sum_part / len(tool_topics_set)
            score += (sum_part * 0.25) + (focus_part * 0.5)

    return score
