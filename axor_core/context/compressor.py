from __future__ import annotations

import re
from dataclasses import dataclass

from axor_core.contracts.context import ContextFragment
from axor_core.contracts.policy import CompressionMode


@dataclass
class CompressionResult:
    fragments: list[ContextFragment]
    before_tokens: int
    after_tokens: int
    compression_ratio: float
    strategies_applied: list[str]


class ContextCompressor:
    """
    Reduces context size while preserving facts and decisions.

    What gets compressed:
        verbose assistant prose   → key decisions only
        oversized command outputs → tail + summary header
        repeated errors           → single collapsed entry
        old turn content          → rolling summary after N turns

    What never gets compressed:
        facts (file paths, function signatures, error causes)
        explicit decisions made by agent
        pending intents
        the current turn

    Compression is mode-aware:
        AGGRESSIVE  → all strategies, tight limits
        BALANCED    → most strategies, moderate limits
        LIGHT       → only obvious waste (huge outputs, exact duplicates)
    """

    # Max tokens for a single tool result fragment before truncation
    # LIGHT still truncates — just at a much higher threshold
    _MAX_TOOL_OUTPUT: dict[CompressionMode, int] = {
        CompressionMode.AGGRESSIVE: 200,
        CompressionMode.BALANCED:   500,
        CompressionMode.LIGHT:      2000,
    }

    # Turns before history is rolled into a summary
    # LIGHT still summarizes — just much later
    _SUMMARY_AFTER_TURNS: dict[CompressionMode, int] = {
        CompressionMode.AGGRESSIVE: 3,
        CompressionMode.BALANCED:   6,
        CompressionMode.LIGHT:      20,   # long but not never
    }

    # Max tokens for a single prose fragment before extraction
    # LIGHT keeps more prose intact
    _MAX_PROSE_TOKENS: dict[CompressionMode, int] = {
        CompressionMode.AGGRESSIVE: 200,
        CompressionMode.BALANCED:   500,
        CompressionMode.LIGHT:      1500,
    }

    def compress(
        self,
        fragments: list[ContextFragment],
        mode: CompressionMode,
        current_turn: int,
    ) -> CompressionResult:
        """
        Compress context fragments.

        Key principle: waste elimination is ALWAYS on regardless of mode.
        Mode controls aggressiveness — not whether optimization happens.

        FragmentValue semantics (applied before all other strategies):
            PINNED    → never compressed, never evicted — always pass through
            KNOWLEDGE → dedup + error collapse only; no truncation, no prose compression
            WORKING   → all strategies (default)
            EPHEMERAL → truncated aggressively regardless of mode; evicted first

        Always applied to WORKING + EPHEMERAL (mode-independent):
            - exact deduplication
            - repeated error collapsing
            - path normalization
            - empty fragment removal

        Mode-dependent (threshold varies by mode):
            - tool output truncation   (LIGHT=2000t, BALANCED=500t, AGGRESSIVE=200t)
            - prose summarization      (LIGHT after 20 turns, AGGRESSIVE after 3)
            - prose token cap          (LIGHT=1500t, BALANCED=500t, AGGRESSIVE=200t)
        """
        before_tokens = sum(f.token_estimate for f in fragments)
        strategies: list[str] = []

        # ── Separate by FragmentValue ──────────────────────────────────────────
        pinned    = [f for f in fragments if f.value == "pinned"]
        knowledge = [f for f in fragments if f.value == "knowledge"]
        working   = [f for f in fragments if f.value == "working"]
        ephemeral = [f for f in fragments if f.value == "ephemeral"]

        # ── PINNED: never touched ──────────────────────────────────────────────
        # pass through as-is — no strategies applied

        # ── KNOWLEDGE: gentle treatment ────────────────────────────────────────
        knowledge, applied = self._deduplicate(knowledge)
        if applied: strategies.append("deduplicate_knowledge")
        knowledge, applied = self._collapse_errors(knowledge)
        if applied: strategies.append("collapse_errors_knowledge")
        # no truncation, no prose compression for knowledge fragments

        # ── EPHEMERAL: aggressive regardless of mode ───────────────────────────
        ephemeral = [f for f in ephemeral if f.content.strip()]  # remove empty
        # truncate at AGGRESSIVE thresholds regardless of actual mode
        ephemeral, applied = self._truncate_tool_outputs(ephemeral, CompressionMode.AGGRESSIVE)
        if applied: strategies.append("truncate_ephemeral")
        ephemeral, applied = self._cap_prose_size(ephemeral, CompressionMode.AGGRESSIVE)
        if applied: strategies.append("cap_prose_ephemeral")

        # ── WORKING: standard pipeline ─────────────────────────────────────────
        result = list(working)

        # 1. remove empty
        result, applied = self._remove_empty(result)
        if applied: strategies.append("remove_empty")

        # 2. exact deduplication
        result, applied = self._deduplicate(result)
        if applied: strategies.append("deduplicate")

        # 3. collapse repeated errors
        result, applied = self._collapse_errors(result)
        if applied: strategies.append("collapse_errors")

        # 4. normalize paths
        result = self._normalize_paths(result)

        # 5. truncate oversized tool outputs
        result, applied = self._truncate_tool_outputs(result, mode)
        if applied: strategies.append("truncate_tool_outputs")

        # 6. cap individual prose fragments
        result, applied = self._cap_prose_size(result, mode)
        if applied: strategies.append("cap_prose_size")

        # 7. summarize old prose turns
        result, applied = self._compress_prose(result, mode, current_turn)
        if applied: strategies.append("compress_prose")

        # ── Reassemble: pinned first, then knowledge, working, ephemeral ───────
        final = pinned + knowledge + result + ephemeral

        after_tokens = sum(f.token_estimate for f in final)
        ratio = after_tokens / before_tokens if before_tokens > 0 else 1.0

        return CompressionResult(
            fragments=final,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            compression_ratio=ratio,
            strategies_applied=strategies,
        )

    # ── Strategies ─────────────────────────────────────────────────────────────

    def _remove_empty(
        self,
        fragments: list[ContextFragment],
    ) -> tuple[list[ContextFragment], bool]:
        """Remove fragments with no meaningful content. Always runs."""
        result = [f for f in fragments if f.content.strip()]
        return result, len(result) < len(fragments)

    def _cap_prose_size(
        self,
        fragments: list[ContextFragment],
        mode: CompressionMode,
    ) -> tuple[list[ContextFragment], bool]:
        """
        Cap individual prose fragments that exceed per-mode token limit.
        Even LIGHT mode caps at 1500 tokens per prose block —
        a single assistant response should never dominate the context.
        """
        max_tokens = self._MAX_PROSE_TOKENS[mode]
        result = []
        applied = False

        for f in fragments:
            if f.kind in ("assistant_prose", "reasoning") and f.token_estimate > max_tokens:
                truncated = self._smart_truncate(f.content, max_tokens)
                result.append(ContextFragment(
                    kind=f.kind,
                    content=truncated,
                    token_estimate=max_tokens,
                    source=f.source,
                    relevance=f.relevance,
                    value=f.value,
                    turn=f.turn,
                ))
                applied = True
            else:
                result.append(f)

        return result, applied

    def _truncate_tool_outputs(
        self,
        fragments: list[ContextFragment],
        mode: CompressionMode,
    ) -> tuple[list[ContextFragment], bool]:
        max_tokens = self._MAX_TOOL_OUTPUT[mode]
        result = []
        applied = False

        for f in fragments:
            if f.kind == "tool_result" and f.token_estimate > max_tokens:
                truncated = self._smart_truncate(f.content, max_tokens)
                result.append(ContextFragment(
                    kind=f.kind,
                    content=truncated,
                    token_estimate=max_tokens,
                    source=f.source,
                    relevance=f.relevance,
                    value=f.value,
                    turn=f.turn,
                ))
                applied = True
            else:
                result.append(f)

        return result, applied

    def _collapse_errors(
        self,
        fragments: list[ContextFragment],
    ) -> tuple[list[ContextFragment], bool]:
        """
        Collapse repeated error messages into a single entry.
        "Error X appeared 3 times" vs 3 separate fragments.
        """
        error_signatures: dict[str, list[int]] = {}
        for i, f in enumerate(fragments):
            if f.kind in ("tool_result", "fact"):
                for line in f.content.splitlines():
                    if any(kw in line.lower() for kw in ("error:", "exception:", "failed:")):
                        sig = line.strip()[:60]
                        error_signatures.setdefault(sig, []).append(i)

        # find indices to collapse (seen > 1 time)
        to_collapse: set[int] = set()
        collapse_summaries: list[str] = []
        for sig, indices in error_signatures.items():
            if len(indices) > 1:
                to_collapse.update(indices[1:])  # keep first, remove rest
                collapse_summaries.append(f"{sig} (×{len(indices)})")

        if not to_collapse:
            return fragments, False

        result = [f for i, f in enumerate(fragments) if i not in to_collapse]
        if collapse_summaries:
            summary = "Repeated errors (collapsed):\n" + "\n".join(collapse_summaries)
            result.append(ContextFragment(
                kind="fact",
                content=summary,
                token_estimate=len(summary) // 4,
                source="compressor:error_collapse",
                relevance=0.6,
            ))

        return result, True

    def _compress_prose(
        self,
        fragments: list[ContextFragment],
        mode: CompressionMode,
        current_turn: int,
    ) -> tuple[list[ContextFragment], bool]:
        """
        Compress verbose assistant prose from old turns into key decisions.
        Current turn prose is never compressed.
        """
        threshold = self._SUMMARY_AFTER_TURNS[mode]
        result = []
        applied = False
        old_prose: list[str] = []

        for f in fragments:
            is_old = f.turn > 0 and (current_turn - f.turn) >= threshold
            is_prose = f.kind in ("assistant_prose", "reasoning")

            if is_prose and is_old:
                # extract key decisions from prose — always, threshold varies by mode
                key_points = self._extract_key_decisions(f.content)
                if key_points:
                    old_prose.extend(key_points)
                applied = True
            else:
                result.append(f)

        if old_prose:
            summary = "Prior decisions:\n" + "\n".join(f"- {p}" for p in old_prose[:10])
            result.insert(0, ContextFragment(
                kind="fact",
                content=summary,
                token_estimate=len(summary) // 4,
                source="compressor:prose_summary",
                relevance=0.7,
            ))

        return result, applied

    def _deduplicate(
        self,
        fragments: list[ContextFragment],
    ) -> tuple[list[ContextFragment], bool]:
        """Remove exact duplicate fragments (same content)."""
        seen: set[str] = set()
        result = []
        applied = False

        for f in fragments:
            key = f.content.strip()
            if key in seen:
                applied = True
                continue
            seen.add(key)
            result.append(f)

        return result, applied

    def _normalize_paths(self, fragments: list[ContextFragment]) -> list[ContextFragment]:
        """
        Normalize absolute paths in fragment `source` field to relative.
        Content is left untouched — normalizing inside file content or
        tool output can change meaning and break references.
        """
        _ABS_PATH = re.compile(r"/(?:home|Users|root)/[\w.]+/([\w./]+)")

        result = []
        for f in fragments:
            normalized_source = _ABS_PATH.sub(r"./\1", f.source)
            if normalized_source != f.source:
                result.append(ContextFragment(
                    kind=f.kind,
                    content=f.content,
                    token_estimate=f.token_estimate,
                    source=normalized_source,
                    relevance=f.relevance,
                    value=f.value,
                    turn=f.turn,
                ))
            else:
                result.append(f)
        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _smart_truncate(self, content: str, max_tokens: int) -> str:
        """
        For command output: keep head (context) and tail (result),
        drop the middle.
        Better than naive truncation for bash output.
        """
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content

        head_chars = max_chars * 2 // 3
        tail_chars = max_chars - head_chars
        head = content[:head_chars]
        tail = content[-tail_chars:]
        dropped = len(content) - head_chars - tail_chars
        approx_lines = dropped // 80

        return (
            f"{head}\n"
            f"... [{approx_lines} lines omitted by context compressor] ...\n"
            f"{tail}"
        )

    def _extract_key_decisions(self, prose: str) -> list[str]:
        """
        Extract decision-like sentences from assistant prose.
        Heuristic: sentences with decision verbs.
        """
        decision_verbs = re.compile(
            r"\b(decided|chose|using|will use|replaced|renamed|refactored|"
            r"moved|deleted|created|added|fixed|changed|set|configured)\b",
            re.IGNORECASE,
        )
        decisions = []
        for sentence in re.split(r"[.!?]\s+", prose):
            sentence = sentence.strip()
            if decision_verbs.search(sentence) and 10 < len(sentence) < 200:
                decisions.append(sentence)
        return decisions[:5]  # cap at 5 key points per prose block
