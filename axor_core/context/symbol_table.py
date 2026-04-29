from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class SymbolStatus(str, Enum):
    ACTIVE     = "active"      # seen recently, definition current
    DEPRECATED = "deprecated"  # renamed or removed — context fragments with old name penalised
    PENDING    = "pending"     # declared but not yet implemented (TODO/FIXME/pass)


@dataclass
class Symbol:
    name: str
    kind: str               # "function" | "class" | "variable" | "method"
    file: str
    line: int | None
    status: SymbolStatus    = SymbolStatus.ACTIVE
    last_seen_turn: int     = 0
    definition_summary: str = ""
    aliases: list[str]      = field(default_factory=list)  # previous names (rename history)


@dataclass
class PendingIntent:
    """
    An action the agent planned but has not yet completed.
    Extracted from TODO/FIXME comments or explicit plan statements.
    Tracked separately — not kept in main context.
    """
    description: str
    file: str | None
    line: int | None
    turn_created: int
    resolved: bool = False


class SymbolTable:
    """
    Live registry of symbols seen during a session.

    Tracks:
        - function/class/variable definitions across files
        - symbol renames (drift detection)
        - pending intents (TODO, FIXME, planned actions)

    Used by:
        selector.py   — deprioritize context fragments with deprecated symbols
        invalidator.py — invalidate cache entries containing stale symbol names
        compressor.py  — replace verbose symbol discussions with summary references

    Not a full AST parser — uses lightweight regex extraction.
    Adapters may replace this with language-server-backed extraction.
    """

    # Patterns for lightweight symbol extraction
    _PYTHON_FUNC    = re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)
    _PYTHON_CLASS   = re.compile(r"^class\s+(\w+)[\s:(]", re.MULTILINE)
    _TODO_PATTERN   = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)[:\s]+(.+)", re.IGNORECASE)
    _RENAME_PATTERN = re.compile(r"rename[d]?\s+[`'\"]?(\w+)[`'\"]?\s+to\s+[`'\"]?(\w+)[`'\"]?", re.IGNORECASE)

    def __init__(self) -> None:
        self._symbols: dict[str, Symbol] = {}         # name → Symbol
        self._pending_intents: list[PendingIntent] = []
        self._current_turn: int = 0

    def advance_turn(self) -> None:
        self._current_turn += 1

    # ── Symbol registration ────────────────────────────────────────────────────

    def ingest_file(self, path: str, content: str) -> list[str]:
        """
        Extract and register symbols from file content.
        Returns list of newly found symbol names.
        """
        new_symbols = []

        for match in self._PYTHON_FUNC.finditer(content):
            name = match.group(1)
            if name not in self._symbols:
                new_symbols.append(name)
            self._upsert(name, kind="function", file=path)

        for match in self._PYTHON_CLASS.finditer(content):
            name = match.group(1)
            if name not in self._symbols:
                new_symbols.append(name)
            self._upsert(name, kind="class", file=path)

        self._extract_pending_intents(content, path)
        return new_symbols

    # Bound the regex input — assistant output can be arbitrarily long, and
    # though the rename pattern is RE2-clean enough in CPython today, applying
    # it to multi-MB output is wasteful. Cap at 64KB which comfortably covers
    # any realistic single-turn assistant message containing rename language.
    _RENAME_INGEST_MAX_BYTES = 64 * 1024

    def ingest_assistant_text(self, text: str) -> None:
        """
        Detect rename mentions in assistant output.
        e.g. "renamed auth_check to verify_token"
        """
        if not text:
            return
        if len(text) > self._RENAME_INGEST_MAX_BYTES:
            text = text[: self._RENAME_INGEST_MAX_BYTES]
        for match in self._RENAME_PATTERN.finditer(text):
            old_name = match.group(1)
            new_name = match.group(2)
            self.mark_renamed(old_name, new_name)

    def mark_renamed(self, old_name: str, new_name: str) -> None:
        """Mark old symbol as deprecated, register new name."""
        if old_name in self._symbols:
            old = self._symbols[old_name]
            self._symbols[old_name] = Symbol(
                name=old_name,
                kind=old.kind,
                file=old.file,
                line=old.line,
                status=SymbolStatus.DEPRECATED,
                last_seen_turn=self._current_turn,
                definition_summary=old.definition_summary,
                aliases=[new_name],
            )
        # register new name carrying rename history
        existing_aliases = []
        if old_name in self._symbols:
            existing_aliases = [old_name]
        self._upsert(new_name, kind=self._symbols.get(old_name, Symbol(
            name=old_name, kind="function", file="", line=None
        )).kind, file="", aliases=existing_aliases)

    def mark_pending_resolved(self, description_fragment: str) -> None:
        for intent in self._pending_intents:
            if description_fragment.lower() in intent.description.lower():
                intent.resolved = True

    # ── Queries ────────────────────────────────────────────────────────────────

    def deprecated_names(self) -> set[str]:
        return {
            name for name, sym in self._symbols.items()
            if sym.status == SymbolStatus.DEPRECATED
        }

    def active_symbols(self) -> list[Symbol]:
        return [s for s in self._symbols.values() if s.status == SymbolStatus.ACTIVE]

    def unresolved_intents(self) -> list[PendingIntent]:
        return [i for i in self._pending_intents if not i.resolved]

    def symbol(self, name: str) -> Symbol | None:
        return self._symbols.get(name)

    def text_contains_deprecated(self, text: str) -> bool:
        """Check if text references any deprecated symbol names."""
        deprecated = self.deprecated_names()
        return any(name in text for name in deprecated)

    def relevance_penalty(self, text: str) -> float:
        """
        Penalty to apply to a context fragment's relevance score
        if it contains deprecated symbol names.
        Higher = more penalised.
        """
        deprecated = self.deprecated_names()
        hits = sum(1 for name in deprecated if name in text)
        return min(1.0, hits * 0.3)

    def pending_summary(self) -> str:
        """Compact summary of unresolved intents for context injection."""
        unresolved = self.unresolved_intents()
        if not unresolved:
            return ""
        lines = [f"- {i.description}" + (f" ({i.file})" if i.file else "")
                 for i in unresolved[:10]]
        return "Pending:\n" + "\n".join(lines)

    # ── Private ────────────────────────────────────────────────────────────────

    def _upsert(
        self,
        name: str,
        kind: str,
        file: str,
        line: int | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        existing = self._symbols.get(name)
        self._symbols[name] = Symbol(
            name=name,
            kind=kind,
            file=file or (existing.file if existing else ""),
            line=line,
            status=SymbolStatus.ACTIVE,
            last_seen_turn=self._current_turn,
            definition_summary=existing.definition_summary if existing else "",
            aliases=aliases or (existing.aliases if existing else []),
        )

    def _extract_pending_intents(self, content: str, path: str) -> None:
        for i, line in enumerate(content.splitlines(), 1):
            match = self._TODO_PATTERN.search(line)
            if match:
                description = match.group(2).strip()
                # avoid duplicates
                if not any(p.description == description for p in self._pending_intents):
                    self._pending_intents.append(PendingIntent(
                        description=description,
                        file=path,
                        line=i,
                        turn_created=self._current_turn,
                    ))
