"""Phase 4 hygiene regressions: sanitizer case-insensitive reserved-prefix,
symbol-table whole-word deprecation match, errors __all__ completeness, and the
unified token estimator."""

from __future__ import annotations

import pytest

import axor_core.errors as errors
from axor_core.contracts.extension import ExtensionTool
from axor_core.context.symbol_table import SymbolTable
from axor_core.extensions.sanitizer import ExtensionSanitizer
from axor_core.tokens import estimate_tokens

pytestmark = pytest.mark.adversarial


def _tool(name: str) -> ExtensionTool:
    return ExtensionTool(name=name, description="x", parameters={}, source="ext")


def test_sanitizer_reserved_prefix_is_case_insensitive():
    san = ExtensionSanitizer()
    tools = (_tool("AXOR_INTERNAL_backdoor"), _tool("__dunder_evil"),
             _tool("Axor_Internal_x"), _tool("legit_tool"))
    kept = {t.name for t in san._sanitize_tools(tools)}
    assert kept == {"legit_tool"}      # all reserved-prefix variants dropped


def test_symbol_deprecation_is_whole_word():
    st = SymbolTable()
    st.ingest_file("m.py", "def foo():\n    pass\n")
    st.mark_renamed("foo", "bar")
    assert "foo" in st.deprecated_names()
    # 'foobar' / 'food' must NOT trip the deprecated-'foo' penalty (substring bug)
    assert st.relevance_penalty("foobar and food are unrelated") == 0.0
    assert st.text_contains_deprecated("foobar") is False
    # a real whole-word mention does
    assert st.relevance_penalty("we still call foo() here") > 0.0
    assert st.text_contains_deprecated("call foo() now") is True


def test_errors_all_is_complete():
    # __all__ must list every public exception, not a stale subset.
    exported = set(errors.__all__)
    defined = {
        name for name in dir(errors)
        if name.endswith("Error") and isinstance(getattr(errors, name), type)
    }
    missing = defined - exported
    assert not missing, f"errors.__all__ missing: {sorted(missing)}"


@pytest.mark.parametrize("text,expected", [
    ("", 0),
    (None, 0),
    ("a" * 4, 1),
    ("a" * 40, 10),
    (12345, 1),         # non-strings are stringified ("12345" -> 5//4 = 1)
])
def test_estimate_tokens(text, expected):
    assert estimate_tokens(text) == expected
