"""JCS canonicalizer (RFC 8785 subset) — conformance and purity.

The shared test vectors live in the platform repo (packages/axor-backend/
tests/vectors/jcs.json) because both the adapter (here) and the backend read
them; here we exercise the properties that matter for signing.
"""
from __future__ import annotations

import pytest

from axor_core.kernel.jcs import CanonicalizationError, canonicalize


def test_object_keys_sorted_lexicographically() -> None:
    assert canonicalize({"b": 1, "a": 2, "c": 3}) == b'{"a":2,"b":1,"c":3}'


def test_arrays_preserve_order_objects_do_not() -> None:
    assert canonicalize({"z": [3, 2, 1], "a": [{"y": 1, "x": 2}]}) == (
        b'{"a":[{"x":2,"y":1}],"z":[3,2,1]}'
    )


def test_utf16_code_unit_order_differs_from_code_point_order() -> None:
    """The property that makes json.dumps(sort_keys=True) non-conformant: a
    non-BMP key (U+1F600, surrogate pair with first unit 0xD83D) must sort
    BEFORE U+FB33, even though by code point it comes after."""
    out = canonicalize({"\U0001f600": 1, "דּ": 2}).decode("utf-8")
    assert out.index("\U0001f600") < out.index("דּ")


def test_mandatory_escapes_only() -> None:
    got = canonicalize({"s": '"\\\b\f\n\r\t\x00\x1f'}).decode("utf-8")
    assert got == '{"s":"\\"\\\\\\b\\f\\n\\r\\t\\u0000\\u001f"}'


def test_non_ascii_emitted_literally_not_escaped() -> None:
    assert canonicalize({"k": "café ☃"}) == '{"k":"café ☃"}'.encode("utf-8")


def test_scalars() -> None:
    assert canonicalize({"t": True, "f": False, "n": None}) == (
        b'{"f":false,"n":null,"t":true}'
    )
    assert canonicalize(-0) == b"0"
    assert canonicalize(10_000_000_000) == b"10000000000"


@pytest.mark.parametrize("value", [1.5, {"a": 2.0}, {"a": {"b": [0.0]}}, [1, 2.5]])
def test_floats_rejected(value: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonicalize(value)


def test_non_string_key_rejected() -> None:
    with pytest.raises(CanonicalizationError):
        canonicalize({1: "x"})


def test_output_is_bytes() -> None:
    assert isinstance(canonicalize({"a": 1}), bytes)
