"""The cross-product contract schemas, owned here.

Three artifacts cross a product boundary in this ecosystem: a recorded trace, a
tool manifest, and the Lab → Control Plane deploy package. None of them had a
single owner. `trace` and `tool-manifest` were documented as axor-core-owned
while the only real definitions lived in axor-lab — which kept a directory named
`_shared_from_axor_core` holding narrower stubs of files this package did not
have. `cp-deploy` had no schema anywhere, and its producer and consumer, written
independently in two repositories, had already drifted: the consumer's entire
trace-replay path keys on `regression_traces`, which the producer has never
emitted.

These tests are the owner's half. Each consumer proves its own artifacts against
the same files, loaded by import.
"""
from __future__ import annotations

import pytest

from axor_core.contracts.schemas import SCHEMA_NAMES, load, validate


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_every_schema_loads_and_declares_itself(name: str) -> None:
    schema = load(name)
    assert schema["$id"].endswith(f"{name}.schema.json")
    assert schema["$schema"].startswith("https://json-schema.org/")
    # A schema that constrains nothing is a file, not a contract. `required` is
    # how the object-shaped ones say it; `predicate` is a top-level `oneOf` over
    # its own $defs and says it that way instead.
    assert schema.get("required") or schema.get("oneOf"), f"{name} constrains nothing"


def test_an_unknown_schema_name_is_refused_not_guessed() -> None:
    with pytest.raises(KeyError, match="unknown schema"):
        load("cp-deployy")


def _package(**over: object) -> dict:
    base = {
        "schema_version": "axor-cp-deploy/v1",
        "verified": True,
        "kernel": "axor-core@0.11.0",
        "policy": {"profile": "default"},
        "config_hash": "sha256:abc",
        "parametric_config_hash": "sha256:def",
        "tool_manifests": [{"schema_version": "tool-manifest/v1", "id": "post",
                            "args_schema": {}, "side_effecting": True,
                            "effect": {"default_class": "EXPORT",
                                       "driving_args": ["body"]}}],
        "regressions": [{"trace_id": "tr1", "expected_verdict": "DENY",
                         "trace_ref": "sha256:x", "expected_sequence": ["DENY"]}],
        "source": {"bundle_id": "b1", "condition_id": "c1"},
    }
    base.update(over)
    return base


def test_a_well_formed_deploy_package_validates() -> None:
    assert validate("cp-deploy", _package()) == []


def test_every_missing_required_field_is_reported_at_once() -> None:
    """A rejection is complete. A validator that stops at the first problem makes
    a caller fix one field, run again, and find the next."""
    errors = validate("cp-deploy", {})
    assert len(errors) == len(load("cp-deploy")["required"])


def test_a_template_dump_is_distinguishable_from_evidence() -> None:
    """`verified` is what separates an evidence-backed export from a config
    template. The schema types it; the consumer refuses false."""
    assert validate("cp-deploy", _package(verified=False)) == []
    assert validate("cp-deploy", _package(verified="yes")) != []


def test_a_pin_id_too_long_for_the_corpus_key_is_refused() -> None:
    """The Control Plane stores a pin as `lab:{trace_id}` in a 64-character key.
    A longer id used to be TRUNCATED there, so two pins agreeing on their first
    60 characters became one row and one side of the case was lost. The bound
    belongs in the format, not only in the consumer that discovered it."""
    fits = _package(regressions=[{"trace_id": "t" * 60, "expected_verdict": "DENY",
                                  "trace_ref": "sha256:x", "expected_sequence": ["DENY"]}])
    over = _package(regressions=[{"trace_id": "t" * 61, "expected_verdict": "DENY",
                                  "trace_ref": "sha256:x", "expected_sequence": ["DENY"]}])
    assert validate("cp-deploy", fits) == []
    assert validate("cp-deploy", over) != []


def test_a_trace_ref_must_be_a_content_hash() -> None:
    bad = _package(regressions=[{"trace_id": "tr1", "expected_verdict": "DENY",
                                 "trace_ref": "whatever", "expected_sequence": ["DENY"]}])
    assert validate("cp-deploy", bad) != []


def test_a_verdict_outside_the_pair_is_refused() -> None:
    bad = _package(regressions=[{"trace_id": "tr1", "expected_verdict": "MAYBE",
                                 "trace_ref": "sha256:x", "expected_sequence": ["MAYBE"]}])
    assert validate("cp-deploy", bad) != []


def test_the_tool_manifest_schema_constrains_the_effect_class() -> None:
    """The consumer used to re-state this enum by hand. It agrees today; nothing
    kept it so."""
    ok = {"schema_version": "tool-manifest/v1", "id": "post", "args_schema": {},
          "side_effecting": True,
          "effect": {"default_class": "EXPORT", "driving_args": ["b"]}}
    assert validate("tool-manifest", ok) == []
    bad = {**ok, "effect": {"default_class": "TRANSMOGRIFY", "driving_args": []}}
    assert validate("tool-manifest", bad) != []


def test_a_deploy_package_validates_its_manifests_as_manifests() -> None:
    """`tool_manifests` is a `$ref`, not a re-statement.

    The deploy schema first said only `{"type": "object"}` here and left the
    manifest rules to whoever read it — which is the exact shape of the drift
    this package exists to end. A broken manifest inside a structurally perfect
    package must be reported by the deploy schema itself."""
    broken = _package(tool_manifests=[{"schema_version": "tool-manifest/v1",
                                       "id": "post", "args_schema": {},
                                       "side_effecting": True,
                                       "effect": {"default_class": "TRANSMOGRIFY",
                                                  "driving_args": []}}])
    errors = validate("cp-deploy", broken)
    assert errors and any("tool_manifests[0]" in e for e in errors), errors


def test_a_manifest_that_resolves_its_effect_from_args_is_not_rejected() -> None:
    """`effect.resolve[].when` is a `predicate`, which is why `predicate` is
    owned here too. While it was not, this manifest — entirely valid — failed
    with `unknown schema ref predicate.schema.json`: the reference did not
    resolve, so the strictest kind of manifest was the one that could not pass.
    """
    resolving = {
        "schema_version": "tool-manifest/v1", "id": "send_email",
        "args_schema": {"type": "object"}, "side_effecting": True,
        "effect": {
            "default_class": "WRITE", "driving_args": ["to", "body"],
            "resolve": [{"when": {"to": {"not_in": ["$inputs.company_domains"]}},
                         "class": "EXPORT"}],
        },
    }
    assert validate("tool-manifest", resolving) == []
    assert validate("cp-deploy", _package(tool_manifests=[resolving])) == []
