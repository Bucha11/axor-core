from __future__ import annotations

from axor_core.contracts.intent import Intent, IntentKind
from axor_core.policy.normalizer import IntentNormalizer


def _tool_intent(tool: str, args: dict) -> Intent:
    return Intent(kind=IntentKind.TOOL_CALL, payload={"tool": tool, "args": args}, node_id="n")


def test_system_path_under_similar_prefix_is_not_treated_as_workdir() -> None:
    normalizer = IntentNormalizer(workdir="/home/user/project")
    normalized = normalizer.normalize(
        _tool_intent("read", {"path": "/home/user/project-secret/file.txt"})
    )
    assert normalized.target_kind == "system_path"


def test_private_network_url_with_scheme_is_detected() -> None:
    normalizer = IntentNormalizer()
    normalized = normalizer.normalize(
        _tool_intent("curl", {"url": "http://10.0.0.1/admin"})
    )
    assert normalized.target_kind == "private_network"
    assert normalized.destination_kind == "private_network"


def test_external_url_remains_external() -> None:
    normalizer = IntentNormalizer()
    normalized = normalizer.normalize(
        _tool_intent("curl", {"url": "https://example.com/data"})
    )
    assert normalized.target_kind == "external_url"
    assert normalized.destination_kind == "external_domain"


def test_relative_parent_write_is_outside_workdir() -> None:
    normalizer = IntentNormalizer(workdir="/tmp/project")
    normalized = normalizer.normalize(
        _tool_intent("write", {"path": "../outside.txt", "content": "x"})
    )
    assert normalized.writes_outside_workdir is True


def test_sibling_prefix_write_is_outside_workdir() -> None:
    normalizer = IntentNormalizer(workdir="/tmp/project")
    normalized = normalizer.normalize(
        _tool_intent("write", {"path": "/tmp/project-secret/file.txt", "content": "x"})
    )
    assert normalized.writes_outside_workdir is True

