"""Tests for signed daemon session grants."""

from __future__ import annotations


from axor_core.capability.session_grant import (
    SESSION_KEY_ENV,
    issue_grant,
    session_key_configured,
    verify_grant,
)


def test_no_key_means_no_grant(monkeypatch):
    monkeypatch.delenv(SESSION_KEY_ENV, raising=False)
    assert not session_key_configured()
    assert issue_grant(["read"]) is None
    assert verify_grant("anything") is None


def test_roundtrip(monkeypatch):
    monkeypatch.setenv(SESSION_KEY_ENV, "k")
    token = issue_grant(["read", "search"])
    body = verify_grant(token)
    assert body is not None
    assert body["allowed_tools"] == ["read", "search"]


def test_tampered_token_rejected(monkeypatch):
    monkeypatch.setenv(SESSION_KEY_ENV, "k")
    token = issue_grant(["read"])
    b64, _, sig = token.partition(".")
    forged = b64 + "." + ("0" * len(sig))
    assert verify_grant(forged) is None


def test_wrong_key_rejected(monkeypatch):
    monkeypatch.setenv(SESSION_KEY_ENV, "k1")
    token = issue_grant(["read"])
    monkeypatch.setenv(SESSION_KEY_ENV, "k2")
    assert verify_grant(token) is None


def test_expired_grant_rejected(monkeypatch):
    monkeypatch.setenv(SESSION_KEY_ENV, "k")
    token = issue_grant(["read"], ttl_seconds=-1.0)
    assert verify_grant(token) is None


def test_grant_cannot_be_forged_without_key(monkeypatch):
    # A process that does not hold the key cannot mint a verifiable grant.
    monkeypatch.setenv(SESSION_KEY_ENV, "secret")
    legit = issue_grant(["read", "write"])
    # Attacker tries with a guessed key.
    forged = issue_grant(["read", "write", "bash"], key=b"guess")
    assert verify_grant(legit) is not None
    assert verify_grant(forged) is None
