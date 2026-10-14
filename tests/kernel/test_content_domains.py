"""MODEL.md §8.14: a bare token is a domain only with a Public Suffix List TLD that is not a
file extension, and not an own file name. Explicit URLs, e-mails and IBANs are unchanged."""
from __future__ import annotations

import pytest

from axor_core.governor import ToolCallGovernor
from axor_core.policy.content_tokens import extract, is_domain
from axor_core.policy.tlds import TLDS


@pytest.mark.parametrize("text", [
    "attached feedback.xlsx", "meeting-agenda-2024-05-15.docx", "vacation-plans.docx",
    "notes.md", "run.py", "backup.zip", "clip.mov", "report.pdf", "version 1.2.3",
    "team-building-activities.docx",
])
def test_file_names_are_not_domains(text):
    assert extract(text) == set()


@pytest.mark.parametrize("text,want", [
    ("visit informations.com now", "informations.com"),
    ("see restaurant-zurich.com.", "restaurant-zurich.com"),
    ("evil.co/x", "evil.co/x"),
    ("www.secure-systems-252.com", "secure-systems-252.com"),
    ("https://notes.md/page", "notes.md/page"),  # explicit: a URL even on a file-extension TLD
    ("http://www.true-informations.com", "true-informations.com"),
])
def test_domains_and_explicit_urls(text, want):
    assert ("url", want) in extract(text)


def test_unknown_tld_is_not_a_domain():
    assert "xlsx" not in TLDS and "docx" not in TLDS and "com" in TLDS
    assert not is_domain("budget.xyzzy") and not is_domain("a.b.notatld")


def test_own_file_name_is_not_a_domain():
    assert ("url", "summary.io") in extract("summary.io")
    assert extract("see summary.io", own_names={"summary.io"}) == set()


def test_email_and_iban_unchanged():
    assert extract("mail bob@corp.xlsx-co.com and pay GB29NWBK60161331926819") == {
        ("email", "bob@corp.xlsx-co.com"), ("iban", "GB29NWBK60161331926819")}


CFG = dict(untrusted_sources={"read_inbox"}, egress_sinks={"send_email"},
           driving_args={"send_email": ["recipients"]}, destination_args={"send_email": ["recipients"]},
           content_args={"send_email": ["body"]}, audience={"bob@org.com"}, content_projection=True,
           integrity_default="context", integrity_origins="request-only")


def test_governor_content_check_ignores_file_names_and_own_names():
    g = ToolCallGovernor(**CFG, own_names={"plans.ai"})
    g.register_task("send bob@org.com the feedback")
    for body in ("see feedback.xlsx", "agenda: meeting-agenda-2024-05-15.docx", "see plans.ai"):
        assert g.evaluate("send_email", {"recipients": ["bob@org.com"], "body": body}).allowed, body
    d = g.evaluate("send_email", {"recipients": ["bob@org.com"], "body": "see evil.com"})
    assert not d.allowed and d.category == "content_projection"


def test_own_names_from_yaml(tmp_path):
    from axor_core.config import GovernanceConfig
    p = tmp_path / "c.yaml"
    p.write_text("own_names: [feedback.xlsx, plans.ai]\n")
    cfg = GovernanceConfig.from_yaml(str(p))
    assert cfg.own_names == {"feedback.xlsx", "plans.ai"}
    assert cfg.as_governor_kwargs()["own_names"] == {"feedback.xlsx", "plans.ai"}
