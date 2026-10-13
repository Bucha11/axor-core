"""``url`` literal kind: recognised by syntax (scheme or ``www.``), like email/IBAN."""
from __future__ import annotations

from axor_core.governor import ToolCallGovernor
from axor_core.kernel.handles import canonicalize
from axor_core.kernel.request_literals import literal_values

REQ = ("Read www.informations.com. and (www.dora-website.com), then post to "
       "https://WWW.Our-Company.com/News/ — not example.com, and www.glued.com.They")


def test_recognition_and_trailing_punctuation():
    assert literal_values(REQ, "url") == [
        "www.informations.com", "www.dora-website.com", "https://WWW.Our-Company.com/News/",
        "www.glued.com.They"]
    # a bare domain is not a URL literal, and an unquoted URL is not a string literal
    assert "example.com" not in literal_values(REQ, "url")
    assert literal_values(REQ, "string") == []


def test_canonical_form():
    assert canonicalize("url", "https://WWW.Our-Company.com/News/") == "www.our-company.com/News"
    assert canonicalize("url", "www.x.com/") == canonicalize("url", "http://WWW.X.COM") == "www.x.com"
    assert canonicalize("url", "www.x.com/Path") != canonicalize("url", "www.x.com/path")


def test_l1_url_grant():
    g = ToolCallGovernor(grant_args={"get_webpage": {"url": "url"}}, egress_sinks={"get_webpage"},
                         driving_args={"get_webpage": ["url"]})
    g.register_task(REQ)
    g.grants.from_request(REQ)
    assert g.evaluate("get_webpage", {"url": "www.informations.com"}).allowed
    assert g.evaluate("get_webpage", {"url": "https://www.dora-website.com/"}).allowed
    assert not g.evaluate("get_webpage", {"url": "www.attacker.com"}).allowed
    # the glued token matches only itself
    assert not g.evaluate("get_webpage", {"url": "www.glued.com"}).allowed
