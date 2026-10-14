"""Fuzz tests of the content projection's extractor (MODEL.md §8.4): a token embedded in random
text is found in canonical form; text with no '@', no dotted host and no IBAN shape yields none."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from axor_core.policy.content_tokens import canonical, extract

label = st.from_regex(r"[a-z][a-z0-9-]{0,10}[a-z0-9]", fullmatch=True)
tld = st.sampled_from(["com", "org", "net", "io", "ch", "de"])
host = st.builds(lambda ls, t: ".".join(ls + [t]), st.lists(label, min_size=1, max_size=3), tld)
url = st.builds(lambda pre, h, path: pre + h + path,
                st.sampled_from(["", "http://", "https://", "www.", "https://www."]),
                host, st.sampled_from(["", "/", "/a", "/a/b-c", "/x?y=1"]))
email = st.builds(lambda u, h: f"{u}@{h}", st.from_regex(r"[a-z][a-z0-9._+-]{0,10}", fullmatch=True), host)
iban = st.builds(lambda cc, d, rest: cc + d + rest, st.from_regex(r"[A-Z]{2}", fullmatch=True),
                 st.from_regex(r"[0-9]{2}", fullmatch=True), st.from_regex(r"[A-Z0-9]{11,30}", fullmatch=True))
filler = st.from_regex(r"[A-Za-z ,!?]{0,40}", fullmatch=True)
sep = st.sampled_from([" ", "\n", " (", ": ", " <"])


@settings(max_examples=400)
@given(filler, sep, url, sep, filler)
def test_url_found(a, s1, u, s2, b):
    got = extract(a + s1 + u + s2 + b)
    assert ("url", canonical("url", u)) in got


@settings(max_examples=400)
@given(filler, email, filler)
def test_email_found_and_its_domain_is_not_a_url(a, e, b):
    got = extract(a + " " + e + " " + b)
    assert ("email", canonical("email", e)) in got
    assert not any(k == "url" and v == e.split("@")[1].lower() for k, v in got)


@settings(max_examples=400)
@given(filler, iban, filler)
def test_iban_found(a, i, b):
    assert ("iban", i) in extract(a + " " + i + " " + b)


@settings(max_examples=400)
@given(st.from_regex(r"[A-Za-z ,!?\n]{0,200}", fullmatch=True))
def test_plain_text_yields_nothing(t):
    assert extract(t) == set()
