"""Property-based fuzzing of the two highest-value security invariants.

1. path_matches_allowlist must never *false-accept*: if it returns True, the
   resolved candidate is genuinely contained by a resolved allowed root.
2. The SSRF host parser must classify every integer/dotted encoding of a
   private / loopback / link-local address as non-external.

These guard against the classes of bypass found in the security review
(path-traversal escape, encoded-IP SSRF) — not just the specific examples.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path

from hypothesis import given, settings, strategies as st

from axor_core.capability.lease_validator import path_matches_allowlist
from axor_core.node.normalizer import IntentNormalizer
from axor_core.security.net import parse_host_to_ip as _parse_host_to_ip


# ── 1. path allowlist soundness (no false-accept) ────────────────────────────

# Path segments including traversal, hidden dirs, and odd-but-legal names.
_segment = st.sampled_from(
    ["..", ".", "a", "b", "etc", "repo", "sub", "x.txt", "..%2f", " ", "f"]
)
_paths = st.lists(_segment, min_size=1, max_size=8).map(lambda parts: "/".join(parts))
_roots = st.sampled_from(["/repo", "/repo/sub", "/srv/data", "/home/u/proj"])


def _resolved_within(path: str, root: str) -> bool:
    cand = Path(os.path.expanduser(path)).resolve(strict=False)
    r = Path(os.path.expanduser(root)).resolve(strict=False)
    return cand == r or r in cand.parents


@given(rel=_paths, root=_roots, absolute=st.booleans())
@settings(max_examples=400)
def test_allowlist_never_false_accepts(rel: str, root: str, absolute: bool):
    candidate = (root + "/" + rel) if absolute else rel
    if path_matches_allowlist(candidate, [root]):
        # If accepted, it MUST genuinely resolve within the root.
        assert _resolved_within(candidate, root), (
            f"false-accept: {candidate!r} accepted for root {root!r}"
        )


@given(root=_roots)
@settings(max_examples=50)
def test_allowlist_rejects_obvious_escape(root: str):
    assert not path_matches_allowlist(root + "/../../../etc/passwd", [root])
    assert not path_matches_allowlist("/etc/passwd", [root])


# ── 2. SSRF host classification across encodings ─────────────────────────────

_private_ints = st.one_of(
    st.integers(min_value=int(ipaddress.ip_address("10.0.0.0")),
                max_value=int(ipaddress.ip_address("10.255.255.255"))),
    st.integers(min_value=int(ipaddress.ip_address("192.168.0.0")),
                max_value=int(ipaddress.ip_address("192.168.255.255"))),
    st.integers(min_value=int(ipaddress.ip_address("169.254.0.0")),
                max_value=int(ipaddress.ip_address("169.254.255.255"))),
    st.integers(min_value=int(ipaddress.ip_address("127.0.0.0")),
                max_value=int(ipaddress.ip_address("127.255.255.255"))),
)


def _encodings(value: int) -> list[str]:
    dotted = str(ipaddress.ip_address(value))
    return [dotted, str(value), hex(value), oct(value).replace("0o", "0o")]


@given(value=_private_ints)
@settings(max_examples=400)
def test_encoded_private_ips_never_classified_external(value: int):
    norm = IntentNormalizer()
    for host in _encodings(value):
        addr = _parse_host_to_ip(host)
        assert addr is not None, f"failed to parse host encoding {host!r}"
        assert addr.is_private or addr.is_loopback or addr.is_link_local
        kind = norm._classify_url_target(f"http://{host}/x")
        assert kind in ("localhost", "private_network"), (
            f"SSRF bypass: {host!r} classified as {kind!r}"
        )


@given(
    octets=st.lists(st.integers(min_value=1, max_value=223), min_size=4, max_size=4),
)
@settings(max_examples=200)
def test_public_ips_classified_external(octets: list[int]):
    ip = ".".join(str(o) for o in octets)
    addr = ipaddress.ip_address(ip)
    # Only assert for genuinely-public addresses.
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast:
        return
    kind = IntentNormalizer()._classify_url_target(f"http://{ip}/")
    assert kind == "external_url"
