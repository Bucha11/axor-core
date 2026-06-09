"""Host parsing and network-destination classification (SSRF-aware).

Centralises the logic that decides whether a URL host is loopback, a private /
link-local range, or genuinely external — including the integer-encoded IPv4
forms (decimal, hex, octal) commonly used to bypass naive metadata-IP matching.
"""

from __future__ import annotations

import ipaddress
import re

_IPAddr = ipaddress.IPv4Address | ipaddress.IPv6Address

# One inet_aton component: decimal, 0x-hex, or 0-prefixed octal.
_ATON_PART = re.compile(r"(?:0x[0-9a-f]+|0[0-7]*|[1-9][0-9]*)")


def _parse_aton_part(part: str) -> int | None:
    """Parse one inet_aton-style component (decimal / 0x hex / leading-0 octal)."""
    if not _ATON_PART.fullmatch(part):
        return None
    if part.startswith("0x"):
        return int(part, 16)
    if part.startswith("0") and len(part) > 1:
        return int(part, 8)
    return int(part)


def _parse_inet_aton(host: str) -> ipaddress.IPv4Address | None:
    """Parse the legacy inet_aton() IPv4 forms that real resolvers (glibc, curl,
    browsers) accept but ``ipaddress.ip_address`` rejects:

        0177.0.0.1        dotted octal       -> 127.0.0.1
        0x7f.0x0.0x0.0x1  dotted hex         -> 127.0.0.1
        127.1             short form (a.b)   -> 127.0.0.1
        192.168.1         short form (a.b.c) -> 192.168.0.1
        2852039166        single integer     -> 169.254.169.254

    In a.b / a.b.c forms the final component fills the remaining bytes, matching
    inet_aton semantics. Returns None when the host is not such a form — the
    caller must then treat it as a DNS name, never as "safe".
    """
    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return None
    values: list[int] = []
    for part in parts:
        value = _parse_aton_part(part)
        if value is None:
            return None
        values.append(value)
    # Leading parts are single bytes; the LAST part fills the remaining bytes.
    *heads, tail = values
    if any(v > 0xFF for v in heads):
        return None
    tail_bytes = 4 - len(heads)
    if tail >= 1 << (8 * tail_bytes):
        return None
    packed = 0
    for v in heads:
        packed = (packed << 8) | v
    packed = (packed << (8 * tail_bytes)) | tail
    return ipaddress.IPv4Address(packed)


def parse_host_to_ip(host: str) -> _IPAddr | None:
    """Parse a URL host into an IP address, including obfuscated forms.

    Handles dotted IPv4/IPv6, IPv4-mapped IPv6 (unwrapped to the IPv4 address so
    range checks see the real target), 0o-prefixed octal, and every inet_aton
    legacy form: single integer (decimal/hex/octal), dotted octal/hex, and
    short forms (``127.1``). Returns None when the host is a regular DNS name.
    """
    host = host.strip().lower().rstrip(".")
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        # ::ffff:192.168.1.1 must be classified as the IPv4 it targets.
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            return addr.ipv4_mapped
        return addr
    # Python-style 0o octal (not inet_aton, but cheap to accept).
    if re.fullmatch(r"0o[0-7]+", host):
        value = int(host, 8)
        if value <= 0xFFFFFFFF:
            return ipaddress.IPv4Address(value)
        return None
    return _parse_inet_aton(host)


def classify_host(host: str) -> str | None:
    """Classify a host string.

    Returns one of: "localhost", "private_network", "external_url", or None
    when the host is empty. Private / link-local / reserved / unspecified ranges
    (covering cloud metadata at 169.254.169.254 and 0.0.0.0) classify as
    "private_network" regardless of how the address is encoded.
    """
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return None
    if host == "localhost" or host.endswith(".localhost"):
        return "localhost"
    addr = parse_host_to_ip(host)
    if addr is None:
        if host.endswith(".local"):
            return "private_network"
        return "external_url"
    if addr.is_loopback:
        return "localhost"
    if (
        addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_unspecified
        or addr.is_multicast
    ):
        return "private_network"
    return "external_url"
