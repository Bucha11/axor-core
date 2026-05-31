"""Host parsing and network-destination classification (SSRF-aware).

Centralises the logic that decides whether a URL host is loopback, a private /
link-local range, or genuinely external — including the integer-encoded IPv4
forms (decimal, hex, octal) commonly used to bypass naive metadata-IP matching.
"""

from __future__ import annotations

import ipaddress
import re

_IPAddr = ipaddress.IPv4Address | ipaddress.IPv6Address


def parse_host_to_ip(host: str) -> _IPAddr | None:
    """Parse a URL host into an IP address, including obfuscated integer forms.

    Handles dotted IPv4/IPv6 as well as decimal (``2852039166``), hex
    (``0xA9FEA9FE``) and octal integer encodings that resolve to the same
    address. Returns None when the host is a regular DNS name.
    """
    host = host.strip().lower()
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        if re.fullmatch(r"0x[0-9a-f]+", host):
            value = int(host, 16)
        elif re.fullmatch(r"0o[0-7]+", host) or re.fullmatch(r"0[0-7]+", host):
            value = int(host, 8)
        elif host.isdigit():
            value = int(host)
        else:
            return None
        if 0 <= value <= 0xFFFFFFFF:
            return ipaddress.ip_address(value)
    except ValueError:
        return None
    return None


def classify_host(host: str) -> str | None:
    """Classify a host string.

    Returns one of: "localhost", "private_network", "external_url", or None
    when the host is empty. Private / link-local / reserved ranges (covering
    cloud metadata at 169.254.169.254) classify as "private_network" regardless
    of how the address is encoded.
    """
    host = (host or "").strip().lower()
    if not host:
        return None
    if host == "localhost":
        return "localhost"
    addr = parse_host_to_ip(host)
    if addr is None:
        if host.endswith(".local"):
            return "private_network"
        return "external_url"
    if addr.is_loopback:
        return "localhost"
    if addr.is_private or addr.is_link_local or addr.is_reserved:
        return "private_network"
    return "external_url"
