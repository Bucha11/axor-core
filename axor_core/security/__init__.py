"""Single source of truth for security-critical primitives.

Path canonicalization and host/SSRF classification used to be duplicated across
the normalizer, lease validator and policy ceiling. Divergence between those
copies was the root of the resource-aliasing and SSRF-encoding bypass classes.
Everything that resolves a path or classifies a network host now routes through
this package so the rules cannot drift.
"""

from axor_core.security.net import classify_host, parse_host_to_ip
from axor_core.security.paths import (
    lexical_normalize,
    path_matches_allowlist,
    path_within,
    paths_within,
    resolve_path,
)

__all__ = [
    "classify_host",
    "parse_host_to_ip",
    "lexical_normalize",
    "path_matches_allowlist",
    "path_within",
    "paths_within",
    "resolve_path",
]
