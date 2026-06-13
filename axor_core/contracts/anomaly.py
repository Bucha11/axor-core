from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AnomalyClass(str, Enum):
    NORMAL     = "normal"     # [0.0, 0.40)
    SUSPICIOUS = "suspicious" # [0.40, 0.75)
    CRITICAL   = "critical"   # [0.75, 1.0]


@dataclass(frozen=True)
class NormalizedIntent:
    """
    Behavioral abstraction of a tool call.

    Produced by IntentNormalizer from a raw Intent. It is a structural projection
    only — the gates and any observe-only detection layer read it, never raw tool
    output, webpage content, or chain-of-thought.
    """
    tool: str
    operation: str          # file_read | file_write | test | search | network_request
                            # | package_install | execute_generated_code | other
    target_kind: str        # workdir | system_path | secret | external_url
                            # | localhost | private_network | cloud_metadata | docker_socket
    destination_kind: str   # none | localhost | same_origin | external_domain | private_network
    provenance: str         # user | repo | official_docs | external_web | unknown

    reads_secret_like_data: bool
    writes_outside_workdir: bool
    executes_generated_code: bool
    after_external_read: bool
    after_secret_access: bool

    data_flow: str          # none | local_to_local | local_to_external | external_to_shell

    # axor-sentinel fields — populated by ReputationEnricher; default 0.0 (unknown)
    target_resource_reputation: float = 0.0
    target_container_reputation: float = 0.0


@dataclass(frozen=True)
class AnomalyResult:
    """
    Output of an observe-only behavioral scorer (the optional detection layer,
    e.g. axor-sentinel). It is telemetry: a score never returns an allow/deny and
    never feeds the gates — at most it tightens degradation via the opt-in floor.

    score:   0.0–1.0 continuous risk score
    cls:     discretized class
    reasons: human-readable trigger reasons
    """
    score: float
    cls: AnomalyClass
    reasons: tuple[str, ...] = field(default_factory=tuple)

