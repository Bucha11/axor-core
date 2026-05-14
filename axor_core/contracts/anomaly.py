from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class AnomalyClass(str, Enum):
    NORMAL     = "normal"     # [0.0, 0.40)
    SUSPICIOUS = "suspicious" # [0.40, 0.75)
    CRITICAL   = "critical"   # [0.75, 1.0]


@dataclass(frozen=True)
class NormalizedIntent:
    """
    Behavioral abstraction of a tool call.

    Produced by IntentNormalizer from a raw Intent.
    This is the only thing the AnomalyDetector and LLMVerifier see —
    they never receive raw tool output, webpage content, or chain-of-thought.
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


@dataclass(frozen=True)
class AnomalyResult:
    """
    Output of AnomalyDetector.score().

    score:   0.0–1.0 continuous risk score
    cls:     discretized class
    reasons: human-readable trigger reasons
    """
    score: float
    cls: AnomalyClass
    reasons: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class LLMVerifier(Protocol):
    """
    Optional gray-zone verifier for ambiguous behavioral trajectories.

    Receives normalized intent sequences only — never raw content.
    Injected into MLAnomalyDetector; axor-core never imports implementations.
    """

    async def verify(
        self,
        window: list[NormalizedIntent],
        task_signal_hint: str,
        policy_name: str,
    ) -> AnomalyResult:
        """
        Review the intent sequence and return an AnomalyResult.

        window:           last N NormalizedIntents (the observation window)
        task_signal_hint: e.g. "FOCUSED/MUTATIVE/coding"
        policy_name:      active policy name for context
        """
        ...


@runtime_checkable
class AnomalyDetector(Protocol):
    """
    Behavioral anomaly scorer.

    Pluggable layer 2 of the governance cascade.
    axor-core defines this protocol; axor-classifier-simple implements it.
    Core never imports the implementation.
    """

    async def score(
        self,
        window: list[NormalizedIntent],
        task_signal_hint: str = "",
        policy_name: str = "",
    ) -> AnomalyResult:
        """
        Score the behavioral window.

        window: the last N NormalizedIntents from this session
        Returns AnomalyResult with score, class, and trigger reasons.
        """
        ...
