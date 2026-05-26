from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from axor_core.contracts.anomaly import NormalizedIntent
    from axor_core.contracts.intent import Intent


@runtime_checkable
class ReputationEnricher(Protocol):
    """
    Populates reputation fields on a NormalizedIntent from the axor-sentinel snapshot.

    axor-core defines this protocol; axor-sentinel provides the implementation.
    Core never imports the implementation — dependency direction is one-way.

    enrich() receives both the normalized intent and the raw intent so the
    enricher can derive a canonical resource_id from the tool args without
    axor-core needing to know about the resource graph.

    Returns a new NormalizedIntent with target_resource_reputation and
    target_container_reputation populated from the current snapshot.
    If the resource is unknown the original intent is returned unchanged.
    """

    def enrich(
        self,
        normalized: "NormalizedIntent",
        intent: "Intent",
    ) -> "NormalizedIntent":
        """
        Return a new NormalizedIntent with reputation fields set.

        Must not raise — unknown resources silently return 0.0.
        Must not query Neo4j — reads pre-loaded snapshot only (invariant A-6).
        """
        ...
