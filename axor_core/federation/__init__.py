"""Federation — provenance across authenticated peers.

Two levels govern a value arriving from another peer:

L1 — any value crossing a process/peer boundary is re-minted untrusted (the safe
     default; no trust in the peer's labels).
L2 — a value from an authenticated peer in a federated domain running a compatible
     kernel, carrying a valid provenance receipt, has its provenance restored (we
     trust the peer's labels instead of re-minting). A forged or tampered receipt is
     denied; an incompatible kernel or non-federated domain degrades to L1.
"""

from axor_core.federation.signing import (
    Signer,
    Verifier,
    HmacSigner,
    Ed25519Signer,
    Ed25519Verifier,
    generate_ed25519_keypair,
)
from axor_core.federation.receipt import (
    LocalIdentity,
    FederationPeer,
    FederationReceipt,
    mint_receipt,
    mint_output_receipt,
    verify_receipt,
    restore_root,
)
from axor_core.federation.transport import (
    PeerTransport,
    InMemoryPeerNetwork,
    peer_tool,
    receipt_to_dict,
    receipt_from_dict,
    to_wire,
    from_wire,
)
from axor_core.federation.gateway import (
    FederationError,
    FederationGateway,
    FederationLevel,
)
from axor_core.federation.value import FederatedValue

__all__ = [
    "Signer", "Verifier", "HmacSigner", "Ed25519Signer", "Ed25519Verifier",
    "generate_ed25519_keypair",
    "LocalIdentity", "FederationPeer", "FederationReceipt",
    "mint_receipt", "mint_output_receipt", "verify_receipt", "restore_root",
    "FederationGateway", "FederationError", "FederationLevel",
    "FederatedValue",
    "PeerTransport", "InMemoryPeerNetwork", "peer_tool",
    "receipt_to_dict", "receipt_from_dict", "to_wire", "from_wire",
]
