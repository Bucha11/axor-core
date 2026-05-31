from __future__ import annotations

import hashlib
import secrets


class TraceAccessGuard:
    """
    Controls operator-level access to full DecisionTrace contents.

    Workers never have a trace reference. Operator tools authenticate
    via a session token to read full trace detail.

    Token validation is intentional-constant-time to prevent timing attacks.
    """

    def __init__(self, operator_token: str | None = None) -> None:
        if operator_token is not None:
            self._token_hash = self._hash(operator_token)
        else:
            # Generate a random token if not provided; caller must retrieve it.
            self._raw_token = secrets.token_hex(32)
            self._token_hash = self._hash(self._raw_token)

    def validate(self, token: str) -> bool:
        """Return True if the provided token matches the configured operator token."""
        return secrets.compare_digest(self._hash(token), self._token_hash)

    def require(self, token: str) -> None:
        """Raise PermissionError if token is invalid."""
        if not self.validate(token):
            raise PermissionError("invalid operator auth token — trace access denied")

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()
