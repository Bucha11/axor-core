from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from axor_core.contracts.anomaly import NormalizedIntent


@runtime_checkable
class ProviderNormalizer(Protocol):
    """
    Converts raw provider-specific tool call events to NormalizedIntent.

    Each provider adapter implements this protocol.
    axor-core never imports adapter implementations — only this contract.

    Fail-closed contract:
    - Any exception from normalize() must be treated as a deny by the caller.
    - Unknown provider format must raise UnknownProviderFormatError.
    - Malformed or incomplete tool call must raise NormalizerError.
    - No silent pass-through of unrecognised formats.
    """

    def normalize(self, raw_event: Any) -> NormalizedIntent:
        """
        Convert a raw provider tool call event to NormalizedIntent.

        raw_event: provider-specific tool call representation (e.g. Anthropic
                   ToolUseBlock, OpenAI ToolCall dict, LangChain invocation dict).

        Raises:
            UnknownProviderFormatError: event type is not recognised for this provider.
            NormalizerError: event is recognised but malformed/incomplete.
        """
        ...
