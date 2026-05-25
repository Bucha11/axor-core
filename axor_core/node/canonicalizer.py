from __future__ import annotations

import hashlib
import os

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.canonical import (
    CanonicalizedIntent,
    ExportClass,
    OperationClass,
    PathClass,
    ProviderClass,
    ToolCategory,
)
from axor_core.contracts.taint import TaintState


def _tool_category(operation: str, target_kind: str) -> ToolCategory:
    if operation == "file_read":
        return ToolCategory.FILE_READ
    if operation == "file_write":
        return ToolCategory.FILE_WRITE
    if operation in ("execute_generated_code", "test", "other") and target_kind not in (
        "external_url",
        "cloud_metadata",
    ):
        return ToolCategory.SHELL
    if operation == "network_request":
        return ToolCategory.NETWORK
    if operation == "search":
        return ToolCategory.SEARCH
    if operation == "package_install":
        return ToolCategory.PACKAGE
    return ToolCategory.UNKNOWN


def _path_class(target_kind: str) -> PathClass:
    mapping: dict[str, PathClass] = {
        "workdir": PathClass.WORKDIR,
        "system_path": PathClass.SYSTEM,
        "secret": PathClass.SECRET,
        "external_url": PathClass.EXTERNAL_URL,
        "cloud_metadata": PathClass.CLOUD_METADATA,
        "docker_socket": PathClass.DOCKER_SOCKET,
        "private_network": PathClass.PRIVATE_NETWORK,
        "localhost": PathClass.PRIVATE_NETWORK,
        "none": PathClass.NONE,
    }
    return mapping.get(target_kind, PathClass.NONE)


def _operation_class(operation: str) -> OperationClass:
    if operation == "file_read":
        return OperationClass.READ
    if operation == "file_write":
        return OperationClass.WRITE
    if operation in ("execute_generated_code", "test"):
        return OperationClass.EXECUTE
    if operation == "network_request":
        return OperationClass.NETWORK
    return OperationClass.UNKNOWN


def _arg_length_bucket(args: dict) -> int:
    total = sum(len(str(v)) for v in args.values())
    if total == 0:
        return 0
    if total < 256:
        return 1
    if total <= 4096:
        return 2
    return 3


def _taint_summary(taint_state: TaintState | None) -> str:
    if taint_state is None or not taint_state.is_tainted:
        return "clean"
    sources = ",".join(sorted(s.value for s in taint_state.sources))
    return f"tainted:{sources}"


def _path_hash(tool: str, args: dict) -> str:
    path = str(args.get("path", args.get("file_path", args.get("url", ""))))
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def _path_extension(tool: str, args: dict) -> str:
    path = str(args.get("path", args.get("file_path", "")))
    # Truncate at first control character so newline/null injection cannot
    # bleed into the extension string passed to the verifier.
    for i, c in enumerate(path):
        if ord(c) < 32 or c == "\x7f":
            path = path[:i]
            break
    _, ext = os.path.splitext(path)
    return ext.lower()[:8]  # cap at 8 chars — never a raw path


class IntentCanonicalizer:
    """
    Converts NormalizedIntent → CanonicalizedIntent.

    Strips all raw strings so that Layer 3 (LLMVerifier) receives only
    canonical feature representations. Prevents prompt injection via
    tool outputs, filenames, or web content reaching the verifier prompt.
    """

    def canonicalize(
        self,
        normalized: NormalizedIntent,
        args: dict | None = None,
        taint_state: TaintState | None = None,
        lease_summary: str = "none",
        node_depth: int = 0,
        provider: str = "unknown",
    ) -> CanonicalizedIntent:
        args = args or {}
        path = str(args.get("path", args.get("file_path", args.get("url", ""))))
        return CanonicalizedIntent(
            tool_category=_tool_category(normalized.operation, normalized.target_kind),
            path_class=_path_class(normalized.target_kind),
            path_depth=len([p for p in path.split("/") if p]),
            path_extension=_path_extension(normalized.tool, args),
            path_hash=_path_hash(normalized.tool, args),
            path_is_absolute=path.startswith("/"),
            path_is_outside_workspace=normalized.writes_outside_workdir,
            argument_shape=",".join(sorted(args.keys())),
            argument_length_bucket=_arg_length_bucket(args),
            reads_secret=normalized.reads_secret_like_data,
            writes_outside=normalized.writes_outside_workdir,
            executes_generated=normalized.executes_generated_code,
            after_external_read=normalized.after_external_read,
            after_secret_access=normalized.after_secret_access,
            data_flow=normalized.data_flow,
            operation_class=_operation_class(normalized.operation),
            export_class=ExportClass.EXTERNAL if normalized.destination_kind in (
                "external_domain", "cloud_metadata"
            ) else (
                ExportClass.LOCAL if normalized.destination_kind == "localhost"
                else ExportClass.NONE
            ),
            provider_class=ProviderClass(provider) if provider in ProviderClass._value2member_map_ else ProviderClass.UNKNOWN,
            taint_state_summary=_taint_summary(taint_state),
            lease_state_summary=lease_summary,
            node_depth=node_depth,
        )
