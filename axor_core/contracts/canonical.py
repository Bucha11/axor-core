from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ToolCategory(str, Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SHELL = "shell"
    NETWORK = "network"
    SEARCH = "search"
    PACKAGE = "package"
    SPAWN = "spawn"
    EXPORT = "export"
    UNKNOWN = "unknown"


class PathClass(str, Enum):
    WORKDIR = "workdir"
    SYSTEM = "system"
    SECRET = "secret"
    EXTERNAL_URL = "external_url"
    CLOUD_METADATA = "cloud_metadata"
    DOCKER_SOCKET = "docker_socket"
    PRIVATE_NETWORK = "private_network"
    NONE = "none"


class OperationClass(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    SPAWN = "spawn"
    UNKNOWN = "unknown"


class ExportClass(str, Enum):
    LOCAL = "local"
    EXTERNAL = "external"
    NONE = "none"


class ProviderClass(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LANGCHAIN = "langchain"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CanonicalizedIntent:
    """
    Canonical feature representation of a NormalizedIntent for Layer 3 (LLMVerifier).

    Contains ONLY canonical features — no raw strings, no raw paths, no raw content.
    This prevents prompt injection via tool outputs or web content reaching the verifier.

    All fields are enums, ints, floats, booleans, or hashed digests.
    """
    tool_category: ToolCategory
    path_class: PathClass
    path_depth: int
    path_extension: str          # e.g. ".py", ".sh" — normalised, not raw path
    path_hash: str               # sha256[:16] of path — for dedup, not content
    path_is_absolute: bool
    path_is_outside_workspace: bool
    argument_shape: str          # e.g. "path,content" — sorted arg key names
    argument_length_bucket: int  # 0=empty, 1=small(<256), 2=medium, 3=large(>4096)
    reads_secret: bool
    writes_outside: bool
    executes_generated: bool
    after_external_read: bool
    after_secret_access: bool
    data_flow: str               # one of the known data_flow enum values
    operation_class: OperationClass
    export_class: ExportClass
    provider_class: ProviderClass
    taint_state_summary: str     # "clean" | "tainted" | "tainted:web,mcp" etc.
    lease_state_summary: str     # "none" | "active:N_uses_remaining" etc.
    node_depth: int
