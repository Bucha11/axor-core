from __future__ import annotations

import ipaddress
from pathlib import Path
import re
from urllib.parse import urlparse

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.intent import Intent

# Tools that perform file writes / mutations
_WRITE_TOOLS = frozenset({"write", "edit", "create", "str_replace_editor", "str_replace_based_edit_tool"})
# Tools that execute shell commands
_BASH_TOOLS = frozenset({"bash", "run_bash", "execute", "shell", "cmd"})
# Tools that perform network requests
_NETWORK_TOOLS = frozenset({"curl", "fetch", "http_request", "web_fetch", "web_search"})
# Tools that read files
_READ_TOOLS = frozenset({"read", "view", "cat", "read_file"})
# Tools that install packages
_PACKAGE_TOOLS = frozenset({"npm", "pip", "cargo", "gem", "brew", "apt"})
# Paths that indicate secret-like targets
_SECRET_PATTERNS = re.compile(
    r"(\.env|\.secret|id_rsa|id_ed25519|\.pem|\.key|credentials|token|password|api[_-]?key)",
    re.IGNORECASE,
)
# Paths that indicate system paths outside workdir
_SYSTEM_PATH_PATTERN = re.compile(r"^/(etc|usr|var|sys|proc|root|home/)")
_SSH_PATH_PATTERN = re.compile(r"~/\.ssh|/\.ssh/")
_CLOUD_METADATA_PATTERN = re.compile(r"169\.254\.169\.254|metadata\.google\.internal")
_DOCKER_SOCKET_PATTERN = re.compile(r"/var/run/docker\.sock")
_URL_PATTERN = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)


class IntentNormalizer:
    """
    Converts raw Intents into NormalizedIntents for anomaly detection.

    Tracks per-session state to detect cross-intent behavioral patterns:
    - write→execute linking (executesGeneratedCode)
    - provenance chains (afterExternalRead, afterSecretAccess)
    """

    def __init__(self, workdir: str | None = None) -> None:
        self._workdir = str(Path(workdir).expanduser().resolve(strict=False)) if workdir else None
        self._written_files: set[str] = set()
        self._external_read_seen: bool = False
        self._secret_accessed: bool = False

    def normalize(self, intent: Intent) -> NormalizedIntent:
        tool = intent.payload.get("tool", "")
        args = intent.payload.get("args", {})

        operation = self._classify_operation(tool, args)
        target_kind = self._classify_target(tool, args)
        destination_kind = self._classify_destination(tool, args)
        provenance = self._classify_provenance(tool, args)

        reads_secret = self._reads_secret(tool, args)
        writes_outside = self._writes_outside_workdir(tool, args)
        executes_generated = self._executes_generated_code(tool, args)
        data_flow = self._classify_data_flow(tool, args, operation, target_kind)

        self._update_state(tool, args, operation, target_kind, provenance)

        return NormalizedIntent(
            tool=tool,
            operation=operation,
            target_kind=target_kind,
            destination_kind=destination_kind,
            provenance=provenance,
            reads_secret_like_data=reads_secret,
            writes_outside_workdir=writes_outside,
            executes_generated_code=executes_generated,
            after_external_read=self._external_read_seen,
            after_secret_access=self._secret_accessed,
            data_flow=data_flow,
        )

    def _is_system_path(self, path: str) -> bool:
        if not _SYSTEM_PATH_PATTERN.match(path):
            return False
        if self._workdir and self._path_is_within(path, self._workdir):
            return False
        return True

    def _path_is_within(self, path: str, root: str) -> bool:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(root) / candidate
        candidate = candidate.resolve(strict=False)
        root_path = Path(root).expanduser().resolve(strict=False)
        if candidate == root_path:
            return True
        try:
            candidate.relative_to(root_path)
            return True
        except ValueError:
            return False

    # ── Operation classification ───────────────────────────────────────────────

    def _classify_operation(self, tool: str, args: dict) -> str:
        t = tool.lower()
        if t in _WRITE_TOOLS:
            return "file_write"
        if t in _READ_TOOLS:
            return "file_read"
        if t in _PACKAGE_TOOLS:
            return "package_install"
        if t in _NETWORK_TOOLS:
            return "network_request"
        if t in _BASH_TOOLS:
            cmd = str(args.get("command", args.get("cmd", ""))).lower()
            return self._classify_bash_operation(cmd)
        if "search" in t or "grep" in t or "find" in t:
            return "search"
        if "test" in t or "pytest" in t:
            return "test"
        return "other"

    def _classify_bash_operation(self, cmd: str) -> str:
        if any(kw in cmd for kw in ("curl ", "wget ", "fetch ", "http://", "https://")):
            return "network_request"
        if any(kw in cmd for kw in ("pip install", "npm install", "cargo build", "gem install")):
            return "package_install"
        if any(kw in cmd for kw in ("pytest", "python -m pytest", "jest", "cargo test", "go test")):
            return "test"
        if any(kw in cmd for kw in ("python ", "node ", "ruby ", "bash ", "./")):
            # Check if this executes a file that was written this session
            return "execute_generated_code" if self._cmd_executes_written_file(cmd) else "other"
        return "other"

    def _cmd_executes_written_file(self, cmd: str) -> bool:
        for written in self._written_files:
            if written in cmd:
                return True
        return False

    # ── Target classification ──────────────────────────────────────────────────

    def _classify_target(self, tool: str, args: dict) -> str:
        path = str(args.get("path", args.get("file_path", args.get("file", ""))))
        cmd = str(args.get("command", args.get("cmd", args.get("url", ""))))
        target = path or cmd

        if _DOCKER_SOCKET_PATTERN.search(target):
            return "docker_socket"
        if _CLOUD_METADATA_PATTERN.search(target):
            return "cloud_metadata"
        if _SSH_PATH_PATTERN.search(target):
            return "secret"
        if _SECRET_PATTERNS.search(target):
            return "secret"
        if self._is_system_path(target):
            return "system_path"

        url_target = args.get("url", args.get("command", ""))
        if isinstance(url_target, str):
            url_kind = self._classify_url_target(url_target)
            if url_kind:
                return url_kind

        return "workdir"

    # ── Destination classification ─────────────────────────────────────────────

    def _classify_destination(self, tool: str, args: dict) -> str:
        cmd = str(args.get("command", args.get("cmd", "")))
        url = str(args.get("url", ""))
        target = cmd + " " + url

        if _CLOUD_METADATA_PATTERN.search(target):
            return "cloud_metadata"
        url_kind = self._classify_url_target(target)
        if url_kind == "external_url":
            return "external_domain"
        if url_kind == "private_network":
            return "private_network"
        if url_kind == "localhost":
            return "localhost"
        return "none"

    # ── Provenance classification ──────────────────────────────────────────────

    def _classify_provenance(self, tool: str, args: dict) -> str:
        if self._external_read_seen:
            return "external_web"
        url = str(args.get("url", ""))
        if self._classify_url_target(url) == "external_url":
            return "external_web"
        cmd = str(args.get("command", ""))
        if self._classify_url_target(cmd) == "external_url":
            return "external_web"
        path = str(args.get("path", args.get("file_path", "")))
        if path.startswith(".") or "/" not in path:
            return "repo"
        return "unknown"

    # ── Derived flags ──────────────────────────────────────────────────────────

    def _reads_secret(self, tool: str, args: dict) -> bool:
        path = str(args.get("path", args.get("file_path", "")))
        cmd = str(args.get("command", ""))
        return bool(_SECRET_PATTERNS.search(path) or _SSH_PATH_PATTERN.search(path)
                    or _SECRET_PATTERNS.search(cmd))

    def _writes_outside_workdir(self, tool: str, args: dict) -> bool:
        if tool.lower() not in _WRITE_TOOLS:
            return False
        path = str(args.get("path", args.get("file_path", "")))
        if not path:
            return False
        if self._workdir:
            return not self._path_is_within(path, self._workdir)
        normalized = str(Path(path).expanduser())
        return bool(
            self._is_system_path(path)
            or normalized.startswith("/tmp")
            or path.startswith("~/")
            or normalized == ".."
            or normalized.startswith("../")
        )

    def _executes_generated_code(self, tool: str, args: dict) -> bool:
        t = tool.lower()
        if t in _BASH_TOOLS:
            cmd = str(args.get("command", args.get("cmd", ""))).lower()
            return self._cmd_executes_written_file(cmd)
        return False

    def _classify_data_flow(
        self, tool: str, args: dict, operation: str, target_kind: str
    ) -> str:
        if target_kind == "external_url" and operation == "network_request":
            return "local_to_external"
        if operation in ("execute_generated_code", "other") and self._external_read_seen:
            return "external_to_shell"
        if target_kind in ("workdir", "system_path", "localhost"):
            return "local_to_local"
        if target_kind == "external_url":
            return "local_to_external"
        return "none"

    def _classify_url_target(self, text: str) -> str | None:
        for url in _URL_PATTERN.findall(text):
            parsed = urlparse(url)
            host = (parsed.hostname or "").strip().lower()
            if not host:
                continue
            if host == "localhost":
                return "localhost"
            try:
                addr = ipaddress.ip_address(host)
            except ValueError:
                if host.endswith(".local"):
                    return "private_network"
                return "external_url"
            if addr.is_loopback:
                return "localhost"
            if addr.is_private or addr.is_link_local:
                return "private_network"
            return "external_url"
        return None

    # ── State update ───────────────────────────────────────────────────────────

    def _update_state(
        self, tool: str, args: dict, operation: str, target_kind: str, provenance: str
    ) -> None:
        if operation == "file_write":
            path = args.get("path", args.get("file_path", ""))
            if path:
                self._written_files.add(str(path))

        if target_kind == "external_url" and operation in ("network_request", "file_read"):
            self._external_read_seen = True

        if target_kind == "secret" or self._reads_secret(tool, args):
            self._secret_accessed = True
