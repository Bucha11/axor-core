"""The shared provenance-arming map — the counterpart to the deny gates.

Where :mod:`axor_core.policy.gates` is the *decision* half of the per-value model
(does this call's driving value carry taint?), this is the *arming* half: when a
tool's output comes back, what causal_root does it introduce? Both the streaming
:class:`~axor_core.node.intent_loop.IntentLoop` and the synchronous
:class:`~axor_core.governor.ToolCallGovernor` must answer this identically — a
divergence here silently desynchronises what the two paths consider tainted — so
the mapping lives once, here, and both register their outputs through it.

Pure: it maps (tool name, normalized intent, declared roles) → a ``CausalRoot`` (or
``None`` for a clean read that arms nothing). It does not touch the taint engine;
the caller registers the returned root.
"""
from __future__ import annotations

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.taint import TaintSource, TrustedOrigin
from axor_core.taint.causal_root import CausalRoot


def output_root(
    tool_name: str,
    normalized: NormalizedIntent | None,
    *,
    untrusted_sources: "frozenset[str] | set[str]",
    sensitive_sources: "frozenset[str] | set[str]",
) -> CausalRoot | None:
    """The causal_root a tool's output introduces, or ``None`` for a clean read.

    A deployment's declared role wins over the normalizer heuristic: a tool named
    as a sensitive source reads secrets (untrusted + sensitive, arming the floor);
    a tool named as an untrusted source reads attacker-influenceable data
    (untrusted). For any other tool the normalizer's structural classification
    decides — an external/network read is untrusted; a secret/system read is
    untrusted + sensitive; everything else is clean and arms nothing.
    """
    if tool_name in sensitive_sources:
        return CausalRoot.external_read(TaintSource.FILE, sensitive=True)
    if tool_name in untrusted_sources:
        return CausalRoot.external_read(TaintSource.WEB)
    if normalized is None:
        return None
    if (
        normalized.target_kind in ("external_url", "cloud_metadata", "docker_socket")
        or normalized.operation == "network_request"
    ):
        return CausalRoot.external_read(TaintSource.WEB)
    if (
        normalized.target_kind in ("secret", "system_path")
        or normalized.reads_secret_like_data
        or normalized.writes_outside_workdir
    ):
        sensitive = (
            normalized.target_kind == "secret" or normalized.reads_secret_like_data
        )
        return CausalRoot.external_read(TaintSource.FILE, sensitive=sensitive)
    return None


def is_trusted_tool(
    tool_name: str,
    benign_tools: "frozenset[str] | set[str]",
    *,
    require_tool_roles: bool,
) -> bool:
    """Whether a clean read's output seeds the trusted-origin index (context-default
    integrity). Called only for a tool whose ``output_root`` is ``None``.

    Under STRICT roles only an explicitly declared ``benign_tools`` read is trusted.
    Outside STRICT a read the normalizer classifies clean is trusted too — parity
    with today, where such a read registers nothing and its values are clean
    (docs/rfc-integrity-context-default.md §9, decision 1).
    """
    return tool_name in benign_tools or not require_tool_roles


def seed_operator_trusted(taint: object, value_policies: "dict | None") -> None:
    """Register every ``enum`` allowlist member as an operator-trusted value.

    No-op for a backend without ``register_trusted`` (a custom trust model that
    does not implement the context-default contract) and harmless in ``"clean"``
    mode, where the index is never consulted.
    """
    register = getattr(taint, "register_trusted", None)
    if register is None:
        return
    for preds in (value_policies or {}).values():
        for p in preds or ():
            if getattr(p, "kind", None) == "enum":
                allowed = getattr(p, "allowed", None) or ()
                register(sorted(allowed, key=repr), TrustedOrigin.OPERATOR)


def derive_driving_root(
    taint: object,
    args: dict,
    driving: "frozenset[str] | set[str] | list[str] | None",
    *,
    integrity_sink: bool = False,
) -> object:
    """The driving root the taint gate decides on — shared by both paths and by
    the recorded payload, so decision and record cannot disagree.

    For an integrity sink under context-default integrity, numbers in the driving
    args must be trusted values too (``include_scalars``): an amount the attacker
    names is as much a choice as a recipient.
    """
    from axor_core.policy.gates import driving_subset

    subset = driving_subset(args or {}, driving)
    if integrity_sink and getattr(taint, "integrity_default", None) == "context":
        return taint.derive_value(subset, include_scalars=True)  # type: ignore[attr-defined]
    return taint.derive_value(subset)  # type: ignore[attr-defined]


def source_tokens(sources: object) -> list[str]:
    """Taint sources as their declared string values (`web`, `mcp`, …).

    ``str()`` on a ``(str, Enum)`` member yields ``TaintSource.WEB`` — a Python
    implementation detail, and these tokens go into recorded governance
    artifacts that other languages read and hash.
    """
    return sorted(getattr(s, "value", None) or str(s) for s in sources or ())


class ValueRefLedger:
    """Mints a value id per registered tool output and answers which ids a
    later value was derived from.

    The kernel event schema is written in value REFS: ``TOOL_RESULT`` mints
    ``value_ref``, ``TOOL_CALL`` binds ``arg_refs = {arg: value_ref}``, the
    replay fold resolves those refs against ``state.tainted_refs``, and
    ``subgraph`` walks them as edges. Neither wrapping path spoke that
    vocabulary — both derive taint from CONTENT — so both recorded verdicts with
    no refs at all. Replay still folded them (it falls back to ``driving_root``),
    but every downstream consumer that reasons over the value graph got nothing:
    the Control Plane's incident converter reads ``arg_refs`` to bind a sink's
    driving argument to the value that tainted it, and with no refs it cannot
    reproduce a recorded DENY, so it refuses the whole run.

    Refs are minted here and resolved with the SAME content derivation the gates
    decide on — one :class:`ValueTaintLedger` per registered value, queried with
    the same ``derive`` — so a ref can never claim a link the verdict did not
    turn on, or miss one it did.

    Ids are ``v1``, ``v2``, … in registration order: deterministic, so two runs
    of the same session produce byte-identical traces.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        # (ref, single-value ledger, root) in registration order
        self._entries: list[tuple[str, object, object]] = []

    def mint(self, content: object, root: object) -> str:
        """Register ``content`` under a fresh ref and return it."""
        from axor_core.taint.ledger import ValueTaintLedger

        ledger = ValueTaintLedger()
        ledger.register(content, root)  # type: ignore[arg-type]
        ref = f"v{len(self._entries) + 1}"
        self._entries.append((ref, ledger, root))
        return ref

    def refs_for(self, value: object) -> list[str]:
        """Every minted ref whose content this value carries, oldest first."""
        out: list[str] = []
        for ref, ledger, _root in self._entries:
            derived = ledger.derive(value)  # type: ignore[attr-defined]
            if derived.is_tainted or derived.sensitive:
                out.append(ref)
        return out

    def ref_for(self, value: object) -> str | None:
        """The most recent ref this value carries, or None.

        ``arg_refs`` is one ref per argument, so an argument splicing two
        registered values can only name one. The most recent is the closer
        cause; the joined taint the verdict turned on is recorded whole in
        ``driving_root`` either way, so this choice narrows the value GRAPH, not
        the verdict.
        """
        refs = self.refs_for(value)
        return refs[-1] if refs else None


def declared_roles(
    tool_name: str,
    *,
    untrusted_sources: "frozenset[str] | set[str]" = frozenset(),
    sensitive_sources: "frozenset[str] | set[str]" = frozenset(),
    egress_sinks: "frozenset[str] | set[str]" = frozenset(),
    imperative_sinks: "frozenset[str] | set[str]" = frozenset(),
    positional_sinks: "frozenset[str] | set[str]" = frozenset(),
    integrity_sinks: "frozenset[str] | set[str]" = frozenset(),
) -> "dict[str, bool]":
    """The data-flow roles the OPERATOR declared for this tool.

    ``normalized`` is the normalizer's structural guess from the tool's name and
    arguments, and it does not recognise a deployment's own vocabulary: a tool
    called ``send_email`` normalises to ``destination_kind: none``, an ordinary
    local call. What makes it an egress sink is the operator declaring it one —
    ``egress_sinks={"send_email"}`` — which is what ``taint_gate`` actually keys
    on when it denies.

    That declaration was never recorded, so a consumer reading the trace saw a
    DENY on a call that looks, by every field present, like a harmless local
    operation. The Control Plane's converter reads egress-ness off
    ``destination_kind`` alone, concluded the run contained no sink at all, and
    refused it — "no recorded egress consequence" — for a run whose whole point
    was a blocked exfiltration.
    """
    return {
        "untrusted_source": tool_name in untrusted_sources,
        "sensitive_source": tool_name in sensitive_sources,
        "egress_sink": tool_name in egress_sinks,
        "imperative_sink": tool_name in imperative_sinks,
        "positional_sink": tool_name in positional_sinks,
        "integrity_sink": tool_name in integrity_sinks,
    }


def normalized_payload(normalized: object) -> "dict[str, object]":
    """The structural projection a recorded call was gated on.

    ``kernel.replay.normalized_from_payload`` reads exactly these keys and
    defaults every missing one to the benign value (``operation="other"``,
    ``destination_kind="none"``, …). Omitting the block therefore does not make
    replay abstain — it makes replay re-gate the call as a local no-op, so the
    ssrf / consequence / carrier gates silently cannot fire and a recorded DENY
    comes back ALLOW. The Control Plane reads ``destination_kind`` from here to
    tell an egress sink from a read at all.
    """
    return {
        "operation": normalized.operation,  # type: ignore[attr-defined]
        "target_kind": normalized.target_kind,  # type: ignore[attr-defined]
        "destination_kind": normalized.destination_kind,  # type: ignore[attr-defined]
        "provenance": normalized.provenance,  # type: ignore[attr-defined]
        "reads_secret_like_data": bool(normalized.reads_secret_like_data),  # type: ignore[attr-defined]
        "writes_outside_workdir": bool(normalized.writes_outside_workdir),  # type: ignore[attr-defined]
        "executes_generated_code": bool(normalized.executes_generated_code),  # type: ignore[attr-defined]
        "after_external_read": bool(normalized.after_external_read),  # type: ignore[attr-defined]
        "after_secret_access": bool(normalized.after_secret_access),  # type: ignore[attr-defined]
        "data_flow": normalized.data_flow,  # type: ignore[attr-defined]
    }


def call_payload(
    tool_name: str,
    args: "dict[str, object]",
    *,
    taint: object,
    driving_args: "dict[str, frozenset[str]] | dict[str, set[str]] | None" = None,
    normalized: object = None,
    refs: "ValueRefLedger | None" = None,
    roles: "dict[str, bool] | None" = None,
) -> "dict[str, object]":
    """The provenance a tool-call verdict was reached on.

    ``driving_root`` is the shape the kernel event schema documents for
    TOOL_CALL, and ``kernel.replay.root_from_payload`` reads it directly.

    The per-argument breakdown goes under ``arg_provenance``, NOT ``arg_refs``.
    That distinction is load-bearing and this got it wrong: the schema's
    ``arg_refs`` is ``{arg: value_ref}`` — a mapping to opaque value REFERENCES
    that ``replay._derive_driving_root`` looks up in ``state.tainted_refs`` and
    ``subgraph`` walks as edges. Writing a nested provenance dict there crashed
    the replay fold outright (`TypeError: unhashable type: 'dict'`).

    ``arg_refs`` is populated only when the caller supplies a
    :class:`ValueRefLedger` — the refs then come from the same content
    derivation the gates decided on, so they cannot disagree with the verdict.
    Without one the key is absent and replay takes its documented fallback to
    ``driving_root``.

    Shared by BOTH ways of wrapping an agent — ``ToolCallGovernor`` (the
    synchronous per-call path) and ``IntentLoop`` (the async streaming path) —
    because they are two entry points to one kernel, not two kernels. Each
    computed this provenance to reach a verdict and then recorded only the tool
    name, so every consumer had to re-derive it from raw arguments, which is how
    a second taint ledger gets built downstream. Living in one function is what
    stops the two paths from drifting into two vocabularies.
    """
    from axor_core.policy.gates import driving_subset

    driving = (driving_args or {}).get(tool_name)
    arg_provenance: dict[str, object] = {}
    arg_refs: dict[str, str] = {}
    for name, value in (args or {}).items():
        root = taint.derive_value(value)  # type: ignore[attr-defined]
        arg_provenance[name] = {
            "sources": source_tokens(root.sources),
            "sensitive": bool(root.sensitive),
        }
        if refs is not None:
            ref = refs.ref_for(value)
            if ref is not None:
                arg_refs[name] = ref
    integrity_sink = bool((roles or {}).get("integrity_sink"))
    driving_root = derive_driving_root(
        taint, args or {}, driving, integrity_sink=integrity_sink,
    )
    payload: dict[str, object] = {
        "tool": tool_name,
        # the raw arguments the call was made with. Replay re-gates on these —
        # `evaluate_call` runs the value-policy and positional gates over them —
        # so a payload without `args` re-decides an egress call on an empty dict.
        "args": dict(args or {}),
        "arg_provenance": arg_provenance,
        "driving_args": sorted(driving) if driving else [],
        "driving_root": {
            "sources": source_tokens(driving_root.sources),
            "sensitive": bool(driving_root.sensitive),
        },
        "floor_active": bool(taint.confidentiality_floor_active()),  # type: ignore[attr-defined]
    }
    if arg_refs:
        payload["arg_refs"] = arg_refs
    if getattr(taint, "integrity_default", "clean") == "context":
        _add_context_fields(
            payload, taint, args or {}, driving_subset(args or {}, driving),
            include_scalars=integrity_sink,
        )
    if normalized is not None:
        payload["normalized"] = normalized_payload(normalized)
    if roles is not None:
        payload["roles"] = dict(roles)
    return payload


def _root_payload(root: object) -> "dict[str, object]":
    return {
        "sources": source_tokens(root.sources),  # type: ignore[attr-defined]
        "sensitive": bool(root.sensitive),  # type: ignore[attr-defined]
    }


def _add_context_fields(
    payload: "dict[str, object]", taint: object, args: dict, driving: dict,
    *, include_scalars: bool = False,
) -> None:
    """Record what a context-mode verdict turned on, so it can be re-derived.

    Under ``integrity_default == "context"`` the driving root is
    ``driving_carried ⊔ (context_root if any driving arg is not a trusted value)``.
    Recording the three parts — and, per argument, whether it is a trusted value —
    lets replay re-derive the root under a counterfactual (different driving args,
    synthetic taint, an excised ref) instead of echoing the recorded answer. No
    content is recorded: roots are labels, trust is a boolean and an origin name.
    Legacy (``"clean"``) payloads are left byte-identical.
    """
    payload["integrity_default"] = "context"
    payload["context_root"] = _root_payload(taint.context_root())  # type: ignore[attr-defined]
    payload["driving_carried"] = _root_payload(taint.derive_carried(driving))  # type: ignore[attr-defined]
    provenance = payload.get("arg_provenance")
    if not isinstance(provenance, dict):
        return
    for name, value in args.items():
        entry = provenance.setdefault(name, {})
        # Wrapped as {name: value}: the same leaves the gate's check sees for this
        # argument inside the driving subset (a nested dict's keys are data).
        entry["trusted"] = bool(taint.is_trusted(  # type: ignore[attr-defined]
            {name: value}, include_scalars=include_scalars))
        origin = taint.trusted_origin(value)  # type: ignore[attr-defined]
        entry["trusted_origin"] = getattr(origin, "value", None)


def result_payload(
    tool_name: str, value_ref: str, root: object,
) -> "dict[str, object]":
    """The TOOL_RESULT payload for an executed tool whose output armed taint.

    This is the event that makes a trace have a SOURCE. The replay fold
    registers ``value_ref`` under ``root`` in ``state.tainted_refs`` — which is
    what makes a later ``arg_refs`` binding resolve to anything — and the
    Control Plane's incident converter requires at least one of these before it
    will accept a run at all ("no untrusted source in the recorded run").
    Neither wrapping path emitted one, so every trace either path produced was a
    sequence of verdicts with no explanation of where the taint came from.
    """
    return {
        "tool": tool_name,
        "status": "ok",
        "value_ref": value_ref,
        "root": {
            "sources": source_tokens(root.sources),  # type: ignore[attr-defined]
            "sensitive": bool(root.sensitive),  # type: ignore[attr-defined]
        },
    }
