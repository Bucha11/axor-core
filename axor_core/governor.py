"""Synchronous per-tool-call governor.

The streaming :class:`IntentLoop` is the kernel's primary enforcement path, but
it is built for an executor that *yields* events. Many integrations — a
LangChain tool wrapper, a benchmark harness, any framework that owns its own
agent loop — instead have a plain "the model wants to call tool T with args A;
may it?" question, answered synchronously, with the framework executing the tool
itself.

:class:`ToolCallGovernor` answers exactly that. It runs the same content-blind
and per-value gates the IntentLoop runs — consequence, value policies, the
SSRF/internal-destination gate, positional admission, the carrier/imperative
gate, and the per-value taint check with the confidentiality floor — over a
single ``(tool_name, args)`` pair, and tracks tool outputs in the same per-value
ledger so a later sink carrying untrusted or secret content is gated.

It deliberately does **not** cover capability/lease/degradation: those are
policy-and-session concerns the host framework owns (it decides which tools
exist and whether the session is degraded). What this governs is the data-flow
and action-class core that a tool registry alone cannot express.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.taint import (
    TrustedOrigin,
    resolve_integrity_default,
    resolve_integrity_origins,
)
from axor_core.policy.normalizer import IntentNormalizer
from axor_core.policy.consequence import consequence_class
from axor_core.contracts.trace import (
    IntentDeniedEvent,
    TaintPropagatedEvent,
    TraceEvent,
    TraceEventKind,
)
from axor_core.policy.gates import (
    GateDecision,
    floor_scope_hit,
    carrier_gate,
    grant_gate,
    grant_gate_confirmable,
    consequence_gate,
    integrity_superseded_by_decidable,
    positional_gate,
    ssrf_gate,
    taint_gate,
    value_policy_gate,
)
from axor_core.policy.sinks import INSTRUCTION_COMPLETE_SINKS
from axor_core.policy.provenance import (
    ValueRefLedger,
    call_payload,
    declared_roles,
    derive_driving_root,
    output_root,
    is_trusted_tool,
    seed_operator_trusted,
    result_payload,
    source_tokens,
)
from axor_core.kernel.registration import (
    validate_driving_arg_allowlists,
    validate_egress_allowlists,
    validate_egress_driving_args,
    tool_is_classified,
)
from axor_core.kernel.registration import validate_grant_config
from axor_core.kernel.grants import GrantMinter, GrantStore, compile_tuple_templates
from axor_core.kernel.handles import HandleTable
from axor_core.kernel.confirm import render
from axor_core.taint.engine import TaintEngine

# The closed set of denial categories, and the GATE each one names.
#
# Categories are an internal vocabulary; the gate names are the kernel's own
# nine-gate sequence, and they are what a recorded trace has to carry — a
# consumer asking "which gate denied this?" should not have to reverse-engineer
# it from a category string. Exporting the mapping is what stops every consumer
# from keeping a private copy: axor-lab kept one whose keys were `ssrf`,
# `consequence`, `positional`, `carrier` — names this kernel does not emit (it
# emits `ssrf_gate`, `consequence_gate`, …) — so seven of eleven categories fell
# through unmapped and produced traces that failed schema validation.
#
# Adding a category without adding it here fails `test_gate_vocabulary.py`.
GATE_OF_CATEGORY: dict[str, str] = {
    "taint_enforcement": "taint_floor",
    "audience_floor": "taint_floor",
    "operator_set": "value_policies",
    "content_projection": "carrier",
    "reference_origin": "value_policies",  # MODEL.md §9: a reference of untrusted origin
    "reference_unknown": "value_policies",  # a reference not in the session table
    "reference_secret": "taint_floor",  # a secret reference outside its audience
    "consequence_gate": "consequence",
    "value_policy": "value_policies",
    "ssrf_gate": "ssrf",
    "positional_gate": "positional",
    "carrier_gate": "carrier",
    "message_gate": "message",
    "budget": "budget",
    "degradation": "degradation",
    "capability": "capability",
    # an unclassified tool is refused by the capability gate: it has no declared
    # role, so no capability to act under.
    "unclassified_tool": "capability",
    # task-scoped grants: no live grant covers the call, or a raw value sat in a
    # handle-typed slot. Both are capability refusals — no authority to act.
    "grant_gate": "capability",
    "grant_handle": "capability",
}

DENIAL_CATEGORIES: frozenset[str] = frozenset(GATE_OF_CATEGORY)


def gate_of(category: str) -> str:
    """The gate a denial category names. Unknown categories raise rather than
    passing the raw category through — a category leaking into a field that
    expects a gate name is how an unrecognised denial became an invalid trace
    instead of a loud error."""
    try:
        return GATE_OF_CATEGORY[category]
    except KeyError:
        raise ValueError(
            f"unknown denial category {category!r}; known: {sorted(DENIAL_CATEGORIES)}"
        ) from None


@dataclass
class GovernanceDecision:
    """The outcome of governing one tool call.

    ``allowed`` is the only thing most callers need; ``reason`` and ``category``
    explain a denial (category is the same coarse label the IntentLoop records,
    e.g. ``taint_enforcement``, ``consequence_gate``, ``carrier_gate``).
    """

    allowed: bool
    reason: str = "approved"
    category: str = "approved"
    # The normalized intent for this call, kept so register_output can reuse it
    # without advancing the normalizer's provenance state twice.
    _normalized: NormalizedIntent | None = field(default=None, repr=False)
    # Task-scoped grants (audit): the grant that authorized this call, its source
    # (L1 request / L2 template / L3 structured / L4 confirmation), and whether a
    # confirmation was asked for it. None on a sink without grant args.
    grant_id: str | None = None
    grant_source: str | None = None
    confirmed: bool = False
    # MODEL.md §9: with references on, the arguments the tool must run with (references resolved)
    resolved_args: "dict[str, Any] | None" = None
    reference_origins: "tuple[str, ...]" = ()
    # MODEL.md §9.15: the description the covering grant was computed from (phrase = value)
    description: str | None = None


class ToolCallGovernor:
    """Runs the kernel's per-call data-flow and action-class gates synchronously.

    Hold one instance per agent session: it carries the per-value taint ledger
    and the confidentiality floor across the session's tool calls.

    Typical use::

        gov = ToolCallGovernor()
        decision = gov.evaluate(tool_name, args)
        if not decision.allowed:
            return denial(decision.reason)
        output = run_the_tool(tool_name, args)
        gov.register_output(decision, output)
    """

    def __init__(
        self,
        *,
        positional_sinks: "set[str] | frozenset[str] | None" = None,
        value_policies: dict | None = None,
        consequence_overrides: dict | None = None,
        max_unattended_consequence: ConsequenceClass = ConsequenceClass.CONSEQUENTIAL,
        floor_mode: str = "session",
        audience: "set[str] | frozenset[str] | None" = None,
        own_names: "set[str] | frozenset[str] | None" = None,
        destination_args: "dict[str, list[str]] | None" = None,
        content_args: "dict[str, list[str]] | None" = None,
        operator_sets: "dict[str, dict[str, list[str]]] | None" = None,
        operator_sets_refuse_above: "ConsequenceClass | None" = None,
        template_limits: "dict[str, dict[str, float]] | None" = None,
        consequence_rules: "dict[str, dict] | None" = None,
        content_projection: bool = False,
        operator_sets_confirm: bool = False,
        absent_driving_is_empty: bool = False,
        fallback_to_confirmation: bool = False,
        references: bool = False,
        references_seed: "int | None" = None,
        secret_fields: "set[str] | frozenset[str] | None" = None,
        sender_aliases: "dict[str, list[str]] | None" = None,
        no_sender_tools: "set[str] | frozenset[str] | None" = None,
        l1_by_clause: bool = False,
        description_sources: "dict | None" = None,
        description_targets: "dict | None" = None,
        delegated_sources: "dict | None" = None,
        delegated_max_class: "str | None" = None,
        sender_in_own: bool = False,
        sender_alias_unique: bool = False,
        untrusted_sources: "set[str] | frozenset[str] | None" = None,
        sensitive_sources: "set[str] | frozenset[str] | None" = None,
        egress_sinks: "set[str] | frozenset[str] | None" = None,
        imperative_sinks: "set[str] | frozenset[str] | None" = None,
        integrity_sinks: "set[str] | frozenset[str] | None" = None,
        benign_tools: "set[str] | frozenset[str] | None" = None,
        driving_args: "dict[str, list[str]] | None" = None,
        require_egress_allowlist: bool = False,
        require_tool_roles: bool = False,
        integrity_default: "str | None" = None,
        integrity_origins: "str | None" = None,
        trace: bool = True,
        node_id: str = "",
        grant_args: "dict[str, dict[str, str]] | None" = None,
        grant_templates: "dict[str, list[dict]] | None" = None,
        bounded_damage: "dict[str, list[str]] | None" = None,
        grant_quotas: "dict[str, int] | None" = None,
        grant_store: Any = None,
        handle_table: "HandleTable | None" = None,
        confirmer: Any = None,
        strict_handles: bool = False,
        floor_scope: str = "addressee",
        shared_stores: "dict[str, Any] | None" = None,
        access_control_sinks: "set[str] | frozenset[str] | None" = None,
        sink_origins: "dict[str, str] | None" = None,
        operator_trusted: "set[str] | frozenset[str] | None" = None,
    ) -> None:
        self._positional_sinks = frozenset(positional_sinks or ())
        illegal = {
            s for s in self._positional_sinks if s.lower() in INSTRUCTION_COMPLETE_SINKS
        }
        if illegal:
            raise ValueError(
                "instruction-complete sinks cannot be declared positional: "
                f"{sorted(illegal)} — their codomain admits instructions by "
                "definition; they must stay on the content-derivation path."
            )
        self._value_policies = dict(value_policies or {})
        self._operator_sets = {t: {a: list(vs) for a, vs in args.items()}
                               for t, args in (operator_sets or {}).items()}
        self._o_refuse_above = operator_sets_refuse_above
        self._template_limits = {t: dict(a) for t, a in (template_limits or {}).items()}
        self._consequence_rules = dict(consequence_rules or {})
        self._content_projection = bool(content_projection)
        self._o_confirm = bool(operator_sets_confirm)
        self._absent_empty = bool(absent_driving_is_empty)
        self._fallback_confirm = bool(fallback_to_confirmation)
        self._requests: list[str] = []
        # MODEL.md §9: references (built after the audience is known, below)
        self._references_on = bool(references)
        self._references_seed = references_seed
        self._consequence_overrides = dict(consequence_overrides or {})
        self._ceiling = max_unattended_consequence
        if floor_mode not in ("session", "audience"):
            raise ValueError(f"unknown floor_mode {floor_mode!r}; expected session | audience")
        self._floor_mode = floor_mode
        self._audience = frozenset(str(v).casefold() for v in (audience or ()))
        self._own_names = frozenset(str(v).lower() for v in (own_names or ()))
        self._destination_args = {k: tuple(v) for k, v in (destination_args or {}).items()}
        self._content_args = {k: tuple(v) for k, v in (content_args or {}).items()}
        # Deployment tool taxonomy. axor's normalizer recognises generic tool
        # names (read/write/bash/curl/...); a real deployment renames its tools
        # (send_money, get_webpage, read_inbox, ...). Declaring those tools' roles
        # here is how an operator tells the kernel which calls produce untrusted
        # data and which are egress sinks — the normalizer heuristics still apply
        # for any tool not declared. A declared role takes precedence.
        self._untrusted_sources = frozenset(untrusted_sources or ())
        self._sensitive_sources = frozenset(sensitive_sources or ())
        self._egress_sinks = frozenset(egress_sinks or ())
        self._imperative_sinks = frozenset(imperative_sinks or ())
        # State-changing sinks whose driving args the attacker must not choose
        # (integrity check only — no confidentiality floor, no allowlist obligation).
        self._integrity_sinks = frozenset(integrity_sinks or ())
        # What the confidentiality floor holds while a secret read is outstanding:
        # the egress sinks (addressee), + declared shared stores, or every state change.
        from axor_core.policy.gates import FLOOR_SCOPES, validate_shared_stores
        if floor_scope not in FLOOR_SCOPES:
            raise ValueError(f"unknown floor_scope {floor_scope!r}; expected one of {sorted(FLOOR_SCOPES)}")
        errs = validate_shared_stores(shared_stores)
        if errs:
            raise ValueError("; ".join(errs))
        self._floor_scope = floor_scope
        # Per-sink integrity-origin mode (O4'): request-only | any-trusted | none
        # (none: no origin check on the sink's driving args). Default: the global mode.
        from axor_core.policy.gates import validate_sink_origins
        errs = validate_sink_origins(sink_origins)
        if errs:
            raise ValueError("; ".join(errs))
        self._sink_origins = dict(sink_origins or {})
        self._shared_stores = {t: (r if r == "always" else {a: frozenset(str(x) for x in ids)
                                                           for a, ids in r.items()})
                               for t, r in (shared_stores or {}).items()}
        # Explicitly-benign reads (trusted output that need not be tainted). Kept so
        # the per-call STRICT role check can tell "declared benign" from "forgot to
        # classify" — without it, both would fail open to a clean read.
        self._benign_tools = frozenset(benign_tools or ())
        # STRICT role obligation, enforced LAZILY per call: the governor never sees
        # the full tool universe (it is asked one (tool, args) at a time), so it
        # cannot validate completeness at construction the way GovernedSession does.
        # Instead it denies any unclassified tool the moment it is used — closing the
        # fail-open default where a renamed, undeclared tool normalises to a benign
        # no-op and slips past every gate.
        self._require_tool_roles = require_tool_roles
        # STRICT (either obligation): a tool with no explicit consequence class is
        # CATASTROPHIC — the governor has no human gate, so it is refused.
        self._strict = require_egress_allowlist or require_tool_roles
        # Per-sink driving arguments — the fields the taint decision keys on. Empty
        # = whole-args (safe, coarse). Declaring them narrows to the destination /
        # instruction field so untrusted content to a trusted destination is not
        # over-blocked.
        self._driving_args = {k: frozenset(v) for k, v in (driving_args or {}).items()}
        self._xrefs = None
        if self._references_on:
            from axor_core.kernel.references import ReferenceTable
            seed = self._references_seed
            if seed is None:
                import secrets
                seed = secrets.randbits(64)
            self._xrefs = ReferenceTable(own=frozenset(self._audience) | frozenset(self._own_names),
                                         seed=seed, requests=self._requests,
                                         secret_fields=frozenset(secret_fields or ()),
                                         sender_aliases={str(k).casefold(): list(v) for k, v in
                                                         (sender_aliases or {}).items()},
                                         sender_own=bool(sender_in_own),
                                         alias_unique=bool(sender_alias_unique))
            self._no_sender_tools = frozenset(no_sender_tools or ())
        # STRICT obligation: every egress sink must carry a destination allowlist
        # (an enum value_policy) — the sound, paraphrase-proof control. Fail closed
        # at construction rather than ship an egress sink on content-derivation alone.
        # None = the STRICT default: the governor has no mode knob, so STRICT is
        # its fail-closed obligations (either one) — "context" then, else "clean".
        resolved_integrity = resolve_integrity_default(
            integrity_default, strict=require_egress_allowlist or require_tool_roles,
        )
        resolved_origins = resolve_integrity_origins(
            integrity_origins, strict=require_egress_allowlist or require_tool_roles,
        )
        if require_egress_allowlist:
            errors = validate_egress_allowlists(self._egress_sinks, self._value_policies)
            errors += validate_driving_arg_allowlists(
                self._egress_sinks, self._driving_args, self._value_policies
            )
            if resolved_integrity == "context":
                errors += validate_egress_driving_args(
                    self._egress_sinks, self._driving_args, self._integrity_sinks
                )
            if errors:
                raise ValueError("strict egress allowlist: " + "; ".join(errors))
        # Task-scoped grants. A sink with grant args runs only under a live grant
        # covering its driving tuple; grants are minted from trusted inputs only
        # (GrantMinter), never from anything register_output sees (O4).
        self._grant_args = {k: dict(v) for k, v in (grant_args or {}).items()}
        self._access_control = frozenset(access_control_sinks or ())
        # A CATASTROPHIC sink, when the ceiling is below it, takes L3/L4 grounds only —
        # no L1/L2 grant is minted for it, as for an access-control sink.
        self._grant_l34_only = self._access_control | frozenset(
            t for t in (grant_args or {})
            if consequence_class(t, overrides=self._consequence_overrides)
            is ConsequenceClass.CATASTROPHIC
            and max_unattended_consequence < ConsequenceClass.CATASTROPHIC)
        grant_errors = validate_grant_config(self._grant_args, grant_templates, bounded_damage,
                                             grant_quotas, self._access_control)
        if grant_errors:
            raise ValueError("grant config: " + "; ".join(grant_errors))
        self._handles = handle_table if handle_table is not None else HandleTable()
        self._grants = grant_store if grant_store is not None else GrantStore()
        # The kernel store or the locked spend-side wrapper; both expose find/reserve.
        inner = getattr(self._grants, "inner", self._grants)
        self._minter = GrantMinter(inner, self._handles, self._grant_args,
                                   quotas=dict(grant_quotas or {}),
                                   access_control=self._grant_l34_only,
                                   by_clause=bool(l1_by_clause))
        if grant_templates:
            self._minter.from_templates(grant_templates)
        # MODEL.md §9.15: description grants
        self._desc = None
        self._desc_targets = dict(description_targets or {})
        self._described: dict[tuple[str, str], set] = {}
        self._desc_done: set = set()
        self._desc_notes: dict[str, str] = {}
        # §9.16 (hypothesis H-A, not in c9v4): tool -> the argument naming the source
        self._delegated = dict(delegated_sources or {})
        self._delegated_max = (ConsequenceClass[str(delegated_max_class).upper()]
                               if delegated_max_class else None)
        self._delegated_done: set = set()
        if description_sources:
            from axor_core.kernel.descriptions import DescriptionTable
            own = {str(v).casefold() for v in (set(self._audience) | set(self._own_names))}
            self._desc = DescriptionTable(dict(description_sources), is_request=self._is_request_value,
                                          is_own=lambda v: str(v).casefold() in own)
        # L2 tuple templates over the driving args (all-O ones are issued now).
        self._minter.tuple_templates = compile_tuple_templates(
            self._value_policies, self._grant_args, self._driving_args, self._grant_l34_only,
            operator_sets=self._operator_sets, template_limits=self._template_limits)
        self._minter.from_tuple_templates_start()
        self._confirmer = confirmer
        self._strict_handles = strict_handles
        self.confirmations = 0
        self._normalizer = IntentNormalizer()
        self._taint = TaintEngine(node_id=node_id, integrity_default=resolved_integrity,
                                  integrity_origins=resolved_origins)
        # Context-default integrity: operator allowlist members are values the
        # attacker cannot author. Seeding them keeps the trusted-origin proof
        # consistent with the enum supersession (T4) those members already carry.
        seed_operator_trusted(self._taint, self._value_policies)
        # Operator-declared sets are trusted per slot (MODEL.md §8.2): a value clears only in
        # the (tool, argument) whose set holds it — see _driving_outside_own — never globally.
        # Operator-vouched values outside any allowlist: origin only, no refusal.
        for v in sorted(operator_trusted or ()):
            self._taint.register_trusted([v], TrustedOrigin.OPERATOR)
        # The value-ref vocabulary the kernel event schema is written in. The
        # gates decide on content derivation and need no ids; the RECORD does —
        # `arg_refs` and `value_ref` are what let a consumer bind a denied sink
        # argument back to the read that tainted it.
        self._refs = ValueRefLedger()
        self._node_id = node_id
        # Trace recording (default on). Each verdict builds a full provenance
        # payload for replay/audit; that payload — not the gate decision — is the
        # bulk of per-call latency. trace=False skips it for a latency-sensitive
        # standalone deployment (the 'thin membrane'): decisions are identical, but
        # no TraceEvents are emitted, so replay/audit sees no record. The control
        # plane / replay path keeps the default True.
        self._trace = trace
        # The SAME TraceEvents the IntentLoop emits. Both are ways of wrapping an
        # agent, so both must feed the one instrumentation path: axor-wrap's
        # trace bridge turns these into kernel-schema Events, and the plane and
        # Lab read that. A gating path that decides without recording forces its
        # consumers to invent a second trace — which is exactly what happened.
        self._trace_events: list[TraceEvent] = []

    # ── decision ────────────────────────────────────────────────────────────────

    def _call_payload(
        self,
        tool_name: str,
        args: dict[str, Any],
        normalized: NormalizedIntent | None = None,
    ) -> dict[str, Any]:
        """The provenance this call was judged on — the SHARED builder.

        Both ways of wrapping an agent record this identical shape, from one
        function, so the two cannot drift into two vocabularies.
        """
        return call_payload(
            tool_name, args, taint=self._taint, driving_args=self._driving_args,
            normalized=normalized, refs=self._refs,
            roles=declared_roles(
                tool_name,
                untrusted_sources=self._untrusted_sources,
                sensitive_sources=self._sensitive_sources,
                egress_sinks=self._egress_sinks,
                imperative_sinks=self._imperative_sinks,
                positional_sinks=self._positional_sinks,
                integrity_sinks=self._integrity_sinks,
            ),
        )

    def evaluate(self, tool_name: str, args: dict[str, Any]) -> GovernanceDecision:
        """Decide whether this tool call may execute. Pure — no side effects on
        the ledger (call :meth:`register_output` after a tool actually runs).

        With references on (MODEL.md §9) the call's references are resolved first; the
        decision carries ``resolved_args`` — the values the tool must run with."""
        if self._xrefs is None:
            return self._evaluate(tool_name, args, None)
        resolved, ctx = self._reference_context(tool_name, args)
        d = self._evaluate(tool_name, resolved, ctx)
        d.resolved_args = resolved
        d.reference_origins = tuple(ctx.get("origins", ()))
        return d

    def _reference_context(self, tool_name: str, args: dict[str, Any]) -> "tuple[dict, dict]":
        """§9.4–9.8 for one call: resolved args and the reference verdicts the gate folds in."""
        from axor_core.kernel.references import EMAIL, IBAN, ID_KEYS, SECRET, SENDER, UNTRUSTED, URL
        t = self._xrefs
        resolved: dict[str, Any] = {}
        used: dict[str, list] = {}
        unknown: list = []
        for k, v in (args or {}).items():
            resolved[k], used[k], unk = t.resolve(v, k)
            unknown += unk
        ctx: dict[str, Any] = {"deny": None, "unproven": None, "origin_decided": False}
        if unknown:
            ctx["deny"] = GateDecision(reason=f"reference: {unknown[0]!r} is not in this session's table",
                                       category="reference_unknown")
            return resolved, ctx
        all_refs = [r for rs in used.values() for r in rs]
        ctx["origins"] = sorted({r.origin for r in all_refs})
        if any(r.origin == SECRET for r in all_refs):  # §9.7: the audience must see the source
            dests = [x for a in self._destination_args.get(tool_name, ()) for x in _leaves(resolved.get(a))]
            if any(not t.is_own(x) for x in dests if x not in (None, "")):
                ctx["deny"] = GateDecision(reason="reference: a secret is resolved only into a sink whose "
                                                  "audience sees its source", category="reference_secret")
                return resolved, ctx

        def ident(a: str, x: Any) -> bool:
            if a in ID_KEYS:
                return x not in (None, "")
            return isinstance(x, str) and any(p.fullmatch(x) for p in (EMAIL, IBAN, URL))
        why = None
        decided = True
        checked = 0  # driving args actually passed and decided by origin
        declared = self._driving_args.get(tool_name) or frozenset()
        for a in declared:
            if a not in args or args.get(a) in (None, "", [], {}):
                continue
            checked += 1
            # §9.15: the value the request described is proven for this sink and argument only
            described = self._described.get((tool_name, a), ())
            refs = [r for r in used.get(a) or []
                    if not (r.origin == UNTRUSTED and str(r.value).strip().casefold() in described)]
            if any(r.origin in (UNTRUSTED, SECRET) for r in refs):
                why = why or f"{a} carries a reference of {refs[0].origin} origin"
            raw = [x for x in _leaves(args.get(a)) if not (x in t.by_name)]
            for x in raw:
                rx = x
                if not ident(a, rx):
                    decided = False  # not an identifier kind: §8.14's checks decide
                    continue
                if t.shown_raw(rx) or str(rx).strip().casefold() in described:
                    continue  # R / own / described
                why = why or (f"{a}={rx!r} rebuilds a value this session hid" if str(rx).casefold()
                              in t.hidden_values() else f"{a}={rx!r} is neither from the request nor own")
        for a in self._content_args.get(tool_name, ()):  # §9.6
            if any(r.origin in (UNTRUSTED, SECRET) for r in used.get(a) or []):
                why = why or f"the content of {a} carries a reference of untrusted origin"
        if why:
            ctx["unproven"] = GateDecision(reason="reference: " + why + "; needs a confirmation of the exact call",
                                           category="reference_origin")
        # §9.9 decides integrity only for driving args that were passed; with none passed, nothing
        # was decided by origin and §8.10's fallback (every argument checked, L4) still applies
        ctx["origin_decided"] = decided and checked > 0
        return resolved, ctx

    def present_output(self, decision: "GovernanceDecision", output: Any) -> Any:
        """Register an executed tool's output and return what the agent may see: with references
        on (§9), target identifiers of untrusted origin replaced by references; otherwise the
        output unchanged."""
        self.register_output(decision, output)
        if self._xrefs is None or not isinstance(output, str):
            return output
        ni = decision._normalized
        tool = ni.tool if ni is not None else ""
        before = {n for n in self._xrefs.by_name}
        view = self._xrefs.mint(output, sensitive=tool in self._sensitive_sources,
                                sender_records=tool not in self._no_sender_tools)
        # §9.5: a named sender's value is a ground as an L1 grant on that value
        from axor_core.kernel.references import SENDER
        for n, r in list(self._xrefs.by_name.items()):
            if n not in before and r.origin == SENDER:
                # as a request literal: named (operator-set rule), trusted origin, L1 grants
                self._taint.register_trusted(str(r.value), TrustedOrigin.TASK)
                if self._grant_args:
                    self.grants.from_request(str(r.value))
        return view

    # ── description grants (MODEL.md §9.15) ──────────────────────────────────
    def _is_request_value(self, value: Any) -> bool:
        from axor_core.kernel.references import _canon
        v = _canon(str(value)).strip()
        if len(v) < 2:
            return False
        import re as _re
        pat = _re.compile(r"(?<![\w@.])" + _re.escape(v) + r"(?![\w@])")
        return any(pat.search(_canon(r)) for r in self._requests)

    def observe_structured(self, tool_name: str, args: "dict[str, Any] | None", value: Any) -> list:
        """§9.15: the platform's structured output of an executed read (a list / records / a
        name -> value map — never text). Recomputes the request's descriptions and mints a grant on
        each newly computed value; returns the new [(description phrase, sink, arg, values)]."""
        if self._desc is None or not self._desc.observe(tool_name, args or {}, value):
            return []
        return self._describe()

    def observe_delegated(self, tool_name: str, args: "dict[str, Any] | None", output: Any) -> list:
        """§9.16 (hypothesis H-A): a read of a source S the request names (a URL / a path literal).
        Values of kind k in S's output become grant candidates for every sink whose leading
        (destination) argument has kind k; one grant (one use) per pair (S, sink)."""
        arg = self._delegated.get(tool_name)
        if not arg or not isinstance(output, str):
            return []
        src = (args or {}).get(arg)
        if not isinstance(src, str) or not src or not self._is_request_value(src):
            return []
        from axor_core.kernel.handles import split_kind
        from axor_core.kernel.request_literals import literal_values
        out = []
        for sink, kinds in self._grant_args.items():
            if self._delegated_max is not None and consequence_class(
                    sink, overrides=self._consequence_overrides) > self._delegated_max:
                continue
            for a in self._destination_args.get(sink, ()):
                k = split_kind(kinds.get(a, ""))[0] if a in kinds else ""
                if k not in ("email", "iban", "url"):
                    continue
                vals = list(dict.fromkeys(literal_values(output, k)))
                key = (tool_name, src.casefold(), sink, a)
                if not vals or key in self._delegated_done:
                    continue
                self._delegated_done.add(key)
                from axor_core.kernel.references import _canon
                req = next((r for r in self._requests if _canon(src) in _canon(r)), self._requests[0])
                note = f"delegated: {tool_name}({src}) -> {sink}.{a}"
                gs = self._minter.from_description(req, sink, a, vals, members=True, note=note, uses=1)
                if not gs:
                    continue
                for g in gs:
                    self._desc_notes[g.id] = note
                self._described.setdefault((sink, a), set()).update(str(v).strip().casefold() for v in vals)
                out.append((note, vals))
        return out

    def _describe(self) -> list:
        from axor_core.kernel.descriptions import covered, parse
        out = []
        member_ents = [e for e, t in self._desc_targets.items() if (t or {}).get("members")]
        for req in self._requests:
            for d in parse(req):
                if covered(req, d) is not None:
                    continue  # §9.15a (a): a constraint the grammar does not parse
                got = self._desc.compute(d, member_ents)
                if got is None:
                    continue
                vals, members = got
                ents = [d.entity] if d.entity else member_ents
                for ent in ents:
                    tg = (self._desc_targets.get(ent) or {}).get("members" if members else "value") or ()
                    for t in tg:
                        sink, _, arg = str(t).partition(".")
                        key = (sink, arg, tuple(str(v) for v in vals), members)
                        if key in self._desc_done or sink not in self._grant_args:
                            continue
                        # §9.15a (b): never on a CATASTROPHIC sink (nor BENIGN, which needs no grant)
                        if consequence_class(sink, overrides=self._consequence_overrides) not in (
                                ConsequenceClass.REVERSIBLE, ConsequenceClass.CONSEQUENTIAL):
                            continue
                        self._desc_done.add(key)
                        note = f"description: {d.phrase!r} = {list(vals)}"
                        gs = self._minter.from_description(req, sink, arg, vals, members=members, note=note)
                        if not gs:
                            continue
                        for g in gs:
                            self._desc_notes[g.id] = note
                        self._described.setdefault((sink, arg), set()).update(
                            str(v).strip().casefold() for v in vals)
                        out.append((d.phrase, sink, arg, list(vals)))
        return out

    def _evaluate(self, tool_name: str, args: dict[str, Any], ctx: "dict | None") -> GovernanceDecision:
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": tool_name, "args": args},
            node_id="",
            sequence=0,
        )
        normalized = self._normalizer.normalize(intent)

        def _denied(reason: str, category: str) -> GovernanceDecision:
            """EVERY denial goes through here. A denial that returns directly is
            a verdict with no trace: replay sees an intent with no decision, and
            a consumer never learns the kernel blocked anything. Skipped when
            trace is off — the denial verdict is unchanged, only its record."""
            if self._trace:
                self._trace_events.append(
                    IntentDeniedEvent(
                        kind=TraceEventKind.INTENT_DENIED,
                        node_id=self._node_id,
                        sequence=len(self._trace_events),
                        intent_kind=IntentKind.TOOL_CALL.value,
                        reason=reason,
                        payload={**self._call_payload(tool_name, args, normalized),
                                 "category": category},
                    )
                )
            return GovernanceDecision(
                allowed=False, reason=reason, category=category,
                _normalized=normalized,
            )

        def _deny(gd: GateDecision) -> GovernanceDecision:
            return _denied(gd.reason, gd.category)

        # 0. STRICT role completeness (lazy): an unclassified tool fails closed
        #    rather than defaulting to a clean benign read. This is the fix for the
        #    fail-open-on-unknown-tool default — without an explicit role the kernel
        #    cannot know a renamed tool exfiltrates or reads a secret.
        if self._require_tool_roles and not tool_is_classified(
            tool_name,
            untrusted_sources=self._untrusted_sources,
            sensitive_sources=self._sensitive_sources,
            egress_sinks=self._egress_sinks,
            positional_sinks=self._positional_sinks,
            benign_tools=self._benign_tools,
            value_policies=self._value_policies,
            integrity_sinks=self._integrity_sinks,
        ):
            return _denied(
                f"tool {tool_name!r} has no declared data-flow role; STRICT mode "
                "refuses an unclassified tool (it would default to a clean read "
                "and arm no floor). Declare it as a source/sink/positional/"
                "value_policy or explicitly benign.",
                "unclassified_tool",
            )

        # 1. consequence — content-blind action-class gate. Above the unattended
        #    ceiling the only ground the governor has is a task-scoped grant: a grant
        #    sink is decided at step 7 (and must be covered there); any other sink is
        #    refused here.
        call_class = self._call_class(tool_name, args, normalized.operation)
        gd = None
        if tool_name not in self._consequence_rules:
            gd = consequence_gate(
                tool_name, normalized.operation, self._ceiling, self._consequence_overrides,
                strict=self._strict,
            )
        elif call_class > self._ceiling:
            gd = GateDecision(
                reason=(f"consequence gate: this call to '{tool_name}' is {call_class.name}, "
                        f"exceeding the unattended ceiling {self._ceiling.name}; a ground "
                        "(a grant, or L3/L4 for CATASTROPHIC) is required"),
                category="consequence_gate")
        l34_needed = tool_name in self._access_control or (
            call_class is ConsequenceClass.CATASTROPHIC and self._ceiling < ConsequenceClass.CATASTROPHIC)
        needs_ground = gd is not None and bool(self._grant_args.get(tool_name))
        if gd is not None and not needs_ground:
            return _deny(gd)
        consequence_denial = gd

        # 2. value policies — operator-registered decidable predicates.
        gd = value_policy_gate(tool_name, args, self._value_policies)
        if gd is not None:
            return _deny(gd)
        # 2b. organisation sets refuse on irreversible actions (revision 13): before
        #     grants and confirmation, so a confirmation cannot override it.
        # §8.5: an unproven destination/object needs a confirmation of the exact call; it is
        # refused here only when there is no human to ask (or no grant args to bind it to)
        if ctx is not None and ctx["deny"] is not None:
            return _deny(ctx["deny"])
        unproven = self._operator_set_refusal(tool_name, args)
        if ctx is not None and ctx["unproven"] is not None:
            unproven = ctx["unproven"]  # §9.5/9.6: a ground only by L4 (the origin is the reason)
        if unproven is not None and (not self._o_confirm or self._confirmer is None
                                     or not self._grant_args.get(tool_name)):
            return _deny(unproven)
        gd = self._content_refusal(tool_name, args)
        if gd is not None:
            return _deny(gd)

        # 3. SSRF / internal-destination gate.
        gd = ssrf_gate(tool_name, normalized)
        if gd is not None:
            return _deny(gd)

        _mode = self._sink_origins.get(tool_name)
        # §8.2: a driving arg whose value is in its slot's own set is proven; when every
        # passed driving arg is, integrity is carried by the sets (as for an enum allowlist)
        _outside = self._driving_outside_own(tool_name, args)
        _declared = self._driving_args.get(tool_name)
        own_proven = bool(_declared) and any(a in args for a in _declared) and _outside == frozenset(
            a for a in _declared if a not in args)
        # §8.7: an explicitly empty driving list — nothing for provenance to check
        no_driving = tool_name in self._driving_args and not self._driving_args[tool_name]
        if self._absent_empty and _declared and not any(
                args.get(a) not in (None, "", [], {}) for a in _declared):
            no_driving = True  # §8.8: none passed (or all empty) — nothing to check
        driving_root = derive_driving_root(
            self._taint, args, (_outside if _outside and not own_proven else _declared),
            integrity_sink=(tool_name in self._integrity_sinks
                            or tool_name in self._egress_sinks),
            origins=None if _mode in (None, "none") else _mode,
        )

        # 4. positional admission — for declared instruction-incomplete sinks.
        gd = positional_gate(tool_name, args, self._positional_sinks)
        if gd is not None:
            return _deny(gd)

        # 5. carrier / imperative-channel gate.
        gd = carrier_gate(tool_name, args, normalized, driving_root, self._imperative_sinks)
        if gd is not None:
            return _deny(gd)

        # 6. per-value taint — integrity + confidentiality floor. A sink whose
        # driving args are fully guarded by satisfied decidable predicates carries
        # its integrity axis there (a stronger, content-blind control), so the
        # content-taint integrity check is superseded; the floor still applies.
        _taint_kw = dict(
            integrity_sinks=self._integrity_sinks,
            floor_extra=floor_scope_hit(tool_name, args, self._floor_scope,
                                        self._integrity_sinks, self._shared_stores),
        )
        _floor_on = self._taint.confidentiality_floor_active() and self._floor_mode == "session"
        gd = taint_gate(
            tool_name, normalized, driving_root, _floor_on, self._egress_sinks,
            integrity_superseded=integrity_superseded_by_decidable(
                tool_name, args, self._driving_args, self._value_policies
            ) or self._integrity_superseded_by_grant(tool_name, args) or own_proven or no_driving
            or self._sink_origins.get(tool_name) == "none"  # O4' low: no origin check
            or (ctx is not None and ctx["origin_decided"]),  # §9.9: origin decides integrity
            **_taint_kw,
        )
        if gd is not None and self._fallback_confirm and self._confirmer is not None \
                and self._grant_args.get(tool_name) and _declared \
                and not any(args.get(a) not in (None, "", [], {}) for a in _declared) \
                and taint_gate(tool_name, normalized, driving_root, _floor_on, self._egress_sinks,
                               integrity_superseded=True, **_taint_kw) is None:
            # §8.10: the fallback's integrity failure (no driving argument passed) asks for
            # a confirmation of the exact call, as an unproven destination/object does
            if unproven is None:
                unproven = gd
            gd = None
        if gd is not None:
            return _deny(gd)

        # 7. task-scoped grant — last, so a confirmation is only ever asked for a
        #    call every other gate would admit.
        gd, grant = grant_gate(
            tool_name, args, self._grant_args, self._grants, self._handles,
            strict_handles=self._strict_handles,
        )
        if unproven is not None and gd is None and grant is None:
            # a call no grant gate decides (empty leading tuple): ask anyway
            gd = GateDecision(reason="grant gate: an unproven destination/object needs a "
                              "confirmation", category="grant_gate")
        if (needs_ground and l34_needed or unproven is not None) and grant is not None \
                and grant.source.value not in ("L3", "L4"):
            # a CATASTROPHIC call: a request or template grant is not a ground — ask instead
            gd = GateDecision(reason="grant gate: this call needs an L3/L4 ground", category="grant_gate")
            grant = None
        confirmed = False
        if grant_gate_confirmable(gd) and self._confirmer is not None:
            action = render(tool_name, args, self._grant_args[tool_name], self._handles)
            self.confirmations += 1
            if self._confirmer.confirm(action, (tool_name, dict(args))):
                grant = self._minter.from_confirmation(tool_name, args)
                confirmed = grant is not None
                gd = None if grant is not None else gd
        if gd is not None:
            return _deny(unproven if unproven is not None else gd)
        if unproven is not None and not (grant is not None and grant.source.value in ("L3", "L4")):
            return _deny(unproven)
        if needs_ground:
            # over the ceiling: admitted only through a grant, and a CATASTROPHIC one
            # only through L3/L4 (no L1/L2 grant is minted for it).
            l34 = grant is not None and grant.source.value in ("L3", "L4")
            if grant is None or (l34_needed and not l34):
                return _deny(consequence_denial)
        gd = self._audience_floor(tool_name, args, grant)
        if gd is not None:
            return _deny(gd)

        if self._trace:
            self._trace_events.append(
                TraceEvent(
                    kind=TraceEventKind.INTENT_APPROVED,
                    node_id=self._node_id,
                    sequence=len(self._trace_events),
                    payload={
                        **self._call_payload(tool_name, args, normalized),
                        # audit: which grant authorized this call, from where
                        **({"grant": {"id": grant.id, "source": grant.source.value,
                                      "confirmed": confirmed}}
                           if grant is not None else {}),
                    },
                )
            )
        return GovernanceDecision(
            allowed=True, _normalized=normalized,
            grant_id=grant.id if grant is not None else None,
            grant_source=grant.source.value if grant is not None else None,
            confirmed=confirmed,
            description=self._desc_notes.get(grant.id) if grant is not None else None,
        )

    def _own_slot(self, tool_name: str, arg: str, value: object) -> bool:
        members = (self._operator_sets.get(tool_name) or {}).get(arg)
        if not members:
            return False
        vals = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
        mem = {str(m).casefold() for m in members}
        return bool(vals) and all(str(v).casefold() in mem for v in vals)

    def _driving_outside_own(self, tool_name: str, args: dict):
        """The driving args minus those whose value is in that slot's own set (§8.2)."""
        driving = self._driving_args.get(tool_name)
        if driving is None:
            return None
        return frozenset(a for a in driving
                         if not (a in args and self._own_slot(tool_name, a, args.get(a))))

    def _own_targeted(self, tool_name: str, args: dict) -> bool:
        """Every passed destination/object value is in its slot's own set (vacuous: own)."""
        for a in (self._operator_sets.get(tool_name) or {}):
            v = args.get(a)
            if v is None or v == "" or v == []:
                continue
            if not self._own_slot(tool_name, a, v):
                return False
        return True

    def _call_class(self, tool_name: str, args: dict, operation) -> ConsequenceClass:
        """κ(x) = effect × target × magnitude (MODEL.md §8.1); per tool when no rule."""
        cls = consequence_class(tool_name, operation=operation,
                                overrides=self._consequence_overrides, strict=self._strict)
        rule = self._consequence_rules.get(tool_name)
        if not rule:
            return cls
        if "own" in rule and self._own_targeted(tool_name, args):
            cls = rule["own"]
        m = rule.get("magnitude")
        if m:
            try:
                if float(args.get(m["arg"])) > m["above"]:
                    cls = max(cls, m["class"])
            except (TypeError, ValueError):
                pass
        return cls

    def _content_refusal(self, tool_name: str, args: dict) -> "GateDecision | None":
        """Content projection (MODEL.md §8.4): every URL / e-mail / IBAN in a content argument
        must be own (the audience) or named by the request; before grants and confirmation."""
        if not self._content_projection:
            return None
        content = self._content_args.get(tool_name)
        if not content:
            return None
        from axor_core.policy.content_tokens import extract
        tokens = set()
        for a in content:
            tokens |= extract(args.get(a), self._own_names)
        if not tokens:
            return None
        own = extract(sorted(self._audience), self._own_names)
        named = extract(self._requests, self._own_names)
        bad = sorted(t for t in tokens if t not in own and t not in named)
        if not bad:
            return None
        return GateDecision(
            reason=("content projection: " + ", ".join(f"{k} {v!r}" for k, v in bad[:3])
                    + " in the content is neither own nor named by the request"),
            category="content_projection")

    def _named_by_request(self, tool_name: str, arg: str, value: object) -> bool:
        if self._taint.trusted_origin(value) is TrustedOrigin.TASK:
            return True
        if str(value).strip().casefold() in self._described.get((tool_name, arg), ()):
            return True  # §9.15: the request described it (this sink and argument only)
        spec = (self._grant_args.get(tool_name) or {}).get(arg)
        if not spec or not self._requests:
            return False
        from axor_core.kernel.handles import canonicalize, split_kind
        from axor_core.kernel.request_literals import literal_values
        kind = split_kind(spec)[0]
        if kind == "string":
            return False
        try:
            want = canonicalize(kind, value)
        except ValueError:
            return False
        for req in self._requests:
            for lit in literal_values(req, kind):
                try:
                    if canonicalize(kind, lit) == want:
                        return True
                except ValueError:
                    continue
        return False

    def _operator_set_refusal(self, tool_name: str, args: dict) -> "GateDecision | None":
        """Revision 13: on a sink above ``operator_sets_refuse_above``, a value of an
        O-set argument must be in the O-set or named by the request."""
        if self._o_refuse_above is None or tool_name not in self._operator_sets:
            return None
        if consequence_class(tool_name, overrides=self._consequence_overrides) <= self._o_refuse_above:
            return None
        for arg, members in self._operator_sets[tool_name].items():
            allowed = {str(m).casefold() for m in members}
            v = args.get(arg)
            for x in (v if isinstance(v, (list, tuple, set, frozenset)) else [v]):
                if x is None or x == "" or str(x).casefold() in allowed:
                    continue
                if self._named_by_request(tool_name, arg, x):
                    continue
                return GateDecision(
                    reason=(f"organisation set: {tool_name}.{arg}={x!r} is outside the "
                            "organisation and not named in the request, on an irreversible "
                            "action"),
                    category="operator_set",
                )
        return None

    def _audience_floor(self, tool_name: str, args: dict, grant) -> "GateDecision | None":
        """Addressed confidentiality floor (``floor_mode="audience"``): while a secret
        read is outstanding, every destination of an egress sink must be in the
        audience or have a ground for the disclosure — fixed by the request (origin
        TASK) or an L3/L4 grant on the call. A write to a declared shared store is a
        destination outside the audience. An internal write is not held."""
        if self._floor_mode != "audience" or not self._taint.confidentiality_floor_active():
            return None
        # a ground: the request fixed it (an L1 grant, matched by kind), or L3/L4
        ground = grant is not None and grant.source.value in ("L1", "L3", "L4")
        outside: list[str] = []
        if tool_name in self._egress_sinks:
            for a in self._destination_args.get(tool_name, ()):
                v = args.get(a)
                for x in (v if isinstance(v, (list, tuple, set, frozenset)) else [v]):
                    if x is None or x == "":
                        continue
                    if str(x).casefold() in self._audience:
                        continue
                    if self._taint.trusted_origin(x) is TrustedOrigin.TASK:
                        continue
                    outside.append(f"{a}={x!r}")
        if floor_scope_hit(tool_name, args, "shared_stores", self._integrity_sinks,
                           self._shared_stores):
            # only a write that can carry the secret discloses it; a tool whose
            # content args are not declared is held (unknown = held)
            content = self._content_args.get(tool_name)
            if content is None or any(args.get(a) not in (None, "", [], {}) for a in content):
                outside.append("a shared store")
        if not outside or ground:
            return None
        return GateDecision(
            reason=("confidentiality (addressed floor): a secret read is outstanding and "
                    f"{', '.join(outside)} is outside its audience with no ground "
                    "(not fixed by the request, no L1/L3/L4 grant)"),
            category="audience_floor",
        )

    def _integrity_superseded_by_grant(self, tool_name: str, args: dict | None = None) -> bool:
        """A grant sink whose every driving arg is a grant arg carries its integrity
        axis on the grant (step 7 denies an uncovered call), so content-taint
        integrity is superseded there, as for a decidable value policy — unless the
        call has an empty leading tuple (no grant arg passed): then no grant decides
        and the integrity check stays."""
        kinds = self._grant_args.get(tool_name)
        if not kinds:
            return False
        if args is not None:
            from axor_core.policy.gates import empty_grant_tuple
            if empty_grant_tuple(tool_name, args, self._grant_args):
                return False
        driving = self._driving_args.get(tool_name)
        return driving is not None and set(driving) <= set(kinds)

    # ── grants ────────────────────────────────────────────────────────────────

    @property
    def grants(self) -> GrantMinter:
        """Mint grants from trusted inputs: ``from_request`` (L1), ``from_templates``
        (L2), ``from_structured`` (L3). L4 is the confirmer passed at construction."""
        return self._minter

    @property
    def handles(self) -> HandleTable:
        return self._handles

    @property
    def grant_store(self) -> Any:
        return self._grants

    def spend(self, decision: GovernanceDecision, args: dict[str, Any],
              call_key: str | None = None) -> str | None:
        """Reserve one use of the grant that authorized ``decision`` and return the
        spend's idempotency key (``<grant id>#<n>``); None when the call needed no
        grant. The executor commits after the effect ran (or releases)."""
        if decision.grant_id is None or decision._normalized is None:
            return None
        r = self._grants.reserve(decision._normalized.tool, args, self._handles,
                                 call_key=call_key, grant_id=decision.grant_id)
        return r.key if r is not None else None

    @property
    def trace_events(self) -> "list[TraceEvent]":
        """Every verdict this governor reached, in call order.

        The kernel reports what it DECIDED; whether the caller then blocked the
        call is the caller's business (a Lab ungoverned arm observes without
        enforcing). So a denial is recorded here even when the wrapper chose to
        execute anyway — otherwise an ungoverned run would have no verdicts to
        replay, and the ungoverned/governed comparison would have nothing to
        compare.
        """
        return list(self._trace_events)

    def drain_trace_events(self) -> "list[TraceEvent]":
        """Take the events and reset, for a caller streaming per trial."""
        events, self._trace_events = self._trace_events, []
        return events

    # ── ledger ────────────────────────────────────────────────────────────────

    def register_output(
        self, decision: GovernanceDecision, output: Any
    ) -> None:
        """Record an executed tool's output in the per-value ledger.

        Mirrors the IntentLoop's registration: external/web reads taint the
        produced value untrusted; secret/system reads also mark it sensitive,
        which arms the confidentiality floor. A clean read registers nothing.
        Call this only for calls that actually executed.

        An arming read is also RECORDED, as a TAINT_PROPAGATED event carrying the
        minted value ref and its root. Without it the trace has verdicts and no
        source: replay has nothing to put in ``tainted_refs``, so a later
        ``arg_refs`` binding resolves to nothing, and the Control Plane refuses
        the run outright ("no untrusted source in the recorded run"). A clean
        read still records nothing — it introduces no provenance to report.
        """
        ni = decision._normalized
        tool_name = ni.tool if ni is not None else ""
        # Shared arming map — identical to the IntentLoop's, so the two paths cannot
        # drift on what a tool's output taints.
        root = output_root(
            tool_name, ni,
            untrusted_sources=self._untrusted_sources,
            sensitive_sources=self._sensitive_sources,
        )
        if root is None:
            # Clean read — no provenance to register. In context mode a trusted
            # tool's output seeds the trusted-origin index (a value copied from it
            # is not tainted by the context root).
            # A decision with no normalized intent names no tool: never trusted.
            if tool_name and is_trusted_tool(
                tool_name, self._benign_tools, require_tool_roles=self._require_tool_roles
            ):
                self._taint.register_trusted(output, TrustedOrigin.TOOL)
            return
        self._taint.register_value(output, root)
        if self._trace:
            self._trace_events.append(
                TaintPropagatedEvent(
                    kind=TraceEventKind.TAINT_PROPAGATED,
                    node_id=self._node_id,
                    sequence=len(self._trace_events),
                    taint_source=",".join(source_tokens(root.sources)),
                    taint_scope="value",
                    payload=result_payload(tool_name, self._refs.mint(output, root), root),
                )
            )

    # ── introspection ────────────────────────────────────────────────────────

    def register_task(self, task: object) -> None:
        """Record the user's task as trusted (context-default integrity).

        The synchronous governor never sees the prompt; the host framework does.
        Call this with the user's own task text (each turn) so a destination the
        user named is not tainted by the context root. Only the user's text belongs
        here — a task written by a model is not trusted."""
        self._taint.register_trusted(task, TrustedOrigin.TASK)
        if isinstance(task, str):
            self._requests.append(task)

    def register_trusted(
        self, value: object, origin: TrustedOrigin = TrustedOrigin.OPERATOR
    ) -> None:
        """Record an operator-vouched value as trusted (context-default integrity)."""
        self._taint.register_trusted(value, origin)

    def confidentiality_floor_active(self) -> bool:
        """True once a secret read has armed the egress floor this session."""
        return self._taint.confidentiality_floor_active()


def _leaves(x: Any) -> list:
    if isinstance(x, dict):
        return [y for v in x.values() for y in _leaves(v)]
    if isinstance(x, (list, tuple, set, frozenset)):
        return [y for v in x for y in _leaves(v)]
    return [] if x is None else [x]
