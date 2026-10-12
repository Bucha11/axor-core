"""Real handlers for end-to-end tests: the check is made by the kernel, the effect
by these. Each handler is written the way such handlers are commonly written —
with a default, a coercion, an alias — because those are exactly the places where
"the value the predicate checked" and "the value the handler consumed" can part.

Nothing here is a mock of the kernel. The tests drive the real IntentLoop / governor
and read the handler's own state (a ledger, a public bucket) to decide what
actually happened.
"""
from __future__ import annotations

import os

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.policy import EscalationPolicy, ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.node.intent_loop import IntentLoop

NODE = "n1"
ALLOWED_IBAN = "GB29NWBK60161331926819"
ATTACKER_IBAN = "US133000000121212121212"


class Bank:
    """One account. ``transfers`` is the observable effect."""

    def __init__(self, balance: float = 50_000.0) -> None:
        self.balance = balance
        self.transfers: list[tuple[str, float]] = []

    def send_money(self, recipient=None, amount=None, subject="", date=None, iban=None):
        # Legacy alias: older clients send `iban`; it wins when present.
        to = iban if iban is not None else recipient
        # No amount means "sweep the account" — a convenience default.
        amt = self.balance if amount is None else float(amount)  # float() parses "1e5"
        self.balance -= amt
        self.transfers.append((f"{to}", amt))  # the account string the bank books
        return {"status": "sent", "to": f"{to}", "amount": amt}


class SendMoney(ToolHandler):
    def __init__(self, bank: Bank) -> None:
        self.bank = bank
        self.received: list[dict] = []

    @property
    def name(self) -> str:
        return "send_money"

    async def execute(self, args: dict):
        self.received.append(dict(args))
        return self.bank.send_money(**args)


class Publish(ToolHandler):
    """Copy a file from the reports root to a public bucket. The path is handed to
    the filesystem resolver — symlinks and all — so the bytes published depend on
    the filesystem, not only on the string that was checked."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.public: list[bytes] = []

    @property
    def name(self) -> str:
        return "publish"

    async def execute(self, args: dict):
        with open(os.path.join(self.root, args["path"]), "rb") as f:
            self.public.append(f.read())
        return "published"


def envelope(tools, *, ceiling: ConsequenceClass = ConsequenceClass.CONSEQUENTIAL,
             call_approval: bool = False, allow_escalation: bool | None = None,
             ) -> ExecutionEnvelope:
    policy = ExecutionPolicy(
        name="t",
        tool_policy=ToolPolicy(allow_read=True, extra_allowed=tuple(tools)),
        max_unattended_consequence=ceiling,
        escalation_policy=EscalationPolicy(
            allow_escalation=call_approval if allow_escalation is None else allow_escalation,
            grantable_tools=tuple(tools),
            require_human=True, max_escalations=3, call_approval=call_approval,
        ),
    )
    lineage = LineageSummary(
        node_id=NODE, parent_id=None, depth=0, ancestry_ids=[], inherited_restrictions=[]
    )
    ctx = ContextView(
        node_id=NODE, working_summary="t", visible_fragments=[], active_constraints=[],
        lineage=lineage, token_count=0, compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset(tools), allow_children=False,
        allow_nested_children=False, allow_context_expansion=False,
        allow_export=False, allow_mutation=True, max_child_depth=0,
    )
    return ExecutionEnvelope(
        node_id=NODE, task="t", context=ctx, policy=policy, capabilities=caps,
        export_contract=ExportContract(
            mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0
        ),
        lineage=lineage, cancel_token=make_token(),
    )


def loop(*handlers: ToolHandler, **kw) -> IntentLoop:
    ex = CapabilityExecutor()
    for h in handlers:
        ex.register(h)
    return IntentLoop(ex, [], **kw)


def call(tool: str, args: dict, tool_use_id: str = "tu1") -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": tool, "args": args, "tool_use_id": tool_use_id},
        node_id=NODE,
    )
