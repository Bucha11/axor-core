"""Emit config/<suite>_strict.yaml — the STRICT arm, built without task knowledge.

Every input is either the suite's tool list or the environment's INITIAL state
(what a deployment knows before any task runs); no user prompt, ground truth or
injection task is read. Per suite:

  roles         every tool classified (reads -> untrusted, or sensitive where it
                returns the user's own data; destination-bearing calls -> egress;
                other state changes -> integrity sinks), consequence class for all
  allowlists    every egress destination argument ∈ initial-state counterparties
                (banking: transaction/standing-order counterparties; slack: users,
                channels, initially present web pages; workspace: contact list;
                travel: the user's own address)
  contracts     every guarded sink's fields, kinds and absence behaviour, as read
                off the stock handler (experiments/review_offline/contract_audit)
  integrity     context default; origin setting left to the STRICT default
                (request-only), so no record declarations are needed

The script prints the declaration volume per suite (what an operator writes) and
refuses to emit a config the STRICT constructor rejects.

Run: uv run --no-project --with agentdojo==0.1.35 --with pyyaml --with-editable . \
       python examples/agentdojo/build_strict.py
"""
from __future__ import annotations

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

ID, IDS, NUM, TXT, INT, FLAG = "identifier", "identifier_list", "number", "text", "interpreted", "flag"


def f(kind, **kw):
    return {"kind": kind, **kw}


OPT_NONE = {"unused_when_absent": True, "null_is_absent": True}
OPT_FALSY = {"unused_when_absent": True, "falsy_is_absent": True}

EMAIL_CONTRACT = {"closed": True, "fields": {
    "recipients": f(IDS), "subject": f(TXT), "body": f(TXT),
    "attachments": f("structured", **OPT_NONE), "cc": f(IDS, **OPT_NONE), "bcc": f(IDS, **OPT_NONE)}}


def _initial(name):
    from agentdojo.task_suite.load_suites import get_suite
    suite = get_suite("v1", name)
    return suite, suite.load_and_inject_default_environment({}).model_dump()


def banking():
    suite, d = _initial("banking")
    acct = d["bank_account"]
    payees = sorted(({t["recipient"] for t in acct["transactions"]}
                     | {t["sender"] for t in acct["transactions"]}
                     | {t["recipient"] for t in acct["scheduled_transactions"]}) - {"me"})
    sinks = ["send_money", "schedule_transaction", "update_scheduled_transaction"]
    pay = {"recipient": f(ID), "amount": f(NUM, decimal_string=True), "subject": f(TXT), "date": f(INT)}
    return suite, {
        "untrusted_sources": ["read_file", "get_most_recent_transactions", "get_scheduled_transactions"],
        "sensitive_sources": ["get_user_info"],
        "benign_tools": ["get_balance", "get_iban"],
        "egress_sinks": sinks,
        "integrity_sinks": ["update_password", "update_user_info"],
        "driving_args": {**{s: ["recipient"] for s in sinks},
                         "update_password": ["password"],
                         "update_user_info": ["first_name", "last_name", "street", "city"]},
        "value_policies": {s: [{"arg": "recipient", "kind": "enum", "allowed": payees}] for s in sinks},
        "field_contracts": {
            "send_money": {"closed": True, "fields": pay},
            "schedule_transaction": {"closed": True, "fields": {**pay, "recurring": f(FLAG)}},
            "update_scheduled_transaction": {"closed": True, "fields": {
                "id": f(NUM, integral=True), "recipient": f(ID, **OPT_FALSY),
                "amount": f(NUM, decimal_string=True, **OPT_FALSY), "subject": f(TXT, **OPT_FALSY),
                "date": f(INT, **OPT_FALSY), "recurring": f(FLAG, **OPT_FALSY)}},
        },
    }


def slack():
    suite, d = _initial("slack")
    s = d["slack"]
    users, channels, pages = sorted(s["users"]), sorted(s["channels"]), sorted(d["web"]["web_content"])
    return suite, {
        "untrusted_sources": ["get_channels", "read_channel_messages", "read_inbox",
                              "get_webpage", "get_users_in_channel"],
        "egress_sinks": ["send_direct_message", "send_channel_message", "post_webpage"],
        "integrity_sinks": ["add_user_to_channel", "invite_user_to_slack", "remove_user_from_slack",
                            "get_webpage"],
        "driving_args": {"send_direct_message": ["recipient"], "send_channel_message": ["channel"],
                         "post_webpage": ["url"], "get_webpage": ["url"],
                         "add_user_to_channel": ["user", "channel"],
                         "invite_user_to_slack": ["user", "user_email"], "remove_user_from_slack": ["user"]},
        "value_policies": {
            "send_direct_message": [{"arg": "recipient", "kind": "enum", "allowed": users}],
            "send_channel_message": [{"arg": "channel", "kind": "enum", "allowed": channels}],
            "post_webpage": [{"arg": "url", "kind": "enum", "allowed": pages}],
        },
        "field_contracts": {
            "send_direct_message": {"closed": True, "fields": {"recipient": f(ID), "body": f(TXT)}},
            "send_channel_message": {"closed": True, "fields": {"channel": f(ID), "body": f(TXT)}},
            "post_webpage": {"closed": True, "fields": {"url": f(ID), "content": f(TXT)}},
        },
    }


def travel():
    suite, d = _initial("travel")
    me = [d["user"]["email"]]
    reads = [t.name for t in suite.tools if t.name.startswith(("get_", "check_", "search_"))
             and t.name != "get_user_information"]
    return suite, {
        "untrusted_sources": reads,
        "sensitive_sources": ["get_user_information"],
        "egress_sinks": ["send_email"],
        "integrity_sinks": ["reserve_hotel", "reserve_restaurant", "reserve_car_rental",
                            "create_calendar_event", "cancel_calendar_event"],
        "driving_args": {"send_email": ["recipients", "cc", "bcc"], "reserve_hotel": ["hotel"],
                         "reserve_restaurant": ["restaurant"], "reserve_car_rental": ["company"],
                         "create_calendar_event": ["title", "participants"],
                         "cancel_calendar_event": ["event_id"]},
        "value_policies": {"send_email": [{"arg": a, "kind": "enum", "allowed": me}
                                          for a in ("recipients", "cc", "bcc")]},
        "field_contracts": {"send_email": EMAIL_CONTRACT},
    }


def workspace():
    suite, d = _initial("workspace")
    contacts = sorted({c["email"] for c in d["inbox"].get("contact_list", [])} | {d["inbox"]["account_email"]})
    reads = [t.name for t in suite.tools if t.name.startswith(("get_", "search_", "list_"))]
    return suite, {
        "untrusted_sources": reads,
        "egress_sinks": ["send_email", "share_file", "add_calendar_event_participants",
                         "create_calendar_event"],
        "integrity_sinks": ["delete_email", "delete_file", "cancel_calendar_event",
                            "reschedule_calendar_event", "append_to_file", "create_file"],
        "driving_args": {"send_email": ["recipients", "cc", "bcc"], "share_file": ["email"],
                         "add_calendar_event_participants": ["participants"],
                         "create_calendar_event": ["participants"],
                         "delete_email": ["email_id"], "delete_file": ["file_id"],
                         "cancel_calendar_event": ["event_id"], "reschedule_calendar_event": ["event_id"],
                         "append_to_file": ["file_id"], "create_file": ["filename"]},
        "value_policies": {
            "send_email": [{"arg": a, "kind": "enum", "allowed": contacts} for a in ("recipients", "cc", "bcc")],
            "share_file": [{"arg": "email", "kind": "enum", "allowed": contacts}],
            "add_calendar_event_participants": [{"arg": "participants", "kind": "enum", "allowed": contacts}],
            "create_calendar_event": [{"arg": "participants", "kind": "enum", "allowed": contacts}],
        },
        "field_contracts": {
            "send_email": EMAIL_CONTRACT,
            "share_file": {"closed": True, "fields": {"file_id": f(ID), "email": f(ID), "permission": f(ID)}},
            "add_calendar_event_participants": {"closed": True, "fields": {"event_id": f(ID), "participants": f(IDS)}},
            "create_calendar_event": {"closed": True, "fields": {
                "title": f(TXT), "start_time": f(INT), "end_time": f(INT),
                "description": f(TXT, default=""), "participants": f(IDS, **OPT_NONE),
                "location": f(TXT, **OPT_NONE)}},
        },
        "consequence_overrides": {"delete_email": "catastrophic", "delete_file": "catastrophic"},
    }


def build(name, fn):
    from axor_core.config import GovernanceConfig
    from axor_core.governor import ToolCallGovernor
    from axor_core.kernel.registration import validate_role_completeness
    suite, body = fn()
    tools = [t.name for t in suite.tools]
    sinks = set(body.get("egress_sinks", [])) | set(body.get("integrity_sinks", []))
    cons = {t: ("consequential" if t in sinks else "benign") for t in tools}
    cons.update(body.get("consequence_overrides", {}))
    cfg_dict = {"mode": "strict", "integrity_default": "context", **body, "consequence_overrides": cons}
    cfg = GovernanceConfig.from_dict(cfg_dict)
    ToolCallGovernor(**cfg.as_governor_kwargs())  # STRICT construction obligations
    errs = validate_role_completeness(
        tools, untrusted_sources=cfg.untrusted_sources, sensitive_sources=cfg.sensitive_sources,
        egress_sinks=cfg.egress_sinks, positional_sinks=cfg.positional_sinks,
        benign_tools=cfg.benign_tools, value_policies=cfg.value_policies,
        integrity_sinks=cfg.integrity_sinks)
    if errs:
        raise SystemExit(f"{name}: " + "; ".join(errs))
    out = os.path.join(HERE, "config", f"{name}_strict.yaml")
    header = (f"# STRICT arm for AgentDojo {name} — generated by build_strict.py from the tool\n"
              f"# list and the environment's initial state only (no task, no ground truth).\n")
    with open(out, "w") as fh:
        fh.write(header + yaml.safe_dump(cfg_dict, sort_keys=False, width=100))
    volume = {
        "tools": len(tools),
        "role_entries": sum(len(body.get(k, [])) for k in
                            ("untrusted_sources", "sensitive_sources", "benign_tools",
                             "egress_sinks", "integrity_sinks")),
        "driving_args": sum(len(v) for v in body["driving_args"].values()),
        "allowlist_entries": sum(len(p["allowed"]) for ps in body["value_policies"].values() for p in ps),
        "contract_fields": sum(len(c["fields"]) for c in body["field_contracts"].values()),
        "yaml_lines": sum(1 for _ in open(out)),
    }
    return volume


if __name__ == "__main__":  # pragma: no cover
    for name, fn in (("banking", banking), ("slack", slack), ("travel", travel), ("workspace", workspace)):
        print(name, build(name, fn))
