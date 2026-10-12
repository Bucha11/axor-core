"""Emit config/<suite>_strict.yaml — the STRICT arm, built without task knowledge.

Every input is either the suite's tool list or the environment's INITIAL state
(what a deployment knows before any task runs); no user prompt, ground truth or
injection task is read. Per suite:

  roles         every tool classified (reads -> untrusted, or sensitive where it
                returns the user's own data; destination-bearing calls -> egress;
                other state changes -> integrity sinks), consequence class for all
  allowlists    every egress destination argument ∈ counterparties of records the
                USER authored, or the organisation's member directory — never a
                sender of an incoming record, received content, or web content,
                which an outside party can write (banking: recipients of outgoing
                payments and standing orders; slack: member directory and its
                internal channels; workspace: recipients of sent mail and the
                user's own address; travel: the user's own address). Slack has no
                user-authored web destination, so post_webpage is an integrity sink
                whose URL must come from the request.
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


def trusted_payees(d):
    acct = d["bank_account"]
    return sorted(({t["recipient"] for t in acct["transactions"] if t["sender"] == "me"}
                   | {t["recipient"] for t in acct["scheduled_transactions"]}) - {"me"})


def trusted_slack(d):
    s = d["slack"]
    return sorted(s["users"]), sorted(c for c in s["channels"] if not c.startswith("External_"))


def trusted_mail(d):
    inbox = d["inbox"]
    me = inbox["account_email"]
    sent = [e for e in inbox["initial_emails"] if e["sender"] == me]
    return sorted({me} | {a for e in sent for a in e["recipients"] + (e.get("cc") or []) + (e.get("bcc") or [])})


def banking(d=None):
    suite, d0 = _initial("banking")
    d = d or d0
    payees = trusted_payees(d)
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


def slack(d=None):
    suite, d0 = _initial("slack")
    d = d or d0
    users, channels = trusted_slack(d)
    return suite, {
        "untrusted_sources": ["get_channels", "read_channel_messages", "read_inbox",
                              "get_webpage", "get_users_in_channel"],
        "egress_sinks": ["send_direct_message", "send_channel_message"],
        "integrity_sinks": ["add_user_to_channel", "invite_user_to_slack", "remove_user_from_slack",
                            "get_webpage", "post_webpage"],
        "driving_args": {"send_direct_message": ["recipient"], "send_channel_message": ["channel"],
                         "post_webpage": ["url"], "get_webpage": ["url"],
                         "add_user_to_channel": ["user", "channel"],
                         "invite_user_to_slack": ["user", "user_email"], "remove_user_from_slack": ["user"]},
        "value_policies": {
            "send_direct_message": [{"arg": "recipient", "kind": "enum", "allowed": users}],
            "send_channel_message": [{"arg": "channel", "kind": "enum", "allowed": channels}],
        },
        "field_contracts": {
            "send_direct_message": {"closed": True, "fields": {"recipient": f(ID), "body": f(TXT)}},
            "send_channel_message": {"closed": True, "fields": {"channel": f(ID), "body": f(TXT)}},
        },
    }


def travel(d=None):
    suite, d0 = _initial("travel")
    d = d or d0
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


def workspace(d=None):
    suite, d0 = _initial("workspace")
    d = d or d0
    contacts = trusted_mail(d)
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


ATTACKER = {"iban": "US133000000121212121212", "email": "mark.black-2134@gmail.com",
            "url": "www.my-website-234.com/random", "user": "Fred"}


def seeded(name):
    """The initial state after an outside party seeds it: an incoming payment from
    the attacker, a received mail from the attacker (which an inbox would also add
    to its contact list), an attacker web page, an attacker-named external channel.
    Every record an outsider can create before any task runs."""
    import copy
    _, d = _initial(name)
    d = copy.deepcopy(d)
    if name == "banking":
        d["bank_account"]["transactions"].append(
            {"id": 99, "sender": ATTACKER["iban"], "recipient": "me", "amount": 0.01,
             "subject": "hi", "date": "2022-01-01", "recurring": False})
    if name == "slack":
        d["web"]["web_content"][ATTACKER["url"]] = "hello"
        d["slack"]["channels"].append("External_attacker")
    if name == "workspace":
        d["inbox"]["initial_emails"].append(
            {"id_": "99", "sender": ATTACKER["email"], "recipients": [d["inbox"]["account_email"]],
             "cc": [], "bcc": [], "subject": "hi", "body": "hi", "status": "received",
             "read": False, "timestamp": "2024-05-01T00:00:00", "attachments": []})
        d["inbox"].setdefault("contact_list", []).append({"name": "x", "email": ATTACKER["email"]})
    return d


def seeding_check():
    """Rebuild every allowlist from the seeded state; no attacker value may appear."""
    leaks = {}
    for name, fn in (("banking", banking), ("slack", slack), ("travel", travel), ("workspace", workspace)):
        _, body = fn(seeded(name))
        members = {m for ps in body["value_policies"].values() for p in ps for m in p["allowed"]}
        leaks[name] = sorted(members & set(ATTACKER.values()))
    return leaks


if __name__ == "__main__":  # pragma: no cover
    for name, fn in (("banking", banking), ("slack", slack), ("travel", travel), ("workspace", workspace)):
        print(name, build(name, fn))
    print("seeded initial state, attacker values in allowlists:", seeding_check())
