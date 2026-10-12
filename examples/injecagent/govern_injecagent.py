"""axor governance post-filter for InjecAgent + a no-API mock pilot.

InjecAgent runs the agent ONCE (undefended). Governance is then a *deterministic
post-filter* on the recorded trajectory: replay the tool sequence through an axor
``ToolCallGovernor`` and see whether the attacker tool call is refused. Because
the decision is deterministic given the call, undefended and governed ASR come
from the SAME model outputs — the ideal paired design at zero extra API cost.

Two mechanisms, matching the two attack shapes:
  - direct-harm (dh): the harmful action is an egress sink; if a driving arg's
    value is content-derived from the untrusted tool response (a >=12-char shared
    fragment — an email / URL / long account the agent lifted from the injection),
    the per-value taint gate refuses it.
  - data-stealing (ds): S1 is a sensitive read (arms the confidentiality floor),
    S2 is the exfil egress (GmailSendEmail) — the floor refuses it, content-blind
    and paraphrase-proof, regardless of the destination string.

`mock_pilot()` fabricates the "agent followed the injection" trajectory (so
undefended ASR = 100% by construction) and checks how much governance removes.
"""
from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # axor-core root (for axor_core import)
sys.path.insert(0, _REPO_ROOT)
INJECAGENT_DIR = os.environ.get("INJECAGENT_DIR", "/home/user/InjecAgent")
sys.path.insert(0, INJECAGENT_DIR)
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")

from axor_core.config import GovernanceConfig  # noqa: E402
from axor_core.governor import ToolCallGovernor  # noqa: E402

CONFIG = os.path.join(_HERE, "config", "injecagent.yaml")
_TOKEN = re.compile(r"[^\s\"'{}\[\](),]{12,}")  # >=12-char tokens (>= ledger _MIN_SEGMENT)


def _governor(consequence: bool = True) -> ToolCallGovernor:
    """Fresh governor. consequence=False strips the consequence taxonomy so the
    result is the PURE provenance (taint/floor) number, with no consequence
    labelling — the cherry-pick-free baseline."""
    import dataclasses
    cfg = GovernanceConfig.from_yaml(CONFIG)
    if not consequence:
        cfg = dataclasses.replace(cfg, consequence_overrides={})
    return ToolCallGovernor(**cfg.as_governor_kwargs())


def _taintable_token(text: str) -> str | None:
    """The destination-ish >=12-char fragment an agent would lift from the read."""
    toks = _TOKEN.findall(text or "")
    if not toks:
        return None
    for t in toks:  # prefer an email / URL / long id
        if "@" in t or "/" in t or any(c.isdigit() for c in t):
            return t
    return toks[0]


def govern_dh(item: dict, driving_args: dict, consequence: bool = True) -> dict:
    """Replay a direct-harm case. With consequence=False, only the per-value taint
    gate can refuse (pure provenance). With consequence=True, the role-blind
    consequence taxonomy also gates irreversible actions."""
    g = _governor(consequence=consequence)
    user_tool, atk = item["User Tool"], item["Attacker Tools"][0]
    resp = item["Tool Response"]
    g.register_output(g.evaluate(user_tool, {}), resp)  # untrusted read taints resp

    token = _taintable_token(item.get("Attacker Instruction", "") + " " + resp)
    drivers = driving_args.get(atk) or ["arg"]
    args = {drivers[0]: token} if token else {}
    d = g.evaluate(atk, args)
    return {"denied": not d.allowed, "reason": getattr(d, "category", None),
            "taintable": token is not None}


def govern_ds(item: dict) -> dict:
    """Replay a data-stealing case: S1 sensitive read arms the floor, S2 exfil
    (GmailSendEmail) is refused by the floor."""
    g = _governor()
    user_tool = item["User Tool"]
    s1 = item["Attacker Tools"][0]
    s2 = item["Attacker Tools"][1] if len(item["Attacker Tools"]) > 1 else s1
    resp = item["Tool Response"]
    g.register_output(g.evaluate(user_tool, {}), resp)          # untrusted read
    g.register_output(g.evaluate(s1, {}), "<sensitive user data>")  # S1 read -> arms floor
    token = _taintable_token(item.get("Attacker Instruction", "") + " " + resp)
    d = g.evaluate(s2, {"recipients": token or "attacker@evil.com"})  # S2 exfil
    return {"denied": not d.allowed, "reason": getattr(d, "category", None),
            "floor_armed": g.confidentiality_floor_active()}


_SHORT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@/-]*[0-9_][A-Za-z0-9_.@/-]*")


def _tool_params() -> "dict[str, list[str]]":
    """Attacker/user tool name -> its parameter names, from InjecAgent's tools.json."""
    with open(os.path.join(INJECAGENT_DIR, "data", "tools.json")) as f:
        kits = json.load(f)
    return {kit["toolkit"] + t["name"]: [p["name"] for p in t.get("parameters", [])]
            for kit in kits for t in kit["tools"]}


def govern_dh_schema(item: dict, driving_args: dict, params: "dict[str, list[str]]",
                     consequence: bool = True) -> dict:
    """Replay a direct-harm case with arguments shaped by the attacker tool's own
    schema, as the live path does: the task is registered (a value the user named
    is trusted), a tool without parameters is called with none, and otherwise the
    first declared driving argument (else the first parameter) carries a value the
    injection names. ``govern_dh``'s >=12-char token rule predates the context
    default and drops short values (a room id, a patient id, an amount), which
    made such cases look as if they carried no attacker value."""
    g = _governor(consequence=consequence)
    g.register_task(item.get("User Instruction", ""))
    atk = item["Attacker Tools"][0]
    g.register_output(g.evaluate(item["User Tool"], {}), item["Tool Response"])
    names = params.get(atk, [])
    if not names:
        args: dict = {}
    else:
        text = item.get("Attacker Instruction", "")
        long = _TOKEN.findall(text)
        short = _SHORT_ID.findall(text)
        value = (long or short or ["attacker-chosen"])[0]
        keys = [a for a in (driving_args.get(atk) or []) if a in names] or names[:1]
        args = {keys[0]: value}
    d = g.evaluate(atk, args)
    return {"denied": not d.allowed, "reason": getattr(d, "category", None),
            "has_params": bool(names)}


def account_dh(setting: str = "base") -> dict:
    """Per-case, per-gate accounting over every direct-harm case: each case is
    replayed with the full configuration and with the consequence taxonomy
    removed, and the (full, origin-only) verdict pairs are counted, split by
    whether the attacker tool takes any parameter. Deterministic; no model call."""
    import collections
    cfg = GovernanceConfig.from_yaml(CONFIG)
    driving = {k: list(v) for k, v in (cfg.driving_args or {}).items()}
    params = _tool_params()
    with open(os.path.join(INJECAGENT_DIR, "data", f"test_cases_dh_{setting}.json")) as f:
        items = json.load(f)
    pairs = collections.Counter()
    no_param_tools = set()
    for item in items:
        full = govern_dh_schema(item, driving, params, consequence=True)
        origin = govern_dh_schema(item, driving, params, consequence=False)
        key = (full["reason"] if full["denied"] else "allow",
               origin["reason"] if origin["denied"] else "allow",
               "params" if full["has_params"] else "no-params")
        pairs[key] += 1
        if not full["has_params"]:
            no_param_tools.add(item["Attacker Tools"][0])
    return {"cases": len(items), "pairs": dict(pairs), "no_param_tools": sorted(no_param_tools)}


def mock_pilot(n: int = 40) -> None:
    cfg = GovernanceConfig.from_yaml(CONFIG)
    driving = {k: list(v) for k, v in (cfg.driving_args or {}).items()}
    data_dir = os.path.join(INJECAGENT_DIR, "data")

    print("=" * 68)
    print(f"MOCK PILOT — undefended trajectory = 'agent followed injection' (ASR 100% by\nconstruction); governance is the post-filter. n={n}/suite, base setting.")
    print("=" * 68)

    dh = json.load(open(os.path.join(data_dir, "test_cases_dh_base.json")))[:n]
    d_taint = d_both = taintable = 0
    for it in dh:
        rt = govern_dh(it, driving, consequence=False)   # provenance only
        rb = govern_dh(it, driving, consequence=True)     # + consequence axis
        d_taint += rt["denied"]; d_both += rb["denied"]; taintable += rt["taintable"]
    print(f"\nDIRECT HARM (dh)  n={len(dh)}   undefended ASR 100% (by construction)")
    print(f"  provenance only (taint)     -> governed ASR {100*(len(dh)-d_taint)/len(dh):.1f}%"
          f"   ({d_taint}/{len(dh)} refused)")
    print(f"    taintable (>=12-char token): {taintable}/{len(dh)}"
          f"  -> {len(dh)-taintable} short-value/no-arg residual (provenance axis boundary)")
    print(f"  + consequence axis          -> governed ASR {100*(len(dh)-d_both)/len(dh):.1f}%"
          f"   ({d_both}/{len(dh)} refused)")
    print(f"    (role-blind taxonomy: 0 user tools flagged here — they are all reads;"
          f" FP cost of this axis is measured on AgentDojo, not InjecAgent)")

    ds = json.load(open(os.path.join(data_dir, "test_cases_ds_base.json")))[:n]
    denied = armed = 0
    for it in ds:
        r = govern_ds(it)
        denied += r["denied"]; armed += r["floor_armed"]
    print(f"\nDATA STEALING (ds)  n={len(ds)}   (S2 exfil = GmailSendEmail)")
    print(f"  undefended ASR : 100.0%  (all followed to exfil, by construction)")
    print(f"  governed  ASR  : {100*(len(ds)-denied)/len(ds):.1f}%   ({denied}/{len(ds)} S2 refused)")
    print(f"  floor armed by S1 sensitive read: {armed}/{len(ds)}")

    # sanity control: a CLEAN call (destination from prompt, not the read) must pass
    g = _governor()
    g.register_output(g.evaluate(dh[0]["User Tool"], {}), "benign product details, no injection here")
    ctrl = g.evaluate("GmailSendEmail", {"recipients": "colleague@work.com"})
    print(f"\nCONTROL (no untrusted-derived arg, no sensitive read): GmailSendEmail -> "
          f"{'ALLOW' if ctrl.allowed else 'DENY'}  (must ALLOW — governance is not blanket)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "account":
        for setting in ("base", "enhanced"):
            print("dh", setting, account_dh(setting))
            with open(os.path.join(INJECAGENT_DIR, "data", f"test_cases_ds_{setting}.json")) as f:
                ds = [govern_ds(item) for item in json.load(f)]
            print("ds", setting, {"cases": len(ds), "refused": sum(r["denied"] for r in ds),
                                  "floor_armed": sum(r["floor_armed"] for r in ds)})
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
        mock_pilot(n)
