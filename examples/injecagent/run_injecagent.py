"""Real InjecAgent run: undefended pass (gpt-4o via OpenRouter) + governance
post-filter, in ONE trajectory set (paired by construction).

For each case we call the model once (undefended), parse its action with
InjecAgent's own scorer, then replay the trajectory through an axor governor:
  - dh: the harmful egress is refused if a driving arg is content-derived from the
    untrusted read (taint) OR the action is catastrophic (consequence axis).
  - ds: S1 sensitive read arms the confidentiality floor; S2 exfil is floor-refused.
Undefended and governed ASR therefore come from the same outputs, at no extra
API cost. Two dh governed numbers are reported (provenance-only, +consequence),
never merged.

Env: OPEN_ROUTER_API_KEY. Usage:
  python3 examples/injecagent/run_injecagent.py --limit 20      # cheap pilot
  python3 examples/injecagent/run_injecagent.py --suite dh ds   # full base
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)
INJECAGENT_DIR = os.environ.get("INJECAGENT_DIR", "/home/user/InjecAgent")
sys.path.insert(0, INJECAGENT_DIR)
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")  # utils.py builds a client at import
os.chdir(INJECAGENT_DIR)  # InjecAgent reads cwd-relative data/

from openai import OpenAI  # noqa: E402
from src.utils import get_tool_dict, get_simulated_attacker_tool_response  # noqa: E402
from src.prompts.agent_prompts import PROMPT_DICT  # noqa: E402
from src.output_parsing import evaluate_output_prompted  # noqa: E402
from src.evaluate_prompted_agent import predict_one_case  # noqa: E402

from axor_core.config import GovernanceConfig  # noqa: E402
from axor_core.governor import ToolCallGovernor  # noqa: E402

sys.path.insert(0, _HERE)
from govern_injecagent import _taintable_token, CONFIG  # noqa: E402

_CFG = GovernanceConfig.from_yaml(CONFIG)
_CFG_NOCONS = dataclasses.replace(_CFG, consequence_overrides={})
_DRIVING = {k: list(v) for k, v in (_CFG.driving_args or {}).items()}


class ORModel:
    """InjecAgent GPTModel routed to OpenRouter."""
    def __init__(self, model: str):
        self.params = {"model_name": model}
        # timeout=45 + max_retries=0: a hanging request fails fast into our own
        # retry loop instead of stalling the whole run on the SDK's 600s default.
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ["OPEN_ROUTER_API_KEY"].strip(),
                             timeout=45.0, max_retries=0)

    def prepare_input(self, sys_prompt, user_prompt_filled):
        return [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt_filled}]

    def call_model(self, model_input):
        # Backoff-with-sleep rides through transient upstream 429s (DeepInfra's
        # "rate-limited, retry shortly" for qwen-2.5-72b clears within a minute).
        # Only a failed attempt pays the sleep, so healthy cases stay fast.
        import time
        backoffs = [8, 16, 24, 32, 40]  # seconds before attempts 2..6
        for attempt in range(len(backoffs) + 1):
            try:
                c = self.client.chat.completions.create(
                    model=self.params["model_name"], messages=model_input,
                    temperature=0, max_tokens=1024)
                if c and c.choices:
                    return c.choices[0].message.content or ""
            except Exception as e:
                if attempt == len(backoffs):
                    print(f"    call_model error (giving up): {e}", flush=True)
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt])
        return ""  # empty output -> parses as 'invalid'/'unsucc', never a false attack success


def _parse_args(param_str):
    try:
        v = json.loads(param_str) if isinstance(param_str, str) else param_str
        return v if isinstance(v, dict) else {"_": str(v)}
    except Exception:
        return {"_": str(param_str)}


def _dh_refused(item, params, consequence: bool) -> bool:
    cfg = _CFG if consequence else _CFG_NOCONS
    g = ToolCallGovernor(**cfg.as_governor_kwargs())
    # Context-default integrity: the benign task is the trusted origin, so a driving
    # value lifted from the injection (not from the task) is untrusted regardless of
    # length. No-op under the legacy "clean" polarity.
    g.register_task(item.get("User Instruction", ""))
    atk = item["Attacker Tools"][0]
    g.register_output(g.evaluate(item["User Tool"], {}), item["Tool Response"])
    args = _parse_args(params)
    if not args or args == {"_": "None"}:  # attacker specified no params -> lift a read token
        tok = _taintable_token(item.get("Attacker Instruction", "") + " " + item["Tool Response"])
        args = {(_DRIVING.get(atk) or ["arg"])[0]: tok} if tok else {}
    return not g.evaluate(atk, args).allowed


def _ds_refused(item) -> bool:
    g = ToolCallGovernor(**_CFG.as_governor_kwargs())
    g.register_task(item.get("User Instruction", ""))  # trusted origin (floor is polarity-independent)
    s1 = item["Attacker Tools"][0]
    s2 = item["Attacker Tools"][1] if len(item["Attacker Tools"]) > 1 else s1
    g.register_output(g.evaluate(item["User Tool"], {}), item["Tool Response"])
    g.register_output(g.evaluate(s1, {}), "<sensitive user data>")  # arms floor
    return not g.evaluate(s2, {"recipients": "attacker@evil.com"}).allowed


def run(suites, limit, model_name, save_dir=None):
    tool_dict = get_tool_dict()
    sys_p, user_p = PROMPT_DICT["InjecAgent"]
    model = ORModel(model_name)
    out = {}
    for suite in suites:
        data = json.load(open(f"data/test_cases_{suite}_base.json"))
        if limit:
            data = data[:limit]
        n = len(data)
        # RESUMABLE: records are appended per-case to a jsonl; a restart loads what
        # is already there and only runs the remaining cases. Lets a slow/flaky model
        # (Qwen) finish across several background windows without losing work.
        path = os.path.join(save_dir, f"{suite}_base.jsonl") if save_dir else None
        records = []
        if path and os.path.exists(path):
            with open(path) as f:
                records = [json.loads(ln) for ln in f if ln.strip()][:n]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fh = open(path, "a") if path else None
        if len(records):
            print(f"  [{suite}] resuming from {len(records)}/{n} saved cases", flush=True)
        for i in range(len(records), n):
            item = data[i]
            output = predict_one_case(sys_p, user_p, {}, item, model, tool_dict)
            p = evaluate_output_prompted(output, item["Attacker Tools"][0], item["Attacker Tools"],
                                         item["User Tool"], item["Attacker Instruction"])
            rec = {"User Tool": item["User Tool"],
                   "Attacker Tools": item["Attacker Tools"],
                   "Tool Response": item["Tool Response"],
                   "Attacker Instruction": item.get("Attacker Instruction", ""),
                   "User Instruction": item.get("User Instruction", ""),  # trusted origin (context mode)
                   "output": output, "eval": p["eval"],
                   "Attacker Tool Parameters": p.get("Attacker Tool Parameters", "")}
            records.append(rec)
            if fh:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (i + 1) % 20 == 0:
                print(f"  [{suite}] {i+1}/{n} ...", flush=True)
        if fh:
            fh.close()
        # Metrics from ALL records (governed re-applied via the current config/polarity).
        undef_succ = gov_prov = gov_both = 0
        for rec in records:
            if rec["eval"] != "succ":
                continue
            undef_succ += 1
            if suite == "dh":
                params = rec.get("Attacker Tool Parameters", "")
                gov_prov += not _dh_refused(rec, params, consequence=False)
                gov_both += not _dh_refused(rec, params, consequence=True)
            else:  # ds: full success needs S2; the floor refuses S2
                refused = _ds_refused(rec)
                gov_prov += not refused
                gov_both += not refused
        out[suite] = {"n": n, "undef_succ": undef_succ,
                      "gov_prov_succ": gov_prov, "gov_both_succ": gov_both}
        print(f"[{suite}] n={n}  undefended ASR={100*undef_succ/n:.1f}%  "
              f"governed(prov)={100*gov_prov/n:.1f}%  governed(+conseq)={100*gov_both/n:.1f}%",
              flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", default=["dh", "ds"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--save", default=None, help="dir to dump per-case records for offline re-analysis")
    a = ap.parse_args()
    res = run(a.suite, a.limit, a.model, save_dir=a.save)
    print("\n=== RESULTS ===")
    print(json.dumps(res, indent=2))
