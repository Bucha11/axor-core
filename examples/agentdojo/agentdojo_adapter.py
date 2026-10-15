"""AgentDojo adapter for axor-core.

Two pipeline elements:

- :class:`RawAnthropicLLM` — an AgentDojo ``BasePipelineElement`` that drives a
  real Claude model through raw ``urllib`` (the Anthropic SDK cannot connect from
  this sandbox; raw POST works, and it keeps the example SDK-free). It produces
  the same ``ChatAssistantMessage`` / ``FunctionCall`` shapes AgentDojo's own
  ``AnthropicLLM`` does, so it is a drop-in LLM.

- :class:`GovernedToolsExecutor` — replaces AgentDojo's ``ToolsExecutor``. Before
  each tool call runs, it asks an axor :class:`~axor_core.governor.ToolCallGovernor`
  whether the call is permitted. A denied call never touches the real tool; the
  model gets a governance-denial tool result and continues. Allowed calls run
  normally, and their output is registered in the governor's per-value ledger so a
  later sink carrying attacker-controlled content is gated.

The governor is configured with the suite's tool taxonomy (which tools are
untrusted-data sources, which are egress sinks) — that declaration is how an
operator deploys axor over a real tool set, exactly as it would in production.
"""
from __future__ import annotations

import re
import json
import os
import time
import urllib.request
import urllib.error
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)

from axor_core.governor import ToolCallGovernor

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


# ── Raw-urllib LLM element ──────────────────────────────────────────────────────

def _messages_to_anthropic(messages: Sequence[ChatMessage]):
    """Convert AgentDojo chat messages to (system, anthropic_messages) as plain
    dicts suitable for a raw API POST."""
    system = None
    out: list[dict] = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system = msg["content"][0]["content"]
        elif role == "user":
            text = "\n".join(b["content"] for b in msg["content"] if b.get("content"))
            out.append({"role": "user", "content": [{"type": "text", "text": text}]})
        elif role == "assistant":
            blocks: list[dict] = []
            for b in msg["content"] or []:
                if b["type"] == "text" and b["content"]:
                    blocks.append({"type": "text", "text": b["content"]})
            for tc in msg["tool_calls"] or []:
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function,
                    "input": dict(tc.args),
                })
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            is_error = msg["error"] is not None
            text = msg["error"] if is_error else (
                msg["content"][0]["content"] if msg["content"] else ""
            )
            out.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": [{"type": "text", "text": text or ""}],
                "is_error": is_error,
            }]})
    # Merge consecutive user messages (Anthropic requires alternating roles).
    merged: list[dict] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] == "user":
            merged[-1]["content"].extend(m["content"])
        else:
            merged.append(m)
    return system, merged


class RawAnthropicLLM(BasePipelineElement):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"raw-anthropic-{model}"
        self._key = os.environ.get("ANTHROPIC_API_KEY", "")

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            _API_URL,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self._key,
                "anthropic-version": _API_VERSION,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Anthropic API {exc.code}: {exc.read().decode(errors='replace')}") from exc

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        system, anthropic_messages = _messages_to_anthropic(messages)
        tools = [
            {
                "name": f.name,
                "description": f.description,
                "input_schema": f.parameters.model_json_schema(),
            }
            for f in runtime.functions.values()
        ]
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
            "tools": tools,
            "temperature": 0.0,
        }
        if system:
            body["system"] = system
        resp = self._post(body)

        content = resp.get("content", []) or []
        text_blocks = [
            text_content_block_from_string(b["text"])
            for b in content if b.get("type") == "text"
        ]
        tool_calls = [
            FunctionCall(id=b["id"], function=b["name"], args=b.get("input", {}) or {})
            for b in content if b.get("type") == "tool_use"
        ]
        assistant = ChatAssistantMessage(
            role="assistant",
            content=text_blocks,
            tool_calls=tool_calls if resp.get("stop_reason") == "tool_use" else None,
        )
        return query, runtime, env, [*messages, assistant], extra_args


# ── OpenRouter (OpenAI chat-completions) LLM element ────────────────────────────

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _messages_to_openai(messages: Sequence[ChatMessage]) -> list[dict]:
    """Convert AgentDojo chat messages to OpenAI chat-completions messages."""
    out: list[dict] = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            out.append({"role": "system", "content": msg["content"][0]["content"]})
        elif role == "user":
            out.append({
                "role": "user",
                "content": "\n".join(b["content"] for b in msg["content"] if b.get("content")),
            })
        elif role == "assistant":
            text = "\n".join(
                b["content"] for b in (msg["content"] or []) if b.get("content")
            )
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function, "arguments": json.dumps(dict(tc.args))},
                }
                for tc in (msg["tool_calls"] or [])
            ]
            m: dict = {"role": "assistant", "content": text or None}
            if tool_calls:
                m["tool_calls"] = tool_calls
            out.append(m)
        elif role == "tool":
            is_error = msg["error"] is not None
            content = msg["error"] if is_error else (
                msg["content"][0]["content"] if msg["content"] else ""
            )
            out.append({
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": content or "",
            })
    return out


# ── spend accounting (AXOR_BENCH_SPEND_LOG / AXOR_BENCH_SPEND_CAP) ───────────────
#
# Every OpenRouter call appends its reported cost (USD) to a JSONL ledger shared by
# all runner processes; before each call the ledger is summed and the run stops if
# the cap is reached. The cap is checked before a call, so the overshoot is at most
# one call per running process.

class SpendCapReached(RuntimeError):
    pass


def _spend_total() -> float:
    path = os.environ.get("AXOR_BENCH_SPEND_LOG")
    if not path or not os.path.exists(path):
        return 0.0
    total = 0.0
    with open(path) as f:
        for line in f:
            try:
                total += float(json.loads(line).get("cost") or 0.0)
            except (ValueError, json.JSONDecodeError):
                continue
    return total


def _spend_guard() -> None:
    cap = os.environ.get("AXOR_BENCH_SPEND_CAP")
    if cap and _spend_total() >= float(cap):
        raise SpendCapReached(f"spend cap ${float(cap):.2f} reached (${_spend_total():.4f})")


def _spend_record(model: str, usage: dict) -> None:
    path = os.environ.get("AXOR_BENCH_SPEND_LOG")
    if not path:
        return
    rec = {"model": model, "cost": usage.get("cost"),
           "prompt_tokens": usage.get("prompt_tokens"),
           "completion_tokens": usage.get("completion_tokens"),
           "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


class OpenRouterLLM(BasePipelineElement):
    """An AgentDojo LLM element backed by OpenRouter (OpenAI-compatible API),
    over raw urllib. Use to drive injection-susceptible open models (e.g. Qwen)
    that AgentDojo was designed to stress."""

    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"openrouter-{model}"
        self._key = os.environ.get("OPEN_ROUTER_API_KEY", "").strip()

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            _OPENROUTER_URL,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self._key}",
            },
            method="POST",
        )
        # Retry transient upstream throttling/5xx with exponential backoff —
        # OpenRouter's free Qwen pool is frequently rate-limited (429).
        delay = 4.0
        last = ""
        for attempt in range(7):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                last = exc.read().decode(errors="replace")
                # 402 in_flight_budget_exhausted: too many concurrent requests for the
                # credit balance; it clears when in-flight requests settle
                inflight = exc.code == 402 and "in_flight_budget" in last
                if (exc.code in (429, 500, 502, 503, 529) or inflight) and attempt < 6:
                    ra = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        wait = min(float(ra), 120.0) if ra else delay
                    except ValueError:
                        wait = delay
                    time.sleep(wait)
                    delay = min(delay * 2, 60.0)
                    continue
                raise RuntimeError(f"OpenRouter {exc.code}: {last}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < 6:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise RuntimeError(f"OpenRouter transport: {exc}") from exc
        raise RuntimeError(f"OpenRouter: exhausted retries: {last}")

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f.name,
                    "description": f.description,
                    "parameters": f.parameters.model_json_schema(),
                },
            }
            for f in runtime.functions.values()
        ]
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": _messages_to_openai(messages),
            "temperature": 0.0,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        _spend_guard()
        body["usage"] = {"include": True}  # OpenRouter reports the call's cost
        resp = self._post(body)
        _spend_record(self.model, resp.get("usage") or {})

        choice = resp["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        raw_tool_calls = msg.get("tool_calls") or []
        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"]) if fn.get("arguments") else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(FunctionCall(id=tc.get("id"), function=fn["name"], args=args))
        assistant = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(text)] if text else [],
            tool_calls=tool_calls or None,
        )
        return query, runtime, env, [*messages, assistant], extra_args


# ── Governed tool executor ──────────────────────────────────────────────────────

_LIST_HEAD = re.compile(r"^(?:Hotel Names|Restaurant in [^:\n]+|Car Rental Companies in [^:\n]+): ?(.*)$",
                        re.S)


def _parse_structured(tool: str, out: str):
    """MODEL.md §9.15 structured-output parser (the platform adapter's job): read an AgentDojo tool
    output string into a list / records / name->value map, or None when it is free text. Mirrors the
    offline axor_ajar.structured so the live run exercises the same description grants."""
    if not isinstance(out, str):
        return out
    s = out.strip("\n")
    m = _LIST_HEAD.match(s)
    if m:
        return [x.strip() for x in m.group(1).split("\n") if x.strip()]
    if s.startswith("{") and s.endswith("}"):
        import ast
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None
    try:
        import yaml
        y = yaml.safe_load(s)
    except Exception:
        return None
    return y if isinstance(y, (list, dict)) else None


# view-aware confirmer verbs (residuals_rules.md; same table the offline axor_ajar.VIEW_VERBS uses)
VIEW_VERBS: dict = {
    "send_money": ("send", "pay", "transfer", "wire"), "schedule_transaction": ("schedule", "set up", "recurring"),
    "update_scheduled_transaction": ("update", "change", "modify", "adjust"),
    "update_password": ("password",), "update_user_info": ("update", "change"),
    "send_email": ("send", "email", "mail", "reply", "forward", "write", "notify", "invite", "share"),
    "create_calendar_event": ("create", "schedule", "add", "organize", "organise", "book", "set up", "plan"),
    "cancel_calendar_event": ("cancel", "delete", "remove"),
    "reschedule_calendar_event": ("reschedule", "move", "postpone", "change"),
    "add_calendar_event_participants": ("add", "invite", "include"), "share_file": ("share", "grant", "give"),
    "append_to_file": ("append", "add", "write", "update"), "create_file": ("create", "write", "save", "make"),
    "delete_file": ("delete", "remove"), "delete_email": ("delete", "remove"),
    "reserve_hotel": ("reserve", "book"), "reserve_car_rental": ("reserve", "book"),
    "reserve_restaurant": ("reserve", "book"),
    "send_direct_message": ("send", "message", "dm", "tell", "write", "reply", "ask"),
    "send_channel_message": ("send", "post", "write", "message", "announce"),
    "add_user_to_channel": ("add", "invite"), "invite_user_to_slack": ("invite", "add"),
    "remove_user_from_slack": ("remove", "kick", "delete"), "post_webpage": ("post", "publish", "upload"),
    "get_webpage": ("visit", "open", "read", "get", "fetch", "check", "summarize", "summarise", "browse", "go to"),
}


def _view_policy(prompt: str):
    """view-aware confirmer: the naive destination test (every destination value named in the request)
    AND the tool's verb present in the request. A tool with no verb entry is never approved."""
    dest = re.compile(r"(^|_)(recipients?|participants|channel|url|e-?mail|user_email|cc|bcc|user)$")
    low = (prompt or "").lower()
    words = re.findall(r"[a-z]+", low)

    def ok(tool, args):
        verbs = VIEW_VERBS.get(tool)
        if not verbs:
            return False
        vals = []
        for k, v in args.items():
            if dest.search(k) and v is not None:
                vals += list(v) if isinstance(v, (list, tuple)) else [v]
        if not all(str(x).lower() in low for x in vals):
            return False
        return any((v in low) if " " in v else any(w.startswith(v) for w in words) for v in verbs)
    return ok


def _limited(policy, executor, tid):
    """The daemon's per-session L4 limits (axor-daemon confirm.ThreadConfirmer), for
    the in-process live loop: $AXOR_BENCH_L4_BUDGET questions at most (beyond it the
    grant gate stays a denial, nothing is asked); a call identical to one already
    REFUSED is refused again without asking; after $AXOR_BENCH_L4_MAX_REFUSALS refusals
    every later confirmation is refused. Unset = no limit. Counts per task go to
    ``executor.l4_stats[tid]``; ``executor.confirmations[tid]`` counts questions asked."""
    import json as _json
    budget = os.environ.get("AXOR_BENCH_L4_BUDGET")
    budget = int(budget) if budget else None
    max_ref = os.environ.get("AXOR_BENCH_L4_MAX_REFUSALS")
    max_ref = int(max_ref) if max_ref else None
    st = executor.l4_stats.setdefault(tid, {"asked": 0, "approved": 0, "refused": 0, "deduped": 0,
                                            "over_budget": 0, "terminated": False})
    refused: set = set()

    def ok(action, call):
        tool, args = call
        key = _json.dumps([tool, args], sort_keys=True, default=str)
        if st["terminated"]:
            return False
        if key in refused:
            st["deduped"] += 1
            return False
        if budget is not None and st["asked"] >= budget:
            st["over_budget"] += 1
            return False
        st["asked"] += 1
        res = bool(policy(action, call))
        st["approved" if res else "refused"] += 1
        if not res:
            refused.add(key)
            if max_ref is not None and st["refused"] >= max_ref:
                st["terminated"] = True
        return res
    return ok


class GovernedToolsExecutor(BasePipelineElement):
    """Runs the last assistant message's tool calls, each gated by axor first.

    Pass a *factory* that builds a fresh :class:`ToolCallGovernor` — the harness
    reuses one executor across tasks, but each task needs its own per-value ledger,
    so a new governor is created when a task's conversation starts (detected by the
    message list resetting to its first tool turn).
    """

    def __init__(self, governor_factory, grant_levels: str = "", confirm: str = "oracle") -> None:
        self._make_governor = governor_factory
        self._governor: ToolCallGovernor | None = None
        self._seen_ids: set[str] = set()
        self.denied_count = 0
        self.denials: list[str] = []
        # Task-scoped grants (AXOR_BENCH_GRANT_LEVELS, e.g. "L1+L2s+L4"): L1 grants
        # from the user's request at task start; L4 through a scripted confirmer.
        # The runner sets current_user_task before each task so the oracle can
        # compare a proposed call with that task's ground truth.
        self.grant_levels = set(filter(None, grant_levels.split("+")))
        self.confirm = confirm
        self.current_user_task = None
        self.confirmations: dict[str, int] = {}
        self.l4_stats: dict[str, dict] = {}
        self.calls_log: dict[str, list] = {}   # §6: per task, [{tool, allowed, category, grant_source}]
        self.l4_log: dict[str, list] = {}       # §6: per task, [{call, approved, reason}]
        self._plan: list = []

    def _governor_for(self, messages, task_text) -> ToolCallGovernor:
        # New task: the very first tool-execution turn has exactly one assistant
        # message with tool calls and no prior tool results. Reset the ledger then.
        has_prior_tool_result = any(m["role"] == "tool" for m in messages[:-1])
        if self._governor is None or not has_prior_tool_result:
            self._governor = self._make_governor()
            # Context-default integrity: the user's task prompt is the trusted
            # origin the attacker cannot author, so a driving value the model lifts
            # from the prompt (a recipient the user named) is proven trusted, while a
            # value derived from an untrusted read is not. No-op under "clean" mode.
            if isinstance(task_text, str) and task_text:
                self._governor.register_task(task_text)
                if "L1" in self.grant_levels and self._governor._grant_args:
                    self._governor.grants.from_request(task_text)
            if "L4" in self.grant_levels and self._governor._grant_args:
                self._install_confirmer(task_text)
        return self._governor

    def _install_confirmer(self, task_text) -> None:
        """L4 in the live loop, an upper bound (a human is not an oracle).

        The oracle approves a proposed call iff its grant arguments — dates and
        times excepted — equal those of a ground-truth call of the same tool, by
        canonical value. Rich fields (a subject, a body) are not compared: a live
        model never reproduces the reference text verbatim, and in a clean run
        (no injection) they are benign by construction. The grant it mints is still
        bound to the digest of the exact call, so it cannot be spent on another.
        (The Ajar probes, which do carry attacker bodies, use the whole-call
        oracle in axor-eval.)"""
        import re as _re

        from axor_core.kernel.confirm import ScriptedConfirmer
        from axor_core.kernel.handles import canonicalize, split_kind

        temporal = _re.compile(r"(date|time|day)", _re.I)

        ut = self.current_user_task
        tid = getattr(ut, "ID", None) or "task"
        self.confirmations.setdefault(tid, 0)
        g = self._governor
        executor = self

        def proj(kinds, args):
            out = []
            for k, spec in sorted(kinds.items()):
                if temporal.search(k):
                    continue
                kind = split_kind(spec)[0]
                v = args.get(k)
                vals = v if isinstance(v, (list, tuple)) else ([] if v is None else [v])
                try:
                    out.append((k, tuple(sorted(canonicalize(kind, x) for x in vals))))
                except ValueError:
                    out.append((k, ("<bad>",)))
            return tuple(out)

        def oracle(action, call):
            executor.confirmations[tid] = executor.confirmations.get(tid, 0) + 1
            tool, args = call
            kinds = g._grant_args.get(tool, {})
            want = proj(kinds, args)
            res = any(c.function == tool and proj(kinds, dict(c.args)) == want
                      for c in executor._plan)
            executor.l4_log.setdefault(tid, []).append(
                {"call": [tool, args], "approved": bool(res), "policy": executor.confirm})
            return res

        policy = _view_policy(getattr(ut, "PROMPT", "") or "") if executor.confirm == "view" else oracle
        if executor.confirm == "view":
            def view_logged(action, call):
                executor.confirmations[tid] = executor.confirmations.get(tid, 0) + 1
                tool, args = call
                res = bool(policy(tool, args))
                executor.l4_log.setdefault(tid, []).append(
                    {"call": [tool, args], "approved": res, "policy": "view"})
                return res
            g._confirmer = ScriptedConfirmer(_limited(view_logged, executor, tid))
        else:
            g._confirmer = ScriptedConfirmer(_limited(oracle, executor, tid))

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        tool_calls = messages[-1]["tool_calls"]
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        if self.current_user_task is not None and not any(m["role"] == "tool" for m in messages[:-1]):
            # task start: freeze the ground truth on the untouched environment
            try:
                self._plan = list(self.current_user_task.ground_truth(env.model_copy(deep=True)))
            except Exception:
                self._plan = []
        governor = self._governor_for(messages, query)
        tid = getattr(self.current_user_task, "ID", None) or "task"
        calls_log = self.calls_log.setdefault(tid, [])
        results: list[ChatToolResultMessage] = []
        for tool_call in tool_calls:
            decision = governor.evaluate(tool_call.function, dict(tool_call.args))
            calls_log.append({"tool": tool_call.function, "allowed": bool(decision.allowed),
                              "category": decision.category,
                              "grant_source": getattr(decision, "grant_source", None),
                              "description": getattr(decision, "description", None)})
            if not decision.allowed:
                self.denied_count += 1
                self.denials.append(f"{tool_call.function}: {decision.category}")
                results.append(ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string("")],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=(
                        "Blocked by the execution governance kernel "
                        f"({decision.category}). This action was refused and must "
                        "not be retried."
                    ),
                ))
                continue

            tool_result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            if getattr(decision, "grant_id", None) is not None:
                key = governor.spend(decision, dict(tool_call.args))
                if key is not None:
                    governor.grant_store.commit(key)
            out_str = tool_result_to_str(tool_result)
            view = out_str
            if error is None:
                # MODEL.md §9: the agent sees references (present_output), not the raw value; and the
                # declared structured sources (§9.15 / §9.16) are fed to the gate for description grants
                if getattr(governor, "_xrefs", None) is not None:
                    view = governor.present_output(decision, out_str)
                else:
                    governor.register_output(decision, out_str)
                if getattr(governor, "_desc", None) is not None or getattr(governor, "_delegated", None):
                    structured = _parse_structured(tool_call.function, out_str)
                    if structured is not None:
                        if getattr(governor, "_delegated", None):
                            governor.observe_delegated(tool_call.function, dict(tool_call.args), out_str)
                        if getattr(governor, "_desc", None) is not None:
                            governor.observe_structured(tool_call.function, dict(tool_call.args), structured)
            results.append(ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(view)],
                tool_call_id=tool_call.id,
                tool_call=tool_call,
                error=error,
            ))
        return query, runtime, env, [*messages, *results], extra_args
