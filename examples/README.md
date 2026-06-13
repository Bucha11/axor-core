# Examples

Runnable examples that exercise the kernel. They are intentionally minimal and
dependency-free — not production adapters (the full Claude adapter lives in
`axor-claude`). Organised by topic:

| Directory | What it shows |
|---|---|
| [`config/`](config/) | The declarative `governance.yaml` — a fully-commented template for the whole data-flow taxonomy (sources, sinks, allowlists, driving args, consequence overrides, A2A federation). |
| [`live_demo/`](live_demo/) | Live end-to-end governance of a **real** Claude model: the confidentiality floor denies an egress after a secret read. Plus `anthropic_executor.py`, a minimal raw-`urllib` `Invokable` you can reuse as a provider adapter. |
| [`agentdojo/`](agentdojo/) | The [AgentDojo](https://github.com/ethz-spylab/agentdojo) prompt-injection benchmark: undefended vs governed, two LLM backends, and the honest results (`agentdojo_results.md`). |
| [`attacks/`](attacks/) | Two attack experiments that stress the taint mechanism — nested-document (`nnsi_experiment.py`) and the split-document judge-bypass (`split_doc_experiment.py`), each with an honest results writeup. |
| [`openclaw_degradation/`](openclaw_degradation/) | The OpenClaw incident replayed through the real governor: the consequence axis catches the un-denied restart/shutdown; honest residuals kept visible. |
| [`trajectory/`](trajectory/) | The stateful-trajectory extension point — a `TrajectoryObserver` (stove-on-too-long) that tightens degradation from session history + tool results. |

## Running

Each script runs as a module from the repo root, e.g.:

```
export ANTHROPIC_API_KEY=sk-ant-...
python -m examples.live_demo.live_governance_demo

pip install agentdojo
AXOR_BENCH_BACKEND=openrouter python -m examples.agentdojo.run_agentdojo
python -m examples.attacks.nnsi_experiment
python examples/trajectory/demo.py
```

Each subdirectory has its own README or a results `.md` with the full story, the
exact commands, and an honest read of what the run does and does not show.
