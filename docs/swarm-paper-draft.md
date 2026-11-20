# Swarm paper — working draft

Status: **draft for discussion.** Separate from `paper-outline.md` (the single-agent
paper); nothing here depends on it. Every claim about the system is anchored to code or
a test; anything not yet built is marked **GAP** and listed in §6. Every external number
carries its source and its evidential status (§0.1).

---

## 0. Working title and thesis

**Title (working):** *Bounded Swarms: Evals Are Not Endorsements — Compositional
Authority, Provenance and Control for Recursive LLM Agent Systems*

**One-sentence thesis.** When a swarm replaces human review with evals, every eval
becomes an authority channel; a swarm is safe to "let cook" only if every node boundary
(a) checks results without endorsing them, (b) attenuates authority, (c) carries
provenance without laundering, and (d) admits runtime control that can only narrow — and
these compose over *any* agent graph (trees, peers, cycles, cross-federation edges), not
just trees.

### 0.1 Motivation and its sources (for the introduction)

- **The operating model (industry context, one sentence at most).** A venture essay
  describes "swarm-native" companies as ones that define reward functions and evals,
  remove functional and bureaucratic barriers, and let the swarms run; the author
  reports, anecdotally and for practitioners he knows, token spend 1,000–10,000× a median
  engineer's and ~100× the top 5% of Claude Code users (M. Huang, *Swarm Riders and
  Swarm-Native Companies*, Paradigm, 10 Nov 2026). **Status: anecdotal, no method — cite
  as context, never as evidence.** The essay does not discuss permissions, oversight,
  trust, provenance or failure modes.
- **The bridge.** The barriers removed are, in practice, human review. If review is
  removed, something must take its place at the boundaries between agents. The essay's
  core skill — turning non-verifiable problems into verifiable ones via evals — then puts
  evals on the authority path: an output that passes an eval is what downstream nodes act
  on. If passing an eval raises the output's trust, the eval is an endorsement oracle and
  an attacker inside the node optimises against it. The essay leaves this open; it is the
  question this paper answers (P1).
- **Evidence (citable).** Composition makes individually robust agents less secure: up
  to 3.8× harmful-task success for specialist star topologies vs a standalone agent at
  similar benign accuracy (Hagag et al., arXiv 2604.23459). The first by-construction
  multi-agent defense (Multi-CaMeL, arXiv 2610.05640) covers trees only and lists
  sub-agent result substitution as unchecked.
- **Optional footnote, not in "Why now".** A deployment report of a production swarm
  with real permissions and no pull-request review: J. Sansbury, *How I Built the Most
  AI-Pilled Company in the World* (post on X, Oct 2026,
  `x.com/jsnnsa/status/2115623382721126826`), reporting <10 people, ~77k changes in 30
  days, ~$3.1M/month in tokens. **Status: self-reported by the company; figures reached
  us through secondary summaries — verify against the post itself before citing, or
  drop.** Do not attribute these figures to the Paradigm essay (it only names Spawn as
  building a game platform).

---

## 1. Positioning (the table a reviewer will build anyway)

| | sub-agent result check | cross-agent provenance | authority ceiling on delegation | non-tree topologies | runtime control | formal | unmodified agents |
|---|---|---|---|---|---|---|---|
| Multi-CaMeL (2610.05640) | **no** (stated limitation) | yes (`variables` channel) | — | **no** (trees only, stated) | — | partial (CFI defs) | no (interpreter per agent) |
| ControlValve (2510.17276) | — (judges transitions, not outputs) | LLM-judged rule only | graph-level (which agent, in which order) | cycles in CFG | permit/reject/re-plan | **no** (stated) | yes |
| Architecture Matters (2604.23459) | — | — | — | star/chain/mesh studied | — | — | — (measurement) |
| Authorization Propagation (2605.05440) | — | — | required (R1–R7) | DAG model | discussed | model, no theorem | — (position) |
| **This work** | **label-neutral acceptance** | yes, labels as data | yes, at spawn | **delegation + lateral + peer + cycles** | **tighten-only, signed widening** | theorem over event order | yes (CLI agents via hooks) |

**The claimed delta, stated narrowly.** Not "labels cross boundaries" (Multi-CaMeL has
that). (i) Evals at node boundaries are treated as an authority channel and proved unable
to endorse; (ii) authority, provenance and control as monotone lattices whose
composition holds over arbitrary graphs; (iii) enforced in front of unmodified agents.
Concede up front: on trees with a pre-committed plan Multi-CaMeL's channel separation is
stronger against framing on the integrity axis; MultiAgentDojo is a comparison, not the
headline.

---

## 2. Threat model

- **Adversary-controlled content** may enter at any node through any read (web, MCP,
  repo file, issue, another agent's message).
- **Compromised node.** Inside its envelope, a node's model is assumed fully
  attacker-controlled: it may propose any call, spawn or message, lie in its answer, and
  **shape its output to pass the parent's eval**. The guarantee must hold for every model
  behaviour at every node.
- **Compromised control-plane backend** may replay, reorder or forge desired-state
  writes. It must not be able to widen (adapter enforces narrowing; signatures checked
  against operator keys from the adapter's own config — `axor_core/kernel/state.py`,
  control-plane protocol §3/§6).
- **Out of scope** (stated): compromised kernel/host process; implicit control-flow
  leaks; covert channels; availability (control may stop useful work); selection among
  trusted values; **semantic correctness of results** — an eval decides acceptance, the
  paper does not claim accepted results are correct.

---

## 3. System model

A run is a directed multigraph `G = (N, E)` that grows during execution.

- **Node** `n` = a `GovernedNode` (`axor_core/node/wrapper.py:72`) with state
  `σ_n = (A_n, L_n, B_n, Λ_n, D_n)`:
  - `A_n` authority — tool policy, export mode, depth (`ExecutionPolicy`);
  - `L_n` degradation level in `NORMAL < CAUTIOUS < RESTRICTED < LOCKED < TERMINAL`;
  - `B_n` budget — call cap, tokens;
  - `Λ_n` per-value ledger + context root (`axor_core/taint/engine.py`);
  - `D_n` desired state from the control plane (`stopped`, `paused`, caps, one-shot
    injection/excision).
- **Edges** (`axor_core/kernel/messaging.py:19-22`): `delegation` (spawn/return),
  `lateral` (intra-federation), `peer` (inter-federation). Unknown kind = deny.
- **Acceptance** at a delegation edge: `acc_r(out, R_parent) → {accept, reject}` declared
  by the child's role `r`, evaluated by the parent over records the parent owns.
- **Events** are totally ordered per node and causally ordered across nodes by
  send→receive and spawn→return. The proof is over this event order, not over graph
  shape — which is what makes cycles unproblematic.

Three lattices:
1. **Authority** `(𝒜, ⊑)` — meet = intersection of permissions; spawn takes a meet.
2. **Labels** `(𝒞, ⊑)` — causal roots, join = `CausalRoot.mint`; `cross_process_in` is
   the fail-closed root for unknown provenance.
3. **Control** `(L × B × D, ≤)` — level up, caps down, `stopped` absorbing.

Acceptance is deliberately **not** a lattice operation: it maps to `{accept, reject}`
and to control (P6), never to labels.

---

## 4. Properties and the theorem

**P1 — Evals are not endorsements (label-neutral acceptance). [central]**
For every role `r`, child output `out` and parent records `R`:
`root_parent(out) after acc_r = root_parent(out) before acc_r`, in both outcomes.
Acceptance is computed by the parent from records the child cannot write (the parent's
approved-call log, its own engine), and a reject maps only to a declared tighten-only
action (`retry_narrower | escalate | drop | degrade`, P6).
*Why it is central:* with endorsement-on-accept, the eval is an oracle — an attacker in
the child optimises output to pass it, and the accepted value enters the parent trusted.
The only label-lowering path in the kernel is `endorse_value`, which requires a
`GovernanceAuthority` and has no runtime caller (`axor_core/taint/engine.py:331`).
*Code today:* the structural findings check (`axor_core/node/findings.py`) is evaluated
in the parent (`wrapper.py:756-762`) and is already label-neutral — the checked answer is
registered with the child's root, "re-encoding launders nothing" (`wrapper.py:764-781`).
**GAP G2:** a pluggable `ChildRole.accept` with declared reject actions.
*Known limit of the findings check:* structural, not semantic — it decides that the child
opened the file it cites, not that the claim is true. P1 is exactly what keeps that
limit from becoming a trust bug.

**P2 — Authority attenuation at spawn.** `A_child ⊑ A_parent`, `depth_child ≤
max_depth_parent`, `export_child ⪰ export_parent`.
*Code:* `_validate_child_policy` (`axor_core/node/spawn.py`).
*Tests:* `tests/node/test_spawn_validation.py` (write/bash/spawn/extra tools/export/depth).

**P3 — Control inheritance at spawn (no spawn-laundering).** `L_child ≥ L_parent`.
*Code:* `inherit_degradation` (`axor_core/node/spawn.py`).
*Tests:* `test_child_inherits_parent_level_narrow_or_preserve`,
`test_child_engine_is_independent_after_spawn` (`tests/node/test_message_bus.py`).

**P4 — Label fidelity on edges.**
- Intra (`delegation`, `lateral`): the receiver registers `root_recv ⊒ root_sender(v)`;
  a missing root fails closed to `cross_process_in` (`fold_carried_root`).
- Peer: the carried root is a claim; the local root is re-derived through the trust
  ladder — L1 re-mints untrusted, L2 signed assertions discount but never yield clean,
  forged assertions fall to L0 (`node/messaging.py:_receive_peer`, `federation/ladder.py`).
- A received value is never added to the receiver's `Trusted` index.
*Tests:* `test_carried_not_laundered`, `test_clean_value_stays_clean_intra`,
`test_peer_inbound_l1_remints_despite_clean_claim`,
`test_peer_inbound_l2_signed_assertion_discounts_never_clean`,
`test_peer_inbound_forged_assertion_falls_to_l0_evidenced`.

**P5 — No laundering by topology.** Hop count, cycles, spawn and re-encoding do not lower
labels: the fold is a join, so a value returning to its origin carries the union
(`test_cycle_cannot_launder`); a child's returned output is registered in the parent with
the child's root (federation on) or re-minted untrusted (federation off)
(`wrapper.py:764-792`; `test_federation_on_restores_child_secret_provenance`,
`test_federation_off_remints_child_output_untrusted_nonsensitive`).

**P6 — Control monotonicity (automation only narrows).** Every automatic signal —
denials, trajectory observers, a stale/crashed child, (GAP) acceptance rejection,
(GAP) sibling circuit breaker, (GAP) subtree `level_floor` — can only raise `L`, lower
caps, or stop. Lowering `L` requires a `GovernanceAuthority` (governance clearance in
`degradation/engine.py`); the LOCKED TTL moves *up* to TERMINAL. The effective call cap is
`min(own, plane)` read at every check (`intent_loop.py:_call_cap`); `stopped` is absorbing
under merge (`kernel/state.py:DesiredState.merge`). Widening is reachable only through a
signed operator command or a host-held authority.
*Tests (existing):* `test_child_crash_is_a_fact_not_a_clean_return`,
`test_locked_sender_denied_and_traced`. *(GAP: a property test that no automatic path
lowers any control coordinate.)*

**Theorem (Compositional containment, informal).** For any execution over any graph `G`
(including cycles and peer edges), any model behaviour at every node, and any acceptance
functions, every effect executed at node `n` at event `t`:
1. is permitted by `⨅_{m ∈ path(root→n)} A_m` (P2);
2. was gated with a driving-argument label `⊒` the join of every untrusted/sensitive
   source that causally reached those arguments by `t` through tracked edges —
   **independently of any acceptance outcome on the way** (P1, P4, P5) — so the per-node
   perimeter guarantee (single-node K4) lifts to the graph;
3. was executed under a control state no wider than any automatic signal observed by `n`
   or its ancestors before `t` allows (P3, P6).

*Corollary (eval-gaming is useless for laundering).* An attacker who fully controls a
child and knows the parent's acceptance function gains, by passing it, no label the
child's output did not already carry; at most it gains acceptance — i.e. its output is
used under the same gates as before.

*Proof sketch.* Induction over the causal event order. Each event type (tool call, spawn,
return, acceptance, send, receive, control delta) updates authority by meet, labels by
join (acceptance: identity on labels) and control monotonically; each update is monotone
on its lattice; the gates are pure functions of (authority, label projection, control)
(`kernel/replay.evaluate_call`). Cycles add events, not fixpoint obligations, because the
fold is per-event and joins are idempotent. Pen-and-paper over a closed constructor set;
a Lean/Coq target, not mechanized.

**Conditions (where the theorem stops — say it in the paper).**
- C1 *Complete mediation* at every node: CLI agents are mediated by hooks; the hook
  channel is an in-band, same-uid socket (`axor_wrap/agents/sidecar.py`) — GAP G6. For
  P1 this matters directly: acceptance reads the approved-call log, which for CLI nodes
  comes from that channel.
- C2 *Label sources are complete*: an untracked read (e.g. a workspace file the sidecar
  does not classify as a source) enters clean — GAP G5.
- C3 *Integrity mode*: under `integrity_default="clean"` a paraphrased untrusted value
  enters clean; the integrity clause needs `"context"` (`contracts/taint.py:
  resolve_integrity_default`) — GAP G1.
- C4 *Envelope roots computed by the kernel*: `make_envelope` derives labels from the
  sender's engine, but `InMemoryMessageBus.send` accepts any envelope — GAP G3.

---

## 5. Implementation status

| Piece | Status | Where |
|---|---|---|
| Structural findings check, parent-side, label-neutral | done | `node/findings.py`, `wrapper.py:756-781` |
| Pluggable label-neutral acceptance + reject actions | **GAP G2 (minimal slice)** | — |
| Recursive `GovernedNode`, spawn ceiling, depth | done | `node/wrapper.py`, `node/spawn.py` |
| Degradation inheritance at spawn | done | `node/spawn.py:inherit_degradation` |
| Child output provenance (federation on/off) | done | `wrapper.py:764-792` |
| Crash = fact, parent → CAUTIOUS | done | `wrapper.py:696-726` |
| Message bus: intra fold, peer ladder, signatures, cycles | done (library) | `node/messaging.py`, `kernel/messaging.py` |
| Bus wired into the node as an intent | **GAP G4 (minimal slice)** | only tests + `axor-proxy/governed.py` demo |
| Signed desired state, cascade stop | done | `kernel/state.py`, `axor-backend/plane.py:180` |
| Subtree `level_floor`, subtree caps/pause | GAP G7 | — |
| Upward/sibling propagation, circuit breaker | GAP G8 | `axor_wrap/agents/tree.py` stages |
| Subtree budget pre-allocation | GAP G9 | child tokens recorded after return (`wrapper.py:815-837`) |
| CLI-agent trees (Claude Code, Codex) | done, depth 1 | `axor_wrap/agents/tree.py` |
| Native harness sub-agents inside a leaf | GAP G10 | `delegate` gated; `allow_spawn` not passed to sidecar |

---

## 6. Gaps to close before evaluation

**Minimal slice (needed for the paper's central claims): G1, G2, G3, G4.**

- **G1** Trees default to `integrity_default="context"` (closes C3).
- **G2** `ChildRole.accept`: acceptance evaluated in the parent over parent-owned records;
  declared reject actions (`retry_narrower | escalate | drop | degrade`), all tighten-only;
  trace `CHILD_ACCEPTED` / `CHILD_REJECTED`. Negative controls: (a) an
  endorse-on-accept variant lets a gamed output reach an egress sink; (b) the shipped
  variant denies it. Estimated +1–2 weeks over the earlier slice.
- **G3** The bus computes the envelope root from the registered sender engine; a
  caller-supplied root is ignored or must be `⊒` it.
- **G4** `send_message` as a node intent through the same gate pipeline.

After the slice:
- **G5** Sidecar classifies untracked workspace reads as a source.
- **G6** Hook-channel authentication (per-node token not in the model's environment) — or
  state it as a condition for write-capable CLI nodes.
- **G7** `DesiredState.level_floor` + subtree scoping, narrowing-only at the adapter.
- **G8** Child level ≥ RESTRICTED or rejection is a fact at the parent; k-of-n sibling
  breaker per stage.
- **G9** Subtree budget slices allocated at spawn, enforced in the child, remainder
  returned.
- **G10** `AXOR_SPAWN` to the sidecar; `swarm: none | opaque | governed` per leaf.

Each G ships with an adversarial test that fails on today's code.

---

## 7. Evaluation design

**RQ1 — Eval gaming (the central experiment).** A compromised child knows the parent's
acceptance function and optimises its output to pass it while carrying an
attacker-derived value toward a parent sink. Acceptance functions: structural findings
check; test-suite pass; schema check; LLM-judge. Arms: (a) endorse-on-accept (the naive
swarm-rider design), (b) P1. Metrics: acceptance rate of gamed outputs (expected high in
both — P1 does not claim to stop gaming), **laundering rate** (gamed value reaches a sink
gated clean; expected >0 in (a), 0 in (b)), utility cost of (b) vs (a) on benign tasks.

**RQ2 — Containment beyond trees.** New suite (*TopoDojo*, name TBD) on AgentDojo
environments, under topologies Multi-CaMeL excludes: sibling infection over a lateral
edge; cycles with re-encoding at each hop; peer edge across federations (declared L1/L2,
forged assertion); spawn-laundering attempt; mesh with a shared scratchpad. Metrics: ASR
and laundering rate per hop count `k = 1..4`; over-taint rate; utility cost.

**RQ3 — Containment on trees, against the closest prior.** MultiAgentDojo (use it if
released, otherwise port the task/injection definitions from 2610.05640). ASR and utility
cost vs own undefended baseline, same backbone where possible; Multi-CaMeL numbers as
reported, version-pinned. Expect to be behind on utility on shared-channel tasks — report
it once, plainly.

**RQ4 — Adaptive attacks on the remaining properties.** Paraphrase across hops
(P4/C3); branch steering inside a permitted plan; forged hook reports (C1, which also
attacks acceptance inputs); control-plane replay and widening attempts (P6); budget
exhaustion to deny siblings.

**RQ5 — Runtime control.** Mid-run compromise; time-to-contain, blast radius (nodes and
effects after the first malicious act), collateral (benign work stopped) for: none,
cascade stop, rejection→degrade, `level_floor`, sibling breaker.

**RQ6 — Real swarms.** Case study with `axor-wrap` trees of Claude Code and Codex on real
repositories: poisoned README/issue read by a reader node, test-suite acceptance on the
implementer; containment, overhead, rejections, operator interventions.

**Ablation = negative-control matrix.** Remove one property at a time (endorse-on-accept;
no carried labels; no spawn ceiling; no degradation inheritance; automatic widening
allowed) and show the corresponding attack succeeds.

**Baselines.** Undefended; per-agent axor without carried labels; endorse-on-accept
acceptance; Multi-CaMeL (reported); ControlValve-style LLM judge (minimal
re-implementation, or reported).

---

## 8. Paper skeleton (12–13 pp. body)

1. Introduction — review is replaced by evals; evals become an authority channel;
   composition breaks single-agent defenses; contributions (≈1.25 pp)
2. Motivating example — an implementer node games the test-suite eval to carry an
   attacker value into a deploy step (≈0.75 pp)
3. Threat model (≈0.75 pp)
4. System model, three lattices, acceptance (≈1.5 pp)
5. Properties, theorem, eval-gaming corollary, conditions (≈2.5 pp)
6. Design and implementation — acceptance, kernel, bus, control plane, CLI-agent
   mediation (≈1.5 pp)
7. Evaluation RQ1–RQ6 + ablation (≈3.5 pp)
8. Related work (≈0.75 pp)
9. Limitations (≈0.5 pp)
Appendix: proof, suite definitions, per-task tables.

---

## 9. Risks

- **"P1 is trivial."** A reviewer may say "just don't endorse." Answer with RQ1: the
  naive design is what swarm practice converges to (accepted output is acted on as
  trusted), and the measured laundering rate under it is the contribution's evidence. If
  RQ1(a) shows little laundering, the central claim weakens — run RQ1 first.
- **Concurrent work.** Multi-CaMeL is the obvious comparison; the answer is the
  acceptance/topology/authority/control axis, not utility.
- **Utility.** Content-derivation taint over-blocks on shared-channel tasks; RQ3 may be
  lost. Do not make it the headline.
- **Benchmark credibility.** Release TopoDojo and the RQ1 harness, include tasks where we
  fail, run tree baselines on it where they apply.
- **Condition C1 for CLI agents.** If G6 is not closed, RQ6 must state that write-capable
  CLI nodes are mediated by a same-uid channel, which also feeds acceptance.
- **Sources.** No figure in "Why now" may come from a secondary summary (§0.1).

## 10. Work order

Minimal slice G1–G4 → RQ1 harness and a first RQ1(a)/(b) run (go/no-go for the framing)
→ G7, G8 → TopoDojo + adaptive attacks → MultiAgentDojo port → RQ6 case study → G9,
G10, G5, G6 as the evaluation demands.
