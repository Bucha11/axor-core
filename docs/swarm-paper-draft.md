# Swarm paper — working draft

Status: **draft for discussion.** Separate from `paper-outline.md` (the single-agent
paper); nothing here depends on it. Every claim about the system is anchored to code or
a test; anything not yet built is marked **GAP** and listed in §6.

---

## 0. Working title and thesis

**Title (working):** *Bounded Swarms: Compositional Authority, Provenance and Control for
Recursive LLM Agent Systems*

**One-sentence thesis.** A swarm is safe to "let cook" only if every node boundary
(a) attenuates authority, (b) carries provenance without laundering, and (c) admits
runtime control that can only narrow — and these three compose over *any* agent graph
(trees, peers, cycles, cross-federation edges), not just trees.

**Why now.** Swarms run in production: a <10-person team reports a permanent agent fleet
with real permissions, no pull requests, ~77k changes/30 days, ~$3M/month in tokens
(Spawn, self-reported). Composition makes individually robust agents *less* secure (up
to 3.8× attack success, Hagag et al. 2604.23459). The first by-construction multi-agent
defense (Multi-CaMeL, 2610.05640) covers trees only and leaves authority, budgets,
runtime control and sub-agent result substitution open.

---

## 1. Positioning (the table a reviewer will build anyway)

| | cross-agent provenance | authority ceiling on delegation | non-tree topologies | runtime control | sub-agent result check | formal | unmodified agents |
|---|---|---|---|---|---|---|---|
| Multi-CaMeL (2610.05640) | yes (`variables` channel) | — | **no** (trees only, stated) | — | **no** (stated limitation) | partial (CFI defs) | no (interpreter per agent) |
| ControlValve (2510.17276) | LLM-judged rule only | graph-level (which agent, in which order) | cycles in CFG | permit/reject/re-plan | — | **no** (stated) | yes |
| Architecture Matters (2604.23459) | — | — | star/chain/mesh studied | — | — | — | — (measurement only) |
| Authorization Propagation (2605.05440) | — | required (R1–R7) | DAG model | discussed | — | model, no theorem | — (position) |
| **This work** | yes, labels as data | yes, at spawn | **delegation + lateral + peer + cycles** | **tighten-only, signed widening** | label-neutral acceptance | theorem over event order | yes (CLI agents via hooks) |

**The claimed delta, stated narrowly:** not "labels cross boundaries" (Multi-CaMeL has
that) but the *conjunction* — authority, provenance and control as three monotone
lattices whose composition holds over arbitrary graphs, enforced in front of unmodified
agents. Concede up front: on trees with a pre-committed plan Multi-CaMeL's channel
separation is stronger against framing on the integrity axis; our results on
MultiAgentDojo are a comparison, not the headline.

---

## 2. Threat model

- **Adversary-controlled content** may enter at any node through any read (web, MCP,
  repo file, issue, another agent's message).
- **Compromised node.** Inside its envelope, a node's model is assumed fully
  attacker-controlled: it may propose any call, any spawn, any message, and lie in its
  answer. The guarantee must hold for *every* model behaviour at every node.
- **Compromised control-plane backend** may replay, reorder or forge desired-state
  writes. It must not be able to widen (adapter enforces narrowing; signatures checked
  against operator keys from the adapter's own config — `axor_core/kernel/state.py`,
  control-plane protocol §3/§6).
- **Out of scope** (stated, not hidden): compromised kernel/host process; implicit
  control-flow leaks; covert channels (timing, token counts); availability (control may
  stop useful work); selection among trusted values; semantic correctness of results
  (acceptance checks are structural).

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
  `lateral` (intra-federation sibling/any), `peer` (inter-federation). Unknown kind = deny.
- **Events** are totally ordered per node and causally ordered across nodes by
  send→receive and spawn→return. The proof is over this event order, not over graph
  shape — which is what makes cycles unproblematic.

Three lattices:
1. **Authority** `(𝒜, ⊑)` — meet = intersection of permissions; spawn takes a meet.
2. **Labels** `(𝒞, ⊑)` — causal roots, join = `CausalRoot.mint`; `cross_process_in` is
   the top of the integrity axis for unknown provenance.
3. **Control** `(L × B × D, ≤)` — level up, caps down, `stopped` absorbing.

---

## 4. Properties and the theorem

**P1 — Authority attenuation at spawn.** `A_child ⊑ A_parent`, `depth_child ≤
max_depth_parent`, `export_child ⪰ export_parent`.
*Code:* `_validate_child_policy` (`axor_core/node/spawn.py`).
*Tests:* `tests/node/test_spawn_validation.py` (write/bash/spawn/extra tools/export/depth).

**P2 — Control inheritance at spawn (no spawn-laundering).** `L_child ≥ L_parent`.
*Code:* `inherit_degradation` (`axor_core/node/spawn.py`).
*Tests:* `test_child_inherits_parent_level_narrow_or_preserve`,
`test_child_engine_is_independent_after_spawn` (`tests/node/test_message_bus.py`).

**P3 — Label fidelity on edges.**
- Intra (`delegation`, `lateral`): the receiver registers `root_recv ⊒ root_sender(v)`;
  a missing root fails closed to `cross_process_in` (`fold_carried_root`).
- Peer: the carried root is a *claim*; the local root is re-derived through the trust
  ladder — L1 re-mints untrusted, L2 signed assertions discount but never yield clean,
  forged assertions fall to L0 (`node/messaging.py:_receive_peer`, `federation/ladder.py`).
- A received value is never added to the receiver's `Trusted` index.
*Tests:* `test_carried_not_laundered`, `test_clean_value_stays_clean_intra`,
`test_peer_inbound_l1_remints_despite_clean_claim`,
`test_peer_inbound_l2_signed_assertion_discounts_never_clean`,
`test_peer_inbound_forged_assertion_falls_to_l0_evidenced`.

**P4 — No laundering by topology.** Hop count, cycles, spawn, and re-encoding do not lower
labels: the fold is a join, so a value returning to its origin carries the union
(`test_cycle_cannot_launder`); a child's returned output is registered in the parent with
the child's root (federation on) or re-minted untrusted (federation off)
(`wrapper.py:764-792`; `test_federation_on_restores_child_secret_provenance`,
`test_federation_off_remints_child_output_untrusted_nonsensitive`); a checked findings
answer carries the same root as the raw answer (`wrapper.py:764-767`).

**P5 — Control monotonicity (automation only narrows).** Every automatic signal —
denials, trajectory observers, a stale/crashed child, (GAP) eval rejection, (GAP) sibling
circuit breaker, (GAP) subtree `level_floor` — can only raise `L`, lower caps, or stop.
Lowering `L` requires a `GovernanceAuthority` (`degradation/engine.py`, governance
clearance); the LOCKED TTL moves *up* to TERMINAL. The effective call cap is
`min(own, plane)` read at every check (`intent_loop.py:_call_cap`); `stopped` is absorbing
under merge (`kernel/state.py:DesiredState.merge`). Widening is reachable only through a
signed operator command or a host-held authority.
*Tests (existing):* `test_child_crash_is_a_fact_not_a_clean_return`,
`test_locked_sender_denied_and_traced`. *(GAP: a property test that no automatic path
lowers any control coordinate.)*

**P6 — Label-neutral acceptance (GAP: `ChildRole.accept`).** A parent-side acceptance
check `acc(out, R_parent) ∈ {accept, reject}` is computed only from records the parent
owns (approved calls, its own engine), never changes `root(out)`, and on reject triggers
a declared tighten-only action (`retry_narrower | escalate | drop | degrade`). Today only
the structural `findings` check exists (`axor_core/node/findings.py`) and it is already
label-neutral.

**Theorem (Compositional containment, informal).** For any execution over any graph `G`
(including cycles and peer edges) and any model behaviour at every node, every effect
executed at node `n` at event `t`:
1. is permitted by `⨅_{m ∈ path(root→n)} A_m` (P1);
2. was gated with a driving-argument label `⊒` the join of every untrusted/sensitive
   source that causally reached those arguments by event `t` through tracked edges
   (P3, P4), so the per-node perimeter guarantee (single-node K4) lifts to the graph;
3. was executed under a control state no wider than any automatic signal observed by
   `n` or its ancestors before `t` allows (P2, P5);
and no acceptance outcome changed any label (P6).

*Proof sketch.* Induction over the causal event order. Each event type (tool call,
spawn, return, send, receive, control delta) updates authority by meet, labels by join
and control monotonically; each update function is monotone on its lattice; the gates
are pure functions of (authority, label projection, control) (`kernel/replay.evaluate_call`).
Cycles add events, not fixpoint obligations, because the fold is per-event and joins are
idempotent. Pen-and-paper, closed constructor set; a Lean/Coq target, not mechanized.

**Conditions (where the theorem stops — say it in the paper).**
- C1 *Complete mediation* at every node: CLI agents are mediated by hooks; a hook channel
  is an in-band, same-uid socket (`axor_wrap/agents/sidecar.py`) — see GAP G6.
- C2 *Label sources are complete*: an untracked read (e.g. a workspace file the sidecar
  does not classify as a source) enters clean — GAP G5.
- C3 *Integrity mode*: under `integrity_default="clean"` a paraphrased untrusted value
  enters clean; the theorem's integrity clause needs `"context"` (`contracts/taint.py:
  resolve_integrity_default`) — GAP G1.
- C4 *Envelope roots computed by the kernel*: `make_envelope` derives labels from the
  sender's engine, but `InMemoryMessageBus.send` accepts any envelope — GAP G3.

---

## 5. Implementation status

| Piece | Status | Where |
|---|---|---|
| Recursive `GovernedNode`, spawn ceiling, depth | done | `node/wrapper.py`, `node/spawn.py` |
| Degradation inheritance at spawn | done | `node/spawn.py:inherit_degradation` |
| Child output provenance (federation on/off) | done | `wrapper.py:764-792` |
| Crash = fact, parent → CAUTIOUS | done | `wrapper.py:696-726` |
| Message bus: intra fold, peer ladder, signatures, cycles | done (library) | `node/messaging.py`, `kernel/messaging.py` |
| Bus wired into the node as an intent | **GAP G4** | only tests + `axor-proxy/governed.py` demo |
| Signed desired state, cascade stop | done | `kernel/state.py`, `axor-backend/plane.py:180` |
| Subtree `level_floor`, subtree caps/pause | **GAP G7** | — |
| Structural findings check | done | `node/findings.py` |
| Pluggable label-neutral acceptance | **GAP G2** | — |
| Upward/sibling propagation, circuit breaker | **GAP G8** | `axor_wrap/agents/tree.py` stages |
| Subtree budget pre-allocation | **GAP G9** | child tokens recorded after return (`wrapper.py:815-837`) |
| CLI-agent trees (Claude Code, Codex) | done, depth 1 | `axor_wrap/agents/tree.py` |
| Native harness sub-agents inside a leaf | **GAP G10** | `delegate` gated; `allow_spawn` not passed to sidecar |

---

## 6. Gaps to close before evaluation (ordered)

- **G1** Trees default to `integrity_default="context"` (closes C3).
- **G2** `ChildRole.accept` + declared reject actions; trace `CHILD_REJECTED`.
- **G3** The bus computes the envelope root from the registered sender engine; a
  caller-supplied root is ignored or must be `⊒` it.
- **G4** `send_message` as a node intent through the same gate pipeline.
- **G5** Sidecar classifies untracked workspace reads (at least git-untracked/ignored
  files and fetched content written to disk) as a source.
- **G6** Hook-channel authentication (per-node token not in the model's environment) — or
  state it as a condition for write-capable CLI nodes.
- **G7** `DesiredState.level_floor` + subtree scoping, narrowing-only at the adapter.
- **G8** Child level ≥ RESTRICTED or rejection is a fact at the parent; k-of-n sibling
  breaker per stage.
- **G9** Subtree budget slices allocated at spawn, enforced in the child, remainder
  returned.
- **G10** `AXOR_SPAWN` to the sidecar; `swarm: none | opaque | governed` per leaf.

Each G ships with an adversarial test that fails on today's code (negative control).

---

## 7. Evaluation design

**RQ1 — Containment on trees, against the closest prior.** MultiAgentDojo (Multi-CaMeL's
suite; use it if released, otherwise port their task/injection definitions). Report ASR
and utility cost vs own undefended baseline, same backbone where possible; Multi-CaMeL
numbers as reported, version-pinned. Expect to be behind on utility on shared-channel
tasks — report it once, plainly.

**RQ2 — Containment beyond trees (the headline axis).** New suite, *TopoDojo* (name TBD),
built on AgentDojo environments, same tasks under topologies Multi-CaMeL excludes:
- sibling infection over a lateral edge (A reads injection → messages B → B acts);
- cycles (A→B→A with re-encoding at each hop);
- peer edge across federations (declared L1/L2, forged assertion);
- spawn-laundering attempt (tainted parent spawns a "clean" child to act);
- mesh with a shared scratchpad.
Metrics: ASR per hop count `k = 1..4`; *laundering rate* (fraction of effects whose
driving value was attacker-derived but gated clean); over-taint rate; utility cost.

**RQ3 — Adaptive attacks.** For each property, an attacker that targets it:
paraphrase across hops (P3/C3); branch steering inside a permitted plan; result
substitution against acceptance (P6); forged hook reports (C1); control-plane replay and
widening attempts (P5); budget exhaustion to deny siblings.

**RQ4 — Runtime control.** Inject mid-run compromise; measure time-to-contain, blast
radius (nodes and effects after the first malicious act), and collateral (benign work
stopped) for: none, cascade stop, `level_floor`, sibling breaker.

**RQ5 — Real swarms.** Case study with `axor-wrap` trees of Claude Code and Codex on real
repositories: poisoned README/issue read by a reader node; measure containment, token and
latency overhead, acceptance rejections, operator interventions.

**Ablation = negative-control matrix.** Remove one property at a time (no carried labels;
no spawn ceiling; no degradation inheritance; acceptance allowed to endorse; automatic
widening allowed) and show the corresponding attack succeeds. This is what makes each
condition load-bearing rather than asserted.

**Baselines.** Undefended; per-agent axor without carried labels (the "individually
robust, composed insecure" baseline); Multi-CaMeL (reported); ControlValve-style LLM judge
(re-implemented minimally, or reported).

---

## 8. Paper skeleton (12–13 pp. body)

1. Introduction — swarms in production; composition breaks single-agent defenses;
   contributions (≈1.25 pp)
2. Motivating example — sibling infection in a coding swarm (≈0.75 pp)
3. Threat model (≈0.75 pp)
4. System model and the three lattices (≈1.5 pp)
5. Properties, theorem, conditions (≈2.5 pp)
6. Design and implementation — kernel, bus, control plane, CLI-agent mediation (≈1.5 pp)
7. Evaluation RQ1–RQ5 + ablation (≈3.5 pp)
8. Related work (≈0.75 pp)
9. Limitations (≈0.5 pp)
Appendix: proof, suite definitions, per-task tables.

---

## 9. Risks

- **Concurrent work.** Multi-CaMeL is the obvious reviewer comparison; our answer is the
  topology/authority/control axis, not utility.
- **Utility.** Content-derivation taint over-blocks on shared-channel tasks; on RQ1 we may
  lose. Do not make RQ1 the headline.
- **Benchmark credibility.** A self-built TopoDojo invites "designed to win"; release it,
  include tasks where we fail, and run Multi-CaMeL-style tree baselines on it where they
  apply.
- **Condition C1 for CLI agents.** If G6 is not closed, RQ5 must state that write-capable
  CLI nodes are mediated by a same-uid channel.
- **Scope creep.** Budgets and control are valuable but secondary; if pages are tight,
  keep RQ4 short and push detail to the appendix.

## 10. Work order

G1, G3, G4 (labels sound on all edges) → G2, G7, G8 (acceptance + control) → TopoDojo +
adaptive attacks → MultiAgentDojo port → RQ5 case study → G9, G10, G5, G6 as the
evaluation demands.
