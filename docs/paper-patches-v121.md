# Патчи к Axor_121.pdf

Каждый патч — одна операция над текстом статьи. Якоря (`FIND`) взяты дословно из PDF
(переносы строк и дефисы переноса сняты) — ищите по ним в исходнике. Формат:

```
<<<<<<< FIND            — точный фрагмент, который ищем (уникален в тексте)
=======
>>>>>>> REPLACE         — чем заменить
```

Для вставок: `INSERT AFTER` / `INSERT BEFORE` — якорь остаётся, новый текст ставится
сразу после/перед ним; границы вставки — `>>> BEGIN INSERT` … `<<< END INSERT`.
Для рисунков — только правка подписей/надписей, сам рисунок перерисовать по указанию.

Что меняется по существу (и только это):

1. **T4 доходит до потребителя** — контракты потребления, подготовка аргументов до
   первой проверки, исполнение `decision.args` (§1, §4.3, §5.4, App. A/B).
2. **Утверждение — про исполнение**, K4 — лемма внутри него (§5.1).
3. **Спецификация разрешающих оснований и матрица отрицательных контролей** (новый
   §5.8, новое App. F).
4. **Точность формулировок**: убрать «discharged by fuzzing», «proved/proven» для
   floor, «shared with FIDES»; supersession — замена основания (везде по тексту).
5. **ROPE — один абзац в Related work** (§7), без расширения §6.3.
6. **Ограничения**: предпосылка соответствия обработчика контракту, record closure,
   полнота таблицы оснований (§8).

Числа в §6 не меняются: перемаркировка origin-плеча AgentDojo не изменила
аргументы governor'а (проверено побайтно), а латентность меряется на той же
конфигурации (уточняется только её название, P-27).

---

## Abstract

### P-01 · Abstract — условие гарантии

<<<<<<< FIND
The guarantee is conditional and the conditions are stated: it holds outright on the decidable enum/numeric fraction of the perimeter and is discharged by fuzzing on the rich-syntax residual.
=======
The guarantee is conditional and the conditions are stated: on the enum/numeric fraction of the perimeter it is decided by a procedure, provided each guarded field's consumption is registered — every call is put into that registration before it is checked, so the value checked is the value executed; on the rich-syntax residual fuzzing gives evidence, not a discharge.
>>>>>>> REPLACE

---

## 1 Introduction

### P-02 · §1 «Why this is not free» — граница внутри периметра

<<<<<<< FIND
Perimeter non-interference — an adaptation of classical noninterference [7] to the tool-call perimeter — is proved outright on the decidable fraction by a decision procedure, the residual is named and localized to a rich-syntax fraction where the discharge is fuzzing, and the boundary is enforced at registration: a configuration that guards a rich-syntax field with a decidable predicate is rejected fail-closed, so a fuzzing obligation cannot be silently laundered into a decided one. The guarantee is conditional and its conditions are stated; that split, not the equation, is the technical core of this paper.
=======
Perimeter non-interference — an adaptation of classical noninterference [7] to the tool-call perimeter — is a lemma inside a claim about execution: every executed call satisfies every constraint the operator stated, for any behaviour of the model. That claim needs two conditions beyond the equation. Every input that can permit a call must come from a trusted mechanism, not from the model (T0). And the property a gate checks must be the property the tool consumes (T4): an exact `enum` check says nothing about a handler that defaults an absent argument, reads an alias, or resolves a path. Axor registers each guarded field's consumption in a contract, rejects at registration a predicate the field's consumer makes unfaithful, and puts every call into the contract before the first gate, so the checked arguments are the executed ones. On the enum/numeric fraction this is decided by a procedure; the rich-syntax residual is named and fuzz-tested, which is evidence, not a discharge. The technical core of this paper is these conditions and the evidence that each is load-bearing, not the equation.
>>>>>>> REPLACE

### P-03 · §1 «Results» — одна фраза про отрицательные контроли

INSERT AFTER:
```
the defense is a property of the projection, not of the model.
```
>>> BEGIN INSERT
 Offline, each condition of the guarantee is shown load-bearing in this construction: removing it alone lets a real handler produce the forbidden effect, restoring it removes the effect, and the legitimate call still executes (§5.8).
<<< END INSERT

### P-04 · §1 Contributions — пункт 1

<<<<<<< FIND
the gate's verdict is invariant under reframing of the injection, proved outright on the enum/numeric fraction, with the rich-syntax residual localized, discharged by fuzzing, and the boundary enforced at registration.
=======
the gate's verdict is invariant under reframing of the injection, and every executed call satisfies the stated constraints under two named conditions — permitting grounds from trusted mechanisms (T0) and consumption registered per field (T4) — decided on the enum/numeric fraction, with the rich-syntax residual localized and fuzz-tested, the boundary enforced at registration against the field rather than the predicate kind, and each condition shown load-bearing by a measured negative control.
>>>>>>> REPLACE

---

## 3 Threat model

### P-05 · §3 Non-goals — FIDES

<<<<<<< FIND
implicit control-flow leaks (explicit flow only, a boundary shared with FIDES);
=======
implicit control-flow leaks (explicit flow only — a boundary of ours: FIDES closes it on the integrity axis by checking the trusted context of the call);
>>>>>>> REPLACE

---

## 4 System architecture

### P-06 · §4.3 — один набор аргументов от проверки до исполнения

INSERT AFTER:
```
The anti-drift guarantee is about the shared six.
```
>>> BEGIN INSERT

One argument set runs from check to execution. Before the first gate, a call to a sink with a consumption contract (§5.4) is put into the contract's representation: undeclared fields refused, declared defaults materialised, the declared numeric conversion applied once. That prepared set is what every gate reads, what a call-bound approval is digested over, and what the tool receives: the streaming loop dispatches it, and `ToolCallGovernor` returns it as `decision.args` for the host to execute — the AgentDojo shim and the drop-in wrapper do, and refuse to fall back to the raw proposal when a contract is configured on a kernel that cannot return it. A host that runs a denied call anyway (an observe-only arm) records the executed arguments apart from the checked ones.
<<< END INSERT

### P-07 · Figure 1 — надписи на рисунке

Перерисовать две надписи и добавить шаг 0:

| Было | Стало |
|---|---|
| `5 ssrf / internal-destination … shared core · rich-syntax, fuzz-discharged` | `… shared core · rich-syntax, fuzz-tested` |
| `7 carrier … shared core · rich-syntax, fuzz-discharged` | `… shared core · rich-syntax, fuzz-tested` |
| — | новая строка над «1 capability»: `0 prepare — consumption contract (§5.4) · the set every gate reads and the tool receives` |

---

## 5 Security analysis

### P-08 · §5.1 — заголовок

<<<<<<< FIND
5.1 The Perimeter Non-Interference theorem
=======
5.1 The claim, and the non-interference lemma inside it
>>>>>>> REPLACE

### P-09 · §5.1 — утверждение об исполнении (в начало подраздела)

INSERT BEFORE:
```
The invariant is stated over the raw input x, not over π, otherwise it is a tautology:
```
>>> BEGIN INSERT
The claim is about execution. For every model behaviour M,

  Execute(s, c) ⟹ ∧_{j ∈ J(p)} P_j(s, c),

where each P_j is a constraint the operator's policy p states independently of the others: the tool is permitted, the destination is allowlisted, the amount is in range, no declared egress while a secret read is outstanding, no unattended call above the consequence ceiling. The formula is the shape of the obligation, not a new fact; its content is in the conditions under which it holds. A call reaches an effect through five links — ground, projection, consumption, dispatch, effect — and each has an owner: T0 for the ground (§5.4, App. F), the lemma below and O1–O3 for the projection, T4 for consumption (§5.4), the single prepared argument set for dispatch (§4.3), and independent gates per P_j for the conjunction (§5.5). Untrusted bytes may change a verdict — an out-of-range amount is denied because of what the model wrote; what they must never be is an unchecked ground for widening authority.

The lemma that makes the verdict framing-invariant is perimeter non-interference.
<<< END INSERT

### P-10 · §5.1 — о чём лемма не говорит

INSERT AFTER:
```
admissible exactly when it meets the two conditions of §5.4, a non-interpreting producer and codomain-fidelity.
```
>>> BEGIN INSERT

What the lemma buys is framing-invariance of the verdict. It does not say whether the verdict's subject is the call that runs, whether its permitting inputs were legitimately produced, or whether the checked property is the consumed one; those are the conditions of §5.4 and the dispatch rule of §4.3, and they are where the claim about execution is made.
<<< END INSERT

### P-11 · §5.3 O2 — FIDES

<<<<<<< FIND
only explicit flow is covered, implicit flow is out of scope (a boundary shared with FIDES).
=======
only explicit flow is covered, implicit flow is out of scope (a boundary of ours; FIDES closes it on the integrity axis).
>>>>>>> REPLACE

### P-12 · §5.4 — вступительный абзац (без «proved / discharged»)

<<<<<<< FIND
We do a third thing: we draw a line inside the perimeter, between the part where the guarantee is proved and the part where it degenerates into weird machines and is discharged by fuzzing.
=======
We do a third thing: we draw a line inside the perimeter, between the part where the guarantee is decided by a procedure — given a registered consumer — and the part where it degenerates into weird machines and fuzzing is the only evidence.
>>>>>>> REPLACE

### P-13 · §5.4 — первая группа codomain-fidelity зависит от потребителя

<<<<<<< FIND
The first comprises finite enums and bounded-numeric ranges: two equal projections cannot produce different effects, and this is discharged by a decision procedure (kernel/decidability.py).
=======
The first comprises finite enums and bounded-numeric ranges: if the consumer takes the value as a case split or a number, two equal projections cannot produce different effects, and the check is a decision procedure (kernel/decidability.py). The condition is about the consumer, not the predicate, and the next paragraphs make it registered rather than assumed.
>>>>>>> REPLACE

### P-14 · §5.4 — граница на регистрации: против поля, а не вида предиката

<<<<<<< FIND
Note that the boundary holds at registration: a configuration that guards a rich-syntax field with a decidable predicate is rejected fail-closed, and the fuzzing obligation cannot be silently laundered into a decided one, so a silent misconfiguration is prevented structurally rather than caught by a detector (registration.py).
=======
**The consumer is registered per field.** An exact check decides a value; whether the tool turns that value into the matching effect depends on the tool. A handler may default an absent argument (a missing amount meaning "the whole balance"), read an alias next to the guarded field, parse a string itself (`float("1e5")`), or resolve a path through a symlink — and in each case the predicate's verdict is correct while the effect leaves its codomain. A consumption contract (kernel/consumption.py) names, per sink, every field the handler reads, its consumption kind (number, flag, identifier, identifier list, path, interpreted, text), and what the handler does when the field is absent (refuse, a declared default, not consumed).

The boundary is enforced at registration against the field, not the predicate kind: an `enum` on a path field is rejected with the classifier's fuzz-required verdict, a range on an identifier is rejected, and so is a predicate on a field the contract does not name. Earlier the check keyed on the predicate's own kind, so the rejecting branch was unreachable; it is now reached through a real field. At run time every call is prepared into its contract before the first gate (§4.3), so the checked arguments are the executed ones. Under STRICT a sink with value predicates must have a contract or an explicit attestation; an attestation is the operator's statement and nothing more — core prepares nothing for that sink, so the whole of T4 there is a premise, and the kernel reports which guarded argument rests on which (App. F).

The gap is measured, not hypothetical. With identical value predicates, removing only the contract lets a real handler sweep the account on an absent amount, pay an alias field, and book a list as an account string; restoring it removes each effect while the legitimate call still executes (§5.8). An allowlisted path that is a symlink out of the root publishes a secret without a contract; with one, the configuration is refused at registration. An exponent string and a comma-joined recipient are refused either way — controls, kept so the claim stays proportional.
>>>>>>> REPLACE

### P-15 · §5.4 — «Precisely stated»

<<<<<<< FIND
Precisely stated: this is not a fuzz-tested theorem. It rests on a decision procedure over the enum/numeric fraction, while the interpretable residual is discharged by fuzzing, and that is appropriate precisely because we separated the two groups up front.
=======
Precisely stated: for the defined predicate language over contracted fields, T4 is decided by a procedure; every other consumer needs a separate argument. Not "every string is undecidable", not "an enum is automatically safe", and not "fuzzing proves fidelity" — on the interpretable residual fuzzing is evidence over the inputs drawn, and the split says up front which fraction rests on it.
>>>>>>> REPLACE

### P-16 · Figure 4 — надписи

| Было | Стало |
|---|---|
| `perimeter non-interference holds outright` | `decided by a procedure — given a registered consumer` |
| `• two equal projections cannot be driven to distinct effects` | `• two equal projections cannot be driven to distinct effects — if the consumer is contracted` |
| `fuzzing is the discharge — named, not hidden` | `fuzz-tested — evidence, named, not hidden` |
| `the line is enforced at registration (registration.py): a decidable predicate guarding a rich-syntax field is rejected …` | `the line is enforced at registration against the field's consumption contract (consumption.py): a decidable predicate on a path/interpreted field is rejected — a fuzzing obligation cannot be laundered into a decided one` |

<<<<<<< FIND
Figure 4: The decidability split. Perimeter non-interference holds outright on the enum/numeric fraction; the weird-machine residual is localized to a named rich-syntax fraction, where fuzzing is the discharge.
=======
Figure 4: The decidability split. On the enum/numeric fraction, for fields whose consumption is registered, fidelity is decided by a procedure; the weird-machine residual is localized to a named rich-syntax fraction, where fuzzing is evidence, not a discharge.
>>>>>>> REPLACE

### P-17 · §5.5 — floor: структурная ось, не «доказанная»

<<<<<<< FIND
Confidentiality, by contrast, is the proved axis.
=======
Confidentiality, by contrast, is the structural axis.
>>>>>>> REPLACE

<<<<<<< FIND
one measured and one proved, which makes collapsing them into a single table meaningless.
=======
one measured and one structural, which makes collapsing them into a single table meaningless. "By construction" here is an argument about the decision's inputs — the floor reads a session fact, never the outgoing bytes — not a proof object.
>>>>>>> REPLACE

### P-18 · §5.5 — композиция при отказе компонента

INSERT AFTER:
```
the kernel refuses to let an exec-class sink be declared positional.
```
>>> BEGIN INSERT

The axes are independent gates reading independent inputs, so the failure of one component relaxes only what depends on it. With the integrity labeler replaced by one that returns an arbitrary fixed label, the floor and the consequence ceiling still refuse, and integrity is lost — the test asserts the loss rather than hiding it.
<<< END INSERT

### P-19 · §5.7 — как устанавливается гарантия

<<<<<<< FIND
The guarantee is discharged on the enum/numeric fraction by a decision procedure, and on the rich-syntax residual by fuzzing (§5.4).
=======
On the enum/numeric fraction the guarantee is decided by a procedure for contracted fields; for attested fields it rests on the operator's statement; on the rich-syntax residual fuzzing is evidence (§5.4).
>>>>>>> REPLACE

### P-20 · Новый §5.8 — основания и матрица отрицательных контролей

INSERT AFTER (конец §5.7, перед «6 Evaluation»):
```
The discharge status of each obligation is stated in §8.
```
>>> BEGIN INSERT

5.8 Permitting grounds and negative controls

**Grounds.** T0 says no model produces a trusted-path projection; the claim needs more — every input that can turn a denial into an allow must have a ground the model cannot manufacture. The audit is kept as data (kernel/grounds.py) and rendered as App. F: for each ground, who may issue it, what core checks, what happens when it is absent, what it does not override, and — in a separate column — the premise it rests on that core cannot check. A test holds each row to the code: the named mechanism imports, the named regression exists, the appendix table equals the data. That makes the table a checked specification. Its completeness — that every admitting input has a row — is argued by enumerating what the allow path reads, and no test enforces that enumeration. Determinism is not correctness: a deterministic matcher can still issue the wrong ground, which is why each row names its check, not only its producer.

**Negative controls.** Each condition is removed one at a time; the effects that leave the constraint are counted on the handlers' own state (a ledger, a payee store, a bucket) behind the real loop; the condition is restored and counted again; the legitimate call is checked with the condition in place. The table is generated by the run and a test fails if the document drifts from it.

| Condition | Constraint | Violating effects: present → removed → restored | Legitimate call |
|---|---|---|---|
| consumption contract — presence | amount ≤ 1000 | 0 → 1 → 0 | executes |
| consumption contract — closed field set | recipient ∈ allowlist | 0 → 1 → 0 | executes |
| consumption contract — field shape | recipient ∈ allowlist | 0 → 1 → 0 | executes |
| sensitive source declared (floor arms) | no egress while a secret read is outstanding | 0 → 1 → 0 | executes |
| approval from a trusted rule, not the model | no unattended call beyond what was authorised | 0 → 1 → 0 | executes |
| context-default integrity | an integrity sink's value is not attacker-chosen | 0 → 1 → 0 | executes |
| destination allowlist (integrity blind to the value) | recipient ∈ allowlist | 0 → 1 → 0 | executes |

The counts are per proposal, not rates. They show indispensability within this construction — remove the condition and nothing else in Axor carries the constraint — not that each mechanism is necessary in general: confidentiality, for one, could be carried by a design without a session floor. The approval row is the producer negative control of T0: an approver whose codomain is {yes, no} but which follows the call's arguments executes an unattended consequential transfer, while every value constraint still holds.
<<< END INSERT

---

## 6 Evaluation

### P-21 · §6.2 — «discharge it by fuzzing»

<<<<<<< FIND
on the rich-syntax residual the property is not decidable and we discharge it by fuzzing, which is evidence rather than proof (§5.4).
=======
on the rich-syntax residual the property is not decidable and we test it by fuzzing, which is evidence rather than proof (§5.4).
>>>>>>> REPLACE

### P-22 · §6.4 — supersession как замена основания

<<<<<<< FIND
A statically declared allowlist is non-poisonable by construction.
=======
A statically declared allowlist is non-poisonable by construction; the attacker can still name a member, and membership then stands in for the origin ground on the integrity axis only.
>>>>>>> REPLACE

---

## 7 Related work

### P-23 · §7 ROPE — заменить абзац (один абзац, без расширения §6.3)

<<<<<<< FIND
ROPE [22]. The closest system by mechanism, and the one we compare against under controlled conditions rather than by citation: both decide at the tool call and ask whether the argument traces to a trusted origin, and they differ in what carries it, a ledger over content against a structural projection per sink.
=======
ROPE [22]. The closest system by mechanism, and the one we compare against under controlled conditions rather than by citation: both decide at the tool call and ask whether the argument traces to a trusted origin, and they differ in what carries it, a ledger over content against a structural projection per sink. Its guarantee (origin soundness, under platform-metadata integrity, record integrity and enumeration completeness) is about the origin of the admitted value; the consumer of that value and the completeness of the guarded-tool table are premises. Run on both systems' own enforcement code with the same handlers and an equal-policy ROPE arm, the amount bound turns out to be policy content (the same bound blocks the same call in both), while the absent-argument and alias cases execute under ROPE and under Axor alike until a consumption contract is added — so the difference on T4 is the contract, not Axor as such; a tool missing from the table is admitted by ROPE and refused by Axor's STRICT completeness; confidentiality is outside ROPE's scope. In the other direction, ROPE's record-closure lemma discharges a condition Axor leaves open: a trusted tool's output can be written by an earlier admitted call (§8).
>>>>>>> REPLACE

> Если точный текст абзаца ROPE в исходнике длиннее (он продолжается про §6.3), заменяется
> только первое предложение-якорь; остальная часть абзаца остаётся как есть.

### P-24 · §7 FIDES

<<<<<<< FIND
Shares the explicit-flow-only scope boundary (O2), and axor inherits the same limitation on implicit flows.
=======
Closes the implicit-flow boundary on the integrity axis by checking the trusted context of the call; axor's O2 is explicit-flow only, so the boundary is ours, not shared.
>>>>>>> REPLACE

---

## 8 Limitations and scope

### P-25 · §8 Boundaries — четыре новых пункта

INSERT AFTER (после пункта «“Any trust model” is demonstrated on two instances, not mechanized (§5.6).»):
```
“Any trust model” is demonstrated on two instances, not mechanized (§5.6).
```
>>> BEGIN INSERT

- Handler conformance to its consumption contract is a premise: the kernel makes the check and the dispatch agree with the contract, not the handler's code. For an attested sink the whole of T4 is the operator's statement (§5.4).
- Record closure is not stated or checked. A declared benign tool's output is a trusted ground; if an earlier admitted call can write into it (a payee list written by `add_payee`, read by `list_payees`), a re-typed attacker value is laundered — measured offline unless the writer is declared an integrity sink. ROPE's record-closure lemma covers the analogous condition (§7).
- The permitting-ground table is a checked specification; its completeness is argued by enumerating the allow path, not tested (§5.8, App. F).
- The negative-control matrix shows each condition is load-bearing in this construction, not that each mechanism is necessary in general (§5.8).
<<< END INSERT

### P-26 · §8 последний абзац — «proved»

<<<<<<< FIND
The integrity axis is asymmetric to the confidentiality one: the first is measured, the second is proved. Confidentiality is proved by construction: the floor is paraphrase-proof because the gate reads a session boolean, not the bytes of the egress,
=======
The integrity axis is asymmetric to the confidentiality one: the first is measured, the second is structural. Confidentiality holds by construction: the floor is paraphrase-proof because the gate reads a session boolean, not the bytes of the egress,
>>>>>>> REPLACE

---

## 9 Conclusion

### P-27 · §9 — условие гарантии

<<<<<<< FIND
The non-interference theorem lets it guarantee a framing-invariant defense, with the conditionality stated and localized: proved on the decidable enum/numeric part, discharged by fuzzing on the rich-syntax residual (§5.4). The split's boundary is enforced at registration; kernel purity is a checked property.
=======
Every executed call satisfies the stated constraints under named conditions: permitting grounds come from trusted mechanisms, the checked property is the consumed one — decided on the enum/numeric part for contracted fields, fuzz-tested on the rich-syntax residual (§5.4) — and the arguments checked are the arguments executed. Non-interference makes the verdict framing-invariant; each condition is shown load-bearing by a negative control (§5.8). The split's boundary is enforced at registration against the field; kernel purity is a checked property.
>>>>>>> REPLACE

---

## Appendices

### P-28 · App. A — строка «Codomain-fidelity, decidable fraction»

<<<<<<< FIND
the effective codomain matches the nominal one: for enum / bounded-numeric projections consumed as case-split, two equal projections cannot be split into distinct effects; discharged by a decision procedure
=======
the effective codomain matches the nominal one: for enum / bounded-numeric projections on a field whose consumption contract declares a case-split or numeric consumer, two equal projections cannot be split into distinct effects; decided by a procedure; every call prepared into the contract before the first gate
>>>>>>> REPLACE

Колонка «Enforcing regression» той же строки — дописать:
`kernel/consumption.py (validate_consumption_contracts, prepare_args); tests/kernel/test_consumption.py; tests/adversarial/test_check_vs_consumer.py`
Колонка «Failure mode it pins» — дописать: `a handler default, alias or parser consuming a value the predicate never saw`.

### P-29 · App. A — строка «Codomain-fidelity, rich-syntax fraction»

<<<<<<< FIND
codomain-fidelity is undecidable in general; fuzzing is the discharge, localized to this named fraction
=======
codomain-fidelity is undecidable in general; fuzzing is evidence, not a discharge, localized to this named fraction
>>>>>>> REPLACE

### P-30 · App. A — три новые строки в конец таблицы

INSERT AFTER (после строки «Trust-model agnosticism»):

| Premise | Statement (one line) | Enforcing regression | Failure mode it pins |
|---|---|---|---|
| Permitting grounds (T0, chain) | every input that can permit a call has a stated issuer, check, absence behaviour, non-overridden obligations and premise | kernel/grounds.py; tests/invariants/test_permitting_grounds.py | a ground without a check or a pinning test; the appendix drifting from the data |
| Checked = executed | the prepared argument set is gated, digested for approval, and dispatched | tests/integration/test_end_to_end_chain.py (property-tested, arbitrary and adaptive proposers); tests/kernel/test_consumption.py (replay parity) | executing the raw proposal after checking a prepared one |
| Negative controls | each condition removed → forbidden effect; restored → none; legitimate call executes | tests/negative_controls.py; tests/integration/test_negative_control_matrix.py (document = fresh run) | a condition that is not load-bearing; a stale matrix |

### P-31 · App. B — шаг 0 перед последовательностью

INSERT AFTER:
```
A deny at any gate is final; a later gate can never widen an earlier deny. The adjudicator is advisory and tightening-only. Order and per-gate rationale :
```
>>> BEGIN INSERT
 Before gate 1, a call to a sink with a consumption contract is prepared into it (§5.4); a call outside its contract is denied and recorded under the value_policies gate, and the prepared set is the one every later gate reads.
<<< END INSERT

### P-32 · App. B — таблица гейтов, строки 5 и 7

| Строка | Было | Стало |
|---|---|---|
| 5 ssrf | `rich-syntax fraction, fuzz-discharged; both historical bugs …` | `rich-syntax fraction, fuzz-tested; both historical bugs …` |
| 7 carrier | `rich-syntax fraction, fuzz-discharged` | `rich-syntax fraction, fuzz-tested` |

### P-33 · App. C — конфигурация латентности (точное название)

<<<<<<< FIND
The latency figures of §6.2 are measured on the banking origin configuration under ExecutionMode.STRICT (require_tool_roles, context-default integrity, declared integrity_sinks),
=======
The latency figures of §6.2 are measured on the banking origin configuration — STRICT role completeness (require_tool_roles), context-default integrity, declared integrity_sinks, no egress allowlist; an origin-only arm, not ExecutionMode.STRICT —
>>>>>>> REPLACE

### P-34 · App. D.2 п.2 — участник множества выбирается атакующим

<<<<<<< FIND
An attacker destination is simply not in the set and stays denied.
=======
An attacker destination outside the set stays denied; the attacker can still name a member, and membership then substitutes for the origin ground on the integrity axis only — a substitution of grounds the operator makes by declaring the set, not a relaxation without cost.
>>>>>>> REPLACE

### P-35 · Новое App. F — разрешающие основания

INSERT AFTER (после App. E, перед References):
>>> BEGIN INSERT

F Permitting grounds

Rendered from kernel/grounds.py; a test fails if this table and the data differ. "Checked by core" and "Premise" are separate on purpose: nothing a ground merely rests on is presented as verified.

| Permitting input | Issued by | Checked by core | Premise (not checked by core) | Status |
|---|---|---|---|---|
| Operator policy (A₀) | operator configuration | STRICT refuses to run without an operator-supplied policy | the operator's policy is the intended one | enforced under STRICT |
| Delegation ceiling (G) | operator configuration | grants and leases validated against available ∪ grantable | the ceiling is the intended one | enforced |
| Action class | built-in table + overrides | exact-name lookup; STRICT: unclassified ⇒ catastrophic | the class matches the tool's real irreversibility | enforced under STRICT |
| Sink roles | operator declaration | STRICT role completeness; exec-class cannot be positional | the declared role matches the tool | enforced under STRICT |
| Consumption contract | operator, per field | predicate vs field kind at registration; every call prepared before the first gate; prepared set executes | the handler consumes each field as declared | enforced |
| Consumption attestation | operator, per sink | none beyond presence | the whole of T4 for that sink | premise |
| Integrity supersession | operator, by a closed allowlist on every driving arg | every driving arg present and covered by a closed codomain | the set is the intended one; the attacker can pick a member | enforced |
| Confidentiality floor down | no outstanding sensitive read; governance endorsement | outstanding-read map; authority token, per fingerprint | the governance authority is unreachable by the agent | enforced |
| Trusted value (TrustedOrigin) | task at depth 0; allowlist members; declared benign tool; endorsement | registration from kernel-side code; TOOL needs a benign declaration under STRICT | a benign tool's output is not attacker-writable, including by an earlier admitted call (no record-closure check) | enforced under STRICT |
| Call-bound approval | operator-supplied callback | digest over prepared args; volatile re-check; atomic one-use claim | the callback is a trusted party or a checkable rule, not a model | premise |
| Projection features | the T0 producer modules | import contract + determinism tests | — | enforced |
| Peer provenance label | a peer, by signed receipt | signature, nonce, binding; bounded discount across keysets | the peer's key is held by the peer | enforced |
| Absence of a model-carried mark | the model | — | — | inadmissible as a positive ground |

Completeness of this table is argued, not tested: the allow path admits a call only if consumption preparation, capability, role completeness, consequence, value policies, degradation, SSRF, positional, carrier and per-value taint all admit it, and each of these reads only inputs listed above (adjudicator, budget and admission can only tighten). A new admitting input added to that path without a row here would not be caught by any test.
<<< END INSERT

---

## Не трогать

- §6.3 (контролируемое сравнение с ROPE в его харнессе) и Таблицу 3 — числа и текст
  остаются; новый материал про ROPE — только абзац P-23 в §7.
- Все числа §6, App. C–E.
- Рисунки 2, 3, 5, 6 — без изменений.
