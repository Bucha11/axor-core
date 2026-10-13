# Task-scoped grants

No model output creates authority. A sink that declares `grant_args` runs only under a
live grant covering its whole driving tuple, and a grant is minted only from a trusted
input. Every executed effect therefore lies inside the grants issued from trusted
sources, within their counters.

## Model

```
Grant = (task, sink, {arg: constraint}, uses, expiry, source ∈ {L1, L2, L3, L4})
```

| Source | Input | Default uses | Code |
| --- | --- | --- | --- |
| L1 | literals the user wrote (`kernel/request_literals.py`, regex only) | 1 | `GrantMinter.from_request` |
| L2 | operator templates / enums | unlimited | `GrantMinter.from_templates` (`grant_templates`) |
| L3 | a typed intent from the application | 1 | `GrantMinter.from_structured` |
| L4 | a human confirmed the exact action | 1 | `confirmer=` on the governor |

**A confirmation grant covers the whole call, not only the driving arguments.** It is
bound to `kernel.confirm.call_digest` — the sink and every argument, rich fields
included — and the confirmation renders every field in full. Bound only to the driving
tuple, a confirmed message could be sent to the same recipient with another body.

A grant arg declared with a trailing `?` (`participants: "email?"`) is optional: a call
may omit it, since absence carries no value anywhere, but a value that is passed must be
covered. A structured or confirmed action that omits an arg yields a grant admitting only
that arg's absence, never a value added later.

A constraint is `exact` (one canonical value), `enum` (a closed set) or `limit`
(a numeric bound). A limit is a bounded-damage guarantee, not an endorsement, and is
refused unless the sink.arg is listed in `bounded_damage`. A grant on a free-text field
is refused at registration (exact match on free text is not authority over the effect).

Values are compared as **handles** (`kernel/handles.py`): canonicalized by kind (IBAN,
email, money, number, path, id, string), interned with their origin. A read value whose
canonical form equals a trusted handle's resolves to it; a look-alike gets a fresh
untrusted handle that no grant names. With `strict_handles=True` a grant arg must be
passed as a handle reference; a raw string in that slot is refused (`grant_handle`).

## Where it sits in the gate sequence

`ToolCallGovernor.evaluate` runs the grant gate **last** (step 7), after consequence,
value policy, SSRF, positional, carrier and taint. Two consequences:

* a confirmation is only asked for a call every other gate would admit;
* the confidentiality floor and the consequence class still apply to granted calls.

On a grant sink whose driving args are all grant args, the grant replaces the
content-taint **integrity** check (like decidable supersession, `integrity_superseded`).
The existing enum supersession is the L2 special case with `uses = ∞`; the regressions
in `tests/adversarial/test_decidable_supersession.py` hold unchanged on L2 grants
(`tests/adversarial/test_task_scoped_grants.py`).

## Invariants

| Invariant | Statement | Discharged by |
| --- | --- | --- |
| O1′ | decide reads only π(x) and p_t | grant match is on canonical driving args; property test |
| O4 | G_{t+1} ⊆ G_t ∪ Mint(t); Mint never sees an untrusted read | `GrantMinter` takes trusted inputs only; import-linter contract `o4-no-amplification`; fuzz test |
| O5 | check-and-decrement is linearizable; at most one execution per spend | `capability/grant_store.py` (`LockedGrantStore`, `execute_granted`); idempotency key `<grant id>#<n>`; reconcile on unknown outcome |

`kernel/grants.py` is a pure state machine (Ring 0 has no locks and no clock). The lock,
the clock and the reconcile protocol live in `capability/grant_store.py`, which belongs
in the process that executes effects. **There is no daemon server in this repository
yet** (only `contracts/daemon.py` and the client); `LockedGrantStore` is the in-process
stand-in, and cross-process atomicity is unproven until the daemon hosts it.

## Audit

`GovernanceDecision.grant_id`, `.grant_source` (L1–L4) and `.confirmed`; the
`INTENT_APPROVED` trace payload carries `{"grant": {id, source, confirmed}}`.
`ToolCallGovernor.spend(decision, args)` reserves the use and returns the spend key the
executor commits or releases.

## Known gaps

* L1 builds tuples as the product of the request's literals per kind: a request naming
  two IBANs and two amounts grants all four pairs.
* `IntentNormalizer` is unchanged; the handle-typed slot check lives in the grant gate.
* `opaque_ids` (replacing identifiers in reads by handles) is not implemented.
