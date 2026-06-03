# Antumbra — Experiment Ledger

In Antumbra the **falsifiable validations *are* the milestones** (architecture §7). This ledger records each
one: the claim, how it was tested, the measured result, an explicit **kill criterion**, and a **reproduce**
command. Results to date are toy-scale and honest about what is faked; see the
[Technical Reference §13-14](../docs/technical-reference.md) for the standing scorecard and the ADRs for the
decisions each experiment exercises.

GPU experiments need the CUDA-13 / Windows environment in [running-the-trainer.md](../docs/running-the-trainer.md)
and `python` on `PATH` for the exec verifiers; build with `--features models,cuda --release`. Non-GPU
experiments are plain `cargo test`.

| ID | Claim | Pillar / ADR | Status |
|---|---|---|---|
| [EXP-001](#exp-001--the-trainer-learns-from-verified-outcomes) | An adapter learns from verified outcomes | 1 / 0002,0003,0010 | **passed** (GPU) |
| [EXP-002](#exp-002--the-gate-routes-in-scope-and-refuses-out-of-scope) | Gate routes in-scope, refuses out-of-scope | 2 / 0005 | **passed** |
| [EXP-003](#exp-003--capability-vectors-from-evaluated-behavior) | Capability vectors come from evaluated behavior | 2 / 0004,0005 | **passed** |
| [EXP-004](#exp-004--a-multi-expert-population-routes) | A multi-expert population routes correctly | 3 / 0001,0005 | **passed** (GPU) |
| [EXP-005](#exp-005--an-actionable-boundary-inhibits-routing-in-scope-only) | An actionable boundary inhibits routing, in-scope only | keystone / 0004 | **passed** |
| [EXP-006](#exp-006--experts-answer-serving) | A graduated expert serves a real answer | 1 / 0006 | **passed** (GPU) |
| [EXP-007](#exp-007--live-counterfactual-boundary-recovery) | Live counterfactual boundary recovery + autonomous discovery | keystone / 0004,0006 | **passed** (GPU) |

---

## EXP-001 — the trainer learns from verified outcomes

**Claim.** RAFT (sample K -> verify -> SFT the winners) lifts an adapter's pass-rate purely from verifiable
reward, with no teacher text.

**Method.** Train on the GPU under two verifiers: a Python-free convention rule (`contains_all`) and a real
exec verifier that runs the generated code and asserts behavior.

**Result.** Convention (`corpora/learn.json`): pass-rate `0.06 -> 0.25 -> 0.88 -> 1.00`. Exec
(`corpora/example-tasks.json`, `add(2,3)==5` / `reverse('abc')=='cba'`): `0.38 -> 1.00 -> 1.00 -> 1.00`. Both
graduated a real bf16 adapter.

**Kill criterion.** Pass-rate flat or falling across rounds -> RAFT is not learning from the reward.

**Reproduce.** `antumbra --url surrealkv://./data/a.skv train --corpus corpora/learn.json --generations 1`
(commits `af92436`, `b7955cb`).

## EXP-002 — the gate routes in-scope and refuses out-of-scope

**Claim.** With real embeddings, the gate routes a task to the right specialist and escalates genuinely
out-of-scope tasks — using **relative** coverage, since an absolute cosine floor cannot (sentence-embedding
cosine is compressed into a high band).

**Method.** Seed three described specialists (arith / strings / dates) with real BERT capability vectors; route
matched and out-of-scope queries.

**Result.** In-scope routing 3/3 (margins 0.106 / 0.163 / 0.110); two out-of-scope queries escalated
(0.037 / 0.068) across the 0.08 threshold. The first design (centroid background) was falsified by measurement
and corrected to the top-1-minus-top-2 prototype margin.

**Kill criterion.** No threshold separates in- from out-of-scope on the relative score -> the coverage gate is
inadequate and the learned gate is required earlier.

**Reproduce.** `antumbra --features models -- --url surrealkv://./data/a.skv seed` then
`... route "reverse the characters of a text string"` (commit `6b69eed`).

## EXP-003 — capability vectors from evaluated behavior

**Claim.** An expert's routing vector is learned from the tasks it provably solved, not a hand-written label.

**Method.** Loop test: a graduating shadow reports its solved prompts; assert the graduated expert's
`capability_vec` equals the centroid of their embeddings.

**Result.** Passes (`crates/antumbra-loop/tests/durable_loop.rs::capability_vector_is_learned_from_solved_exemplars`).

**Kill criterion.** Capability vector independent of solved tasks -> routing reflects labels, not behavior.

**Reproduce.** `cargo test -p antumbra-loop` (commit `827f204`).

## EXP-004 — a multi-expert population routes

**Claim.** Several specialists, each trained on its own corpus and graduated into one population, are routed to
correctly.

**Method.** Train arith / strings / lists into one persistent population (GPU), each with a behavior-derived
capability vector; route across them.

**Result.** Population grew 1->2->3; ranking 3/3 correct (arith 0.831 / strings 0.906 / lists 0.765 top-1).
Finding: the abstention threshold is per-population — adjacent specialists (arith ~ lists) compress the margin,
so 0.08 over-escalated and 0.045 separated cleanly. Recorded as a calibration property, not papered over.

**Kill criterion.** A query's true specialist is not top-1 -> capability vectors do not separate experts.

**Reproduce.** `train --corpus corpora/arith.json --run arith` (x3 for strings/lists), then
`route "add two integers"` (commit `8675898`).

## EXP-005 — an actionable boundary inhibits routing, in-scope only

**Claim.** A recovered counterfactual boundary, persisted, makes the gate escalate inside the failure scope and
nowhere else.

**Method.** End-to-end test: `find_scope` recovers C', `finding_to_boundary` persists an actionable boundary
through surql-rs, then `route` a task inside vs outside the failure region — with a no-boundary control.

**Result.** Inside-scope task escalates (boundary cancels its coverage); outside-scope routes normally; the
control (same population, no boundary) routes the inside task straight to the matching expert — isolating the
boundary as the cause.

**Kill criterion.** The boundary inhibits outside its scope, or fails to inhibit inside it -> over/under-
generalized inhibition.

**Reproduce.** `cargo test -p antumbra-store --test keystone_mem` (commit `5e7705c`).

## EXP-006 — experts answer (serving)

**Claim.** A graduated expert can be loaded and serve a real answer (ADR-0006).

**Method.** `CandleServe` loads the shared base + an expert's adapter and generates; `ask` routes a task to a
specialist and serves from its adapter (GPU).

**Result.** `ask "add two integers"` -> arith specialist -> `def add(a, b): return a + b`; `ask "reverse a
string"` -> strings specialist -> `return s[::-1]`. Route -> load adapter -> serve closed end-to-end.

**Kill criterion.** Loaded adapter does not change the base's behavior, or serving errors -> adapter loading or
serving is broken.

**Reproduce.** after EXP-004's training, `ask "Write a Python function add(a, b) that returns their sum."`
(commit `ecd2550`).

## EXP-007 — live counterfactual boundary recovery

**Claim.** The keystone runs live with no fake: a trained expert's competence boundary is recovered by actually
generating and executing, then persisted as actionable — and the governing feature can be **discovered**, not
supplied.

**Method.** Train a narrow **adder** (add-only); probe its boundary **with its own adapter**
(`scope --expert adder-g0`) over an in-scope (`op=add`) vs out-of-scope (`op=multiply`) context, each with its
own exec verifier. Then `--discover` to infer the governing feature from pass/fail.

**Result.** Recovers reliably, including in `--discover` mode (the governing feature is *inferred*, not
supplied): the adder passes `op=add` and fails `op=multiply` -> governing feature `op`, C' `{op: add}`,
actionable boundary stored (`status`: 1 actionable, 0 open). Getting here flushed out three real bugs, each
fixed and committed: (1) best-of-K re-seeded the RNG identically, so the K draws were the same completion -> a
per-call generation nonce; (2) the server reloaded the multi-GB model on every `act` -> cache it once; (3) the
verifier executed untrusted generated code with no timeout, so a runaway draw hung the whole run -> null stdin +
a hard kill timeout. Throughout, the probe held integrity — it returned "stays open" rather than fabricating a
scope.

**Kill criterion.** The probe fabricates a boundary when generation never verifies, OR no actor/temperature
setting yields reliable recovery -> the live keystone is not dependable.

**Reproduce.** train `corpora/add-only.json --run adder`, then
`scope --spec corpora/scope-adder.json --expert adder-g0 --discover` (commits `ed8b0f0`, `7a4779a`, and the
generation-diversity fix).

---

*The mechanisms (search, probe, gate, persistence) are unit-proven and exercised end-to-end; the open frontier
is reliability and scale, not fakes. The single fake retired this cycle was the AcceptabilityProbe; what remains
authored is the candidate governing-feature set, and what remains untested is scale and catastrophic forgetting
(pillar 3).*
