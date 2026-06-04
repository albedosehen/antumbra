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
| [EXP-008](#exp-008--grpo-vs-raft) | GRPO is more sample-efficient than RAFT | 1 / 0011 | **passed, single run** (GPU) |
| [EXP-009](#exp-009--4-bit-qlora-training-memory) | 4-bit base trains a LoRA at f16 quality, ~1/4 resident base | 1 / 0011 | **passed** (GPU) |
| [EXP-010](#exp-010--catastrophic-forgetting-frozen-population-vs-monolithic) | Frozen population retains skills a monolith forgets | 3 / 0001,0010 | **inconclusive** — no forgetting at toy scale (GPU) |
| [EXP-011](#exp-011--durable-correction-against-a-strong-prior) | A one-time correction is captured + routed across a context reset | 1,2 / 0004,0006,0009 | **passed** (GPU) |

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

## EXP-008 — GRPO vs RAFT

**Claim.** GRPO (group-relative policy optimization, ADR-0011) reaches a given pass-rate in fewer sampled
completions than RAFT — the v1 sample-efficiency upgrade.

**Method.** Train the same arith corpus on the GPU with `train --algo raft` and `--algo grpo`, identical knobs
(samples 6, rounds 3, max-new-tokens 48). GRPO uses the base-with-LoRA-off reference (no second model) and steps
per group member.

**Result.** RAFT pass-rate `0.08 -> 0.25 -> 1.00`; GRPO `0.33 -> 0.92 -> 1.00`. GRPO climbs much faster
(0.92 by round 1 vs 0.25). The combined-group backward OOM'd the card (candle retains base activations for the
LoRA backward, so memory scales with the group); fixed by stepping per member.

**Kill criterion.** GRPO does not beat RAFT on sample-efficiency or final pass-rate (and is far more code) ->
shelve GRPO, keep RAFT. *(Cleared directionally; a rigorous win needs multiple seeds and tasks.)*

**Reproduce.** `train --corpus corpora/arith.json --algo grpo` vs `--algo raft` (commits `0159d55`, `4d53853`,
`e6a7a20`, `47fecbc`).

## EXP-009 — 4-bit QLoRA training memory

**Claim.** A Q4_K base trains a LoRA at f16 pass-rate using ~1/4 of the resident base memory (dequant-in-forward,
ADR-0011).

**Method.** Train the arith corpus with the f16 base and with `--quantize-base`, identical knobs (samples 4,
rounds 3, max-new-tokens 32).

**Result.** Both graduated with the **same curve: 4-bit `0.12 -> 0.38 -> 1.00`, f16 `0.12 -> 0.38 -> 1.00`** —
Q4_K dequant-in-forward trains a LoRA at f16 quality. The first attempt OOM'd, and the cause was a real bug,
not a candle limit: **generation ran with autograd tracking on** (the LoRA factors are `Var`s), so the KV cache
retained the whole growing generation graph; with a quantized base every token re-dequantizes the full ~3 GB of
weights and all were retained -> OOM. Training (one forward, immediate `backward_step`) was never the problem;
sampling was. Fixed by a `grad` flag on `LoraLinear` that detaches the LoRA factors during generation (same
values, untracked) — which also makes f16 generation/serving leaner.

**Kill criterion.** 4-bit is slower than, or no smaller than, f16 -> stays north-star-only. *(Cleared: matches
f16 quality; resident base is ~1/4, the per-token re-dequant is the only added cost. A direct VRAM measurement
across base sizes is the rigorous follow-up.)*

**Reproduce.** `train --corpus corpora/arith.json --quantize-base` vs without (commits `db89a36`, and the
generation-detach fix).

## EXP-010 — catastrophic forgetting (frozen population vs monolithic)

**Claim.** A population of frozen expert adapters retains each skill, whereas a single adapter continually
fine-tuned across skills catastrophically forgets the earlier ones (pillar 3 — the case for a population).

**Method.** Two arms over Qwen2.5-Coder-1.5B + LoRA, RAFT (samples 6, rounds 3, max-new-tokens 32). **Population:**
train `adder` on `add-only` and `reverser` on `reverse-only` as two separate frozen adapters. **Monolithic:**
warm-start one adapter from `adder` and continue-train it on `reverse-only` (the new `train --parent`). Then
score each adapter with **no training** (the new `eval` command): the monolith re-scored on the *old* skill (add)
is the forgetting probe.

**Result.** **No catastrophic forgetting at this scale.** The monolith kept add at **1.00** (8/8) while learning
reverse to **0.75** — add was not degraded at all (it even beat `adder`'s own 0.88). Population retained add
(`adder` 0.88) and reverse (`reverser` 0.62). A second design with a deliberately *interfering* pair (same
function name `solve`, reverse vs uppercase, monolith overwritten for 5 rounds) was **inconclusive**: the
ambiguous `solve` prompt never trained (RAFT pass-rate stayed 0.00 — no verified winners), so there was no
learned skill to forget.

**Why.** `add` and `reverse` are different function names that do not compete for the low-rank adapter's
capacity, and the base is competent enough that the *descriptive prompt* carries much of each task — so the
adapter is not the sole skill-carrier, and overwriting it does not erase the behavior. Demonstrating
adapter-level forgetting needs a regime where the adapter is **load-bearing** (the base fails the task zero-shot)
and the two skills **interfere** — a hard toy pair to build: if the base can't do the task, RAFT gets no winners
to train on; if it can, the prompt masks any forgetting.

**Kill criterion.** If the monolith retains the old skill as well as the population (no forgetting gap), the toy
experiment does not support pillar 3's premise at this scale. *(Triggered: no forgetting gap observed. The
population's benefit is a scale / interference-regime question — kept as the open frontier, not claimed as
demonstrated. The trainer primitives it needs — continual warm-start and no-train eval — are now in place.)*

**Reproduce.** `train --corpus corpora/add-only.json --run adder`, `train --corpus corpora/reverse-only.json
--run reverser`, then `train --corpus corpora/reverse-only.json --run mono --parent adapters/adder_g0.safetensors`;
score with `eval --corpus corpora/add-only.json --adapter adapters/mono_g0.safetensors` vs `--adapter
adapters/adder_g0.safetensors`.

## EXP-011 — durable correction against a strong prior

**Claim.** A one-time, externally-supplied correction — held against a strong, *wrong-for-this-context* base
prior — can be captured into a frozen expert and re-applied by routing, so it persists across a context reset
without being re-stated. This is the agent failure Antumbra targets: the model assumes the obvious default
(`package.json` -> npm), is corrected once ("this project uses bun"), and would otherwise repeat the mistake
after the correction falls out of context. Capture is the second intake path beside RAFT discovery (ADR-0004/0009).

**Method.** Clean proxy: project `acme-api` should use `bun add <pkg>`, but the base reaches for `npm` by habit.
The disambiguator lives only in the expert, never the inference prompt (the prompt carries the project, not the
tool). (1) Measure the base floor (`eval` with no adapter). (2) `teach` a frozen expert from 10 verified `bun add`
corrections (varied packages). (3) `eval` it on the training prompts and on **held-out** packages
(`react`, `axios`). (4) From a **fresh process**, `ask` an `acme-api` task and let the gate route + serve.

**Result.** Base floor **0.00** (always npm — a genuinely load-bearing prior). After capture: **1.00 on training
prompts and 1.00 on held-out packages** (`bun add react` for a package never trained — the *pattern* was learned,
not memorized). The expert persisted to the store; a separate `ask` process routed `acme-api` to it and served
` bun add react`. The base floor (npm) is the status-quo regression; the routed expert is the correction
surviving a context reset. Landing this flushed out a real **training bug**: tokenizing the prompt alone vs
inside the full text mis-aligned the completion mask, so the **first** completion token was never supervised —
invisible for in-distribution RAFT (EXP-001), fatal for an out-of-distribution token like `bun`. Fixed by masking
past the shared token prefix, plus supervising an EOS so short completions terminate; both improve all training.

**Population routing (two corrections).** Capturing a second project — `payments-service -> yarn add` — gave a
two-expert population, and the gate routed the **same package** by project identity: `acme-api + react ->
bunexpert -> bun add react`, `payments-service + react -> yarnexpert -> yarn add react` (and the `axios` pair
likewise). Top-1 was correct in all four cases. One case (`payments + axios`) abstained at the default
threshold (relative coverage 0.063 < 0.08) and routed once it was lowered to 0.04 — the two experts are
semantically adjacent ("add a dependency to a project"), compressing the margin, so the gate safely escalates
rather than mis-serving. This is the EXP-004 per-population calibration property, reconfirmed: discrimination is
correct; the threshold only trades confident-serve against safe-abstain.

**Kill criterion.** If the base floor is not low (the prior is not load-bearing), or the captured correction does
not generalize past the trained strings, or a fresh process does not route to it, capture is not a durable
mechanism. *(Cleared: floor 0.00, held-out 1.00, fresh-process routing served the correction; a two-expert
population routes each project to its own correction.)*

**Reproduce.** `eval --corpus corpora/teach-bun-eval.json` (base floor), then `teach --corpus corpora/teach-bun.json
--run bunexpert --rounds 15 --lr 3e-4`, then `eval --corpus corpora/teach-bun-eval.json --adapter
adapters/bunexpert_g0.safetensors`, then `ask "# acme-api project. Shell command to add the 'react' dependency: "`.

---

*The mechanisms (search, probe, gate, persistence) are unit-proven and exercised end-to-end; the open frontier
is reliability and scale, not fakes. The single fake retired this cycle was the AcceptabilityProbe; what remains
authored is the candidate governing-feature set. Catastrophic forgetting (pillar 3) was probed at toy scale
(EXP-010) and did not appear — the monolith retained the old skill — so the population's headline benefit is now
a measured open question that needs a load-bearing-adapter / interference regime, not an untouched assumption.*
