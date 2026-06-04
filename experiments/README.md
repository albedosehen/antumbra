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
| [EXP-012](#exp-012--the-self-improvement-lifecycle-end-to-end) | Fail -> bound -> capture -> retire -> route to the fix, in one loop | keystone / 0004,0005,0006,0009 | **passed** (GPU) |
| [EXP-013](#exp-013--the-learned-router) | A learned router separates specialists from generalists | 2 / 0005,0009 | **passed** (GPU) |
| [EXP-014](#exp-014--adapter-composition) | Experts compose into one served adapter; behavior is dialable | 3 / 0006,0009 | **passed** (GPU) |
| [EXP-015](#exp-015--complementary-composition-the-capability-multiplier) | Composing complementary experts does what neither alone was trained for | 3 / 0006,0009 | **passed** (GPU) |
| [EXP-016](#exp-016--the-closed-serving-loop-route--auto-compose) | One `ask` routes the project expert and auto-composes standing conventions | 2,3 / 0005,0006,0009 | **passed** (GPU) |
| [EXP-017](#exp-017--calibrated-router-out-of-distribution-abstention) | The router abstains on out-of-distribution tasks instead of overconfidently routing | 2 / 0005,0009 | **passed** (GPU) |
| [EXP-018](#exp-018--autonomous-self-improvement-eval-gated) | The system trains itself to a quality bar and stops, no manual driving | 1 / 0002,0010 | **passed** (GPU) |

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

**Boundary probe (ADR-0004), now with relative inhibition.** We also drove the *counterfactual boundary* on this
scenario: a general `generaldeps -> npm` expert (which covers `acme-api` by similarity, coverage 0.877 — the
confident mistake), then `scope --discover` to recover where it is wrong. Recovery worked: probing the expert's
own behavior inferred `governing feature: project`, found `C' = webshop`, and stored an actionable boundary. The
first attempt exposed a real limit — an **absolute cosine-radius** inhibition either fired too weakly to act or,
once strong enough, **over-generalized** (the `webshop` control escalated too); the two contexts are only 0.066
apart in boundary-similarity, below sentence-embedding resolution (the EXP-002/004 compressed-band problem,
reaching the boundary). The fix is the same relative trick that makes the gate work: store **C''s embedding too**
and inhibit on the *margin* `sim(fail) - sim(C')`, so the shared "add a dependency" background cancels and only
the discriminative project signal remains. With that, **`acme-api` escalates (coverage -0.123) while the
`webshop` control routes normally (0.885)** — precise per-project scope. So both halves discriminate per project
now: nearest-expert routing for the correction, relative-margin inhibition for the boundary. Two real fixes
landed: the context vector strips the verify spec before embedding, and `FailureBoundary` carries `ok_context_vec`
for relative inhibition (legacy absolute radius kept as the fallback when C' is not embedded).

**Kill criterion.** If the base floor is not low (the prior is not load-bearing), or the captured correction does
not generalize past the trained strings, or a fresh process does not route to it, capture is not a durable
mechanism. *(Cleared: floor 0.00, held-out 1.00, fresh-process routing served the correction; a two-expert
population routes each project to its own correction; and a recovered boundary escalates the in-scope project
while sparing the control via relative inhibition.)*

**Reproduce.** `eval --corpus corpora/teach-bun-eval.json` (base floor), then `teach --corpus corpora/teach-bun.json
--run bunexpert --rounds 15 --lr 3e-4`, then `eval --corpus corpora/teach-bun-eval.json --adapter
adapters/bunexpert_g0.safetensors`, then `ask "# acme-api project. Shell command to add the 'react' dependency: "`.

## EXP-012 — the self-improvement lifecycle (end to end)

**Claim.** The validated pieces compose into one loop: an agent makes a confidently-wrong call, the failure is
detected and *bounded* (so it stops repeating it), a verified correction is captured into a frozen expert that
*supersedes* the boundary, and the same task now routes to the fix — all owned and persisted, on a 1.5 B base.

**Method.** One continuous run, one persistent store. (0) `teach` a general `npm` deps expert — the habit.
(1) `route` an `acme-api` task. (2) `scope --discover` recovers the boundary from the expert's own behavior;
re-route. (3) `teach` the `bun` correction — the boundary is retired when a captured expert covers its failure
region (`is_covered_by`, the same relative test as the inhibition). (4) re-route and serve.

**Result, stage by stage.** (1) `acme-api` routes to npm (coverage 0.877) — *the mistake*. (2) Boundary recovered
(governing feature `project`, `C' = webshop`); `acme-api` **escalates** (-0.123) while the `webshop` control still
**routes** (0.885) — relative inhibition, precise. (3) `bunexpert` captured (1.00), then
`retired boundary boundary:scope:project (resolved by a captured expert)` — final state **2 experts, 0
boundaries**. (4) `acme-api` routes to `bunexpert` (top-1, 0.897) and serves **`bun add left-pad`**. One caveat:
stage 4 needed a lower abstention threshold (0.015) because the *general* deps expert shadows the *specialist*
(margin 0.020) — the EXP-004 per-population calibration property; the discrimination (top-1) is correct, only the
serve/abstain threshold is the knob. In the real scenario the npm habit is the *base model*, not a competing
expert, so the specialist routes uncontested.

**Kill criterion.** If any link breaks — the mistake is not stopped, the correction is not captured, the boundary
is not retired, or the fix is not routed — the loop is not autonomous. *(Cleared end-to-end; the only open edge is
the general-vs-specialist routing margin, a known item for the learned gate, ADR-0009.)*

**Reproduce.** The staged run: `teach` generaldeps; `route` acme-api; `scope --discover --expert generaldeps-g0`;
`route` (escalates), `route` webshop (spared); `teach` bunexpert (retires the boundary); `route`/`ask`
`--threshold 0.015` (serves bun). Commits `dc79d54` (relative inhibition) + the retirement change.

## EXP-013 — the learned router

**Claim.** The recurring bottleneck across EXP-004/011/012 is the heuristic gate: it routes by raw cosine to each
expert's capability centroid, and frozen sentence embeddings compress general and specific experts into the same
band, so a specialist barely outscores a generalist and the gate abstains. A router that learns a per-dimension
metric from the population's own exemplars — the same relative idea as the gate, but *learned* instead of
hand-set — should separate them cleanly (ADR-0009, the north-star gate in its routing form).

**Method.** Capture three experts whose tasks overlap heavily: `generaldeps -> npm` (general), `bunexpert ->
acme-api bun`, `yarnexpert -> payments yarn`. Route an `acme-api` task with the heuristic gate, then `gate-train`
(learn a diagonal metric over the 26 exemplars, prototypical cross-entropy, CPU) and route again. The route query
uses a **held-out** package (`react`), so a pass means the router *generalized* the project->expert mapping.

**Result.** Heuristic: `acme-api` -> bunexpert is top-1 (0.967) but generaldeps is right behind (0.917) — margin
0.051, below the 0.08 bar, so the gate **escalates** rather than commit to the specialist. Learned: **bunexpert
p=1.000, generaldeps 0.000** — and `webshop -> generaldeps`, `payments -> yarnexpert`, each p=1.000. The learned
metric turned a 0.051 margin into a clean separation, on a held-out package, so the EXP-012 general-vs-specialist
caveat is resolved. Inference is pure arithmetic in `antumbra-core` (the gate stays light); training is a tiny CPU
model in `antumbra-train`; the router persists per-store and `route`/`ask` use it when present, falling back to
the heuristic gate otherwise.

**Kill criterion.** If the learned router does not outseparate the heuristic gate, or overfits (fails on held-out
tasks), the metric is not worth the retraining cost. *(Cleared: 0.051 -> 1.000 separation, held-out package.)*
Open: probability calibration (p=1.000 is overconfident on 26 exemplars; the routing decision generalizes, the
confidence is sharp), and folding boundary inhibition into the learned path.

**Reproduce.** `teach` generaldeps/bunexpert/yarnexpert, `route` acme-api (heuristic escalate), `gate-train
--epochs 500`, `route` acme-api/webshop/payments (each p=1.000).

## EXP-014 — adapter composition

**Claim.** A population is a *capability multiplier*: several frozen experts can be blended into one served
adapter, with the mix controllable (and ultimately driven by the learned router's weights). Because every expert
shares the base rank and scale, the weighted delta-sum is **exact** as a rank-concatenated adapter — no
weight-space interference, no model surgery (ADR-0009).

**Method.** `compose_adapters` stacks each expert's `sqrt(w_i)`-scaled `A`/`B` factors (A along its rank rows, B
along its rank columns) into one rank-`(sum r_i)` adapter; the existing single-adapter forward serves it (with
`alpha = scale * merged_rank` so the effective LoRA scale is unchanged). Validated by dialing the bun-weight from
0 to 1 on an `acme-api` task (held-out package `react`), composing `generaldeps -> npm` with `bunexpert -> bun`.

**Result.** Endpoints reproduce the individual experts exactly (`generaldeps:1.0 -> npm install react`,
`bunexpert:1.0 -> bun add react`), proving the rank-concatenated merge serves correctly. The blend is
**controllable**: `npm` holds through 0.5 bun-weight and flips to `bun` by 0.7 — a sensible crossover, since bun
must overcome the base's npm prior. So composition is real, exact at the ends, and a continuous behavior dial —
owned control a single model does not expose. A unit test confirms the concatenation realizes the exact weighted
delta-sum.

**Kill criterion.** If a composed adapter does not reproduce its experts at the endpoints, or serving the merged
rank fails, composition is broken. *(Cleared.)* Open frontier — the *complementary* multiplier: blending experts
with **non-conflicting** competences (e.g. a project-convention expert + a code-style expert) so the output does
something *neither alone was trained for*. That is the real capability gain over a flagship and the next
experiment; conflicting behaviors (npm vs bun) can only be dialed between, not combined.

**Reproduce.** `teach` bunexpert + generaldeps, then `compose "<acme-api task>" --experts
"generaldeps-g0:W,bunexpert-g0:1-W"` for W in {1.0, 0.7, 0.5, 0.3, 0.0}.

## EXP-015 — complementary composition (the capability multiplier)

**Claim.** The real prize over interpolation (EXP-014): composing two **complementary** experts produces output
**neither alone was trained for** — the population as a genuine capability multiplier, not just a behavior dial.
This is the personalization layer as composable weights: a standing convention applied *within* project-specific
knowledge, from independently-grown owned experts (ADR-0009; the Antumbra-replaces-the-Kushtaka-harness thesis).

**Method.** Two experts with non-conflicting competences: `bunexpert` (acme-api -> `bun add X`, a *project tool*)
and `convexpert` (-> `... --save-exact`, a *cross-project convention*, trained across varied projects/tools so it
isolates the **flag**, not the tool). Compose at several weights on an `acme-api` task (held-out package `react`).

**Result.** bunexpert alone -> `bun add react` (no flag); convexpert alone -> empty (it never saw `acme-api`).
Composed at **`bun:0.4, conv:0.6` -> `bun add react --save-exact`**: the project's tool (`bun`, from bunexpert)
**and** the convention (`--save-exact`, from convexpert) **combined — which neither expert produces alone.** The
window is narrow (0.5/0.5 and 0.6/0.4 gave just `bun`; the convention only surfaces once conv-weight is high
enough), so the combine depends on weight **calibration** — which the learned router (EXP-013) is the natural
source of, and motivates calibrated routing as the next step. The capability gain is real: composing owned,
independently-trained experts yields behavior no single one holds — the thing a single flagship cannot do with
private, user-grown competences.

**Kill criterion.** If no weighting combines the two competences (only one wins, or output garbles), weight-blend
composition cannot do the complementary multiplier and token-/layer-level routing is required. *(Cleared:
`bun:0.4/conv:0.6` combined both; the narrow window is a calibration item, recorded.)*

**Reproduce.** `teach` bunexpert (`teach-bun.json`) + convexpert (`conv-exact.json`), then `compose
"<acme-api task>" --experts "bunexpert-g0:0.4,convexpert-g0:0.6"`.

## EXP-016 — the closed serving loop (route + auto-compose)

**Claim.** The whole gate+composition axis closes into one serving call: the substrate auto-routes the
*contextual* (project) expert **and** auto-composes the *standing* (convention) experts, applying both at once.
This internalizes Kushtaka's two jobs — recall the relevant fact, enforce the standing rule — into the weights:
contextual experts are task-routed (fact recall), standing experts are always composed (rule enforcement).

**Compass (arXiv).** The direction is grounded in the mixture-of-LoRA literature and sharpened against it.
RAMoLE (2406.16989) retrieves relevant LoRAs from a *dynamically-growing pool* and composes them on the fly —
the closest prior art to Antumbra's population + router + composition. DynMoLE (2504.00661) routes by the
*entropy* of the router distribution (the calibration fix for the EXP-015 narrow window). RouteDK (2508.17250)
routes *complementary* knowledge types and balances contributions to avoid conflict. MoLE (2404.13628) warns
naive arithmetic merging can lose capability — handled here because the experts share rank/scale so the merge is
exact. **What these do not have, and Antumbra does:** experts grown *independently from verified self-improvement*
(RAFT/capture), carrying *counterfactual boundaries* with recovery/retirement, and the *contextual-vs-standing*
split that maps Kushtaka's fact/rule duality onto routing-vs-always-compose.

**Method.** Grow `bunexpert` (acme-api, contextual), `convexpert` (`--save-exact`, standing), `generaldeps`
(npm); `gate-train` the learned router; then one `ask` on an `acme-api` task, with and without the standing
convention (`--with convexpert-g0:0.6`).

**Result.** Routed only: the learned router picks `bunexpert` (p=0.725) -> **`bun add react`**. With the standing
convention: same routing, then auto-composition -> **`bun add react --save-exact`** — the project's tool (routed)
**and** the convention (composed), applied automatically in a single call. The loop capture -> route -> compose
runs end to end with no manual weights at serve time beyond naming the standing experts.

**Kill criterion.** If routing and composition cannot run in one serving call, or the standing convention is not
applied, the substrate is not a usable serving layer. *(Cleared.)* Open: derive the standing-expert weights from
a *calibrated* router (entropy-style, per the compass) so even the convention weight is automatic.

**Reproduce.** `teach` bunexpert/convexpert/generaldeps, `gate-train`, then `ask "<acme-api task>" --with
"convexpert-g0:0.6" --self-weight 0.4`.

## EXP-017 — calibrated router (out-of-distribution abstention)

**Claim.** The learned router's softmax is purely *relative* — it always picks a max, so it routes even a task
far from every expert with high confidence. A production substrate must instead **abstain** on genuinely
out-of-scope tasks (escalate to the flagship/generalist) rather than confidently mis-route them. Calibration: an
*absolute* floor on the nearest-centroid similarity in the learned metric space (selective prediction; DynMoLE's
uncertainty gating, arXiv:2504.00661; the EXP-002 relative-OOD lineage).

**Method.** `gate-train` now computes an in-distribution floor (`mean - 2*std` of how tightly exemplars sit to
their own centroid in the learned space). Route in-distribution package tasks vs out-of-distribution
general-knowledge questions; escalate when `top_similarity < floor`.

**Result.** Floor `0.532`. In-distribution: `acme-api -> bunexpert` (sim 0.748), `webshop -> generaldeps`
(sim 0.686) — both routed. Out of distribution: "capital of France?" (sim 0.410), "joke about cats" (0.424),
"summarize Hamlet" (0.354) — **all escalated**. The tell: the softmax was *overconfident* on OOD (it gave
`generaldeps` p=0.982 for the geography question), but the **absolute** similarity flagged it as uncovered. The
floor separates in-distribution (0.69-0.75) from OOD (0.35-0.42) with a clean margin, so the substrate knows what
it does not know.

**Kill criterion.** If the floor cannot separate in-distribution from out-of-distribution (OOD tasks score above
it, or in-distribution tasks below), the router cannot abstain safely. *(Cleared: clean 0.42-vs-0.69 gap.)* Open:
entropy-gated *multi-expert* activation (route several when the distribution is genuinely spread), which needs
tasks that span contextual experts; the standing-convention weight stays operator-set by design (a rule's
strength is a preference, not an inference).

**Reproduce.** `teach` two experts, `gate-train`, then `route` an in-distribution task vs a general-knowledge
question; the latter escalates.

## EXP-018 — autonomous self-improvement (eval-gated)

**Claim.** The substrate improves itself to a quality bar with no manual per-step driving: train -> evaluate ->
if below target, keep training (warm-started from the prior adapter) -> repeat until it passes or the generation
budget runs out. This is genuinely new over the `GenerationLoop`, which spawns *independent* shadows per
generation: `evolve` adds **eval-gated stopping** + **warm-start continuation**, so the system decides when it is
good enough (on the verifier) rather than running a fixed number of rounds.

**Method.** `evolve --corpus add-only.json --target 0.9 --max-gens 4 --rounds 2`. Each generation: load the
current capability (warm-started from the prior adapter, or the bare base on gen 0), `eval_pass_rate`, and if it
is below target run one RAFT pass and continue; otherwise stop.

**Result.** `gen 0: pass-rate 0.25 (base)` -> below target -> trained -> `gen 1: pass-rate 1.00 (warm from g0)` ->
**converged at gen 1**. The controller measured its own gap, trained, re-measured, and halted at the bar — no
human deciding when or whether to train. The closed loop (sample -> verify -> train -> re-evaluate -> stop) runs
on the verifier alone.

**Kill criterion.** If the pass-rate does not rise across generations, or the loop never halts, it is not
self-improving. *(Cleared: 0.25 -> 1.00, auto-stopped at the target.)*

**Full cycle (2026-06-04).** `evolve` now **persists** the converged capability as a routable expert
(behavior-derived capability vector from the solved exemplars) and refreshes the gate — so self-improvement
feeds the population. Evolving two skills into one store grew `adder` and `reverser` autonomously (each
0.25 -> 1.00, converged at gen 1), and the gate then routed `add -> adder` (sim 0.870) and `reverse -> reverser`
(sim 0.979) while "capital of France" escalated (sim 0.298 < floor 0.850). This surfaced a real bug: with one
exemplar per expert the centroid *is* the exemplar, so std ~ 0 and the OOD floor collapsed to 1.0 (rejecting
everything); fixed with a minimum floor margin (`mean - max(2*std, 0.15)`), a no-op for well-sampled populations
(EXP-017's 0.532 floor is unchanged). Still open: drive `evolve` from *serving* failures across a mixed stream
(route -> serve -> verify -> train only the gaps), rather than one corpus per skill.

**Reproduce.** `evolve --corpus corpora/add-only.json --target 0.9 --max-gens 4 --rounds 2`.

---

*The mechanisms (search, probe, gate, persistence) are unit-proven and exercised end-to-end; the open frontier
is reliability and scale, not fakes. The single fake retired this cycle was the AcceptabilityProbe; what remains
authored is the candidate governing-feature set. Catastrophic forgetting (pillar 3) was probed at toy scale
(EXP-010) and did not appear — the monolith retained the old skill — so the population's headline benefit is now
a measured open question that needs a load-bearing-adapter / interference regime, not an untouched assumption.*
