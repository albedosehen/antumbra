# ADR-0024: Typed decisions, and the margin they replace

**Status:** Proposed (gated on the measurement in Validation) · **Date:** 2026-09-22 · **Related:** 0005 (the gate), 0004 (the boundary), 0003 (the critic), 0022 (governed self-improvement and the anchor invariant), 0023 (bounded answers), 0002 (shadows)

## Context

Most of what Antumbra decides is not text. Route or escalate. Is this behaviour in scope here. Is this memory relevant to this query. Did this step help. Each of those is a question with a small, known answer space, and each is currently answered by an uncalibrated scalar compared against a hand-set threshold.

**The gate is the best of them and still has a stated defect.** `antumbra-gate` computes a prototype margin, `sims[0] - sims[1]`, subtracts boundary inhibition, and escalates below `coverage_threshold`. The design is deliberate and cites its sources: relative scoring after RMD (arXiv:2106.09022), because it "defeats the compressed near-OOD cosine band an absolute floor cannot", and the threshold as the selective-prediction risk-coverage knob (arXiv:1705.08500). The record is equally deliberate about what it cannot do:

> a task served *equally well* by two experts has a small margin and will escalate; the prototype-margin conflates "ambiguous between in-scope experts" with "out of scope"

That conflation is structural. One scalar carries two questions, so no threshold separates them. The header names the resolution as ADR-0009's composed model, which is a north star rather than a plan.

**Recall is the worse of them, and was measured.** On 2026-09-22, against `ws:default`: a query built from text taken almost verbatim from a stored memory scored that memory **0.397**, while a one-line stub about an unrelated project scored **0.774** on the same query and ranked first for every query tried, including "banana bread recipe" at 0.774 and "the weather in Reykjavik on a Tuesday" at 0.789. Unrelated content sits at 0.31 to 0.48. A verbatim match therefore lands *inside* the noise band while nonsense sits above it. The cause is ordinary mean-pooling dilution, confirmed by re-storing the offending text and watching the fresh copy score identically: similarity ranks by length, not relevance. ADR-0023 deferred its relevance floor (B-2) for exactly this reason and left the mechanism open.

These are one problem wearing two faces: **an uncalibrated number standing in for a decision that has a type.** The gate is principled about the number and still cannot split two questions out of one. Recall is not principled about the number at all.

Two external systems shipped in September 2026 that answer this shape directly. **Jev** (TypeSafe AI, 15 September) and **Laya** (Convai Innovations, 18 September) are non-autoregressive decision models: given a state and a set of typed questions, they answer all of them in a single forward pass with calibrated probabilities instead of generating prose. Laya is ModernBERT-large plus a two-layer decision head, 421M parameters, Apache 2.0, roughly 33ms on a T4, and its training objective is the part that matters here: reinforcement learning against a **strictly proper scoring rule**, under which expected reward is maximised only by reporting true probabilities.

## Decision

**Answer the system's non-generative decisions with a typed head trained on verifier outcomes, and keep the verifier as the only source of its labels.**

The governing rule:

> **The typed-decision rule.** A decision with a known answer space is asked as a typed question and answered with a calibrated probability, not inferred from a distance. Every such head is trained on outcomes a verifier produced, and its influence is bounded by its measured agreement with that verifier.

The second clause is not decoration. ADR-0022's anchor invariant forbids reward originating from a signal never checked outside the loop, and a decision head trained on its own past answers, or on the critic's, is precisely that. A head trained on verifier outcomes is anchored and admissible. There is no third option, and this record does not create one.

Three primitives, named after Laya's surface because it is the one that demonstrated them: `choice` returns one option from a set with a distribution over all of them, `score` returns an expectation on an ordinal scale, and `noul` returns a calibrated probability that a statement is true.

The head is Antumbra's own, for the reasons set out under Alternatives: the encoder is Apache-2.0 and comes from elsewhere either way, the head is two layers, the calibration objective is a loss rather than an artefact, and a checkpoint that scores near chance zero-shot must be trained on this system's verifier outcomes to be worth anything. Borrowing the vocabulary is free; borrowing the weights buys a warm start and costs ownership of a judgment this system exists to own.

### D-1 · The gate asks two questions instead of reading one number

The conflation dissolves when the two questions are separated rather than summed. A `choice` over the candidate experts answers *which one*, and a `noul` answers *is any of these actually in scope*. The margin never could do both; two questions do it without a composed model and without waiting for ADR-0009.

The gate keeps its current path as the fallback and as the control. `coverage_threshold` is not deleted, because a tuned threshold is the thing the head has to beat (see Validation), and because a head that fails to load must degrade to something that works rather than to nothing.

### D-2 · The relevance floor ADR-0023 could not specify

B-2 asked for a definitive empty state and could not say how to decide emptiness, because the only available signal ranked by length. A `noul` asking whether a recalled memory answers the query returns a calibrated probability in about 33ms, which is the instrument that gap needed. The floor then reads as "no result cleared the floor" rather than as five weak rows for the caller to judge.

Note what this does **not** change: hybrid recall stays. BM25 is currently the only reason a verbatim query returns the right memory at all, and a typed head re-ranks what recall retrieves rather than replacing retrieval.

**The obvious cheaper answer was tried first, and does not work.** Since this record's Context was written, recall gained two corrections: the dense leg is re-ranked by a per-text calibration before fusion, and a cross-encoder (`BAAI/bge-reranker-base`, served by text-embeddings-inference) re-scores the fused pool. A cross-encoder emits a relevance score per `(query, memory)` pair, which is superficially the instrument D-2 asks for, and it is already in the serving path. If it could carry the floor, this record's D-2 would be unnecessary.

It cannot, and the reason is the same shape as the defect it was brought in to fix. Measured on ten real memories: against one fixed nonsense query the separation looks perfect, every irrelevant pair landing within 3.741e-5 to 3.744e-5 while every on-topic pair scored at least 8.6 times higher. Vary the nonsense instead, and the floor moves: the same memory scores 3.7e-5 against "banana bread recipe" and 7.9e-4 against "which strings to use on a fretless bass", a spread of twenty-one times. The weakest genuine match in the first set scored 3.2e-4, which sits *inside* that band. A fixed threshold therefore admits some nonsense and rejects some answers, whichever value it takes.

The cross-encoder is reliable for ORDERING candidates within one query and unreliable as an absolute magnitude ACROSS queries. A floor is an across-query comparison by construction — it must mean the same thing for every query the caller asks. That is precisely the distinction between a ranking signal and a calibrated probability, and it is why `noul` is specified as the latter. The same finding applies to `similarity` (ADR-0023 B-2) and now to the reranker: this system has two ranking signals and no calibrated one.

**So the control was measured, on a set built the way D-2's own labels have to be.** The obvious cheaper move is a threshold on the reranker's output. Whether it suffices is an empirical question, and answering it needed labelled data this deployment does not have — no evaluation run, shadow, boundary or reward signal exists in the store, because the generational loop has never run here.

That blocker is real for D-1 and not for D-2, and the difference is worth stating because it is what makes this record partly measurable today. **D-2's labels can be constructed by a deterministic verifier over the store itself.** A query cut from inside a memory has that memory as its answer and does not have any other memory as its answer. The verifier is substring provenance: checkable, reproducible, and not derived from any model, which is exactly what the typed-decision rule demands of a label. No loop is required.

Built that way over 120 memories, with a hard negative for each — the same query against a different memory, rather than a nonsense query nothing would match — the deployed cross-encoder gives:

| | |
|---|---|
| best single threshold | 6.35e-4 |
| accuracy | 0.800 |
| precision | 0.860 |
| recall | 0.717 |
| F1 | 0.782 |
| always-reject baseline | 0.500 (classes balanced, 120/120) |

**That is the bar, and it is a real one.** An earlier pass of this measurement used ten positives against fifty nonsense negatives and reported 0.950 accuracy against an 0.833 always-reject baseline, which read as failure; the number was an artefact of the class imbalance and of negatives too easy to be informative. On a balanced set with hard negatives the control is a usable floor rather than a broken one.

It is also a lossy floor, and the losses are the argument for going further. Recall 0.717 means **more than a quarter of genuine answers fall below the best threshold** and would be reported as "nothing matched". The score ranges overlap at the bottom — the weakest positive and the weakest negative are both 3.73e-5 — so no threshold separates them cleanly and no monotone recalibration can, since temperature scaling cannot reorder a pair. Four of 120 hard negatives still outscore the median positive.

So Validation 2's bar for D-2 is 0.782 F1, and a typed head earns its place in the serving path by beating it. That is now a number rather than an argument.

> **Superseded — read "The benchmark was answerable by `grep`" below before using any figure in this subsection.** This 0.782 and every score compared against it were measured on a label set a `contains` check solves at F1 1.000. The corrected bar is **0.785**, and the corrected conclusions are in that later section. The paragraphs here are kept because the reasoning that produced them is sound and the flaw was in the data, which is the more useful thing to be able to see.

**The cheap head was tried against that bar and fails, for a reason that rules out a whole class of shortcut.** Before paying for a 400M encoder, the question worth asking is whether the one already in this stack carries enough signal: all-MiniLM-L6-v2 is 22M, loaded, tested, and free. A two-layer head over its frozen output, trained on 800 constructed pairs with the split taken by MEMORY rather than by pair — so the same passage never appears on both sides — scores **0.525 F1, accuracy 0.530** on the held-out half. Balanced classes make chance 0.500. It is barely distinguishable from guessing.

The first explanation offered for that was the pooling: `BertEmbedder::encode_pooled` mean-pools, so feeding it `query: … passage: …` yields a blend of two texts rather than a representation of their relationship. **That explanation was tested and is wrong**, which matters more than the original number.

If pooling a concatenation were the constraint, handing the head the two vectors and their interaction terms explicitly should recover most of it — `[u, v, |u-v|, u⊙v]` is the standard bi-encoder recipe for sentence-pair classification and exists precisely to supply the comparison a cross-encoder gets from attention. Measured on the same 800 pairs and the same split:

| features | accuracy | F1 |
|---|---|---|
| joined, mean-pooled | 0.517 | 0.519 |
| separate, `[u, v, \|u-v\|, u⊙v]` | 0.548 | **0.520** |
| control (cross-encoder threshold) | — | **0.782** |

Both sit on chance. The interaction terms buy three points of accuracy and nothing at all in F1, so **the pairing is not the binding constraint**.

What is left is the representation itself, and it is a defect this record's own Context already measured from the other side. A frozen MiniLM vector of a long memory is a mean-pool of up to 512 tokens into 384 dimensions, and that is the same dilution that makes a 66-character stub score 0.774 against a query about nothing while a 1058-character memory scores 0.224 against a query about its own contents. **The specific twelve-word span the query was cut from does not survive into the vector.** No function of `u` and `v` can recover information neither vector contains, which is why a better pairing changes nothing and why a larger encoder pooled the same way would not help either.

That leaves two routes, and they are the two this session has already found by other paths. Cross-attention: the model reads both texts together and attention locates the matching span, which is what a fine-tuned `ModernBertClassifier` with `ClassifierPooling` does and what the reference design is. Or chunking, so each indexed unit is short enough that its vector still describes it — which is the `memory` / `document_chunk` asymmetry named under D-2 above.

### The benchmark was answerable by `grep`, and the bar above it was too

**Everything from here corrects what is written above. The 0.782 figure, and every score compared against it, was measured on a label set a one-line `contains` check solves perfectly.**

`scripts/d2-labels.sh` built a query by cutting twelve words out of a memory and leaving them in place. So a positive pair was a passage containing its own query verbatim. The task was never "does this memory answer this query"; it was "does this string appear in this text", and the deterministic verifier that made D-2 measurable at all is exactly what made it trivial.

It was found by disbelieving a good result. Chunking was swept across sizes once a GPU made a sweep cost seconds, and the head appeared to beat the control outright — 0.940 F1 at 150 characters, 0.862 at 200, 0.730 at 300. A frozen 22M bi-encoder with a two-layer head does not beat a 278M cross-encoder by sixteen points. The curve was **monotonic in smaller chunks**, which named the mechanism: the smaller the chunk, the more exactly it *is* the query. One `contains` check confirmed it at accuracy 1.000, precision 1.000, recall 1.000.

**The fix keeps the determinism and removes the shortcut: the twelve words are excised from the memory they were cut from.** What remains is the passage around them, which concerns the query's subject without containing its words. The verifier stays checkable, reproducible and model-free — the properties that let D-2 be measured on a deployment with no evaluation runs — while no longer being answerable by grep. `scripts/d2-relevance-baseline.sh` now *reads* that label file rather than rebuilding it, because the two scripts each carried a comment insisting their constructions be identical, and they were: identical and identically wrong.

On the clean set, 800 pairs, 400 held out, split by memory:

| | accuracy | F1 (clean) | F1 (degenerate) |
|---|---|---|---|
| no model — `contains` | 0.500 | **0.000** | **1.000** |
| joined, mean-pooled | 0.517 | 0.519 | 0.519 |
| separate, `[u, v, \|u-v\|, u⊙v]` | 0.548 | 0.522 | 0.522 |
| best chunk, 150 | 0.608 | 0.594 | 0.940 |
| best chunk, 200 | 0.627 | **0.613** | 0.862 |
| best chunk, 300 | 0.570 | 0.525 | 0.730 |
| best chunk, 450 | 0.558 | 0.540 | 0.599 |
| **control (cross-encoder)** | 0.802 | **0.785** | 0.782 |

**The most informative row is the control's.** The cross-encoder scores 0.785 on the clean set against 0.782 on the degenerate one — it never touched the shortcut, and was doing real relevance judgement the whole time. The frozen-encoder head fell from 0.940 to 0.613 because exploiting the shortcut was all it was doing. That asymmetry is the strongest evidence in this record that the control is a real instrument and the cheap head is not.

**So the conclusion is unchanged and its support is not.** Chunking helps, 0.519 to 0.613, which is real but modest; the best chunk size is now 200 and the curve peaks in the middle rather than running to the smallest chunk — agreeing in shape with an independent chunk-and-max-pool measurement for ranking. Chunking remains **necessary and not sufficient** against a 0.785 bar, and a fine-tuned pair encoder with cross-attention is still what D-2 needs.

That the answer survived is luck, not vindication. For one run the evidence said the opposite and said it loudly, and the only thing standing between that run and a decision was refusing to believe a number that was too good.

The harness now prints the `contains` line on every run and the baseline script says `DEGENERATE` if it clears 0.9, so this particular hole cannot reopen quietly. The general lesson is on the record in `docs/adr/0024-typed-decisions.md` and in the probe's own documentation: **a constructed benchmark must be checked against a no-model baseline before any model is compared against it.**

### D-3 · The boundary probe

ADR-0004 models the context-scope of a behaviour as right-here versus wrong-there plus a governing feature. That is a `choice` and a `score`, and `AcceptabilityProbe` is already a port, so the seam exists. This lands after D-1 and D-2, because the thesis of the system is the wrong place to learn how a new instrument behaves.

### D-4 · The critic, deferred with its reason

ADR-0003's densifier scores correctness, which is a `score` question, and a 421M encoder is far cheaper than a 7B shadow. It is deferred because ADR-0022's S-2 binds the critic's influence to its measured rank correlation with verifier outcomes inside a group the verifier has already partitioned, and changing the critic's architecture and its governance in one step would leave neither measured. D-4 waits for S-2.

## Consequences

The escalate decision becomes a probability, so the risk-coverage knob becomes a probability threshold with a meaning rather than a tuned constant with a deployment note.

A second model enters the serving path. It is 421M against a 7 to 8 B base, it shares the resident card, and at 33ms it is cheaper than the generation call it prevents. On the Paradigm sizing it is close to free. It is still one more artefact to version, place and roll back, and `Expert.placed_on` (ADR-0017) is the existing machinery for that.

Laya ships over-confident before temperature calibration, which lands it squarely in ADR-0022's per-generation recalibration step rather than beside it.

The failure mode to watch is a head that is well calibrated on the slices a verifier can see and arbitrary everywhere else. That is the same limit ADR-0022 names for the critic, it is unsolved there too, and it is why D-1 keeps the margin as a fallback rather than deleting it.

## Alternatives considered

**Adopt Jev instead.** Rejected, and not on quality. It is closed, API-only, metered per token, with no paper, no weights and no datasets. Antumbra's argument is that a customer's judgment becomes an asset they hold rather than a prompt in someone else's logs; renting the decision layer from a closed API contradicts that at the point it is sharpest. If the open weights were the weaker artefact this would be a real trade, but the open one is the one that can be fine-tuned on a customer's own verified outcomes and served inside their tenancy.

**Tune the existing threshold harder.** This is the control, not the alternative, and Validation requires beating it. A threshold cannot split one scalar into two answers, so it cannot address the conflation whatever it is set to.

**Wait for ADR-0009's composed model.** That is the resolution the gate header names, and it remains the better long-term answer for blending adapters. It is a north star with no date. This record buys the separation now at 421M.

**Train a head from scratch rather than adopting one.** Originally rejected on cost, on the grounds that "the expensive part is the encoder and the calibration objective, both of which arrive under Apache 2.0". **Reading the artefact overturns the premise, and this is now the recommended path.** The question that settles it is what Laya supplies that this system cannot build, taken item by item.

*The encoder does not come from Laya.* `rl_agent_config.json` names `answerdotai/ModernBERT-large` and the repository does not vendor it. Antumbra fetches the same Apache-2.0 weights from the same place whether it adopts Laya or not. That half of the stated cost was never Laya's to charge.

*The head is two layers.* This record says so itself. `candle-transformers` already ships `modernbert.rs` with `ModernBertClassifier`, so the encoder path needs no dependency and no second runtime; what remains is a small head, and Laya's is custom enough that adopting it means porting rather than loading.

*The calibration objective is a loss, not an artefact.* A strictly proper scoring rule is a published technique, and `antumbra-train/src/objective.rs` is already the place where this system writes a loss against candle tensors. It arrives under Apache 2.0 in the sense that a textbook does.

*The pretrained weights are the only irreplaceable part, and this record already prices them at near zero.* Laya scores 0.362 zero-shot, near chance, which is why "use it zero-shot" is rejected above. Validation requires a head fine-tuned on Antumbra's own verifier outcomes. If the checkpoint must be retrained on this system's data to be useful, it is a warm start rather than a capability.

There is a thesis argument underneath the arithmetic, and it is the same one that rejects Jev. Antumbra's whole claim is that a customer's judgment becomes an asset they hold, built by training small specialists on verified outcomes. **A typed decision head is exactly that specialist.** Building it in-house is not a detour from the architecture; it is the architecture applied to a new question type. Adopting someone else's decision model — even an open one — puts a judgment this system was built to own inside an artefact it did not train and cannot re-derive.

What Laya is still worth is INFORMATION rather than dependency: evidence the approach works at all (0.766 after fine-tuning against a 0.735 teacher-agreement ceiling), a reference head shape, its shipped temperature table as a sanity check on our own calibration, and the degradation past roughly twenty options that Validation 6 already encodes. Read it, measure against it, do not depend on it.

**Use it zero-shot.** Rejected by the vendor's own number: 0.362 on typed decisions, near chance. Laya is a base to fine-tune. That is fatal for an adopter wanting a drop-in API and irrelevant here, because training small specialists on verified outcomes is what this system already is.

## Validation

1. **The control is a tuned threshold, not the shipped one.** Sweep `coverage_threshold` on held-out tasks and record the best risk-coverage curve it achieves. That curve is the bar.
2. A head fine-tuned on existing verifier outcomes beats that curve on **escalation precision and recall**, measured on the same held-out slice, or this record is rejected and says so.
3. The two-question form resolves the documented conflation: on tasks served equally well by two experts, `choice` disagrees between them while `noul` reports in-scope, where the margin escalated.
4. Calibration is reported as expected calibration error **sliced** by task family and answer space size, not as a global average, following ADR-0022's insistence that a global number hides a broken slice.
5. On recall, a verbatim query places its own memory above the noise band, which is the measurement this record's Context shows failing today.
6. Every `choice` question in the system has an answer space under twenty options, asserted by a test, because options share a fixed token budget and accuracy collapses on large label spaces.
7. A head that fails to load degrades to the margin path, proven by a test that removes the artefact.

## Order of work

1. [~] **The measurement, before any integration.** Validation 1 and 2, offline, against existing verifier outcomes. If the head does not beat a tuned threshold, stop here and record it.

   Split by what the labels need. **D-2's half is done, and it returned a negative result, which is the outcome this item explicitly provides for.** Its labels are constructible from the store by a deterministic verifier (see D-2), so the control is measured and the bar is **0.785 F1** over 800 balanced pairs with hard negatives. The cheapest candidate — a two-layer head over the frozen MiniLM already in the stack — was then measured against that bar and does not approach it, across seven featurings including a chunk-size sweep. So this item's own instruction applies: the finding is recorded rather than integrated, and D-2 does not ship on a frozen encoder.

   It took two passes to get there, and the second is the one worth remembering. The first label set left the query verbatim inside its positive memory, so `grep` scored F1 1.000 on it and a chunk sweep appeared to beat the control; the span is now excised and the harness prints the no-model check on every run. **A constructed benchmark gets a no-model baseline before any model is compared against it** — that rule is the durable output of this item, more than either number.

   **D-1's half is blocked, on data rather than on effort**: sweeping `coverage_threshold` needs routing outcomes, and this deployment holds zero evaluation runs, shadows, boundaries and reward signals because the generational loop has never run on it. That half waits for a loop run, and no amount of care with the gate's code substitutes for it.

   **The loop now leaves populations behind, and the sweep is built.** ADR-0022's GPU runs train populations on the workbench corpus, and each run keeps its store and adapters.
   - **Routing:** `antumbra gate-sweep` routes every task the default partition withholds over such a population, with the threshold out of the way, keeping each task's expert and margin.
   - **Scoring:** that expert and the base model answer each task under the same seeds.
   - **The sweep:** the threshold is then swept offline, with escalations answered by the base model, the stand-in for the agent above.
   - **The report:** the curve's best point is the control, and the shipped threshold is printed beside it.

   `scripts/gate-sweep.sh` runs it over a run directory.

   **The first sweep ran, and it measured nothing about the gate.** It ran on 2026-09-25 over the S-3 comparison's credit arm. That population holds one expert, because admission turned every specialist away.
   - **The data:** 97 withheld tasks, the expert scoring 0.573 and the base model 0.366.
   - **Why it is degenerate:** with one expert the gate's relative coverage falls back to absolute similarity, which cleared the threshold on every task. So the best threshold and the shipped one both route everything.
   - **What the control needs:** a population where the margin separates experts, which the grow step's warm start is being run to produce. The sweep runs again over that.

   **The second sweep ran over four experts, and it sets the control at routing everything.** It ran on 2026-09-26 over the uniform arm of ADR-0022's grow comparison under learned-router admission (0331bb2). That population is a generalist and three specialists warm-started from it, each 0.93 to 0.94 like it. The data: 97 withheld tasks, two seeds of four samples.

   | threshold | share routed | accuracy | risk on what is routed |
   | --- | ---: | ---: | ---: |
   | none routed (the base model alone) | 0.00 | 0.366 | - |
   | shipped, 0.08 | 0.15 | 0.412 | 0.367 |
   | best, 0.0003 | 1.00 | 0.616 | 0.384 |
   | all routed | 1.00 | 0.616 | 0.384 |

   - **The shipped margin escalates what it should route.** With experts this alike, the top-two margin is under 0.08 on 85% of the tasks. So the shipped threshold sends them to the base model, and accuracy falls from 0.616 to 0.412.
   - **The margin carries no abstention signal here.** Every expert beats the base model on nearly every task, so the best point on the curve is to route everything. Nothing the margin separates is worth escalating.
   - **What this does and does not measure:** it is the heuristic gate's threshold. A population of two or more is served by the learned router, whose out-of-distribution floor is a different signal, and the sweep does not read it.
   - **The bar it sets for D-1's head:** 0.616 at full coverage and 0.366 at none, on these tasks. A typed gate earns its place only by abstaining where the population would be wrong, and on this population there is little such room.

   **The learned router now learns where tasks should have gone.** Two grow runs ended with the population 0.07 under what its own experts would score routed as well as they could be (ADR-0022 S-3). Meanwhile the router learned only from the text of what each expert had solved. That is the *which one* half of this decision, and the measurement already holds its labels.
   - **The labels:** a contribution measurement with the baseline on scores every expert on every live task. A task one expert beat every other expert on by at least 0.25 (and the base model, where it was scored) is kept as that expert's routing outcome. Each new measurement replaces it, and a task with no clear winner any more loses it.
   - **The training:** the router retrains with each won task's prompt as an exemplar of its winner, beside the capability exemplars, whenever the winners change. `gate-train` does the same unless given `--exemplars-only`. `gate-outcomes` records the winners on a population no loop measured this way.
   - **Where it is judged:** the live tasks now shape routing, so the population's score and headroom on them read high from then on. Routing is judged on the withheld tasks instead. `gate-sweep --learned` routes them with the stored learned router. Its coverage is the nearest-centroid similarity, and it ships at the router's own floor.
   - **The comparison:** one population's router trained with `--exemplars-only` and without, swept on the same withheld tasks under the same seeds.
   - **What would refute it:** the outcome-trained router scoring no higher than the exemplar-only one at full coverage on the withheld tasks. Then the wins taught the live tasks and nothing that carries to new ones, and they come out of the router's training.
2. [~] **D-2, the relevance floor.** It is the smallest surface, it closes ADR-0023's open B-2, and it is the one place where the current signal is measurably broken rather than merely uncalibrated.

   **A floor now ships, and it is not the head this record specifies.** `antumbra-rerank/src/floor.rs` answers `Noul` by mapping the deployed cross-encoder's score through a logistic fitted on the same verifier labels, reaching 0.803 accuracy and 0.797 F1 on a held-out half against the 0.785 the best in-sample threshold manages, with an expected calibration error of 0.033. It satisfies the `TypedDecider` contract on the contract's own terms: log loss is strictly proper, and the labels come from a deterministic verifier rather than from any model's answers.

   That changes what remains rather than removing it. **The cheap thing turned out to be a calibration, not an architecture** — the record assumed a floor needed a new model, and what it needed was a fitted map over a signal already in the serving path. The head is still wanted for what a cross-encoder cannot do: `Choice` and `Score` have no cross-encoder analogue, and D-1's gate is built from exactly those. So D-2's floor is delivered and D-2's head is now a D-1 dependency rather than a floor dependency.

   What is settled: the bar is **0.785 F1** (D-2's corrected table), the frozen-encoder shortcut is closed, and chunking is established as necessary and not sufficient — best at 200 characters, worth 0.519 to 0.613, against 0.785. What remains is the part that was never cheap — a pair encoder that reads both texts together, which is `ModernBertClassifier` with `ClassifierPooling` from the `candle-transformers` already in the tree, fine-tuned on constructed pairs. `scripts/d2-labels.sh` produces its training data today and `crates/antumbra-serve/src/decision_probe.rs` already holds the training and scoring harness to judge it by, so what is missing is the encoder and the fine-tuning run, not the measurement apparatus.

   Two constraints on doing it, both learned rather than assumed. It needs a GPU: the 22M encoder took 161 minutes on CPU for an 800-pair three-way run and did not finish, and ModernBERT-large is 400M with a training pass rather than inference only. And the floor it feeds must sit on the fused score, never on `similarity` — ADR-0023's B-2 correction, which holds for the reranker too.
3. [ ] **D-1, the gate.** Two questions, with the margin retained as fallback and control.
4. [ ] **D-3, the boundary probe**, once D-1 and D-2 have a calibration history.
5. [ ] **D-4, the critic**, after ADR-0022's S-2, not before.

## Notes on the evidence

**What adopting it would actually cost, read off the artefact rather than the card.** Checked on 2026-09-22, because "adopt the open one" is the load-bearing half of the Jev comparison and the shape of the repository decides how much of it transfers.

The weights are not where the name suggests. `convaiinnovations/laya` is card-only — a README and logo assets, no config and no tensors. The artefact is `convaiinnovations/laya-typed-decisions`, which carries `model.safetensors`, `tokenizer/`, and two configs. Apache-2.0 as claimed. Both repositories report zero downloads through the HF API, which is not evidence of quality either way but is worth knowing before citing the model in front of a customer.

`rl_agent_config.json` confirms the card's architecture and adds the parts that decide the port: `encoder: answerdotai/ModernBERT-large`, `head_layers: 2`, `max_len: 1024`, `head_max_len: 256`, `max_prefixes: 6`, and an RL cost structure (`act_costs.escalate: 0.5`, `cost_wrong_act: 3.0`, `amp_dtype: bf16`). **The encoder is referenced, not vendored**, so adopting Laya means fetching ModernBERT-large separately; it is Apache-2.0 too, so this is a step rather than an obstacle.

It also ships its temperature calibration — a `temperature` array and a `temperature_by_options` map keyed by how many options a question offers. So the card's over-confidence caveat is something the artefact addresses rather than something an adopter inherits, and the per-option-count keying is itself an argument that calibration varies with answer-space size, which is what Validation 4 asks to be sliced by.

**The encoder path is in-stack and the head is not.** `candle-transformers` 0.10.2 ships `modernbert.rs` with `ModernBertClassifier`, `ClassifierConfig` and `ClassifierPooling`, so serving a ModernBERT classifier in-process needs no new dependency and no second runtime. Laya's head is not that classifier: it is a custom two-layer decision head with typed-question prefixes and its own RL configuration, so it must be ported rather than loaded. The estimate is therefore encoder free, head written, and the calibration table usable as data.

Laya's figures are from its own model card and are quoted with its own caveats: 0.766 on typed decisions after fine-tuning against a 0.735 teacher-agreement ceiling, 0.362 zero-shot, sharp degradation past roughly twenty options, weak ordinal scoring, and over-confidence before temperature calibration. The launch benchmarks against Jev were published by one party without access to the other's API, and on Banking77's 77 labels Laya scored 0.425 against Jev's 0.870, which is the large-label-space weakness showing up exactly where the model card says it will. None of those numbers are load-bearing here, because Validation requires Antumbra's own measurement on Antumbra's own outcomes.

The recall figures in Context were measured directly against `ws:default` on 2026-09-22 and are reproducible: store any short stub, query anything unrelated, and watch it rank first.

Those figures predate the corrections landed later the same day and should be read as the diagnosis that prompted them rather than as current behaviour. Three things have since changed under this record: the lexical leg was returning an arbitrary slice of its matches rather than its best ones (a missing `ORDER BY`), the dense leg is now re-ranked by a per-text calibration that lifted top-1 from 2/30 to 13/30 on a thirty-memory benchmark, and a cross-encoder re-scores the fused pool. The stub no longer leads any query. What has NOT changed is the part D-2 rests on: there is still no signal whose magnitude means the same thing across two different queries, which is what the cross-encoder measurement in D-2 establishes and what a floor requires.

The priority dispute around Jev's originality (arXiv:2503.23303 and arXiv:2510.01237, the latter being confidence-aware routing into local, retrieval, larger-model and human pathways) is noted because the routing paper is close to what ADR-0005 already does, and because it is the kind of thing worth knowing before citing either system in front of a customer. It does not bear on the decision.
