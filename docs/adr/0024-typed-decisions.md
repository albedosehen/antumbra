# ADR-0024: Typed decisions, and the margin they replace

**Status:** Proposed (gated on the measurement in Validation) · **Date:** 2026-09-22 · **Related:** 0005 (the gate), 0004 (the boundary), 0003 (the critic), 0022 (governed self-improvement and the anchor invariant), 0023 (bounded answers), 0002 (shadows)

## Context

Most of what Antumbra decides is not text. Route or escalate. Is this behaviour in scope here. Is this memory relevant to this query. Did this step help. Each of those is a question with a small, known answer space, and each is currently answered by an uncalibrated scalar compared against a hand-set threshold.

**The gate is the best of them and still has a stated defect.** `antumbra-gate` computes a prototype margin, `sims[0] - sims[1]`, subtracts boundary inhibition, and escalates below `coverage_threshold`. The design is deliberate and cites its sources: relative scoring after RMD (arXiv:2106.09022), because it "defeats the compressed near-OOD cosine band an absolute floor cannot", and the threshold as the selective-prediction risk-coverage knob (arXiv:1705.08500). The record is equally deliberate about what it cannot do:

> a task served *equally well* by two experts has a small margin and will escalate; the prototype-margin conflates "ambiguous between in-scope experts" with "out of scope"

That conflation is structural. One scalar carries two questions, so no threshold separates them. The header names the resolution as ADR-0009's composed model, which is a north star rather than a plan.

**Recall is the worse of them, and was measured.** On 2026-09-22, against `ws:default`: a query built from text taken almost verbatim from a stored memory scored that memory **0.397**, while a one-line stub about an unrelated project scored **0.774** on the same query and ranked first for every query tried, including "banana bread recipe" at 0.774 and "the weather in Reykjavik on a Tuesday" at 0.789. Unrelated content sits at 0.31 to 0.48. A verbatim match therefore lands *inside* the noise band while nonsense sits above it. The cause is ordinary mean-pooling dilution, confirmed by re-storing the offending text and watching the fresh copy score identically: similarity ranks by length, not relevance. ADR-0023 deferred its relevance floor (B-2) for exactly this reason and left the mechanism open.

These are one problem wearing two faces: **an uncalibrated number standing in for a decision that has a type.** The gate is principled about the number and still cannot split two questions out of one. Recall is not principled about the number at all.

Two external systems shipped in September 2026 that answer this shape directly. **Jev** (TypeSafe AI, 15 September) and **Laya** (Convai Innovations, 18 September) are non-autoregressive decision models: given a state and a set of typed questions, they answer all of them in a single forward pass with calibrated probabilities instead of generating prose. Laya is ModernBERT-large plus a two-layer decision head, 421M parameters, Apache 2.0, roughly 33ms on a T4, and its training objective is the part that matters here: reinforcement learning against a **strictly proper scoring rule**, under which expected reward is maximised only by reporting honest probabilities.

## Decision

**Answer the system's non-generative decisions with a typed head trained on verifier outcomes, and keep the verifier as the only source of its labels.**

The governing rule:

> **The typed-decision rule.** A decision with a known answer space is asked as a typed question and answered with a calibrated probability, not inferred from a distance. Every such head is trained on outcomes a verifier produced, and its influence is bounded by its measured agreement with that verifier.

The second clause is not decoration. ADR-0022's anchor invariant forbids reward originating from a signal never checked outside the loop, and a decision head trained on its own past answers, or on the critic's, is precisely that. A head trained on verifier outcomes is anchored and admissible. There is no third option, and this record does not create one.

Three primitives, following Laya's surface because it is the one with open weights: `choice` returns one option from a set with a distribution over all of them, `score` returns an expectation on an ordinal scale, and `noul` returns a calibrated probability that a statement is true.

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

### D-3 · The boundary probe

ADR-0004 models the context-scope of a behaviour as right-here versus wrong-there plus a governing feature. That is a `choice` and a `score`, and `AcceptabilityProbe` is already a port, so the seam exists. This lands after D-1 and D-2, because the thesis of the system is the wrong place to learn how a new instrument behaves.

### D-4 · The critic, deferred with its reason

ADR-0003's densifier scores correctness, which is a `score` question, and a 421M encoder is far cheaper than a 7B shadow. It is deferred because ADR-0022's S-2 binds the critic's influence to its measured rank correlation with verifier outcomes inside a group the verifier has already partitioned, and changing the critic's architecture and its governance in one step would leave neither measured. D-4 waits for S-2.

## Consequences

The escalate decision becomes a probability, so the risk-coverage knob becomes a probability threshold with a meaning rather than a tuned constant with a deployment note.

A second model enters the serving path. It is 421M against a 7 to 8 B base, it shares the resident card, and at 33ms it is cheaper than the generation call it prevents. On the Paradigm sizing it is close to free. It is still one more artefact to version, place and roll back, and `Expert.placed_on` (ADR-0017) is the existing machinery for that.

Laya ships over-confident before temperature calibration, which lands it squarely in ADR-0022's per-generation recalibration step rather than beside it.

The failure mode to watch is a head that is well calibrated on the slices a verifier can see and arbitrary everywhere else. That is the same honest limit ADR-0022 names for the critic, it is unsolved there too, and it is why D-1 keeps the margin as a fallback rather than deleting it.

## Alternatives considered

**Adopt Jev instead.** Rejected, and not on quality. It is closed, API-only, metered per token, with no paper, no weights and no datasets. Antumbra's argument is that a customer's judgment becomes an asset they hold rather than a prompt in someone else's logs; renting the decision layer from a closed API contradicts that at the point it is sharpest. If the open weights were the weaker artefact this would be a real trade, but the open one is the one that can be fine-tuned on a customer's own verified outcomes and served inside their tenancy.

**Tune the existing threshold harder.** This is the control, not the alternative, and Validation requires beating it. A threshold cannot split one scalar into two answers, so it cannot address the conflation whatever it is set to.

**Wait for ADR-0009's composed model.** That is the resolution the gate header names, and it remains the better long-term answer for blending adapters. It is a north star with no date. This record buys the separation now at 421M.

**Train a head from scratch rather than adopting one.** Rejected on cost, not principle. The decision head is two transformer layers over an encoder; the expensive part is the encoder and the calibration objective, both of which arrive under Apache 2.0.

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

   Split by what the labels need. **D-2's half is done**: its labels are constructible from the store by a deterministic verifier (see D-2), so the control is measured and the bar is 0.782 F1 over 240 balanced pairs. **D-1's half is blocked, on data rather than on effort**: sweeping `coverage_threshold` needs routing outcomes, and this deployment holds zero evaluation runs, shadows, boundaries and reward signals because the generational loop has never run on it. That half waits for a loop run, and no amount of care with the gate's code substitutes for it.
2. [ ] **D-2, the relevance floor.** It is the smallest surface, it closes ADR-0023's open B-2, and it is the one place where the current signal is measurably broken rather than merely uncalibrated. Its control has been measured and is recorded in D-2: the best fixed threshold over the deployed cross-encoder reaches 0.950 accuracy with thirteen of fifty negatives outranking the weakest positive, and no temperature fixes an ordering violation. That number is the bar.
3. [ ] **D-1, the gate.** Two questions, with the margin retained as fallback and control.
4. [ ] **D-3, the boundary probe**, once D-1 and D-2 have a calibration history.
5. [ ] **D-4, the critic**, after ADR-0022's S-2, not before.

## Notes on the evidence

Laya's figures are from its own model card and are quoted with its own caveats: 0.766 on typed decisions after fine-tuning against a 0.735 teacher-agreement ceiling, 0.362 zero-shot, sharp degradation past roughly twenty options, weak ordinal scoring, and over-confidence before temperature calibration. The launch benchmarks against Jev were published by one party without access to the other's API, and on Banking77's 77 labels Laya scored 0.425 against Jev's 0.870, which is the large-label-space weakness showing up exactly where the model card says it will. None of those numbers are load-bearing here, because Validation requires Antumbra's own measurement on Antumbra's own outcomes.

The recall figures in Context were measured directly against `ws:default` on 2026-09-22 and are reproducible: store any short stub, query anything unrelated, and watch it rank first.

Those figures predate the corrections landed later the same day and should be read as the diagnosis that prompted them rather than as current behaviour. Three things have since changed under this record: the lexical leg was returning an arbitrary slice of its matches rather than its best ones (a missing `ORDER BY`), the dense leg is now re-ranked by a per-text calibration that lifted top-1 from 2/30 to 13/30 on a thirty-memory benchmark, and a cross-encoder re-scores the fused pool. The stub no longer leads any query. What has NOT changed is the part D-2 rests on: there is still no signal whose magnitude means the same thing across two different queries, which is what the cross-encoder measurement in D-2 establishes and what a floor requires.

The priority dispute around Jev's originality (arXiv:2503.23303 and arXiv:2510.01237, the latter being confidence-aware routing into local, retrieval, larger-model and human pathways) is noted because the routing paper is close to what ADR-0005 already does, and because it is the kind of thing worth knowing before citing either system in front of a customer. It does not bear on the decision.
