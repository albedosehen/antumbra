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

1. [ ] **The measurement, before any integration.** Validation 1 and 2, offline, against existing verifier outcomes. If the head does not beat a tuned threshold, stop here and record it.
2. [ ] **D-2, the relevance floor.** It is the smallest surface, it closes ADR-0023's open B-2, and it is the one place where the current signal is measurably broken rather than merely uncalibrated.
3. [ ] **D-1, the gate.** Two questions, with the margin retained as fallback and control.
4. [ ] **D-3, the boundary probe**, once D-1 and D-2 have a calibration history.
5. [ ] **D-4, the critic**, after ADR-0022's S-2, not before.

## Notes on the evidence

Laya's figures are from its own model card and are quoted with its own caveats: 0.766 on typed decisions after fine-tuning against a 0.735 teacher-agreement ceiling, 0.362 zero-shot, sharp degradation past roughly twenty options, weak ordinal scoring, and over-confidence before temperature calibration. The launch benchmarks against Jev were published by one party without access to the other's API, and on Banking77's 77 labels Laya scored 0.425 against Jev's 0.870, which is the large-label-space weakness showing up exactly where the model card says it will. None of those numbers are load-bearing here, because Validation requires Antumbra's own measurement on Antumbra's own outcomes.

The recall figures in Context were measured directly against `ws:default` on 2026-09-22 and are reproducible: store any short stub, query anything unrelated, and watch it rank first.

The priority dispute around Jev's originality (arXiv:2503.23303 and arXiv:2510.01237, the latter being confidence-aware routing into local, retrieval, larger-model and human pathways) is noted because the routing paper is close to what ADR-0005 already does, and because it is the kind of thing worth knowing before citing either system in front of a customer. It does not bear on the decision.
