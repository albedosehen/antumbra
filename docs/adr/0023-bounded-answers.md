# ADR-0023: Bounded answers, and the budget already written down

**Status:** Proposed · **Date:** 2026-09-22 · **Related:** 0015 (the runtime surface), 0021 (sovereign mode: the session hook and the MCP lint), 0012 (Penumbra memory), 0014 (compartments), 0018 (provenance over extraction)

## Context

Antumbra's runtime surface answers into a context window it does not own. Every tool in `antumbra-mcp` returns rows to an agent whose budget is finite, shared, and already spoken for by the time Antumbra is asked anything. No record has ever said what the surface owes that budget, and the omission has produced one live defect and one silent hazard.

**The surface returns memory content unbounded.** `MemoryView` (`crates/antumbra-mcp/src/server/params.rs:81`) serialises the full `content` of every result. `RecallParams` carries `query`, `top_k`, `network`, `repo` and `branch`, and no way to ask for less or for more. There is no truncation anywhere in the crate: the only occurrences of the word narrow the candidate *pool* to `k`, never the text. A memory's content is prose an agent wrote through `store_memory`, so its length is set by a caller's earlier input and is bounded by nothing.

**Antumbra has already written the budget down, and enforces it on everything except this.** `brief.rs:10` states the constraint plainly:

> It is short because it has to be: a hook's context is capped at 10,000 characters for everything the hook says, and this shares that room with the memories the hook recalls.

The session brief holds itself to `const BUDGET: usize = 2_000`, checked by a test, with a deliberate note that run-time enforcement would be wrong there because "cutting a path short would be worse than running long". The memories that share the other 8,000 characters are held to nothing at all. That asymmetry is the defect: the one thing whose size Antumbra controls is budgeted, and the one thing whose size a caller controls is not.

**Past the cap, the failure is silent.** `bridge.rs:4` records what was measured when the first plan was to pass `AGENTS.md` through the same hook: past the cap "the agent is handed a file path and a 2,000-character preview it is never asked to open", so an 11,468-character file "would have been cut to a fifth, silently". That measurement was taken for a file, and whether the recall path degrades identically has never been measured. It does not need to be. A surface that can exceed a cap whose overflow behaviour is recorded in this repository as silent is a surface that should not be able to exceed it. Three memories of the length this project routinely writes, plus a 2,000-character brief, clear 10,000 without anything reporting that they did — and the thing most likely to be discarded is the brief, which was budgeted precisely so it would survive.

**Nothing is not said as nothing.** The doc comment on `MemoryView::similarity` names the second gap while solving a different one:

> recall always returns `top_k` whether or not anything was relevant, so only this distinguishes a near match from the best of a bad lot

The problem was seen and answered with a measurement the caller must interpret rather than a statement the caller can act on. An agent that cannot tell "nothing matched" from "five weak matches" re-runs with different flags to find out, which costs a turn and more context than the answer did.

This record was prompted by [AXI](https://axi.md), a design spec for agent-facing CLI tools. Most of it does not apply here and the parts that do are noted with the rest of the evidence below; what it supplied was the vocabulary for a defect that was already in the tree.

## Decision

**A surface that answers into a context window is accountable for what it spends there.**

The governing rule:

> **The budget rule.** Any field whose size is set by a caller's earlier input is bounded by default, the bound is visible in the answer, and there is one documented call that lifts it. A surface may spend more than its default only because a caller asked it to.

Three consequences of that rule, in the order they should be built.

### B-1 · A recalled memory is bounded, and says what it left out

`content` is truncated to a default prefix. The row carries the original length and a marker that it was cut, so the agent can see the cut rather than infer it. `RecallParams` gains `full: Option<bool>`; `full: true` returns the whole text and is the documented way to get it.

**The cut is a prefix, never a summary.** No model goes in the recall path. A summarising step would put an unverified signal between the store and the agent, which is the shape ADR-0003 and ADR-0022 exist to refuse, and it would make two identical recalls return different text. A prefix is deterministic, cheap, and plain about being partial.

**Run-time enforcement, not a test-held budget.** `brief.rs` is right to hold its budget with a test rather than a cut, because a path cut short is worse than a path that runs long. Prose is the opposite: a memory cut short is still a usable memory, and the marker tells the agent exactly how to get the rest. So this bound is enforced where it is produced.

**The bound is on content, because that is where the cost is.** A `MemoryView` envelope is eight short fields, on the order of forty to sixty tokens. The prose inside `content` is two hundred to a thousand and more. Re-encoding the envelope in a denser notation saves roughly a dozen tokens a row; bounding the content saves hundreds. Any future work on this surface that attacks the envelope before the content is optimising the wrong axis, and this paragraph exists so that argument does not have to be had twice.

### B-2 · Nothing is said as nothing

Recall gains a relevance floor. When nothing clears it, the answer says so in a form the agent can branch on, rather than returning `top_k` rows for the caller to judge. Results that clear the floor are unchanged.

The floor is a default the caller may lower, because "the best of a bad lot" is occasionally what a caller wants. What it may not be is the only option.

**The floor is on the fused score, and never on `similarity`.** This paragraph corrects the one it replaces, which said `similarity` "still says how good the answer is". It does not, and a floor built on it would cut correct answers and keep nonsense. Two measurements on the live store settle it.

First, `similarity` is not the ordering key. `recall_memories` ranks by reciprocal rank fusion over a dense leg and a BM25 leg; the number attached to each row is the raw dense cosine, computed after fusion for display. Results routinely come back with the first hit scoring lowest, which reads as a ranking fault and is not one.

Second, that number tracks a memory's LENGTH more than its topic. A 66-character stub scores 0.774 against "banana bread recipe", a query it has nothing to do with, while a 1058-character memory scores 0.224 against a query about its own contents. The curve is monotonic: 66 chars 0.774, 130 chars 0.655, 180 to 250 chars 0.42 to 0.48, 300 to 330 chars 0.31 to 0.40, 1000 to 1900 chars 0.05 to 0.17. On one real recall the two on-topic memories held the two lowest cosines in the result (0.219 and 0.104) while three off-topic rows scored higher (0.866, 0.720, 0.561). A floor anywhere between those bands deletes both correct answers and keeps all three wrong ones. The correlation is not weak; it is inverted.

The cause is mean-pooling, not a bad row and not the import: re-storing the worst-behaving memory as a fresh row reproduced its score to the digit. It is a property of averaging a passage into one vector, so no floor on a per-row cosine can be made safe by tuning the threshold.

Two corrections followed and are in the tree: the dense leg is now re-ranked by a per-text calibration before fusion (`antumbra-core/src/calibrate.rs`), and a cross-encoder re-scores the fused pool where one is configured. Both produce a score that survives a comparison across lengths. The floor belongs on that score, after fusion and after any rerank, which is also the only place it can be compared against a threshold that means the same thing for a stub and for a thousand-word memory.

**The floor now ships, and building it corrected the paragraph above.** The claim that the cross-encoder cannot carry a floor is true of a THRESHOLD and false of a CALIBRATION, which this record ruled out without trying. The ten-memory measurement that produced the ruling is real; it was just far smaller than what followed.

Measured over 800 balanced pairs from 400 distinct queries, a logistic fitted on `log10(score)` — `antumbra-core/src/platt.rs`, fit on half, every number below reported on the other half:

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| calibrated probability at its own 0.5 | 0.803 | 0.820 | 0.775 | **0.797** |
| best tuned threshold, chosen *in sample* | 0.802 | 0.862 | 0.720 | 0.785 |

The calibrated probability wins while giving up the advantage the threshold had, which was picking its cut point on the rows it was then scored against. Expected calibration error is 0.033 over ten bins.

**The default floor is 0.30, and the curve rather than the round number chose it.** F1 is flat from 0.30 to 0.60 (0.797, 0.803, 0.797, 0.796), so the choice buys nothing in overall quality and is entirely about which error to prefer:

| floor | precision | recall | genuine answers dropped |
|---|---|---|---|
| 0.30 | 0.732 | 0.875 | 12.5% |
| 0.50 | 0.820 | 0.775 | 22.5% |
| 0.60 | 0.874 | 0.730 | 27.0% |

For a memory the two errors are not symmetric. A relevant memory held back is invisible to the caller and cannot be asked for, because they do not know it exists; a weak one that surfaces is visible and discardable. So the default takes ten points of recall for no F1, and remains movable in both directions by the caller's `floor`.

Two things are weak here and are recorded rather than smoothed over. The worst-calibrated slice is 0.50–0.75, gap 0.083 on 42 samples — and mid-range is exactly where a floor is read, so that is the band a refit should target first. And the labels are constructed by a verifier, a proxy for relevance rather than relevance itself; a calibration fitted on a proxy is a hypothesis about real queries until it meets some.

`antumbra-rerank/src/floor.rs` implements this as a `TypedDecider`, which the port's contract admits because the fit minimises log loss (strictly proper) over labels a deterministic verifier produced rather than over any model's own answers. It answers `Noul` and refuses the other question types instead of guessing. It needs no GPU and no `models` build, so the floor is available on every deployment that configures a reranker rather than only on the ones with a card.

### B-3 · The output shape is linted, and not only Antumbra's

`antumbra claude mcp-lint` (ADR-0021) already reads a server's `tools/list` and reports which tools will break a session or vanish from it. It checks input schemas against the vendor's rules and says nothing about what a tool *returns*.

The same command gains output-shape checks, run against any connected server: an unbounded text field in a collection row, a collection with no stated bound, an empty result indistinguishable from an unmatched one. The lint reports; it does not refuse. This is the same posture as the schema checks beside it, which name a failure the agent would otherwise meet as an unattributed 400.

Two reasons this belongs in the lint rather than in a convention. First, a convention is only as good as the next surface that forgets it, and this one was forgotten by Antumbra's own. Second, the checks are worth more pointed outward than inward: a user of sovereign mode connects servers Antumbra did not write, into a context window Antumbra is trying to protect, and nothing else in the ecosystem tells them which of those servers will empty it.

## Consequences

An agent that needs a full memory pays one extra call for it. That is the trade, taken deliberately: the common case is recall for orientation, where a prefix is sufficient, and the uncommon case is now explicit rather than universal.

The marker has to be unmissable. An agent that acts on a prefix believing it has the whole text is a worse failure than the one this record fixes, and it is the main risk the design carries. The default cut should be generous enough that the common case is not truncated at all, so that the marker's appearance is itself informative.

Truncation does not touch provenance. A cut memory still carries its anchor, its `scope`, and its `orphaned_at`, so the judgement of ADR-0018 is unaffected by how much of the text came back.

The hook's total becomes checkable. With a bound on each memory and a count of memories, the session-start payload has a worst case that can be computed, and `brief.rs`'s existing budget test acquires a sibling that holds the whole hook to the cap rather than holding one part of it.

## Alternatives considered

**Re-encode the envelope in a denser notation (TOON or similar).** Rejected on arithmetic, above: it compresses the part that is already cheap. It would also mean sending a string where the protocol expects structured JSON, losing the typing the surface currently gets for free.

**Lower the default `top_k`.** Rejected. Fewer memories is worse recall, which is the wrong axis to trade; the problem is the size of each row, not the number of rows.

**Summarise server-side.** Rejected: a model in the recall path is non-deterministic, costs latency on the hottest surface, and interposes an unverified signal between the store and the agent.

**Add `max_chars` and no default.** Rejected as insufficient. Defaults are what ships; an option nobody sets is a defect with documentation.

**Leave it, and hold the hook to its cap in the hook.** Rejected because the surface is used outside the hook, by agents with no hook at all, and a bound that only exists at one call site is not a property of the surface.

## Validation

1. A default recall of `k` memories, each longer than the bound, returns a payload under a stated size; a test fails if it does not.
2. The truncation marker is present exactly when the content was cut, and absent when it was not.
3. `full: true` returns the untruncated text, and the marker is absent.
4. A recall where nothing clears the floor is distinguishable, by a field rather than by inference, from a recall where five weak matches did.
5. The session-start payload — the brief plus the memories the hook recalls at its configured `top_k` — has a worst case held under 10,000 characters by a test, extending the discipline `brief.rs` already applies to itself.
6. `mcp-lint` flags a server whose collection rows carry an unbounded text field, demonstrated against a fixture.

## Order of work

1. [x] **B-1, the bound.** `full` on `RecallParams`, the prefix cut, the marker and the original length on `MemoryView`, and validations 1 to 3. This is the defect; it goes first and is shippable alone.
2. [x] **The hook's worst case.** Validation 5, which is the reason the bound matters and is a test rather than a feature. Found already satisfied, on both platforms, and predating this record: each session-start hook caps its own output at 9,500 characters and lists what it left out, and `scripts/hooks/tests/session-start.{ps1,sh}` drive twelve 1,500-character memories (18,000 in all) through them asserting the context stays under 10,000, that the best memory survives, that the omitted count and the kept count sum to twelve, and that the brief precedes any memory. Both suites re-run green after B-1. The item stands as a record of where the guarantee lives, since it is enforced in the hook rather than in the crate and would otherwise be looked for in the wrong place.
3. [x] **B-2, the floor.** The relevance floor and validation 4, both in `antumbra-mcp`: a `floor` the caller may lower, a `DEFAULT_RELEVANCE_FLOOR`, and `nothing_cleared_the_floor` on the result. Validation 4 is a test rather than a claim, and so is its converse — an empty store does NOT set the flag, because "there was nothing to reject" and "everything was rejected" are different answers and the field must not blur them.

   **Both the mechanism and the number are now settled, which was not true when this line was first written.** The floor sits on the fused score after any rerank, the flag makes emptiness branchable, and the probability it compares against is calibrated rather than assumed: `antumbra-rerank/src/floor.rs` maps the cross-encoder's score through a logistic fitted on verifier labels, reaching 0.803 accuracy and 0.797 F1 out of sample with an expected calibration error of 0.033. The default is 0.30, chosen off the precision/recall curve rather than from the round number.

   The claim this item used to rest on — that the system has two ranking signals and no calibrated one — was answered by calibrating one of them rather than by waiting for a new model. ADR-0024's D-2 is still wanted, for the part a cross-encoder cannot do: it answers `Noul` and has no opinion about a `Choice` or a `Score`, which is what D-1's gate needs.
4. [x] **B-3, the lint.** All three output-shape rules are in `mcp-lint`, and validation 6 is a fixture test. Last because it is the generalisation, and generalising before the specific case is settled would encode a guess.

   The first two generalise from B-1: an unbounded text field in a collection row, and a collection with no stated bound. **The third generalises from B-2** — a collection that can come back empty with nothing beside it to say why — and it could be written once B-2's mechanism landed, because what it checks for is the shape B-2 chose. A boolean, an enumerated status or a count beside the rows all satisfy it; so does a description that says what an empty one means, which is the same escape hatch the other two rules offer and the right form for a tool whose empty state is unambiguous.

   **Pointed at antumbra-mcp's own `tools/list`, the third rule found five and they are fixed.** It PASSED `recall_memories`, because `nothing_cleared_the_floor` is exactly what it asks for, and flagged `get_neighbors`, `list_compartments`, `population`, `propose_compartments` and `recall_documents` — the defect B-2 fixed in one place and never reached in the others. That is the record's own claim, that this was "forgotten by Antumbra's own" surface, measured rather than asserted.

   Each was fixed by saying what an empty one means, which is the right form here: none of the five has a relevance floor, so a `nothing_cleared_the_floor` boolean copied across would have been a fabricated signal. The meanings differ and the differences are the point — `recall_documents` empty means nothing matched, `population` empty means the population was never seeded, and `propose_compartments` empty is genuinely ambiguous between "nothing to cluster" and "nothing clustered at this threshold", which the description now says along with the move that resolves it. The rule reports zero on this surface now.

   The remaining 9 of 19 are the bound findings, which generalise from B-1 and are recorded here as measured rather than scheduled.

   Taking the capture needs no live server: `capture_the_tool_list` in `antumbra-mcp` writes it from the static tool router, and the ignored `report_what_a_live_server_returns` in `antumbra-cli` lints it. The two crates do not depend on each other, so they meet through a file.

   One correction the measurement forced, recorded because the rule was wrong before it was right: the first version keyed on the literal word "empty" and flagged `store_memory`'s `auto_proposed`, whose description ("only when the autonomous propose trigger is enabled and fired") explains its absence completely. A server says absence in its own vocabulary, so the check reads a small set of absence phrasings instead of one word.

**A prompt that asks for two things is retrieved as its parts too (2026-10-04).** The floor is only as good as what reaches it. A real prompt, "move the tokens out of ~/.claude.json and record the 23 dependency edges on kushkokwim", recalled neither memory it was about, not even among the 100 candidates the reranker reads, although the reranker scored those two memories 0.965 and 0.854 against it. Each was found first by a query about it alone, so retrieval was sound and the prompt was the problem: its embedding is a blend of two subjects that lands near neither, and its words split between them. `antumbra_core::query::parts` cuts such a prompt at sentences, lines, and an "and", "then" or "also" with a request's worth of words on both sides (at most four parts; a prompt that asks for one thing has none). Recall retrieves for the whole and for each part, fuses the lists by rank (`memory::recall_hybrid_many`), and the reranker and the floor still read the whole prompt. Each extra part costs one embedding and one hybrid retrieval.

## Notes on the evidence

The defect and the quotations above were found by reading the tree on 2026-09-22: `params.rs:81` for `MemoryView`, `brief.rs:10-19` for the budget and its test, `bridge.rs:4-9` for the measured overflow behaviour, and a search for truncation across `antumbra-mcp` that returned nothing touching memory content.

The numbers in B-2 came later the same day, from a 5,538-memory store rather than from reading the tree, and they are the reason that section now says something different from what it first said. Three of them are reproducible from the repository: `antumbra-serve`'s ignored tests `length_curve_across_models`, `chunk_and_max_pool_across_chunk_sizes` and `is_the_length_bias_predictable_enough_to_subtract` print the curve, the chunking trade-off, and the calibration that corrects it. They are ignored because they download model weights, and they are the benchmark to re-run against any future change to this surface rather than reasoning about it. A fourth, `calibrated_ranking_beats_raw_cosine_on_the_real_corpus`, needs a sample of real memories and measured 2/30 top-1 for raw cosine against 13/30 for the calibrated score.

Recording this because the first version of B-2 was written from the tree and was wrong about the tree's behaviour, which is a failure mode worth naming: `similarity` is a plausible thing to floor, the field is right there, and nothing in the code says it is length-dominated. It took a query about banana bread to find out.

The framing came from [AXI](https://axi.md) (Kun Chen), a ten-principle spec for agent-facing CLI tools. Its principles 3 (truncate by default, with an escape hatch) and 5 (definitive empty states) are B-1 and B-2, and the debt is acknowledged here rather than absorbed silently.

What was taken from it stops there, and the reasons are worth recording so the question is not reopened. Its central claim — that a CLI beats MCP for agent tools — does not transfer: Antumbra's MCP connection carries identity, a JWT resolving to a per-identity serving connection whose record session the engine's row permissions filter from (ADR-0013, ADR-0015), and a CLI would re-establish that per invocation. Its principle 7, installing into the agent's session lifecycle from an explicit setup command, is `antumbra claude apply`, arrived at independently in ADR-0021 and with more of an account of what it costs. Its output format, TOON, is declined above on arithmetic; the independent benchmark (arXiv 2603.03306) also finds that "plain JSON generation shows the best one-shot and final accuracy", and TOON's headline saving is measured against pretty-printed JSON rather than the compact form a transport actually sends.

AXI's own benchmarks are published by its author against a single model family and have not been reproduced, and several of the community implementations it catalogues are forks carrying identical descriptions. None of that bears on whether principles 3 and 5 are right — they are, and the tree proves it — but it is the reason this record adopts two of ten rather than the framework.

One asymmetry is worth naming as an opportunity rather than a borrowing. AXI has no conformance suite; its principles are prose. B-3 is a conformance check for the two that matter, pointed at every server a user connects. That is a thing Antumbra can ship that the spec's own author has not.
