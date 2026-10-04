# ADR-0025: Chunked memory retrieval

**Status:** Accepted · **Date:** 2026-10-04 · **Related:** 0023 (bounded answers: the floor and the reranker read what retrieval hands them), 0024 (typed decisions: D-2's label construction, reused here), 0007 (the SurrealDB substrate and its vector and full-text indexes)

## Context

Recall runs in two stages. The first, hybrid retrieval, finds candidates: a dense leg (one 384-dimension vector per memory, `all-MiniLM-L6-v2`, through an HNSW index) fused by rank with a lexical leg (BM25 over the content). The second, the cross-encoder, orders the candidates against the query, and the calibrated floor (ADR-0024 D-2) decides which of them answer it. The second stage was improved on 2026-10-04 (`gte-reranker-modernbert-base`, floor F1 0.797 to 0.885). It can only reorder what the first stage gives it: 30 candidates for a three-row recall, the size the prompt hook asks for, and at most 100.

**The first stage was measured, and it misses most of what a memory says.** `antumbra-bench` now reads D-2's label file: 400 of the user's own memories, each with a query of twelve words cut from about 60% of the way through it and excised from it, so neither leg can match it verbatim. The memories average 3,025 characters.

| dense leg | recall@1 | recall@10 | recall@30 | MRR |
| --- | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2`, one vector per memory (today) | 0.140 | 0.372 | 0.552 | 0.214 |
| `bge-small-en-v1.5`, one vector | 0.125 | 0.357 | 0.532 | 0.202 |
| `gte-modernbert-base` (reads 8,192 tokens), one vector, 384 of 768 dimensions | 0.155 | 0.393 | 0.585 | 0.240 |
| `all-MiniLM-L6-v2`, chunks of 400 characters | **0.290** | **0.670** | **0.820** | **0.419** |
| `all-MiniLM-L6-v2`, chunks of 600 characters | 0.285 | 0.650 | 0.787 | 0.406 |
| `all-MiniLM-L6-v2`, chunks of 800 characters | 0.242 | 0.647 | 0.810 | 0.376 |
| `all-MiniLM-L6-v2`, chunks of 1,000 characters | 0.245 | 0.647 | 0.805 | 0.372 |
| `gte-modernbert-base`, chunks of 1,000 characters | 0.270 | 0.637 | 0.765 | 0.387 |

- **One vector per memory is the limit, not the model.** MiniLM reads the first 256 tokens, about 1,000 characters, so most of a long memory is outside its vector. A model that reads the whole memory does little better: one vector for 3,000 characters is a blur of all of them, and a passage from the middle is a small part of the blur. Three points of recall@30 for a model change, twenty-seven for chunking.
- **The smallest chunk measured is the best,** at every cutoff. One caution on reading it: these queries are twelve-word fragments, about the size of a 400-character chunk's content, and a fragment favors a chunk its own size; a person's prompt is usually longer. 400 is still the choice, because it is best on every column and the reranker reads the whole memory either way.
- **The lexical leg adds nothing on this set** (hybrid within 0.003 of dense), because the words are excised. It is what finds an identifier or an error code verbatim, and it stays.
- **A real case.** A prompt naming the dependency edges recorded that morning did not recall the memory about them, which a query from its own opening words reached at a cosine of only 0.38: most of its 1,000 characters were a shell command, and the vector described the command.

## Decision

**Every memory is also indexed as chunks, and the dense leg reads both.**

1. **The chunk index.** A `memory_chunk` table holds, per memory, its text cut into pieces of about 400 characters between words, each overlapping the one before by about a fifth, with each piece's vector, the memory it belongs to, its workspace and compartment, and a hash of the memory's content. It has its own HNSW index. Its select rule is the memory table's own (`memory_select_rule`), over the copied workspace and compartment, as `document_chunk`'s is, so a chunk is visible exactly when its memory is. Only the server writes it.
2. **The dense leg reads both indexes.** Recall searches the memory vectors, as it does today, and the chunk vectors, and scores each memory by its best vector, its whole one or its nearest piece, through the same calibration, before fusing with the lexical leg. Scores and not ranks: fused by rank, a memory whose pieces are all far from the query tied a whole memory that was near it, which a test caught. A memory not yet chunked is found as it is found today, so the chunk index is additive: it can be empty, partial or rebuilt without a recall going wrong. A memory that fits in one piece gets no chunks, since its whole vector already reads all of it.
3. **The chunks are derived, and the server keeps them.** Memories are written by many paths (the agent's `store_memory`, the CLI's intakes, the GitHub App, sync between nodes), and the store crate embeds nothing. So the chunker is one background task in the server: on start it chunks every memory whose chunks are missing or whose hash differs (the hash covers the content and the piece size, so a new size re-cuts everything), and after that, each minute, it follows memories changed since its last pass, with the workspace's own embedder, the one its queries are embedded with. Each hour a pass reads every memory again, for one synced in under an older `updated_at` and for the chunks of a memory purged since. A chunk is never synced; each node derives its own. A memory forgotten (a tombstone) is dropped from recall by the memory rows the chunks resolve to, and its chunks with it on the next pass. A compartment or network that changes moves the chunks' copy without re-embedding. `--chunk-in-flight` (`ANTUMBRA_CHUNK_IN_FLIGHT`, default 4) bounds how many pieces are embedded at once, for a node whose embedder also serves its queries; zero stops the keeper.
4. **The whole-memory vector stays.** It is what similarity, clustering and consolidation read, and what the dense leg reads for a memory with no chunks yet.

## Consequences

- **Positive:** the first stage finds about a quarter more of what a memory says, measured, with the model already deployed; the reranker and the floor get the candidates their measurement assumed.
- **Negative:** about ten times the vectors (a 3,000-character memory is nine or ten overlapping chunks): for the 6,000 memories on kuskokwim, about 55,000 vectors, some 85 MB, and a first pass of as many embeddings, minutes on the GPU and longer on a CPU node, in the background. A recall runs one more HNSW query.
- **Neutral:** the chunks are a cache of the content, so a wrong chunker is fixed by changing it and letting the pass rebuild.

## Alternatives considered

- **A better embedder, one vector per memory.** Measured above: three points.
- **A long-context embedder with chunking.** Measured above: no better than MiniLM with chunking, and a schema change for its 768 dimensions.
- **Writing chunks in each write path.** Every path would need an embedder, and the one that forgot would leave memories the chunk leg cannot see. A derived index kept by one task cannot be forgotten by a writer.
- **Chunks in place of the whole-memory vector.** Loses the additive property: a memory not yet chunked would be invisible to the dense leg.
- **Stemming the lexical leg.** It would help "record" find "recorded", and it would also stem identifiers, which the content analyzer leaves whole on purpose (`CONTENT_ANALYZER` in `antumbra-store/src/schema.rs`). Not decided here.

## Validation

1. **The bench, again, over the deployed path.** `antumbra-bench` with the label file, its chunked mode at the chosen size, must hold recall@30 at or above 0.80 with MiniLM. _Kill:_ the production dense leg, both indexes fused, scores below the one-vector baseline of 0.552 at recall@30.
2. **Recall on the store.** The memories a query from their own middles misses today are found once the first pass has run.
3. **Cost.** A three-row recall stays under the prompt hook's ten seconds and the session hook's five, warm.

**Validation 1, measured 2026-10-04: passed.** `ANTUMBRA_BENCH_CHUNK_INDEX=1` stores each memory as the server does, whole and with its pieces in `memory_chunk`, and recalls through `recall_hybrid`'s dense leg. MiniLM:

| path | recall@1 | recall@10 | recall@30 | MRR |
| --- | ---: | ---: | ---: | ---: |
| one vector per memory (before) | 0.140 | 0.372 | 0.552 | 0.214 |
| the deployed path: whole and pieces, each memory by its best vector | 0.285 | 0.665 | 0.820 | 0.414 |

That is the chunks-of-400 measurement above to within half a point, so scoring a memory by its best vector, rather than storing its pieces as memories of their own, loses nothing.

**Found on the way.** Probing production with the same 400 queries showed the relevance floor scoring each memory's 900-character prefix rather than the text the reranker had read. A recall shaped like the prompt hook's (three rows, floor on) returned the right memory for 60.0% of the queries, against 96.8% with the floor off. Fixed in #169. It is a separate defect from this record's, and both measurements are needed to read validation 2.

**Validation 2, measured on kuskokwim 2026-10-04: nothing to recover there.** The production probe runs all 400 label queries through `recall_memories` before and after the first pass. These are hybrid recalls, and the store's memories still contain the words each query was cut from. Ten of the 400 memories are no longer in the store: a recall from each one's own opening does not find it either. Before the chunk index, the other 390 were all in the top ten with the floor off, 387 in the top three, and 366 first. Afterwards the counts were the same except one more first. So with verbatim words, the lexical leg and the reranker already find what a fragment from the middle points to. The chunk index's gain is where a query shares no words with its memory, which validation 1 measures and this label set cannot show in production.

**Validation 3, measured on kuskokwim 2026-10-04: passed.** Hook-shaped recalls (three rows, floor on), warm, 400 queries:

| | p50 | p90 | max |
| --- | ---: | ---: | ---: |
| before the chunk index | 312 ms | 528 ms | 1,501 ms |
| with it, as first deployed | 863 ms | 1,989 ms | 2,775 ms |
| with it, memories fetched by key (#171) | 577 ms | 1,184 ms | 1,557 ms |

Every one is well inside the prompt hook's ten seconds and the session hook's five. The first deploy's cost was mostly fetching the memories that only the chunk leg found, through a filter that read every row of the workspace (178 ms for 150 memories, against 4 ms by key).

**The first pass.** kuskokwim cut 6,314 memories into 38,817 pieces in 654 s on its GPU embedder, with no failures. A restart's full pass with nothing to cut takes 3.7 s, and each later pass cuts the memories written since. On shaman, the first pass ran the Orin Nano's llama.cpp `embed-server` out of memory twice. llama.cpp keeps a host-memory prompt cache (`--cache-ram`, 8,192 MiB by default) that grows with every distinct text embedded, and the pass is about 39,000 distinct texts. Replaying 4,000 of the user's pieces against a test instance took it from 329 MB to 926 MB and still climbing; with `--cache-ram 0` it stayed flat at 600 MB. With the embed-server restarted with `--cache-ram 0` and `MemoryMax=2G` (a systemd drop-in on the Orin), shaman's keeper went back on at two pieces at a time. It finished the first pass in 150 s (952 more memories, 4,844 pieces, no failures), while the embed-server went from 626 MB to 687 MB and stayed there.

## Order of work

1. [x] **The measurement and this record**, with the bench's label corpus, dense mode and chunked mode (`ANTUMBRA_BENCH_LABELS`, `ANTUMBRA_BENCH_CHUNK_CHARS`), against `all-MiniLM-L6-v2`, `bge-small-en-v1.5` and `gte-modernbert-base` served by text-embeddings-inference on kuskokwim.
2. [x] **The chunk index and the chunker**: the table, the fused dense leg, the server's pass, tests. The bench cuts with the same `antumbra_core::chunk::split` the server does.
3. [x] **Deploy, the first pass, and validations 2 and 3** on kuskokwim, then shaman (2026-10-04).
