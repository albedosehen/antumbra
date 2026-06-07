# Antumbra - Technical Reference

Engineer-facing reference for the implementation: crate map, domain model, the port seams, the algorithms as
actually coded, the build matrix, and the validation results to date. For the *why*, read
[architecture.md](architecture.md), the [ADRs](adr/), and the [glossary](glossary.md). For the metaphor, the
[README](../README.md). For GPU setup, [running-the-trainer.md](running-the-trainer.md).

This document describes what is built and measured, and is explicit about what is not.

## 1. The thesis in three pillars

Antumbra grows a population of small, frozen LoRA experts (the **umbra**) by spawning short-lived shadow
adapters (the **penumbra**), training them on **verified outcomes**, and graduating the ones that work. The
keystone is the **antumbra**: the counterfactual boundary of each expert's competence - knowing where an expert
is right, where it is wrong, and when to escalate.

| Pillar | Claim | Status |
|---|---|---|
| 1. Improve a small expert from verified outcomes alone | engine | **demonstrated** (toy scale) - RAFT lifts pass-rate to 1.0 on GPU |
| 2. Know each expert's scope: route in, refuse/escalate out | keystone (routing half) | **demonstrated** - relative-coverage gate, capability vectors from evaluated behavior |
| 3. Compose a growing population without forgetting | payoff | **partial** - population grows + routes; composition and forgetting tests are future |

The deepest keystone claim - recovering a counterfactual `C'` for a failure boundary (ADR-0004) - now runs
**live and autonomous**: a real generate-then-verify probe recovers the boundary on the GPU, the governing
feature is *discovered* from pass/fail (not supplied), and the gate inhibits routing within the recovered scope.
The candidate context set is still authored, and scale remains untested.

## 2. Crate map

Thirteen crates, single Rust workspace. The default build is light (no ML deps); candle is gated behind `models`.

| Crate | ADR | Responsibility | Heavy deps (feature) |
|---|---|---|---|
| `antumbra-core` | 0001-0004, 0012-0014 | Domain types (incl. `Memory`, `Compartment`, `Grant`, identity ids), state machines, port traits, `penumbra` clustering (`propose_compartments`), `testing` fakes. No I/O. | - (`testing` feature for fakes) |
| `antumbra-store` | 0007, 0012-0014 | SurrealDB data layer, **surql-rs builders only** (no hand-written SurrealQL): schema-as-code, repositories, HNSW recall, the Penumbra memory store + graph + **compartments**, record-access auth and the **engine-enforced tenant/compartment ACL**. | `surrealdb` (kv-mem, kv-surrealkv) |
| `antumbra-embed` | 0015 | HTTP embedder: an OpenAI-compatible `/embeddings` client behind the core `Embedder` port, shared by the MCP server and the operator console so route/ask/recall embed with the *same* model the population was built with (dimension enforced, `EMBED_DIM`). | `ureq` |
| `antumbra-sync` | - | Collector/sync (R-1): bidirectional last-write-wins replication of the Penumbra between a local embedded store and a remote authoritative one. | - |
| `antumbra-critic` | 0003 | Verifiers (the reward is the environment): in-process rules + external commands. | - |
| `antumbra-gate` | 0005 | Boundary-conditioned coverage gate: rank by capability, escalate on relative coverage. | - |
| `antumbra-boundary` | 0004 | Counterfactual scope engine (keystone). Seam for `C'` recovery. | - |
| `antumbra-loop` | 0008 | Durable generational loop; writes full lineage to the substrate. | - |
| `antumbra-train` | 0001, 0002, 0010, 0012 | candle QLoRA/RAFT trainer: Qwen2.5-Coder + LoRA, SFT, save; capture/teach intake; **consolidation** (gate + replay), memory-import, and **harness metabolization** (`harness`: successful orchestration traces → capture tasks). | `candle-*`, `tokenizers`, `hf-hub` (`models`); `cuda`/`metal` |
| `antumbra-serve` | 0006 | Candle adapter serving: `CandleServe` (single pinned adapter) and `MultiAdapterServe` (resident base, S-LoRA hot-swap per routed expert), plus the real candle BERT embedder. | `candle-*`, `candle-transformers`, `antumbra-train` (`models`); `cuda`/`metal` |
| `antumbra-cli` | - | Operator CLI: `migrate · schema · experts · status · loop · route · ask · serve · train · teach · evolve · populate · memory-import · metabolize · remember · consolidate · consolidate-compartment · propose-compartments · retire`. | pulls `train`/`serve`/`critic` (`models`) |
| `antumbra-mcp` | 0015 | MCP server over the Penumbra + population: 14 tools (memory, graph, compartments incl. `propose_compartments`, `route`, `answer`), an optional autonomous propose trigger (`--auto-propose`). Two transports: stdio (one bound `(tenant, user)`) and `--http` (networked, multi-tenant per request — JWT claims become `$auth`). | `rmcp`, `axum`, `jsonwebtoken`; `antumbra-serve` (`models`) |
| `antumbra-tui` | 0005 | Interactive operator console (ratatui + tachyonfx): the live animated population/gate, with route-ask through the gate, a live event stream of store changes, drill-down inspection, switchable themes/layouts, multi-monitor high-refresh pacing, fuzzy filter/palette, and operator actions (prune/graduate shadow · freeze/thaw expert · delete boundary) behind a confirm. Headless `snapshot` mode renders an e2e text grid + PNG. | `ratatui`, `tachyonfx`, `antumbra-embed` |

## 3. Domain model (`antumbra-core`)

Plain data types, serializable, with the state-machine logic as methods.

- **`Expert`** (ADR-0001) - a frozen adapter in the population. Fields: `id`, `name`, `base_model`,
  `artifact_uri` (adapter file), `capability_card` (JSON provenance), `capability_vec: Option<Vec<f32>>`,
  `fitness`, `frozen_at`, `generation`, `created_at`. `capability_similarity(query) -> Option<f32>` is cosine
  over `capability_vec`.
- **`Shadow`** (ADR-0002) - an in-training adapter. `ShadowStatus` is a guarded state machine:
  `Spawned -> Exploring -> Scoring -> {Graduated | Pruned}`; illegal transitions error.
- **`FailureBoundary`** (ADR-0004) - a recorded failure region: `behavior`, `fail_context`,
  `near_ok_context`, `governing_features`, `grain`, `context_vec`, `confidence`. `is_actionable()` is false
  until a near-OK counterfactual exists; `inhibition_for(task_vec, radius)` returns the routing penalty.
- **`RewardSignal`** (ADR-0003) - source-tagged (`RewardSource`) per-step reward; `fold_step` accumulates.
- **`EvaluationRun`** (ADR-0007) - one measured run with `EvalStatus`, metrics, regression fingerprint.
- **`Generation`**, **ids** (`ExpertId`, `ShadowId`, `BoundaryId`, `RunId`) - newtypes; `Generation::ZERO`.
- **`generational::{GenerationHead, LoopState}`** - the loop's persisted checkpoint and its state enum.

`cosine_similarity(a, b)` is exported at the crate root.

## 4. Ports (the seams)

Every external capability is a trait, so the loop runs end-to-end against fakes (`antumbra-core::testing`)
and the real engines drop in unchanged.

```rust
trait Trainer    { async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome>; }
trait Serve      { async fn act(&self, req: ActRequest) -> Result<ActOutput>; }
trait Verifier   { async fn verify(&self, req: &VerifyRequest) -> Result<VerifierVerdict>; }
trait Critic     { async fn densify(&self, output: &ActOutput) -> Result<Vec<CriticScore>>; }
trait Embedder   { async fn embed(&self, text: &str) -> Result<Vec<f32>>; fn dim(&self) -> usize; }
trait AcceptabilityProbe { async fn acceptable(&self, behavior: &str, ctx: &Value) -> Result<bool>; }
```

Key payloads:

- `TrainRequest { shadow, base_model, corpus_task_ids, max_steps }`.
- `TrainOutcome { adapter_uri, reward_curve: Vec<f32>, final_fitness, capability_exemplars: Vec<String> }`.
  `reward_curve` is the per-round pass-rate; `capability_exemplars` are the prompts the shadow provably solved.
- `VerifyRequest { run_id, step_idx, dimension, artifact: Value }`; `VerifierVerdict { passed, value }`.

Fakes: `ScriptedTrainer` (graduating / collapsing / `graduating_with_exemplars`), `FixedEmbedder`
(byte-histogram), `EchoServe`, `MarkerVerifier`.

## 5. The generational loop (`antumbra-loop`, ADR-0008)

A resumable state machine in the database: `grow -> explore -> score -> decide -> consolidate -> grow`. The
`GenerationHead` is persisted after **every** transition, so the state value *is* the checkpoint - kill the
process and a fresh loop resumes from the substrate (validated on `surrealkv://`).

Each generation writes its full lineage: the shadow and its status transitions, source-tagged reward signals,
an evaluation run, and then either a **graduated `Expert`** (fitness >= `graduate_threshold`) or, on prune, an
**open-negative `FailureBoundary`** (recorded but not actionable - honest, because nothing should gate routing
on an un-scoped negative until counterfactual search recovers a near-OK `C'`).

`GenerationReport { generation, shadow, fitness, graduated, reward_curve }` is returned per generation.

## 6. The trainer (`antumbra-train`, ADR-0002/0010)

RAFT - reward-ranked fine-tuning, the simplest realization of RLVR.

Per round, for each task: sample `K` completions, **verify** each (ground truth), keep the winners, and SFT the
LoRA on them. The model learns from its *own verified-correct* generations - no teacher text, no policy-gradient
machinery. The per-round pass-rate is the reward curve; an empty winner set means no update that round (never
reinforce nothing).

- **LoRA** (`lora.rs`): frozen base `W` plus trainable low-rank `A`, `B` (B zero-init);
  `y = Wx + (alpha/r) B(Ax)`. `lora_scale = alpha / rank`.
- **Objective** (`objective.rs`): `causal_lm_loss` is completion-masked causal-LM cross-entropy (shift, gather,
  mask the prompt).
- **Model** (`models/qwen.rs`, feature `models`): vendored Qwen2 (GQA, RoPE, RMSNorm, KV cache) with
  LoRA-wrapped projections, loaded from hf-hub. `dtype = f32` on CPU, `bf16` on GPU (f16 overflowed to NaN
  logits - see ADR-0010). Generation is temperature sampling with a greedy fallback on a degenerate
  distribution.
- **`RaftTrainer`** (`trainer.rs`) realizes the `Trainer` port over a `ModelLoader`, a `Corpus`, and a
  `Verifier`. Only the model loader touches the GPU, so the orchestration is tested with fakes.
- **Capability exemplars**: the prompts solved in the final round are returned in `TrainOutcome`, and become
  the graduated expert's capability vector (Section 9).

`RaftConfig`: `base_model` (Qwen2.5-Coder-1.5B), `lora_rank` 16, `lora_alpha` 32, `samples_per_task` 8,
`rounds` 4, `max_new_tokens` 256, `learning_rate` 1e-4, `dtype` bf16.

## 7. The verifier (`antumbra-critic`, ADR-0003)

`CommandVerifier` reads the task's `verify` spec from the request artifact and, optionally, `extract_code`
(pulls the first fenced code block). Then:

1. **In-process rule** `contains_all: [..]` - the completion must contain every listed substring. Deterministic,
   Python-free; used for convention learning.
2. **External command** `program` / `args` / `cwd` - exit code 0 is a pass. The candidate completion is exposed
   to the command via `$ANTUMBRA_COMPLETION`. This is how behavior is checked by *executing* the generated code
   (e.g. `python -c "exec(...); sys.exit(0 if add(2,3)==5 else 1)"`).

The environment is the reward; the (future) critic densifies it but never overrides it.

## 8. The embedder (`antumbra-serve`, feature `models`)

`BertEmbedder` is the real `Embedder`: all-MiniLM-L6-v2 (384-d, ~23M params) via candle-transformers, on CPU
(tiny model; keep the GPU free). `embed` tokenizes, runs BERT, mean-pools over tokens, and L2-normalizes (kNN-OOD
calls normalization critical). The default build keeps the fake `FixedEmbedder`; `EMBED_DIM` is 384.

## 9. Capability vectors from evaluated behavior (ADR-0004/0005)

An expert's capability vector is **not** a hand-written label. On graduation, the loop embeds each prompt the
shadow provably solved (`TrainOutcome.capability_exemplars`) and takes the **centroid** - so the routing vector
is defined by what the expert *demonstrably does*. The solved prompts are also stored in `capability_card` for
provenance. With no exemplars, it falls back to a generic descriptor. (Cosine downstream is scale-invariant, so
the centroid is not renormalized.)

## 10. The gate (`antumbra-gate`, ADR-0005)

`route(task_vec, experts, boundaries, k, cfg) -> GateDecision { chosen, escalate, ranked, coverage }`.

- **Ranking** (which expert): `score(e) = cosine(e.capability_vec, task) - max_boundary_inhibition(task)`,
  top-k.
- **Escalation** (in/out of scope) is an **out-of-distribution** decision and uses **relative coverage**, not
  an absolute cosine floor. Sentence-embedding cosine for short texts is compressed into a high band (~0.6-0.9
  for everything), so an absolute floor cannot separate in- from out-of-scope (measured). The fix, grounded in
  the OOD literature, cancels the shared non-discriminative direction:
  - `coverage = (cos(task, e1) - cos(task, e2)) - inhibition` - the top-1-minus-top-2 prototype margin. The
    shared direction contributes equally to both terms and cancels (the cosine-space realization of Relative
    Mahalanobis Distance, arXiv:2106.09022). Subtracting the population *centroid* instead was measured to fail
    (coverage collapsed to ~0). With a single expert it degrades to the absolute score.
  - Escalate when `coverage < coverage_threshold` (default 0.08) - the abstention / risk-coverage knob of
    selective prediction (arXiv:1705.08500), calibrated per deployment.
- **Boundary inhibition** (ADR-0004) is subtracted from coverage, so a confident boundary forces escalation.

`GateConfig { coverage_threshold: 0.08, inhibition_radius: 0.5 }`. Known v0 limitation: a task served equally
by two experts has a small margin and escalates (ambiguity conflated with out-of-scope); north-star composition
(ADR-0009) dissolves it.

## 11. The substrate (`antumbra-store`, ADR-0007)

SurrealDB via **surql-rs** (`oneiriq-surql`) builders exclusively - no hand-written SurrealQL anywhere. Tables
are `SCHEMALESS` with explicit unique and HNSW vector indexes; the DDL is *generated*. Records use a `key`
column plus a `RecordID` (`id` is reserved). Repositories (`expert`, `shadow`, `reward`, `boundary`,
`evaluation`, `generation`) wrap `crud` + the `Query` builder (including `Query::vector_search` KNN).
`Store::connect` accepts `mem://` (ephemeral), `surrealkv://path` (persistent), or `ws://host/rpc`.

## 12. Build & feature matrix

| Build | Command | Brings in |
|---|---|---|
| Default (light) | `cargo build` / `cargo test` | no ML deps; fakes only; CLI `train` errors out |
| Real models (CPU) | `cargo build -p antumbra-cli --features models` | candle (CPU), BERT embedder, Qwen trainer |
| GPU | `... --features models,cuda` (or `metal`) | candle CUDA/Metal backend |

CUDA 13 on Windows needs a specific env (vcvars, `CUDARC_CUDA_VERSION`, `NVCC_PREPEND_FLAGS`, runtime DLLs on
PATH) and the exec verifier needs `python` on PATH - see [running-the-trainer.md](running-the-trainer.md). A
release-profile `opt-level=1` override on `surrealdb`/`surrealdb-core` works around a rustc ICE.

## 13. Validation to date

- **MT-3 - the trainer learns (GPU, RTX 3090 Ti, bf16).** Convention reward (`contains_all`) pass-rate rose
  `0.06 -> 0.25 -> 0.88 -> 1.00`; the real exec verifier (runs the generated code, asserts behavior) rose
  `0.38 -> 1.00 -> 1.00 -> 1.00`. Both graduated a real adapter. The loop improves an adapter from verified
  outcomes - including outcomes verified by executing the code.
- **Gate - routing and refusal (seeded, real embeddings).** Three described specialists; matched queries routed
  3/3 (margins 0.106 / 0.163 / 0.110) and two out-of-scope queries escalated (0.037, 0.068) across the 0.08
  threshold. The first design (centroid background) was falsified by measurement and corrected to the prototype
  margin.

- **Multi-expert population (GPU, end-to-end).** Three specialists were trained from the base on their own
  exec-verified corpora and graduated into one persistent `surrealkv://` population, each with a capability
  vector derived from the tasks it solved: arith `[0.42, 1.00, 1.00]`, strings `[0.25, 0.50, 0.83]`, lists
  `[0.17, 0.50, 0.83]` - all graduated. Routing across the population (k=1):

  | query | top-1 (score) | coverage | @0.08 default | @0.045 calibrated |
  |---|---|---|---|---|
  | add two integers | arith (0.884) | 0.070 | escalate (false) | **arith** |
  | reverse a string | strings (0.833) | 0.182 | strings | **strings** |
  | largest item in a list | lists (0.765) | 0.054 | escalate (false) | **lists** |
  | train a CNN (out-of-scope) | strings (0.650) | 0.033 | escalate | **escalate** |
  | grill a rack of lamb (out-of-scope) | strings (0.657) | 0.025 | escalate | **escalate** |

  **Ranking was 3/3 correct** - each query's true specialist is top-1. But the two numeric specialists (arith,
  lists) are semantically adjacent, so their in-scope margins are thin (0.070, 0.054); the default 0.08
  threshold - calibrated to the earlier described-vector population - over-escalated them. Recalibrating the
  abstention threshold to 0.045 separates *this* population cleanly (in-scope >= 0.054, out-of-scope <= 0.033)
  for 5/5. This is the predicted selective-prediction behavior: the threshold is a per-deployment risk-coverage
  knob, not a universal constant, and overlapping capability regions compress the margin - which is precisely
  the case for a learned/boundary-conditioned gate (ADR-0004/0009) over a fixed margin.

- **Keystone end-to-end (ADR-0004).** Counterfactual search recovers C' for a behavior failure, the actionable
  boundary persists through surql-rs, and the gate inhibits a perfectly-matching expert **inside** the failure
  scope (forcing escalation) while doing nothing **outside** it; a no-boundary control routes the same task
  straight to that expert, isolating the boundary as the cause
  (`crates/antumbra-store/tests/keystone_mem.rs`). The only fake is the `AcceptabilityProbe` (needs serving).

- **Serving - experts answer (GPU).** `CandleServe` reuses the trainer's Qwen+LoRA model to load a graduated
  expert's adapter and generate. CLI `ask "add two integers"` routed to the arith specialist, loaded
  `arith_g0.safetensors`, and produced `def add(a, b): return a + b`; `ask "reverse a string"` routed to the
  strings specialist and produced `return s[::-1]`. Route -> load adapter -> serve is closed end-to-end.

- **Real AcceptabilityProbe (generate-then-verify).** `GenerateVerifyProbe` (`antumbra-serve`) holds the
  behavior fixed, renders it for a candidate context, serves a completion, and verifies it — generic over the
  `Serve` and `Verifier` ports. Unit tests prove acceptability is decided by serving-and-checking and that
  `find_scope` drives it to recover the governing feature and C' (`probe.rs` tests). Production swaps the fakes
  for `CandleServe` + `CommandVerifier` with no change to the keystone path; that is the keystone's last fake
  retired at the mechanism level.
- **Keystone live and autonomous (GPU).** Probing with **the expert's own adapter** (`scope --expert`), the
  search recovers a real boundary: a narrow **adder** passes `op=add`, fails `op=multiply`, so it recovers
  governing feature `op`, C' `{op: add}`, and stores an **actionable** boundary (`status`: 1 actionable, 0
  open). With `--discover` (`discover_boundary`) the governing feature is **inferred from pass/fail**, not
  supplied, and still resolves to `op`. Real expert -> real generation -> real execution -> counterfactual
  recovery -> persisted boundary, no fake in the path. Getting reliable live recovery flushed out and fixed
  three bugs (identical best-of-K seeds, per-`act` model reload, an unsandboxed verifier that hung on runaway
  generated code); the integrity discipline held throughout (the probe never fabricated a scope).

## 14. What is proven, and what is not

**Proven (toy scale):** an adapter learns from verified outcomes; the loop is durable and resumable; capability
vectors are derived from evaluated behavior; the gate routes to the right specialist and refuses out-of-scope
queries.

The failure-boundary `C'` recovery - the deepest keystone claim - composes end-to-end (search -> actionable
boundary -> persistence -> scoped inhibition), and the `AcceptabilityProbe` is now a real generate-then-verify
mechanism (`GenerateVerifyProbe`) rather than a fake.

v0 **serving** exists: `CandleServe` loads a graduated expert's adapter once (cached) and generates, so route ->
serve is closed (single-adapter).

The keystone is now **fully live and autonomous**: probing with a trained expert's adapter recovers a real
competence boundary end-to-end, and `--discover` infers the governing feature from pass/fail rather than being
told it.

**Not yet:** capability is exercised on small corpora and few experts (no generalization or catastrophic-
forgetting test); candidate governing features are authored rather than discovered; the gate is the heuristic
coverage gate, not the learned latent mixer; multi-adapter hot-swap and the `llama-cpp-2`/`mistral.rs` backends,
GRPO (v1 over RAFT), and GGUF-Q4 quantized backward (MT-4) are future; composition (ADR-0009) is the north star.
