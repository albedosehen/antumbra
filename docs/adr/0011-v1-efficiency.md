# ADR-0011 — v1 efficiency: GRPO, 4-bit QLoRA training, and serving throughput

**Status:** Accepted — GRPO validated, 4-bit validated, #4 hardened (2026-06-03) · **Date:** 2026-06-03 · **Related:** 0010 (the trainer — this details its v1/MT-4), 0002 (shadow plasticity), 0003 (verified reward), 0006 (serving — concurrency lives here), 0009 (north star — where 4-bit pays off)

> **4-bit validated — the first OOM was a generation-tracking bug, not a backward limitation (2026-06-03).**
> Implemented as `BaseWeight::Dense | Quantized` with a `quantize_base` walk (Q4_K via `QTensor::quantize_onto`)
> and dequant-in-forward (`train --quantize-base`). The first A/B OOM'd, and the cause was misdiagnosed as a
> retained-dequant-for-`grad_x` problem. The **real** cause: **generation ran with autograd tracking on.** The
> LoRA factors are `Var`s, so every sampled token's forward was tracked, and the **KV cache retained the whole
> growing generation graph** across all `max_new_tokens` steps. With an f16 base the base weight is shared and
> cheap; with a Q4_K base **each token re-dequantizes the full ~3 GB of weights and all of them were retained**
> in that graph — 32 tokens × 3 GB → OOM. *Training* (`train_one`: one forward, then `backward_step` frees it)
> was never the problem; *sampling* was. **Fix:** a `grad` flag on `LoraLinear` that **detaches the LoRA factors
> during generation** (`sample_one`/`sample_one_with_logprobs` set it off; `train_one`/`forward_logits` set it
> on) — same values, untracked, nothing retained between tokens. With that, the GPU A/B (arith, samples 4 /
> rounds 3, max-new-tokens 32) gives **4-bit `0.12 -> 0.38 -> 1.00` = f16 `0.12 -> 0.38 -> 1.00`**, both
> graduated, no OOM. So Q4_K dequant-in-forward trains a LoRA at f16 quality, the resident base is ~1/4, and the
> per-token re-dequant is the only added cost. (The detach also makes f16 generation/serving leaner — it no
> longer retains a graph it never backprops.)

> **GRPO validated (2026-06-03).** Implemented as `grpo.rs` (CPU-tested core) + `QwenCausalLm: GrpoLm` (GPU) +
> `GrpoTrainer` behind the `Trainer` port, selectable via `train --algo grpo`. The design bets held: reference =
> base with LoRA toggled off (no second model), group size = `K`. GPU A/B on the arith corpus (same knobs,
> samples 6 / rounds 3): **RAFT `0.08 -> 0.25 -> 1.00`, GRPO `0.33 -> 0.92 -> 1.00`** — GRPO climbs much faster
> (0.92 by round 1 vs 0.25), clearing the sample-efficiency kill criterion on this run. One implementation
> correction: a combined-group backward OOMs the card because candle retains the base-forward activations for the
> LoRA backward (memory scales with `G`), so `grpo_step` steps **per group member** (one forward alive at a time),
> as RAFT steps per winner. Honest caveat: a single run on a toy task; a rigorous win needs multiple seeds/tasks.

> Study artifact. v0 is validated end-to-end (ADR-0010 MT-3: RAFT + f16 LoRA learns on the GPU; ADR-0006:
> cached single-adapter serving). This ADR studies the three efficiency levers queued before the
> scale/forgetting study (EXP-010): a more sample-efficient **algorithm** (GRPO), a more memory-efficient
> **base** (4-bit QLoRA), and **serving throughput** (the sync-generate-on-the-runtime-thread question). The
> point is to resolve the feasibility unknowns *before* writing code.

## Context

ADR-0010 named GRPO as the v1 algorithm and GGUF-Q4 as MT-4, but flagged a load-bearing unknown: candle's
quantization is GGUF (Q4_K), not bitsandbytes NF4, and `QMatMul` is inference-only with no backward — it called
true 4-bit QLoRA a "DIY quantized backward." That framing turns out to be wrong in a way that makes 4-bit
*easier* than feared. Separately, ADR-0006's `CandleServe` now caches its model but runs candle's synchronous
generation on the async runtime thread, which is fine for one stream and not for many. This ADR settles all
three.

## Research synthesis

### GRPO (DeepSeekMath, 2402.03300)

The objective, per group of `G` sampled outputs for a prompt `q`:

```
J(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t { min( ρ_{i,t}·Â_i , clip(ρ_{i,t}, 1−ε, 1+ε)·Â_i ) − β·D_KL[π_θ ‖ π_ref] } ]
ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_{θ_old}(o_{i,t} | q, o_{i,<t})       (per-token policy ratio)
Â_i     = (r_i − mean(r)) / std(r)                                          (group-normalized reward)
D_KL    = π_ref/π_θ − log(π_ref/π_θ) − 1                                    (unbiased estimator)
```

It is critic-free (the group mean is the baseline). What it needs over RAFT:

- **per-token log-probs** of the policy — we already compute logits in the forward; `log_softmax` + gather.
- **old-policy log-probs** `π_{θ_old}` — capture the per-token log-probs *at sampling time* (the policy that
  generated the sample); store them with each sample.
- **a frozen reference** `π_ref` for the KL term.
- **group rewards** — already produced (RAFT samples `K` and verifies each; reuse as the group).

**Candle feasibility (favorable).** No new candle capability is required: the policy log-probs come from the
same LoRA forward whose gradients we already take; the PPO ratio, clip, and KL are elementwise tensor math the
autograd handles. The reference model needs **no second base copy**: since `policy = base + LoRA`, `π_ref` is
`base + (frozen LoRA snapshot)` — for GRPO-from-scratch that is the base with LoRA disabled; for refining an
existing expert it is the starting adapter cloned and frozen. One extra forward with the LoRA toggled, not a
second 3 GB model. Old-policy log-probs are likewise just the LoRA state at sampling time.

### 4-bit QLoRA in candle — dequantize-in-forward, *not* a quantized backward

`candle_core::quantized::QTensor::dequantize(&self, device) -> Result<Tensor>` exists and returns an ordinary
`Tensor`. `QTensor`/`QMatMul` have no backward — **but QLoRA never backpropagates into the frozen base.** The
training graph is:

```
y = dequantize(W_q) · x  +  (α/r)·B·(A·x)
        └ constant ┘         └─ the only Vars ─┘
```

`dequantize(W_q)` and `x` are non-`Var` constants; adding a constant does not break autograd; gradients flow
through the additive LoRA term into `A`, `B` exactly as with an f16 base. So 4-bit training is: **store the
frozen base layers as Q4_K `QTensor`s, `dequantize()` each weight inside the forward, matmul, add the LoRA
term.** The "inference-only `QMatMul` / no-backward `QTensor`" limitation is irrelevant because we use
`dequantize()` + the normal differentiable `matmul`, never `QMatMul`, for training.

- **Memory win:** the base in Q4_K is ~1/4 of f16 (a 1.5 B base ≈ 0.9 GB vs ≈ 3 GB; a 7 B base ≈ 4 GB vs
  ≈ 14 GB).
- **Compute cost:** every forward re-dequantizes each weight to f16/bf16 (transient), trading compute and
  peak per-layer memory for resident memory.
- **Where it pays off:** not on a 1.5 B base on 24 GB (ADR-0006: memory is not v0's binding constraint), but on
  **larger bases / the heterogeneous north star (ADR-0009)**, where f16 base + activations + grads approach the
  card. So 4-bit is a *capacity* lever, gated on base size.

### Serving throughput

`CandleServe::act` holds a `tokio::Mutex` over the cached model and runs candle's **synchronous** generation on
the runtime thread. For a single request stream (the v1 reality) this only parks one worker and is harmless.
True concurrency — many adapters answering at once — is the S-LoRA multi-adapter serving ADR-0006 already
defers, and needs a real inference scheduler, not a `spawn_blocking` patch. Minimal hardening (`block_in_place`,
or a dedicated inference thread/actor) is only warranted once a concurrent path exists.

## Decision

1. **GRPO is the v1 algorithm, built as a second `Trainer` (`grpo.rs`), not a rewrite.** It reuses RAFT's
   sample→verify to form the group, captures old-policy per-token log-probs at sampling, computes the
   group-normalized advantage from the verifier reward, and optimizes the clipped-ratio objective with a KL
   penalty to `π_ref = base + frozen-LoRA-snapshot` (LoRA-toggle, no second base). The existing `RaftTrainer`
   stays the default and the honest baseline; GRPO is swapped in behind the `Trainer` port for A/B.

2. **4-bit QLoRA is implemented as dequantize-in-forward, gated on base size.** Add an optional Q4_K base path
   to the model (`QTensor` per frozen projection, `dequantize()` in the forward; LoRA and the SFT/GRPO loss
   unchanged). Default v1 stays f16 on the 1.5 B base; the Q4_K path turns on for 7 B+ bases / ADR-0009. This
   corrects ADR-0010's "DIY quantized backward" to "dequant-in-forward."

3. **Serving stays single-stream for v1; concurrency is part of the S-LoRA serving work (ADR-0006), not a
   piecemeal patch.** The known runtime-thread block under sync generation is documented and deferred; when a
   concurrent serving path is added, harden with `block_in_place` / a dedicated inference thread there.

## Consequences

- **Positive:** GRPO and 4-bit both reuse the existing forward/LoRA/verifier; the reference-via-LoRA-toggle and
  dequant-in-forward avoid the two scariest costs (a second model, a custom backward). The `Trainer` port keeps
  RAFT and GRPO interchangeable, so GRPO is falsifiable against RAFT.
- **Negative:** GRPO adds real bookkeeping (per-token log-prob capture, old/ref forwards, ratio/clip/KL) and is
  easy to get subtly wrong; 4-bit adds per-forward dequant compute and needs Q4_K CUDA dequant verified.
- **Neutral:** serving concurrency is explicitly out of scope here; this ADR is a study and gates the order
  (GRPO and 4-bit before the EXP-010 scale/forgetting study).

## Alternatives considered

- **GRPO with a second frozen reference model.** Rejected: the LoRA toggle gives `π_ref` for free; a second
  base wastes ~3 GB.
- **NF4 (bitsandbytes) for 4-bit.** Not in candle; Q4_K is candle-native and is the same family `llama-cpp-2`
  serves (ADR-0006), so a 4-bit-trained adapter is serveable without re-quantizing.
- **`spawn_blocking` to fix serving concurrency now.** Rejected as premature: there is no concurrent serving
  path yet, and moving a `&mut` model out of a `&self` mutex into a `'static` closure is the wrong shape; do it
  with the S-LoRA engine.
- **Doing the scale/forgetting study (EXP-010) first.** Deferred at the user's direction: land the efficiency levers, then scale-test.

## Validation (each a falsifiable experiment with a kill criterion)

- **GRPO.** On the same tiny verifiable corpora, GRPO reaches a given pass-rate in **fewer sampled completions**
  than RAFT, or a higher final pass-rate. *Kill:* if GRPO does not beat RAFT on sample-efficiency or final
  quality (and is far more code), keep RAFT and shelve GRPO.
- **4-bit.** A Q4_K base trains a LoRA whose graduated pass-rate matches the f16 base within noise, at ~1/4 the
  resident base memory. *Kill:* if Q4_K dequant-in-forward is too slow or its trained adapter underperforms f16
  materially, 4-bit stays a north-star-only lever.

## Open inputs (need before implementation)

- Confirm **Q4_K `dequantize` on CUDA** (not just CPU) and its per-forward cost on the 3090 Ti.
- Decide GRPO knobs: group size `G` (reuse RAFT's `K`), clip `ε`, KL `β`, and whether `π_θ_old` is refreshed
  each step (single-step PPO) or per round.
- Whether GRPO's reward is the bare verifier pass/fail or the critic-densified signal (ADR-0003).
