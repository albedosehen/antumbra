# ADR-0002 - Shadow models hold the plasticity

**Status:** Accepted · **Date:** 2026-05-30 · **Related:** 0001 (frozen experts), 0003 (critic), 0004 (boundary), 0005 (gate), 0008 (loop)

## Context

ADR-0001 froze the experts to stop forgetting - which removed all ability to learn. Plasticity has to live somewhere. This is Complementary Learning Systems theory in ML - a fast plastic explorer feeding a slow stable store - realized via parameter isolation: LoRA (2106.09685), QLoRA (2305.14314), EWC (1612.00796). In the shadow-geometry metaphor, shadows are the **penumbra**: the partial-shadow ring that forms around the umbra, then either deepens into it (graduates) or fades (prunes).

## Decision

Plasticity lives in short-lived, trainable **shadow** adapters that explore around the frozen core and then **graduate** (freeze into a new expert, ADR-0001) or **prune**.

1. A shadow is a **LoRA/QLoRA adapter** over the shared frozen base - shallow, so consolidation can't drift the core. Graduation = freezing the adapter and adding it to the gate's mixture (ADR-0005), not minting a standalone model.
2. **Training is a DIY `candle` QLoRA path in Rust** (locked this session): LoRA over a 4-bit (NF4) frozen base, implemented ourselves (no Rust Unsloth). This is the **highest-effort component** - first-class engineering, not glue. `burn` is the fallback backend.
3. **Learn from verified outcomes, not imitation** (ADR-0003). The training signal is the environment's verdict (a test passes, a command works) - _not_ a teacher model's text. A flagship critic, if used, only _accelerates cold-start exploration and diagnosis_; what gets baked into the adapter is the verified outcome.
4. **The corpus is your selected repos** (for the coding domain). Per-repo conventions are exactly the context-scoped boundaries (ADR-0004) the system learns. This requires a **code-capable shared base** so shadows can _act_ well enough to generate learnable signal.
5. **Anti-collapse reward is mandatory:** a shadow optimized only to _avoid failure_ collapses to empty output. Reward = positive accomplishment + intrinsic curiosity, not just "not-wrong."
6. **Optimizer:** prefer SGD / added-noise over Adam (Adam worsens forgetting in continual settings).
7. **Isolation invariant:** after any shadow run, the frozen base and existing experts must be byte-identical.

```mermaid
stateDiagram-v2
    [*] --> spawning
    spawning --> exploring: attach LoRA adapter; act on the corpus
    exploring --> scoring: environment verifies + critic densifies (ADR-0003)
    scoring --> exploring: keep training on verified outcomes
    scoring --> graduated: fitness above threshold
    scoring --> pruned: stalled or collapsed
    graduated --> [*]: deepen into a frozen expert (umbra); add to gate (ADR-0001/0005)
    pruned --> [*]: discarded; boundary logged (ADR-0004)
```

## Consequences

- **Positive:** new skill without touching frozen weights; shadows are cheap, disposable adapters; graduation is the _only_ path to permanent capability; learning is grounded in your data + execution.
- **Negative:** the DIY candle QLoRA path is real risk (slower than Unsloth, more to build/maintain); shadow collapse is a live failure; a too-tiny base can't act well enough to generate signal (hence code-capable base).
- **Neutral:** the **gate** itself (ADR-0005) is just another trainable component on the same lifecycle.

## Risks (the DIY-candle-QLoRA seam)

| Risk                                          | Mitigation                                                                                      |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 4-bit QLoRA backprop is hand-rolled in candle | De-risk with **plain LoRA over a bf16 base** first; add NF4 once it works.                      |
| Memory training a 7 to 8 B base               | Gradient checkpointing + paged optimizer (DIY); smaller base if needed.                         |
| Cold-start: tiny model flails, no signal      | Flagship critic accelerates early exploration (ADR-0003); wean off as experts graduate.         |
| candle QLoRA stalls the project               | Thin Python Unsloth trainer behind a clean interface - explicitly the _fallback_, not the plan. |

## Alternatives considered

- **Distill from a teacher's outputs (imitation SFT).** Rejected as the _signal_: it's ToS-sensitive and less grounded than verified outcomes. The teacher is a critic/accelerator, not the target.
- **Full fine-tune of a copy, then freeze.** Rejected: expensive, drifts, loses cheap-disposable.
- **Keep a Python ML plane.** Rejected (single-language Rust); Python trainer survives only as a fallback.

## Validation

**(a) Isolation:** train a shadow to a skill the base lacks; assert the frozen base/experts are byte-identical after. **(b) Grounding:** assert shadows trained on _verified outcomes_ (not teacher text) reach success, with the anti-collapse reward preventing empty output. _Kill criterion:_ isolation leaks, or shadows can't learn from verified outcomes → the freeze+shadow premise is unsound; stop before 0003/0004.
