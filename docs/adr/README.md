# Architecture Decision Records

These ADRs capture the **load-bearing decisions** behind Antumbra. **ADR-0004 (the counterfactual boundary) is
the keystone - the thesis the rest of the system serves;** the others are numbered in build-dependency order.
Read them against the [two readings of the chain](../architecture.md#1-the-thesis-and-the-two-readings-of-the-chain).

## Status legend

| Status | Meaning |
|---|---|
| **Accepted** | Decided; build to it. |
| **Keystone** | The central thesis; first-class; the project's defining gate. |
| **Proposed** | Direction set, not yet proven; validated behind a kill criterion. |
| **North star / Deferred** | The target architecture or out-of-scope-for-v0; documented so the seam exists. |

## Index

| ADR | Title | Status | One-line |
|---|---|---|---|
| [0001](0001-frozen-experts.md) | Population of frozen experts | Accepted | Capability = a growing set of small, frozen experts. In v0 each is a **LoRA adapter over a shared base**; freezing stops forgetting. |
| [0002](0002-shadow-plasticity.md) | Shadow models as plasticity | Accepted | Plasticity lives in short-lived trainable shadows (adapters); **train via DIY `candle` QLoRA**; learn from verified outcomes; guard against collapse. |
| [0003](0003-critic-credit-assignment.md) | Critic for credit assignment | Accepted | Verifiable/environment rewards are primary; the critic (rules and/or a flagship) is a **diagnostic densifier**, never the sole signal. |
| [0004](0004-inhibitory-boundaries.md) | Counterfactual boundary | **Keystone** | **The thesis.** Model the *context-scope* of a behavior (right-here / wrong-there + governing feature); built first-class; the make-or-break gate. |
| [0005](0005-orchestrator-router.md) | Router → in-model gate | Accepted | A learned, boundary-conditioned **gate** mixes adapters in latent space; same artifacts give coverage *and* composition; also the escalate-or-answer decision. |
| [0006](0006-hardware-serving.md) | Hardware-adaptive serving | Proposed (**v0 = 1 GPU**) | v0 = single RTX 3090 Ti, one base + adapter library (S-LoRA-style); the fleet + ternary tier are deferred. |
| [0007](0007-surrealdb-substrate.md) | SurrealDB substrate | Proposed | One multi-model DB for every store + vector index + durable flow state, via `surql-rs`. |
| [0008](0008-generational-loop.md) | Durable generational loop | Proposed | grow → explore → score → graduate/prune as a resumable, population-aware state-machine-in-DB. |
| [0009](0009-heterogeneous-composition.md) | Heterogeneous composed model | **North star** | The end goal: genuinely separate frozen experts wired by learned cross-attention bridges (CALM/BTX). v0's gate/boundary/loop/substrate carry over. |
| [0010](0010-candle-qlora-trainer.md) | The candle QLoRA trainer | Accepted (MT-3 validated) | The RAFT reward-ranked LoRA trainer: sample K -> verify -> SFT the winners; candle Qwen2.5-Coder + LoRA; learning validated on the GPU. |

## Format

Each ADR: **Status / Date / Related**, **Context**, **Decision**, **Consequences**, **Alternatives**, and -
where falsifiable - a **Validation** with an explicit **kill criterion**. Diagrams are inline Mermaid.
