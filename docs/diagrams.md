# Antumbra - Diagram Atlas

Every load-bearing diagram in one place. Each is reproduced from its source-of-truth document; follow the link
in the caption to read the surrounding decision. Grouped: **concept → system → flow → schema → per-ADR
mechanism → north star.**

- Concept: [the shadow](#1-the-shadow-umbra--penumbra--antumbra) · [decision chain](#2-the-keystone-decision-chain)
- System: [v0 architecture](#3-v0-system-architecture) · [v0 vs north star](#5-v0-vs-north-star)
- Flow: [training & data](#4-training--data-flow)
- Schema: [entities](#6-schema---entities) · [substrate](#13-adr-0007---surrealdb-substrate)
- Per-ADR: [0001](#7-adr-0001---umbra-the-frozen-population) · [0002](#8-adr-0002---penumbra-the-shadow-lifecycle) ·
  [0003](#9-adr-0003---criticverifier-credit) · [0004](#10-adr-0004---the-boundary-engine-keystone) ·
  [0005](#11-adr-0005---the-boundary-conditioned-gate) · [0006](#12-adr-0006---single-gpu-serving) ·
  [0008](#14-adr-0008---the-generational-loop) · [0009](#15-adr-0009---heterogeneous-composition)

---

## Concept

### 1. The shadow: umbra · penumbra · antumbra

The name is the model. A cast shadow has three regions and so does the system: the **umbra** is the proven
frozen experts; the **penumbra** is the shadows-in-training; the **antumbra** is the keystone boundary where
coverage *inverts* and the system must escalate. Source: [README](../README.md).

```mermaid
flowchart LR
    PEN["PENUMBRA<br/>shadows-in-training - explore,<br/>then deepen or fade"] -->|"graduate (deepen to full shadow)"| UMB["UMBRA<br/>frozen experts<br/>(adapters over a shared base)"]
    PEN -->|"prune"| X["dissipated"]
    UMB -.->|"cast a new shadow"| PEN
    ANT["ANTUMBRA - the keystone<br/>counterfactual scope:<br/>where coverage inverts,<br/>and when to escalate"] -.->|"gates"| UMB
    X -.->|"log why + where it failed"| ANT
```

### 2. The keystone decision chain

Read conceptually, the architecture radiates from ADR-0004; the ADRs are numbered 0001→0009 in the opposite
(build-dependency) order. Source: [architecture §1](architecture.md#1-the-thesis-and-the-two-readings-of-the-chain).

```mermaid
flowchart TD
    T["ADR-0004 · KEYSTONE<br/>counterfactual scope of competence<br/>right-here / wrong-there + governing feature"]
    T --> N1["needs a STABLE substrate<br/>ADR-0001 frozen experts"]
    T --> N2["needs PROBES near the edge<br/>ADR-0002 shadows"]
    T --> N3["needs to MEASURE correctness<br/>ADR-0003 critic + verifiers"]
    T --> N4["is CONSUMED by the gate<br/>ADR-0005 router/gate"]
    T --> N5["is REFINED each generation<br/>ADR-0008 loop"]
    T --> N6["scales to separate models<br/>ADR-0009 north star"]
```

---

## System

### 3. v0 system architecture

One frozen code-capable base + a library of frozen LoRA experts + a learned, boundary-conditioned gate, all in
a single-plane Rust process on one GPU. Source: [architecture §2](architecture.md#2-system-architecture-v0---shared-base-adapters-single-plane-rust).

```mermaid
flowchart TB
    task["task (e.g. a repo task)"] --> GATE
    subgraph RUST["Antumbra - single-plane Rust process"]
        GATE["Learned gate · ADR-0005<br/>boundary-conditioned adapter mixer"]
        BASE["shared frozen base (code-capable)"]
        ADPT["frozen LoRA experts - the population · ADR-0001"]
        BND["boundary engine · ADR-0004<br/>counterfactual scope"]
        CRIT["critic harness · ADR-0003<br/>verifiers + (optional) flagship-as-critic"]
        LOOP["generational loop · ADR-0008"]
        TR["trainer · ADR-0002<br/>candle QLoRA + gate"]
        STORE["store layer · surql-rs"]
    end
    GATE --> ADPT
    ADPT --> BASE
    BASE --> OUT["output"]
    BND -. "scope: which experts are in-scope;<br/>when to escalate" .-> GATE
    LOOP --> TR
    LOOP --> CRIT
    CRIT --> BND
    BND --> STORE
    GATE --> STORE
    GATE <--> GPU["RTX 3090 Ti · 24 GB"]
    TR <--> GPU
    STORE <--> DB[("SurrealDB · graph · vector · doc · flow-state")]
    CRIT -. "out-of-scope / cold-start only" .-> FLAG["optional flagship<br/>escalation tier (shrinks over time)"]
```

### 5. v0 vs north star

What carries over (gate, boundary engine, loop, substrate) and what changes (only the composition substrate).
Source: [architecture §4](architecture.md#4-v0-scope-vs-the-north-star).

```mermaid
flowchart TB
    subgraph V0["v0 - shared-base adapters (build now)"]
        G["RTX 3090 Ti · 24 GB"]
        G --> S1["one frozen code-capable base + frozen LoRA experts"]
        G --> S2["learned boundary-conditioned gate (latent mixing)"]
    end
    subgraph NS["north star - heterogeneous composed model · ADR-0009"]
        H["genuinely separate frozen experts"]
        H --> H1["learned cross-attention bridges (CALM/BTX)"]
        H --> H2["sparse top-k selection + paged experts"]
    end
    V0 -. "gate, boundary engine, loop, substrate all carry over;<br/>only the composition substrate changes" .-> NS
```

---

## Flow

### 4. Training & data flow

The environment is the truth; the critic only densifies; you train on the verified outcome, never the critic's
text. Source: [architecture §3](architecture.md#3-training--data-flow-verifiable-outcomes-not-imitation).

```mermaid
flowchart LR
    CORP["your selected repos<br/>(the corpus)"] --> ACT["expert/shadow ACTS<br/>(runs a command, writes code)"]
    ACT --> ENV["environment judges<br/>tests / build / exec = TRUTH"]
    ENV -->|"fail"| CRIT["critic densifies + diagnoses<br/>(verifier rules and/or flagship)<br/>names the governing feature"]
    CRIT --> BND["counterfactual boundary<br/>(scope + governing feature) · ADR-0004"]
    ENV -->|"verified outcome"| TRAIN["train on the VERIFIED OUTCOME<br/>(not the critic's text) · ADR-0002"]
    BND --> TRAIN
    TRAIN --> GRAD["graduate adapter · ADR-0001"]
```

---

## Schema

### 6. Schema - entities

Conceptual ER view; full DDL is in [ADR-0007](adr/0007-surrealdb-substrate.md). Source:
[architecture §5](architecture.md#5-schema-surrealdb---summary).

```mermaid
erDiagram
    EXPERT ||--o{ SHADOW : "explored-by"
    SHADOW |o--|| EXPERT : "graduates-into"
    EXPERT ||--o{ EVALUATION_RUN : "measured-by"
    ORCHESTRATION_RUN ||--o{ REWARD_SIGNAL : "scored-by"
    FAILURE_BOUNDARY ||--o{ SHADOW : "evidenced-by"
    EXPERT {
        string name
        string base_model
        string artifact_uri
        array  capability_vec
        float  fitness
    }
    FAILURE_BOUNDARY {
        string behavior
        object scope
        array  governing_features
        array  context_vec
        float  confidence
    }
```

---

## Per-ADR mechanism diagrams

### 7. ADR-0001 - umbra: the frozen population

A base + a growing library of frozen adapters, mixed in latent space by the gate. Source:
[ADR-0001](adr/0001-frozen-experts.md).

```mermaid
flowchart LR
    subgraph POP["Umbra - frozen experts (adapters over a shared base)"]
        E1["adapter: deno-repo conventions"]
        E2["adapter: brand-voice draft"]
        E3["adapter: weekly-deck"]
        En["..."]
    end
    BASE["shared frozen base (code-capable)"]
    new["graduated shadow (ADR-0002)"] -->|freeze + add| POP
    POP --> GATE["boundary-conditioned gate (ADR-0005)<br/>mixes adapters in latent space"]
    BASE --> GATE
```

### 8. ADR-0002 - penumbra: the shadow lifecycle

Plasticity lives only in short-lived shadows: spawn → explore → score → graduate (deepen to umbra) or prune.
Source: [ADR-0002](adr/0002-shadow-plasticity.md).

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

### 9. ADR-0003 - critic/verifier credit

Verifiers are primary ground truth; the critic only interpolates dense per-step credit between them and can
never override a verifier. Source: [ADR-0003](adr/0003-critic-credit-assignment.md).

```mermaid
flowchart LR
    step["expert step output"] --> V["Verifiers (PRIMARY)<br/>tests · schema · exec"]
    step --> C["Critic (DENSIFIER)<br/>PRM-style, per-step"]
    V -->|"sparse, trusted<br/>checkpoints"| AGG["reward_signal<br/>(source-tagged)"]
    C -->|"dense credit<br/>between checkpoints"| AGG
    AGG --> SH["shadow training signal (ADR-0002)"]
    V -. "bounds the critic - <br/>it cannot unilaterally steer" .-> C
```

### 10. ADR-0004 - the boundary engine (keystone)

The antumbra made mechanical: hold behavior fixed, vary context until acceptability flips, recover the governing
feature + grain, then inhibit *only in-scope* and steer. Source: [ADR-0004](adr/0004-inhibitory-boundaries.md).

```mermaid
flowchart TB
    F["B judged INCORRECT in context C<br/>(verifier / instruction · ADR-0003)"] --> ENG
    subgraph ENG["Boundary engine - first-class subsystem"]
        SRCH["1 · counterfactual search (context-direction)<br/>hold behavior B fixed; vary CONTEXT<br/>along candidate dimensions; re-probe"]
        SRCH --> FLIP{"acceptability flips?<br/>B becomes correct"}
        FLIP -->|"no"| SRCH
        FLIP -->|"yes, at C-prime"| BND["boundary = SCOPE of B<br/>governing feature + grain<br/>(C incorrect / C-prime correct)"]
        MODEL["2 · scope model<br/>P(B correct given B, context)"] -. "decision surface over context = scope" .-> BND
    end
    BND --> STORE["failure_boundary store · ADR-0007"]
    STORE --> INH["INHIBIT B only WITHIN its incorrect-scope<br/>(global inhibition = false inhibition)"]
    STORE --> STEER["STEER toward in-scope alternatives"]
    INH --> R["Gate · ADR-0005"]
    STEER --> R
    STEER --> L["Loop · ADR-0008<br/>where + what context to probe next"]
```

### 11. ADR-0005 - the boundary-conditioned gate

A learned in-model mixer: score adapters, let the scope gate/steer, blend in-scope experts in latent space, or
escalate out-of-scope (which becomes the next training example). Source: [ADR-0005](adr/0005-orchestrator-router.md).

```mermaid
flowchart TB
    task["task + context"] --> EMB["embed task"]
    EMB --> SCORE["score adapters:<br/>capability_sim + fitness"]
    BND["counterfactual scope · ADR-0004"] -. "gate in-scope; steer; flag out-of-scope" .-> SCORE
    SCORE --> DEC{"in scope?"}
    DEC -->|"yes"| MIX["blend top-k adapters<br/>in latent space"]
    DEC -->|"no / low-confidence"| ESC["escalate to flagship tier<br/>(becomes a training example)"]
    MIX --> OUT["output"]
    MIX --> LOG["persist orchestration_run + reward (0003/0008)"]
    ESC --> LOG
```

### 12. ADR-0006 - single-GPU serving

v0 is one base + an adapter library (S-LoRA-style) served and trained on one RTX 3090 Ti; the fleet, ternary
tier, and heterogeneous composition all defer. Source: [ADR-0006](adr/0006-hardware-serving.md).

```mermaid
flowchart TB
    subgraph V0["v0 - build now"]
        G["RTX 3090 Ti · 24 GB"]
        G --> S1["serve: one base + adapter library (S-LoRA-style)"]
        G --> T1["train: candle QLoRA adapters + gate"]
    end
    subgraph FUTURE["deferred"]
        direction LR
        N9["ADR-0009 heterogeneous composed model<br/>(separate models + cross-attention bridges, paging)"]
        FLEET["fleet: M4 Pro 48GB (MLX) · 3080 mobile · 1080 · Jetson Orin"]
        TERN["native-ternary tier: Bonsai / BitNet"]
    end
    V0 -. "once loop / gate / boundary are proven" .-> FUTURE
```

### 13. ADR-0007 - SurrealDB substrate

One multi-model engine is every store + vector index + graph + durable flow state, reached only through
`surql-rs`. Full DDL lives in the ADR. Source: [ADR-0007](adr/0007-surrealdb-substrate.md).

```mermaid
flowchart TB
    subgraph DB["SurrealDB (one instance)"]
        direction LR
        DOC["document<br/>expert · shadow · device_profile"]
        VEC["vector / HNSW<br/>capability_vec · context_vec"]
        GRAPH["graph / RELATE<br/>graduated_into · explores · placed_on"]
        FLOW["durable flow state<br/>orchestration_run · status checkpoints"]
    end
    surql["surql-rs (Rust)"] --> DB
    R["Router (0005)"] --> surql
    L["Loop (0008)"] --> surql
    CR["Critic (0003)"] --> surql
```

### 14. ADR-0008 - the generational loop

A resumable state-machine-in-DB: grow → explore → score → graduate/prune → consolidate, restartable from the
persisted `status` checkpoint. Source: [ADR-0008](adr/0008-generational-loop.md).

```mermaid
stateDiagram-v2
    [*] --> grow
    grow --> explore: spawn shadows around weak/blank capabilities
    explore --> score: critic + verifiers (ADR-0003)
    score --> graduate: winners (fitness above threshold)
    score --> prune: losers (stalled / collapsed)
    graduate --> consolidate: freeze into experts (ADR-0001)
    prune --> consolidate: log boundary (ADR-0004)
    consolidate --> grow: next generation
    consolidate --> [*]: paused (fully resumable)
```

### 15. ADR-0009 - heterogeneous composition

The north star: sparse top-k selection over genuinely separate frozen experts wired by learned, per-expert
cross-attention bridges, with the scope gating both selection and bridge gain. Source:
[ADR-0009](adr/0009-heterogeneous-composition.md).

```mermaid
flowchart TB
    task["task + context"] --> SEL["sparse selection<br/>top-k in-scope experts"]
    BND["counterfactual scope · ADR-0004"] -. "gate selection + bridge gain" .-> SEL
    SEL --> E1["frozen expert A (paged)"]
    SEL --> E2["frozen expert B (paged)"]
    E1 <--> BR["learned cross-attention bridges<br/>(modular, per-expert)"]
    E2 <--> BR
    BR --> OUT["composed output"]
```

### 16. ADR-0012–0015 - memory, tenancy, compartments, MCP

The newer subsystems carry their diagrams inline in their ADRs, to avoid drift:

- **[ADR-0012](adr/0012-penumbra-memory.md)** - the consolidation arc (penumbra memory → score → capture+replay
  → umbra; contradiction → retire).
- **[ADR-0013](adr/0013-tenant-isolation-identity.md)** - the identity hierarchy and the engine `PERMISSIONS`
  boundary (tenant org → user → compartment → agent; shared umbra, private penumbra).
- **[ADR-0014](adr/0014-compartments.md)** - penumbra → antumbra-clustered compartments → (share / consolidate
  into a private expert).
- **[ADR-0015](adr/0015-mcp-runtime-surface.md)** - the 12-tool MCP surface over the bound `(tenant, user)`.
