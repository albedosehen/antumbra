//! Heavier command handlers kept out of `main.rs` (which is already large):
//! `consolidate` (EXP-021, graduate trusted memories into the population with
//! replay) and `retire` (population-level forgetting, the contradiction hook).
//!
//! These reuse `main.rs`'s `connect` / `make_embedder` / `refresh_router`
//! helpers via `crate::` (a child module sees its parent's private items), so
//! no infrastructure is duplicated.

#[cfg(feature = "models")]
use chrono::Utc;

#[cfg(feature = "models")]
use antumbra_core::{Expert, ExpertId, Generation, RunId};
#[cfg(feature = "models")]
use antumbra_store::repo::expert;
#[cfg(feature = "models")]
use antumbra_store::EMBED_DIM;
#[cfg(feature = "models")]
use antumbra_train::consolidate::{replay_from_tasks, score_memory, ConsolidationPolicy};
#[cfg(feature = "models")]
use antumbra_train::memory::{parse_export, to_task, ImportPolicy, MemoryRecord};
#[cfg(feature = "models")]
use antumbra_train::{
    capture_corrections, CandleModelLoader, Corpus, CorpusTask, JsonCorpus, ModelLoader, RaftConfig,
};

#[cfg(feature = "models")]
use crate::refresh_router;

/// Serialize capture tasks back to the `{id, prompt, verify, completion, skill}`
/// corpus shape (captures carry a completion; seeds do not).
#[cfg(feature = "models")]
fn corpus_to_json(tasks: &[CorpusTask]) -> Vec<serde_json::Value> {
    tasks
        .iter()
        .map(|t| {
            let mut o = serde_json::Map::new();
            o.insert("id".into(), serde_json::json!(t.id));
            o.insert("prompt".into(), serde_json::json!(t.prompt));
            o.insert("verify".into(), t.verify.clone());
            if let Some(c) = &t.completion {
                o.insert("completion".into(), serde_json::json!(c));
            }
            o.insert("skill".into(), serde_json::json!(t.skill()));
            serde_json::Value::Object(o)
        })
        .collect()
}

/// Parameters for [`consolidate`]; mirrors the clap variant so `main.rs`'s arm
/// stays a one-line dispatch. Fields are read only in the `models` build; the
/// stub ignores them.
#[cfg_attr(not(feature = "models"), allow(dead_code))]
pub struct ConsolidateArgs {
    pub source: String,
    pub log: String,
    pub min_recurrence: u32,
    pub min_confidence: f32,
    pub train: bool,
    pub replay_ratio: f64,
    pub run: String,
    pub rounds: usize,
    pub samples: usize,
    pub max_new_tokens: usize,
    pub lr: f64,
    pub grad_accumulation: bool,
}

/// Score a memory export against the consolidation gate, graduate the survivors
/// into per-skill specialists (rehearsing already-consolidated skills via the
/// replay buffer), and append them to the consolidated log (the demotion record
/// and future replay source). The store stays the home of everything that does
/// not graduate.
#[cfg(feature = "models")]
pub async fn consolidate(url: &str, a: ConsolidateArgs) -> anyhow::Result<()> {
    let bytes = std::fs::read(&a.source)?;
    let records = parse_export(&bytes)?;
    let policy = ConsolidationPolicy {
        min_recurrence: a.min_recurrence,
        min_confidence: a.min_confidence,
        ..ConsolidationPolicy::default()
    };

    // Score every memory; a graduate carries its original index (for to_task)
    // and ranking score so a budgeted run could take the strongest first.
    let mut graduates: Vec<(usize, &MemoryRecord, f32)> = Vec::new();
    let mut stays = 0usize;
    for (i, r) in records.iter().enumerate() {
        let v = score_memory(r, &policy);
        let id = r.id.clone().unwrap_or_else(|| format!("mem-{i}"));
        println!("  {} {id}: {}", if v.graduate { "[grad]" } else { "[stay]" }, v.reason);
        if v.graduate {
            graduates.push((i, r, v.score));
        } else {
            stays += 1;
        }
    }
    graduates.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "scored {} memories: {} graduate, {stays} stay in store",
        records.len(),
        graduates.len()
    );

    // Graduates are always trusted captures (they cleared the gate).
    let conv = ImportPolicy { capture_threshold: 0.0 };
    let new_tasks: Vec<CorpusTask> = graduates
        .iter()
        .map(|(i, r, _)| to_task(r, *i, &conv).task)
        .collect();

    // The prior consolidated corpus is the rehearsal source.
    let prior: Vec<CorpusTask> = if std::path::Path::new(&a.log).exists() {
        JsonCorpus::from_file(&a.log)?.tasks(&[])
    } else {
        Vec::new()
    };
    let mut replay = replay_from_tasks(&prior);
    println!("replay buffer: {} prior consolidated example(s)", replay.len());

    if a.train && !new_tasks.is_empty() {
        let store = crate::connect(url).await?;
        let embedder = crate::make_embedder()?;
        let cfg = RaftConfig {
            samples_per_task: a.samples,
            rounds: a.rounds,
            max_new_tokens: a.max_new_tokens,
            learning_rate: a.lr,
            replay_ratio: a.replay_ratio,
            grad_accumulation: a.grad_accumulation,
            ..RaftConfig::default()
        };
        let loader = CandleModelLoader::new(cfg.clone());
        let verifier = antumbra_critic::CommandVerifier;

        // One specialist per skill, each rehearsing already-consolidated skills
        // (and the ones trained earlier this run, appended to `replay` below).
        let mut groups: Vec<(String, Vec<CorpusTask>)> = Vec::new();
        for t in &new_tasks {
            let s = t.skill();
            match groups.iter_mut().find(|(k, _)| *k == s) {
                Some((_, v)) => v.push(t.clone()),
                None => groups.push((s, vec![t.clone()])),
            }
        }
        for (skill, tasks) in groups {
            let name = format!("{}-{skill}", a.run);
            let mut model = ModelLoader::load(&loader, &cfg.base_model, None).await?;
            let out = capture_corrections(
                &mut model,
                &verifier,
                &tasks,
                &RunId::new(name.clone()),
                &cfg,
                &replay,
            )
            .await?;
            let solved = if out.capability_exemplars.is_empty() {
                tasks.iter().map(|t| t.prompt.clone()).collect()
            } else {
                out.capability_exemplars.clone()
            };
            let mut acc = vec![0.0f32; EMBED_DIM];
            for text in &solved {
                for (x, b) in acc.iter_mut().zip(embedder.embed(text).await?) {
                    *x += b;
                }
            }
            let nproto = solved.len().max(1) as f32;
            let now = Utc::now();
            let e = Expert {
                id: ExpertId::new(format!("expert:{name}")),
                name: name.clone(),
                base_model: cfg.base_model.clone(),
                artifact_uri: out.adapter_uri,
                capability_card: serde_json::json!({ "exemplars": solved, "consolidated": true }),
                capability_vec: Some(acc.iter().map(|v| v / nproto).collect()),
                fitness: out.final_fitness,
                frozen_at: Some(now),
                generation: Generation::ZERO,
                owner: None,
                compartment: None,
                created_at: now,
            };
            expert::delete(&store, &e.id).await?; // supersede on re-run
            expert::insert(&store, &e).await?;
            println!(
                "  consolidated '{skill}' -> {name} (internalized {:.2})",
                out.final_fitness
            );
            // Rehearse this freshly-consolidated skill while training the rest.
            replay.extend(replay_from_tasks(&tasks));
        }
        if let Ok(Some(r)) = refresh_router(&store, embedder.as_ref(), 400).await {
            println!("router refreshed over {} experts", r.experts.len());
        }
    } else if !a.train {
        println!("dry run: re-invoke with --train to internalize the graduates");
    }

    // Write-back: append graduates to the consolidated log, deduped by id.
    if !new_tasks.is_empty() {
        let mut merged = prior;
        for t in new_tasks {
            if !merged.iter().any(|m| m.id == t.id) {
                merged.push(t);
            }
        }
        let arr = corpus_to_json(&merged);
        let n = arr.len();
        std::fs::write(&a.log, serde_json::to_vec_pretty(&arr)?)?;
        println!("consolidated log -> {} ({n} entries)", a.log);
    }
    Ok(())
}

#[cfg(not(feature = "models"))]
pub async fn consolidate(_url: &str, _a: ConsolidateArgs) -> anyhow::Result<()> {
    anyhow::bail!("`consolidate` requires building with --features models (candle + a GPU)")
}

/// Parameters for [`consolidate_compartment`].
#[cfg_attr(not(feature = "models"), allow(dead_code))]
pub struct ConsolidateCompartmentArgs {
    pub tenant: String,
    pub user: String,
    pub compartment: String,
    pub min_recurrence: u32,
    pub min_confidence: f32,
    pub rounds: usize,
    pub samples: usize,
    pub max_new_tokens: usize,
    pub lr: f64,
    pub replay_ratio: f64,
}

/// Consolidate a **private compartment** into a **private expert** (ADR-0014/0012,
/// the personalization north star). Gathers the compartment's memories, scores
/// them through the consolidation gate, captures the graduates, and mints an
/// expert tagged `(owner = user, compartment)`. The expert is NOT added to the
/// shared learned router (it would leak); the route tool matches it by centroid
/// for its owner only. Runs as owner (training writes the expert table).
#[cfg(feature = "models")]
pub async fn consolidate_compartment(
    url: &str,
    a: ConsolidateCompartmentArgs,
) -> anyhow::Result<()> {
    use antumbra_core::{CompartmentId, Memory, TenantId, UserId};
    use antumbra_store::repo::{memory, principal};

    let store = crate::connect(url).await?;
    let embedder = crate::make_embedder()?;
    let tenant = TenantId::new(a.tenant.as_str());
    let user = UserId::new(a.user.as_str());
    let comp = CompartmentId::new(a.compartment.as_str());
    principal::provision(&store, &tenant, &user).await?;

    // Gather -> convert -> score -> graduate (all CPU; the gate is arithmetic).
    let mems: Vec<Memory> = memory::list_by_compartment(&store, &tenant, &comp).await?;
    let policy = ConsolidationPolicy {
        min_confidence: a.min_confidence,
        min_recurrence: a.min_recurrence,
        ..ConsolidationPolicy::default()
    };
    let conv = ImportPolicy {
        capture_threshold: 0.0,
    };
    let tasks: Vec<CorpusTask> = mems
        .iter()
        .enumerate()
        .map(|(i, m)| (i, MemoryRecord::from_memory(m)))
        .filter(|(_, r)| score_memory(r, &policy).graduate)
        .map(|(i, r)| to_task(&r, i, &conv).task)
        .collect();
    println!(
        "compartment {} : {} memories, {} graduate",
        a.compartment,
        mems.len(),
        tasks.len()
    );
    if tasks.is_empty() {
        println!("nothing to consolidate");
        return Ok(());
    }

    // Capture into a private expert (the only GPU step; the rest is CPU).
    let cfg = RaftConfig {
        samples_per_task: a.samples,
        rounds: a.rounds,
        max_new_tokens: a.max_new_tokens,
        learning_rate: a.lr,
        replay_ratio: a.replay_ratio,
        ..RaftConfig::default()
    };
    let loader = CandleModelLoader::new(cfg.clone());
    let verifier = antumbra_critic::CommandVerifier;
    let name = format!("expert:{}:{}", a.user, a.compartment);
    let mut model = ModelLoader::load(&loader, &cfg.base_model, None).await?;
    let out = capture_corrections(&mut model, &verifier, &tasks, &RunId::new(name.clone()), &cfg, &[])
        .await?;
    let solved = if out.capability_exemplars.is_empty() {
        tasks.iter().map(|t| t.prompt.clone()).collect()
    } else {
        out.capability_exemplars.clone()
    };
    let mut acc = vec![0.0f32; EMBED_DIM];
    for text in &solved {
        for (x, b) in acc.iter_mut().zip(embedder.embed(text).await?) {
            *x += b;
        }
    }
    let nproto = solved.len().max(1) as f32;
    let now = Utc::now();
    let e = Expert {
        id: ExpertId::new(name.clone()),
        name: name.clone(),
        base_model: cfg.base_model.clone(),
        artifact_uri: out.adapter_uri,
        capability_card: serde_json::json!({ "exemplars": solved, "compartment": a.compartment, "private": true }),
        capability_vec: Some(acc.iter().map(|v| v / nproto).collect()),
        fitness: out.final_fitness,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: Some(user),
        compartment: Some(comp),
        created_at: now,
    };
    expert::delete(&store, &e.id).await?; // supersede on re-run
    expert::insert(&store, &e).await?;
    println!(
        "minted PRIVATE expert {name} for {} (internalized {:.2})",
        a.user, out.final_fitness
    );
    Ok(())
}

#[cfg(not(feature = "models"))]
pub async fn consolidate_compartment(
    _url: &str,
    _a: ConsolidateCompartmentArgs,
) -> anyhow::Result<()> {
    anyhow::bail!(
        "`consolidate-compartment` requires building with --features models (candle + a GPU)"
    )
}

/// Arguments for [`propose_compartments`].
pub struct ProposeCompartmentsArgs {
    pub tenant: String,
    pub user: String,
    /// The inbox compartment whose contents are clustered. Defaults to the MCP
    /// session default `comp:{tenant}:{user}:default`.
    pub inbox: Option<String>,
    pub similarity_threshold: f32,
    pub min_size: usize,
    /// Persist proposals as `Origin::Proposed` compartments and move members in.
    pub apply: bool,
}

/// The antumbra proposes compartments by clustering a user's **unorganized**
/// memory (the inbox compartment plus anything they authored uncompartmented)
/// into competence-coherent regions (ADR-0014). This is the owner/offline
/// surface mirroring the MCP `propose_compartments` tool — cron-able, and the
/// path toward proposing autonomously as the penumbra grows. It needs no model:
/// clustering runs over the embeddings already stored on each memory, so this is
/// available in the default build. Runs as owner (mints compartments / reassigns
/// memory), the same provenance the consolidate ops use.
pub async fn propose_compartments(url: &str, a: ProposeCompartmentsArgs) -> anyhow::Result<()> {
    use antumbra_core::{ClusterConfig, Compartment, CompartmentId, Memory, TenantId, UserId};
    use antumbra_store::repo::{compartment, memory, principal};

    let store = crate::connect(url).await?;
    let tenant = TenantId::new(a.tenant.as_str());
    let user = UserId::new(a.user.as_str());
    principal::provision(&store, &tenant, &user).await?;

    let inbox = a
        .inbox
        .clone()
        .unwrap_or_else(|| format!("comp:{}:{}:default", a.tenant, a.user));
    let inbox_id = CompartmentId::new(inbox.as_str());

    // Pool = the user's unorganized memory: in their inbox, or uncompartmented
    // and authored by them. Deliberately-filed compartments are left alone.
    let pool: Vec<Memory> = memory::list(&store, &tenant)
        .await?
        .into_iter()
        .filter(|m| {
            m.compartment.as_ref() == Some(&inbox_id)
                || (m.compartment.is_none() && m.author.as_ref() == Some(&user))
        })
        .collect();

    let cfg = ClusterConfig {
        similarity_threshold: a.similarity_threshold,
        min_size: a.min_size,
        ..ClusterConfig::default()
    };
    let proposals = antumbra_core::propose_compartments(&pool, &cfg);
    println!(
        "{} unorganized memories -> {} proposal(s)",
        pool.len(),
        proposals.len()
    );

    let now = chrono::Utc::now();
    for (i, p) in proposals.iter().enumerate() {
        println!(
            "  [{i}] '{}' : {} member(s), cohesion {:.3}",
            p.label,
            p.members.len(),
            p.cohesion
        );
        if a.apply {
            let id = format!("comp:{}:{}:proposed:{}", a.tenant, a.user, p.label);
            let c = Compartment::new(
                id.as_str(),
                tenant.clone(),
                user.clone(),
                p.label.clone(),
                now,
            )
            .proposed();
            compartment::create(&store, &c).await?;
            let target = CompartmentId::new(id.as_str());
            let mut moved = 0usize;
            for mid in &p.members {
                if let Some(mut m) = memory::get(&store, &tenant, mid).await? {
                    m.compartment = Some(target.clone());
                    m.updated_at = now;
                    memory::upsert(&store, &m).await?;
                    moved += 1;
                }
            }
            println!("      applied -> {id} ({moved} moved)");
        }
    }
    if !a.apply && !proposals.is_empty() {
        println!("re-run with --apply to create these as proposed compartments");
    }
    Ok(())
}

/// Supersede an expert by name and refresh the router — population-level
/// forgetting. Wire a store's `report_contradiction` against a *consolidated*
/// memory to this: a contradiction retires the expert that memory produced, so
/// the gate stops routing to it (ADR-0004 retire-on-correction at the
/// population scale, since a frozen LoRA cannot be edited per-fact).
#[cfg(feature = "models")]
pub async fn retire(url: &str, expert_name: &str) -> anyhow::Result<()> {
    let store = crate::connect(url).await?;
    let target = expert::list(&store)
        .await?
        .into_iter()
        .find(|e| e.name == expert_name)
        .ok_or_else(|| anyhow::anyhow!("expert '{expert_name}' not found"))?;
    expert::delete(&store, &target.id).await?;
    println!("retired expert {} ({})", target.name, target.id);
    let embedder = crate::make_embedder()?;
    match refresh_router(&store, embedder.as_ref(), 400).await? {
        Some(r) => println!("router refreshed over {} experts", r.experts.len()),
        None => println!("router cleared (fewer than 2 experts remain)"),
    }
    Ok(())
}

#[cfg(not(feature = "models"))]
pub async fn retire(_url: &str, _expert_name: &str) -> anyhow::Result<()> {
    anyhow::bail!("`retire` requires building with --features models (real embedder for the router)")
}

/// Arguments for [`remember`].
pub struct RememberArgs {
    pub tenant: String,
    pub user: String,
    pub compartment: String,
    pub content: String,
    pub network: String,
    pub confidence: f32,
}

/// Seed a memory into a user's compartment from the CLI -- the owner/admin path
/// (the agent-facing writer is the MCP `store_memory` tool). No embedding is
/// attached: consolidation gathers a compartment by membership, not by
/// similarity, so this needs no embedder and runs in the default build. Mint the
/// compartment into a private expert with `consolidate-compartment`.
pub async fn remember(url: &str, a: RememberArgs) -> anyhow::Result<()> {
    use std::hash::{Hash, Hasher};

    use antumbra_core::{CompartmentId, Memory, MemoryNetwork, TenantId, UserId};
    use antumbra_store::repo::{memory, principal};

    let store = crate::connect(url).await?;
    let tenant = TenantId::new(a.tenant.as_str());
    let user = UserId::new(a.user.as_str());
    principal::provision(&store, &tenant, &user).await?;

    let network = match a.network.trim().to_lowercase().as_str() {
        "bank" => MemoryNetwork::Bank,
        "opinion" => MemoryNetwork::Opinion,
        _ => MemoryNetwork::World,
    };
    // Deterministic id from (compartment, content) so re-seeding is idempotent.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    a.compartment.hash(&mut hasher);
    a.content.hash(&mut hasher);
    let id = format!("memory:{:x}", hasher.finish());

    let m = Memory::new(
        id.clone(),
        tenant.clone(),
        network,
        a.content.clone(),
        a.confidence,
        chrono::Utc::now(),
    )
    .by(user.clone(), "cli")
    .in_compartment(CompartmentId::new(a.compartment.as_str()));
    memory::upsert(&store, &m).await?;
    println!(
        "remembered {id} in {} (tenant {}, user {})",
        a.compartment, a.tenant, a.user
    );
    Ok(())
}
