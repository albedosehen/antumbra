//! Autonomous consolidation: graduate a user's reinforced memories from a
//! compartment into a private expert (the hippocampus-to-neocortex path). This
//! is the GPU step the MCP server fires on its own when a memory crosses the
//! gate, and the same step the `consolidate-compartment` CLI command runs, so
//! both mint a private expert identically.

use chrono::Utc;

use antumbra_core::ports::Embedder;
use antumbra_core::{
    CompartmentId, Expert, ExpertId, Generation, Memory, Result, RunId, TenantId, UserId,
};
use antumbra_store::repo::{expert, memory, principal};
use antumbra_store::{Store, EMBED_DIM};
use antumbra_train::consolidate::{score_memory, ConsolidationPolicy};
use antumbra_train::memory::{to_task, ImportPolicy, MemoryRecord};
use antumbra_train::{capture_corrections, CandleModelLoader, CorpusTask, ModelLoader, RaftConfig};

/// What a consolidation produced: the private expert minted (or refreshed) and
/// how many memories graduated into it.
#[derive(Debug, Clone)]
pub struct ConsolidationOutcome {
    pub expert: ExpertId,
    /// The minted adapter's path, so a live engine can hot-register it.
    pub adapter_uri: String,
    pub graduated: usize,
    pub fitness: f32,
}

/// Consolidate one compartment into a private expert: gather its memories, keep
/// the ones that clear the gate (recurrence / stability / verifiability), capture
/// them through the verifier-checked LoRA loop on the GPU, and mint (superseding
/// on re-run) a private expert owned by `user`, routed for its owner by centroid.
/// Returns `None` when nothing in the compartment graduates, so the caller can
/// skip the expensive train and any router refresh.
pub async fn consolidate_compartment(
    store: &Store,
    embedder: &dyn Embedder,
    tenant: &TenantId,
    user: &UserId,
    compartment: &CompartmentId,
    policy: &ConsolidationPolicy,
    cfg: &RaftConfig,
) -> Result<Option<ConsolidationOutcome>> {
    principal::provision(store, tenant, user).await?;

    // Gather -> convert -> score -> keep the graduates (all CPU; the gate is
    // arithmetic). Graduates are trusted captures (they cleared the gate).
    let mems: Vec<Memory> = memory::list_by_compartment(store, tenant, compartment).await?;
    let conv = ImportPolicy {
        capture_threshold: 0.0,
    };
    let tasks: Vec<CorpusTask> = mems
        .iter()
        .enumerate()
        .map(|(i, m)| (i, MemoryRecord::from_memory(m)))
        .filter(|(_, r)| score_memory(r, policy).graduate)
        .map(|(i, r)| to_task(&r, i, &conv).task)
        .collect();
    if tasks.is_empty() {
        return Ok(None);
    }
    let graduated = tasks.len();

    // Capture into a private expert (the only GPU step; the rest is CPU).
    let name = format!("expert:{}:{}", user.as_str(), compartment.as_str());
    let loader = CandleModelLoader::new(cfg.clone());
    let verifier = antumbra_critic::CommandVerifier;
    let mut model = ModelLoader::load(&loader, &cfg.base_model, None).await?;
    let out = capture_corrections(
        &mut model,
        &verifier,
        &tasks,
        &RunId::new(name.clone()),
        cfg,
        &[],
    )
    .await?;

    // A capture that verified nothing leaves the adapter at its untrained init
    // (fitness 0; NaN if the eval itself failed). Minting it would register a
    // non-functional expert that then wins routes and churns escalation, and
    // would supersede any prior working expert. Mint only if it learned something.
    if out.final_fitness <= 0.0 || out.final_fitness.is_nan() {
        eprintln!(
            "[consolidate] {}: capture did not learn (fitness {:.2}); not minting",
            compartment.as_str(),
            out.final_fitness
        );
        return Ok(None);
    }

    // The capability vector is the centroid of the prompts it provably solved.
    let solved: Vec<String> = if out.capability_exemplars.is_empty() {
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
        capability_card: serde_json::json!({
            "exemplars": solved, "compartment": compartment.as_str(), "private": true
        }),
        capability_vec: Some(acc.iter().map(|v| v / nproto).collect()),
        fitness: out.final_fitness,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: Some(user.clone()),
        compartment: Some(compartment.clone()),
        created_at: now,
    };
    expert::delete(store, &e.id).await?; // supersede on re-run
    expert::insert(store, &e).await?;

    Ok(Some(ConsolidationOutcome {
        expert: e.id,
        adapter_uri: e.artifact_uri,
        graduated,
        fitness: out.final_fitness,
    }))
}
