//! Train a user's accepted behaviours into a private standing expert
//! (ADR-0027), on the GPU.
//!
//! Validation 2's procedure, as one step:
//! 1. Score the base model on each behaviour's held-out examples and on
//!    controls no behaviour governs.
//! 2. Collect its own answers to everyday prompts as replay.
//! 3. Train on the behaviours' examples beside that replay.
//! 4. Score again, and mint the expert only when every behaviour clearly
//!    rose and the controls held (`antumbra_train::behave::admit`).
//!
//! The expert is the user's own, private, and one per scope: a repository,
//! or everywhere. A re-run replaces it.

use chrono::Utc;

use antumbra_core::behaviour::{self, Spec, State, Status};
use antumbra_core::ports::Embedder;
use antumbra_core::{Expert, ExpertId, Generation, Memory, Result, RunId, TenantId, UserId};
use antumbra_store::repo::{expert, memory, principal};
use antumbra_store::{Store, EMBED_DIM};
use antumbra_train::behave::{self as plan, Verdict};
use antumbra_train::{
    capture_corrections, eval_pass_rate, CandleModelLoader, CorpusTask, ModelLoader, RaftConfig,
};

/// What a run did: the expert minted, if admitted, and the scores behind it.
#[derive(Debug, Clone)]
pub struct BehaveReport {
    /// The behaviours taught, by id.
    pub behaviours: Vec<String>,
    /// How many base answers were replayed beside them.
    pub replay: usize,
    pub verdict: Verdict,
    /// The minted expert and its adapter, when admitted.
    pub expert: Option<(ExpertId, String)>,
}

/// The private standing expert for a user's behaviours in `scope`.
pub fn expert_id(user: &UserId, scope: &str) -> ExpertId {
    ExpertId::new(format!("expert:{}:behaviour:{scope}", user.as_str()))
}

/// The behaviours a run teaches: accepted, or trained before and still in
/// force, in `scope`, each with a spec. Retired and proposed ones are left out.
fn learnable(memories: &[Memory], scope: &str) -> Vec<(Memory, Spec)> {
    memories
        .iter()
        .filter_map(|m| {
            let state = State::of(&m.evidence)?;
            let wanted = matches!(state.status, Status::Accepted | Status::Trained);
            (wanted && state.scope == scope)
                .then(|| Some((m.clone(), behaviour::spec_of(&m.content)?)))
                .flatten()
        })
        .collect()
}

/// Train `user`'s behaviours in `scope` into their standing expert. `None`
/// when there is nothing to teach. `cfg` sets the base model, epochs and
/// learning rate; scoring is greedy, one answer per task.
pub async fn train_behaviours(
    store: &Store,
    embedder: &dyn Embedder,
    tenant: &TenantId,
    user: &UserId,
    scope: &str,
    cfg: &RaftConfig,
) -> Result<Option<BehaveReport>> {
    principal::provision(store, tenant, user).await?;
    let compartment = behaviour::compartment_id(tenant, user);
    let memories = memory::list_by_compartment(store, tenant, &compartment).await?;
    let taught = learnable(&memories, scope);
    if taught.is_empty() {
        return Ok(None);
    }
    let ids: Vec<String> = taught
        .iter()
        .map(|(m, _)| m.id.as_str().to_string())
        .collect();
    let (mut train, mut held): (Vec<CorpusTask>, Vec<CorpusTask>) = (Vec::new(), Vec::new());
    for (m, spec) in &taught {
        let t = plan::tasks(m.id.as_str(), spec);
        train.extend(t.train);
        held.extend(t.held);
    }
    held.extend(plan::controls());

    let cfg = RaftConfig {
        temperature: 0.0,
        samples_per_task: 1,
        ..cfg.clone()
    };
    let id = expert_id(user, scope);
    let run = RunId::new(id.as_str().to_string());
    let loader = CandleModelLoader::new(cfg.clone());
    let verifier = antumbra_critic::CommandVerifier;
    let mut model = ModelLoader::load(&loader, &cfg.base_model, None).await?;

    let base = eval_pass_rate(&mut model, &verifier, &held, &run, 1).await?;
    let prompts = plan::replay_prompts();
    let answered = eval_pass_rate(&mut model, &verifier, &prompts, &run, 1).await?;
    let answers: Vec<(String, String)> = answered
        .draws
        .into_iter()
        .map(|d| (d.task, d.completion))
        .collect();
    let replay = plan::replay(&prompts, &answers);
    let replayed = replay.len();
    let mut tasks = train.clone();
    tasks.extend(replay);

    let out = capture_corrections(&mut model, &verifier, &tasks, &[], &run, &cfg, &[]).await?;
    let expert_scores = eval_pass_rate(&mut model, &verifier, &held, &run, 1).await?;
    let verdict = plan::admit(&ids, &base.per_task, &expert_scores.per_task);
    if !verdict.admitted {
        return Ok(Some(BehaveReport {
            behaviours: ids,
            replay: replayed,
            verdict,
            expert: None,
        }));
    }

    // Routed for its owner by the centroid of the tasks it was taught.
    let mut acc = vec![0.0f32; EMBED_DIM];
    for t in &train {
        for (x, b) in acc.iter_mut().zip(embedder.embed(&t.prompt).await?) {
            *x += b;
        }
    }
    let n = train.len().max(1) as f32;
    let rules: Vec<&str> = taught.iter().map(|(_, s)| s.rule.as_str()).collect();
    let now = Utc::now();
    let e = Expert {
        id: id.clone(),
        name: id.as_str().to_string(),
        base_model: cfg.base_model.clone(),
        artifact_uri: out.adapter_uri.clone(),
        capability_card: serde_json::json!({
            "behaviours": ids, "rules": rules, "scope": scope,
            "private": true, "standing": true,
            "controls": verdict.controls.iter().map(|c| serde_json::json!({
                "family": c.family, "base": c.base, "expert": c.expert,
            })).collect::<Vec<_>>(),
        }),
        capability_vec: Some(acc.iter().map(|v| v / n).collect()),
        fitness: expert_scores.pass_rate,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: Some(user.clone()),
        compartment: Some(compartment.clone()),
        // The adapter is on this machine's disk, and adapters stay out of sync
        // (ADR-0017), so the row says which machine can open it.
        placed_on: Some(antumbra_core::this_host()),
        created_at: now,
    };
    expert::delete(store, &e.id).await?;
    expert::insert(store, &e).await?;
    for (mut m, _) in taught {
        behaviour::mark_trained(&mut m.evidence, id.as_str());
        m.updated_at = now;
        memory::upsert(store, &m).await?;
    }
    Ok(Some(BehaveReport {
        behaviours: ids,
        replay: replayed,
        verdict,
        expert: Some((e.id, out.adapter_uri)),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::behaviour::Example;

    fn behaviour(id: &str, status: Status, scope: &str) -> Memory {
        let spec = Spec {
            rule: "A rule.".into(),
            must: vec!["x".into()],
            must_not: vec![],
            examples: vec![Example {
                task: "t".into(),
                answer: "x".into(),
            }],
            violations: vec!["y".into()],
        };
        Memory::new(
            id,
            TenantId::new("ws:a"),
            antumbra_core::MemoryNetwork::Opinion,
            behaviour::content(&spec),
            1.0,
            Utc::now(),
        )
        .with_evidence(vec![
            behaviour::status_evidence(status),
            behaviour::scope_evidence(scope),
        ])
    }

    #[test]
    fn a_run_teaches_the_accepted_and_trained_behaviours_of_its_scope() {
        let all = vec![
            behaviour("memory:accepted", Status::Accepted, "everywhere"),
            behaviour("memory:trained", Status::Trained, "everywhere"),
            behaviour("memory:proposed", Status::Proposed, "everywhere"),
            behaviour("memory:retired", Status::Retired, "everywhere"),
            behaviour("memory:other", Status::Accepted, "github.com/a/b"),
        ];
        let ids: Vec<String> = learnable(&all, "everywhere")
            .into_iter()
            .map(|(m, _)| m.id.as_str().to_string())
            .collect();
        assert_eq!(ids, ["memory:accepted", "memory:trained"]);
        assert_eq!(
            expert_id(&UserId::new("user:a"), "everywhere").as_str(),
            "expert:user:a:behaviour:everywhere"
        );
    }
}
