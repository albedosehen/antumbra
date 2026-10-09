//! Behaviors (ADR-0027): how the user wants an agent to act, recorded with a
//! check so the user's own expert can learn it from tasks it governs.
//!
//! Its own router, joined to the others in `engine.rs`. A behavior is a memory
//! in the user's `behavior` compartment; its rule, check and examples live in
//! the content and its status in the evidence (`antumbra_core::behavior`).
//! This is the surface over it: record, list, accept, retire.

use super::*;
use antumbra_core::behavior::{self, Example, Spec, State, Status};
use antumbra_store::repo::behavior as store_behavior;

/// The most behaviors one answer lists, newest first.
const MAX_LISTED: usize = 200;

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ExampleParams {
    /// A task the behavior governs, as a person would ask it.
    pub(super) task: String,
    /// An answer to it that follows the behavior.
    pub(super) answer: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RecordBehaviorParams {
    /// The rule in one sentence, as the user would say it.
    pub(super) rule: String,
    /// Regular expressions every answer that follows the rule matches. At
    /// least one: a check that only refuses passes an empty answer.
    pub(super) must: Vec<String>,
    /// Regular expressions no answer that follows the rule matches.
    #[serde(default)]
    pub(super) must_not: Vec<String>,
    /// At least four tasks the rule governs, each with an answer that follows
    /// it: training holds one in four out to admit the expert by. Vary the
    /// phrasing and the case.
    pub(super) examples: Vec<ExampleParams>,
    /// Answers that break the rule, as an agent that did not know it would
    /// write them. The check must refuse each.
    pub(super) violations: Vec<String>,
    /// The repository it applies to (`host/org/name`), or omit for everywhere.
    pub(super) scope: Option<String>,
    /// The id of a behavior this one replaces; that one is retired.
    pub(super) supersedes: Option<String>,
    /// `true` only when the user stated this rule themselves in this session.
    /// Otherwise it is proposed, for the user to accept.
    pub(super) accepted: Option<bool>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RecordedBehaviorOut {
    /// Whether it was stored. When false, `problems` says why, and nothing
    /// was written: fix them and record it again.
    pub(super) recorded: bool,
    pub(super) id: Option<String>,
    /// `proposed` or `accepted`.
    pub(super) status: Option<String>,
    /// What kept it from being recorded: a check that does not compile, an
    /// example it refuses, a violation it passes, too few examples.
    pub(super) problems: Vec<String>,
    /// Whether the behavior named in `supersedes` was found and retired.
    pub(super) superseded: Option<bool>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ListBehaviorsParams {
    /// Only this status: `proposed`, `accepted`, `trained` or `retired`.
    pub(super) status: Option<String>,
    /// Only this scope: a repository (`host/org/name`) or `everywhere`.
    pub(super) scope: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BehaviorView {
    pub(super) id: String,
    pub(super) rule: String,
    pub(super) scope: String,
    pub(super) status: String,
    pub(super) must: Vec<String>,
    pub(super) must_not: Vec<String>,
    pub(super) examples: usize,
    pub(super) violations: usize,
    pub(super) supersedes: Option<String>,
    pub(super) updated_at: String,
    /// What keeps it from being taught to the expert, when anything does: one
    /// recorded before today's rules (three examples, where four are needed
    /// now) is listed but never trained. Record it again, superseding this
    /// one, to fix it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) problems: Vec<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BehaviorsOut {
    pub(super) behaviors: Vec<BehaviorView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct BehaviorIdParams {
    /// The behavior's `id`, as record_behavior and list_behaviors return it
    /// (`id`, and the earlier `behaviour_id`, are accepted too).
    #[serde(alias = "id", alias = "behaviour_id")]
    pub(super) behavior_id: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BehaviorStatusOut {
    /// Whether it is one of your behaviors.
    pub(super) found: bool,
    /// Its status now.
    pub(super) status: Option<String>,
}

#[tool_router(router = behavior_router, vis = "pub(super)")]
impl McpServer {
    /// Record how the user wants an agent to act, with a check.
    #[tool(
        description = "Record a behavior: a rule for how an agent should act on a class of tasks, with a check a program can apply. Use it when the user states or corrects how to act (a convention, a preference, a correction), so the user's own expert can learn it. Give the rule in one sentence; `must` and `must_not` regular expressions that decide whether an answer follows it; at least four example tasks with answers that follow it (training holds one in four out to admit the expert by); and answers that break it. The check is tested against the examples and violations before anything is stored; if it fails, `problems` says why and nothing is written."
    )]
    pub(super) async fn record_behavior(
        &self,
        Parameters(p): Parameters<RecordBehaviorParams>,
    ) -> Result<Json<RecordedBehaviorOut>, ErrorData> {
        let spec = Spec {
            rule: p.rule.trim().to_string(),
            must: p.must,
            must_not: p.must_not,
            examples: p
                .examples
                .into_iter()
                .map(|e| Example {
                    task: e.task,
                    answer: e.answer,
                })
                .collect(),
            violations: p.violations,
        };
        let problems = spec.problems();
        if !problems.is_empty() {
            return Ok(Json(RecordedBehaviorOut {
                recorded: false,
                id: None,
                status: None,
                problems,
                superseded: None,
            }));
        }
        let status = if p.accepted.unwrap_or(false) {
            Status::Accepted
        } else {
            Status::Proposed
        };
        let scope = behavior::normalize_scope(p.scope.as_deref());
        let embedding = self
            .embedder
            .embed(&behavior::content(&spec))
            .await
            .map_err(err)?;
        let stored = store_behavior::record(
            &self.store,
            &self.tenant,
            &self.user,
            &self.device(),
            MemoryId::new(next_id("memory")),
            &spec,
            status,
            &scope,
            p.supersedes.as_deref(),
            embedding,
        )
        .await
        .map_err(err)?;
        Ok(Json(RecordedBehaviorOut {
            recorded: true,
            id: Some(stored.id.as_str().to_string()),
            status: Some(status.as_str().to_string()),
            problems: Vec::new(),
            superseded: stored.superseded,
        }))
    }

    /// The user's behaviors.
    #[tool(
        description = "List your behaviors, newest first: each rule with its scope, status (proposed, accepted, trained, retired), check patterns, how many examples and violations back it, and `problems` when something keeps it from being taught. Filter by `status` or `scope`."
    )]
    pub(super) async fn list_behaviors(
        &self,
        Parameters(p): Parameters<ListBehaviorsParams>,
    ) -> Result<Json<BehaviorsOut>, ErrorData> {
        let wanted_status = p.status.as_deref().map(Status::parse);
        if let Some(None) = wanted_status {
            return Err(ErrorData::invalid_params(
                "status is one of proposed, accepted, trained, retired",
                None,
            ));
        }
        let wanted_scope = p
            .scope
            .as_deref()
            .map(|s| behavior::normalize_scope(Some(s)));
        let compartment = behavior::compartment_id(&self.tenant, &self.user);
        let mut all = memory::list_by_compartment(&self.store, &self.tenant, &compartment)
            .await
            .map_err(err)?;
        all.sort_by_key(|m| std::cmp::Reverse(m.updated_at));
        let behaviors = all
            .iter()
            .filter_map(|m| {
                let state = State::of(&m.evidence)?;
                let spec = behavior::spec_of(&m.content)?;
                Some((m, state, spec))
            })
            .filter(|(_, state, _)| wanted_status.flatten().is_none_or(|s| s == state.status))
            .filter(|(_, state, _)| wanted_scope.as_ref().is_none_or(|s| *s == state.scope))
            .take(MAX_LISTED)
            .map(|(m, state, spec)| BehaviorView {
                id: m.id.as_str().to_string(),
                problems: spec.problems(),
                rule: spec.rule,
                scope: state.scope,
                status: state.status.as_str().to_string(),
                must: spec.must,
                must_not: spec.must_not,
                examples: spec.examples.len(),
                violations: spec.violations.len(),
                supersedes: state.supersedes,
                updated_at: m.updated_at.to_rfc3339(),
            })
            .collect();
        Ok(Json(BehaviorsOut { behaviors }))
    }

    /// Accept a proposed behavior, so the user's expert learns it.
    #[tool(
        description = "Accept a behavior, so the next training of your expert learns it. Accept only behaviors the user has agreed to."
    )]
    pub(super) async fn accept_behavior(
        &self,
        Parameters(p): Parameters<BehaviorIdParams>,
    ) -> Result<Json<BehaviorStatusOut>, ErrorData> {
        Ok(Json(
            self.set_behavior_status(&p.behavior_id, Status::Accepted)
                .await?,
        ))
    }

    /// Retire a behavior, so no expert learns it again.
    #[tool(
        description = "Retire a behavior that no longer holds, so the next training of your expert leaves it out. It stays readable through list_behaviors."
    )]
    pub(super) async fn retire_behavior(
        &self,
        Parameters(p): Parameters<BehaviorIdParams>,
    ) -> Result<Json<BehaviorStatusOut>, ErrorData> {
        Ok(Json(
            self.set_behavior_status(&p.behavior_id, Status::Retired)
                .await?,
        ))
    }
}

impl McpServer {
    /// Set one of the user's behaviors to `status`; not found when the id is
    /// not a behavior in their behavior compartment.
    async fn set_behavior_status(
        &self,
        id: &str,
        status: Status,
    ) -> Result<BehaviorStatusOut, ErrorData> {
        let set = store_behavior::set_status(&self.store, &self.tenant, &self.user, id, status)
            .await
            .map_err(err)?;
        Ok(BehaviorStatusOut {
            found: set.is_some(),
            status: set.map(|s| s.as_str().to_string()),
        })
    }
}
