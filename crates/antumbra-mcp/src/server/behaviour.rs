//! Behaviours (ADR-0027): how the user wants an agent to act, recorded with a
//! check so the user's own expert can learn it from tasks it governs.
//!
//! Its own router, joined to the others in `engine.rs`. A behaviour is a memory
//! in the user's `behaviour` compartment; its rule, check and examples live in
//! the content and its status in the evidence (`antumbra_core::behaviour`).
//! This is the surface over it: record, list, accept, retire.

use super::*;
use antumbra_core::behaviour::{self, Example, Spec, State, Status};

/// The most behaviours one answer lists, newest first.
const MAX_LISTED: usize = 200;

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ExampleParams {
    /// A task the behaviour governs, as a person would ask it.
    pub(super) task: String,
    /// An answer to it that follows the behaviour.
    pub(super) answer: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RecordBehaviourParams {
    /// The rule in one sentence, as the user would say it.
    pub(super) rule: String,
    /// Regular expressions every answer that follows the rule matches. At
    /// least one: a check that only refuses passes an empty answer.
    pub(super) must: Vec<String>,
    /// Regular expressions no answer that follows the rule matches.
    #[serde(default)]
    pub(super) must_not: Vec<String>,
    /// At least three tasks the rule governs, each with an answer that follows
    /// it. Vary the phrasing and the case.
    pub(super) examples: Vec<ExampleParams>,
    /// Answers that break the rule, as an agent that did not know it would
    /// write them. The check must refuse each.
    pub(super) violations: Vec<String>,
    /// The repository it applies to (`host/org/name`), or omit for everywhere.
    pub(super) scope: Option<String>,
    /// The id of a behaviour this one replaces; that one is retired.
    pub(super) supersedes: Option<String>,
    /// `true` only when the user stated this rule themselves in this session.
    /// Otherwise it is proposed, for the user to accept.
    pub(super) accepted: Option<bool>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RecordedBehaviourOut {
    /// Whether it was stored. When false, `problems` says why, and nothing
    /// was written: fix them and record it again.
    pub(super) recorded: bool,
    pub(super) id: Option<String>,
    /// `proposed` or `accepted`.
    pub(super) status: Option<String>,
    /// What kept it from being recorded: a check that does not compile, an
    /// example it refuses, a violation it passes, too few examples.
    pub(super) problems: Vec<String>,
    /// Whether the behaviour named in `supersedes` was found and retired.
    pub(super) superseded: Option<bool>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ListBehavioursParams {
    /// Only this status: `proposed`, `accepted`, `trained` or `retired`.
    pub(super) status: Option<String>,
    /// Only this scope: a repository (`host/org/name`) or `everywhere`.
    pub(super) scope: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BehaviourView {
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
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BehavioursOut {
    pub(super) behaviours: Vec<BehaviourView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct BehaviourIdParams {
    pub(super) behaviour_id: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BehaviourStatusOut {
    /// Whether it is one of your behaviours.
    pub(super) found: bool,
    /// Its status now.
    pub(super) status: Option<String>,
}

#[tool_router(router = behaviour_router, vis = "pub(super)")]
impl McpServer {
    /// Record how the user wants an agent to act, with a check.
    #[tool(
        description = "Record a behaviour: a rule for how an agent should act on a class of tasks, with a check a program can apply. Use it when the user states or corrects how to act (a convention, a preference, a correction), so the user's own expert can learn it. Give the rule in one sentence; `must` and `must_not` regular expressions that decide whether an answer follows it; at least three example tasks with answers that follow it; and answers that break it. The check is tested against the examples and violations before anything is stored; if it fails, `problems` says why and nothing is written."
    )]
    pub(super) async fn record_behaviour(
        &self,
        Parameters(p): Parameters<RecordBehaviourParams>,
    ) -> Result<Json<RecordedBehaviourOut>, ErrorData> {
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
            return Ok(Json(RecordedBehaviourOut {
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
        let scope = behaviour::normalize_scope(p.scope.as_deref());
        let mut evidence = vec![
            behaviour::status_evidence(status),
            behaviour::scope_evidence(&scope),
        ];
        if let Some(old) = &p.supersedes {
            evidence.push(behaviour::supersedes_evidence(old));
        }
        let content = behaviour::content(&spec);
        let compartment = self.behaviour_compartment().await?;
        let embedding = self.embedder.embed(&content).await.map_err(err)?;
        let id = next_id("memory");
        let m = Memory::new(
            id.clone(),
            self.tenant.clone(),
            MemoryNetwork::Opinion,
            content,
            1.0,
            Utc::now(),
        )
        .with_embedding(embedding)
        .in_compartment(compartment)
        .by(self.user.clone(), self.host.clone())
        .with_evidence(evidence)
        // Trained through its tasks and check (`antumbra behave`), never by
        // the write-time consolidation, which would teach it to echo itself.
        .volatile(true);
        memory::upsert(&self.store, &m).await.map_err(err)?;
        let superseded = match &p.supersedes {
            Some(old) => Some(self.set_behaviour_status(old, Status::Retired).await?.found),
            None => None,
        };
        Ok(Json(RecordedBehaviourOut {
            recorded: true,
            id: Some(id),
            status: Some(status.as_str().to_string()),
            problems: Vec::new(),
            superseded,
        }))
    }

    /// The user's behaviours.
    #[tool(
        description = "List your behaviours, newest first: each rule with its scope, status (proposed, accepted, trained, retired), check patterns, and how many examples and violations back it. Filter by `status` or `scope`."
    )]
    pub(super) async fn list_behaviours(
        &self,
        Parameters(p): Parameters<ListBehavioursParams>,
    ) -> Result<Json<BehavioursOut>, ErrorData> {
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
            .map(|s| behaviour::normalize_scope(Some(s)));
        let compartment = behaviour::compartment_id(&self.tenant, &self.user);
        let mut all = memory::list_by_compartment(&self.store, &self.tenant, &compartment)
            .await
            .map_err(err)?;
        all.sort_by_key(|m| std::cmp::Reverse(m.updated_at));
        let behaviours = all
            .iter()
            .filter_map(|m| {
                let state = State::of(&m.evidence)?;
                let spec = behaviour::spec_of(&m.content)?;
                Some((m, state, spec))
            })
            .filter(|(_, state, _)| wanted_status.flatten().is_none_or(|s| s == state.status))
            .filter(|(_, state, _)| wanted_scope.as_ref().is_none_or(|s| *s == state.scope))
            .take(MAX_LISTED)
            .map(|(m, state, spec)| BehaviourView {
                id: m.id.as_str().to_string(),
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
        Ok(Json(BehavioursOut { behaviours }))
    }

    /// Accept a proposed behaviour, so the user's expert learns it.
    #[tool(
        description = "Accept a behaviour, so the next training of your expert learns it. Accept only behaviours the user has agreed to."
    )]
    pub(super) async fn accept_behaviour(
        &self,
        Parameters(p): Parameters<BehaviourIdParams>,
    ) -> Result<Json<BehaviourStatusOut>, ErrorData> {
        Ok(Json(
            self.set_behaviour_status(&p.behaviour_id, Status::Accepted)
                .await?,
        ))
    }

    /// Retire a behaviour, so no expert learns it again.
    #[tool(
        description = "Retire a behaviour that no longer holds, so the next training of your expert leaves it out. It stays readable through list_behaviours."
    )]
    pub(super) async fn retire_behaviour(
        &self,
        Parameters(p): Parameters<BehaviourIdParams>,
    ) -> Result<Json<BehaviourStatusOut>, ErrorData> {
        Ok(Json(
            self.set_behaviour_status(&p.behaviour_id, Status::Retired)
                .await?,
        ))
    }
}

impl McpServer {
    /// Set one of the user's behaviours to `status`; not found when the id is
    /// not a behaviour in their behaviour compartment.
    async fn set_behaviour_status(
        &self,
        id: &str,
        status: Status,
    ) -> Result<BehaviourStatusOut, ErrorData> {
        let compartment = behaviour::compartment_id(&self.tenant, &self.user);
        let found = memory::get(&self.store, &self.tenant, &MemoryId::new(id))
            .await
            .map_err(err)?
            .filter(|m| m.compartment.as_ref() == Some(&compartment))
            .filter(|m| State::of(&m.evidence).is_some());
        let Some(mut m) = found else {
            return Ok(BehaviourStatusOut {
                found: false,
                status: None,
            });
        };
        behaviour::set_status(&mut m.evidence, status);
        m.updated_at = Utc::now();
        memory::upsert(&self.store, &m).await.map_err(err)?;
        Ok(BehaviourStatusOut {
            found: true,
            status: Some(status.as_str().to_string()),
        })
    }

    /// The user's behaviour compartment, created the first time it is needed.
    async fn behaviour_compartment(&self) -> Result<CompartmentId, ErrorData> {
        let id = behaviour::compartment_id(&self.tenant, &self.user);
        let exists = compartment::list_owned(&self.store, &self.tenant, &self.user)
            .await
            .map_err(err)?
            .iter()
            .any(|c| c.id == id);
        if !exists {
            compartment::create(
                &self.store,
                &Compartment::new(
                    id.clone(),
                    self.tenant.clone(),
                    self.user.clone(),
                    "behaviour",
                    Utc::now(),
                ),
            )
            .await
            .map_err(err)?;
        }
        Ok(id)
    }
}
