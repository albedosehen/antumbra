//! Behaviors: how the user wants an agent to act, recorded with a
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

// The schema an agent reads requires the rule, `must`, the examples and the
// violations (`require_the_spec`), but a call that leaves any out is still
// read, so `problems` names everything it lacks in one answer instead of
// deserialization naming one missing field per call.
#[derive(Deserialize, schemars::JsonSchema)]
#[schemars(transform = require_the_spec)]
pub(super) struct RecordBehaviorParams {
    /// The rule in one sentence, as the user would say it.
    #[serde(default)]
    pub(super) rule: String,
    /// Regular expressions every answer that follows the rule matches. At
    /// least one: a check that only refuses passes an empty answer.
    #[serde(default)]
    pub(super) must: Vec<String>,
    /// Regular expressions no answer that follows the rule matches.
    #[serde(default)]
    pub(super) must_not: Vec<String>,
    /// At least four tasks the rule governs, each with an answer that follows
    /// it: training holds one in three out to admit the expert by. Ten, varied
    /// in phrasing and case, teach a rule about a command or a code form
    /// reliably. A rule about how to go about a task, stated in prose (verify
    /// first, do not restart it, check the state before acting), needs sixteen
    /// to twenty: at ten, most such rules were learned on only some of their
    /// held-out tasks.
    #[serde(default)]
    pub(super) examples: Vec<ExampleParams>,
    /// Answers that break the rule, as an agent that did not know it would
    /// write them. The check must refuse each.
    #[serde(default, deserialize_with = "violation_answers")]
    pub(super) violations: Vec<String>,
    /// The repository it applies to (`host/org/name`), or omit for everywhere.
    pub(super) scope: Option<String>,
    /// The id of a behavior this one replaces; that one is retired.
    pub(super) supersedes: Option<String>,
    /// `true` only when the user stated this rule themselves in this session.
    /// Otherwise it is proposed, for the user to accept.
    pub(super) accepted: Option<bool>,
    /// The ids of the memories it was drawn from, when it comes from what the
    /// store already holds (`behavior_candidates`) rather than from the user.
    /// They are not offered as candidates again.
    #[serde(default)]
    pub(super) sources: Vec<String>,
}

/// The fields [`RecordBehaviorParams`] reads when absent but asks for.
const SPEC_FIELDS: [&str; 4] = ["rule", "must", "examples", "violations"];

/// Lists [`SPEC_FIELDS`] as required again, and drops the empty default
/// `#[serde(default)]` gives each, which would read as leave-it-out.
fn require_the_spec(schema: &mut schemars::Schema) {
    if let Some(props) = schema
        .get_mut("properties")
        .and_then(serde_json::Value::as_object_mut)
    {
        for name in SPEC_FIELDS {
            if let Some(prop) = props
                .get_mut(name)
                .and_then(serde_json::Value::as_object_mut)
            {
                prop.remove("default");
            }
        }
    }
    schema.insert("required".into(), serde_json::json!(SPEC_FIELDS));
}

/// A violation as an agent writes it: the answer itself, or a `{task,
/// answer}` pair written the way an example is.
#[derive(Deserialize)]
#[serde(untagged)]
enum Violation {
    Answer(String),
    Example { answer: String },
}

/// The violations' answers, however each was written.
fn violation_answers<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    Ok(Vec::<Violation>::deserialize(d)?
        .into_iter()
        .map(|v| match v {
            Violation::Answer(answer) | Violation::Example { answer } => answer,
        })
        .collect())
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
    /// The memories it was drawn from, when it came from the store. At most
    /// as many as were cited when it was recorded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) sources: Vec<String>,
    /// What keeps it from being taught to the expert, when anything does: one
    /// recorded before today's rules (three examples, where four are needed
    /// now) is listed but never trained. Record it again, superseding this
    /// one, to fix it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) problems: Vec<String>,
    /// How the last training of its scope went for it: whether it was
    /// learned, by its held-out pass rate against the base model's, and
    /// whether the scope's expert was admitted. An admitted expert serves
    /// the behaviors it learned, without the ones it did not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) last_training: Option<String>,
    /// The last time its scope's served expert was measured, beside the base
    /// model, on examples this behavior gained after that expert was
    /// trained: how many, and how many each passed. Evidence on the
    /// admission, from tasks the training never saw; it changes nothing by
    /// itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) last_audit: Option<String>,
    /// When its last training did not learn it: a held-out task it was
    /// missed on, and the trained expert's answer, which failed the check.
    /// An answer that does the right thing in other words points at the
    /// check; one that does something else points at the examples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) missed: Option<MissedView>,
}

/// A held-out task a behavior was missed on, and the expert's answer.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub(super) struct MissedView {
    pub(super) task: String,
    pub(super) answer: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BehaviorsOut {
    pub(super) behaviors: Vec<BehaviorView>,
    /// The regression confidence admission in this workspace is held to,
    /// once the loop has read the workspace's audits: the last reading's.
    /// Absent, the server recipe's applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tuned_confidence: Option<f64>,
    /// The loop's recent readings of the workspace's audits, newest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) tuning: Vec<TuningView>,
}

/// One reading of a workspace's audits against its regression confidence.
#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct TuningView {
    pub(super) at: String,
    pub(super) from: f64,
    pub(super) to: f64,
    /// What it read and what it did.
    pub(super) said: String,
}

/// How many readings `list_behaviors` returns.
const RECENT_READINGS: usize = 5;

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
        description = "Record a behavior: a rule for how an agent should act on a class of tasks, with a check a program can apply. Use it when the user states or corrects how to act (a convention, a preference, a correction), so the user's own expert can learn it. Give the rule in one sentence; `must` and `must_not` regular expressions that decide whether an answer follows it; at least four example tasks with answers that follow it (training holds one in three out to admit the expert by; ten, varied in phrasing and case, teach a rule about a command or a code form reliably, and a prose rule about how to go about a task needs sixteen to twenty); and answers that break it. The check is tested against the examples and violations before anything is stored; if it fails, `problems` says why and nothing is written."
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
            &p.sources,
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
        description = "List your behaviors, newest first: each rule with its scope, status (proposed, accepted, trained, retired), check patterns, how many examples and violations back it, and `problems` when something keeps it from being taught. One its last training did not learn says, under `missed`, a held-out task and what the expert answered instead. Filter by `status` or `scope`."
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
                last_training: behavior::training(&m.evidence).map(|t| t.describe()),
                last_audit: behavior::audits(&m.evidence).last().map(|a| a.describe()),
                missed: behavior::missed::of(&m.evidence).map(|x| MissedView {
                    task: x.task,
                    answer: x.answer,
                }),
                rule: spec.rule,
                scope: state.scope,
                status: state.status.as_str().to_string(),
                must: spec.must,
                must_not: spec.must_not,
                examples: spec.examples.len(),
                violations: spec.violations.len(),
                supersedes: state.supersedes,
                updated_at: m.updated_at.to_rfc3339(),
                sources: state.sources,
            })
            .collect();
        let readings = antumbra_store::repo::behavior_tuning::recent(
            &self.store,
            &self.tenant,
            RECENT_READINGS,
        )
        .await
        .map_err(err)?;
        let tuned_confidence = readings.first().map(|r| r.to);
        let tuning = readings
            .iter()
            .filter_map(|r| {
                Some(TuningView {
                    at: r.at.to_rfc3339(),
                    from: r.from,
                    to: r.to,
                    said: r.reading()?.describe(),
                })
            })
            .collect();
        Ok(Json(BehaviorsOut {
            behaviors,
            tuned_confidence,
            tuning,
        }))
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
