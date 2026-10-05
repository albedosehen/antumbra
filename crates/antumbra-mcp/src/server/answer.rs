//! The `answer` tool: route a task, then serve it through the covering expert
//! with the user's standing experts composed in (ADR-0027).
//!
//! Its own router, joined to the others in `engine.rs`. A standing expert holds
//! the behaviours a user accepted for a scope: everywhere, or one repository.
//! It is never the routed expert. It joins every answer in its scope, and
//! answers alone a task like the ones it was taught when nothing else covers
//! it.

use super::*;
use antumbra_core::behaviour::{normalize_scope, EVERYWHERE};
use antumbra_core::ports::Serve;
use antumbra_core::Expert;

#[tool_router(router = answer_router, vis = "pub(super)")]
impl McpServer {
    /// Route a task and serve the answer through the covering expert's adapter
    /// (the full recall→route→serve surface). Escalates when nothing covers it
    /// or when no serving engine is configured.
    #[tool(
        description = "Answer a task: route it across the shared population and your private experts, then generate a response through the covering expert's adapter, with your standing behaviours (for everywhere and for `repo`) composed in. Escalates if nothing covers it."
    )]
    pub(super) async fn answer(
        &self,
        Parameters(p): Parameters<AnswerParams>,
    ) -> Result<Json<AnswerOut>, ErrorData> {
        let Some(serve) = self.serve.clone() else {
            return Ok(Json(escalation(
                None,
                "serving not configured (run the server with a serving engine / --features models)"
                    .into(),
            )));
        };
        let v = self.embedder.embed(&p.task).await.map_err(err)?;
        let (routes, why) = self.routed(&v, 1).await.map_err(err)?;
        let top = routes.first().map(|r| ExpertId::new(r.expert_id.clone()));
        // The serving engine snapshots its adapter population at startup; a route
        // to an expert it can't serve (e.g. one minted afterward) escalates cleanly
        // rather than surfacing a "no adapter registered" error.
        if let Some(top) = &top {
            if !serve.can_serve(top) {
                return Ok(Json(escalation(
                    Some(top.as_str().to_string()),
                    "covering expert not resident in the serving engine; escalate".into(),
                )));
            }
        }
        let standing = self
            .standing_experts(p.repo.as_deref(), serve.as_ref())
            .await
            .map_err(err)?;
        // With nothing routed, a standing expert answers only a task like the
        // ones it was taught; anything else is the generalist's.
        let standing: Vec<ExpertId> = standing
            .into_iter()
            .filter(|e| {
                top.is_some()
                    || e.capability_similarity(&v)
                        .is_some_and(|s| s >= PRIVATE_ROUTE_FLOOR)
            })
            .map(|e| e.id)
            .collect();
        let Some(served_by) = top.clone().or_else(|| standing.first().cloned()) else {
            return Ok(Json(escalation(
                None,
                format!(
                    "no in-scope expert ({}); escalate",
                    why.unwrap_or("nothing covers it")
                ),
            )));
        };
        let mut blend: Vec<ExpertId> = top.into_iter().collect();
        blend.extend(standing.iter().cloned());
        // Generation is synchronous compute inside an `async fn`, like a train: it
        // runs off the runtime's workers so other sessions' calls keep moving.
        let request = ActRequest::new(next_id("answer"), p.task, blend);
        let out = heavy::spawn_heavy(async move { serve.act(request).await })
            .await
            .map_err(err)?
            .map_err(err)?;
        Ok(Json(AnswerOut {
            answer: out.final_output,
            expert_id: Some(served_by.as_str().to_string()),
            escalate: false,
            note: None,
            standing: standing.iter().map(|e| e.as_str().to_string()).collect(),
        }))
    }
}

impl McpServer {
    /// The session user's standing experts that `serve` holds: the one for
    /// everywhere, then the one for `repo`.
    async fn standing_experts(
        &self,
        repo: Option<&str>,
        serve: &dyn Serve,
    ) -> antumbra_core::Result<Vec<Expert>> {
        let mut scopes = vec![EVERYWHERE.to_string()];
        let repo = normalize_scope(repo);
        if repo != EVERYWHERE {
            scopes.push(repo);
        }
        let mine: Vec<Expert> = lifecycle::routable(&self.store)
            .await?
            .into_iter()
            .filter(|e| e.owner.as_ref() == Some(&self.user) && serve.can_serve(&e.id))
            .collect();
        Ok(scopes
            .iter()
            .filter_map(|s| {
                mine.iter()
                    .find(|e| e.standing_scope() == Some(s.as_str()))
                    .cloned()
            })
            .collect())
    }
}

fn escalation(expert_id: Option<String>, note: String) -> AnswerOut {
    AnswerOut {
        answer: String::new(),
        expert_id,
        escalate: true,
        note: Some(note),
        standing: Vec::new(),
    }
}
