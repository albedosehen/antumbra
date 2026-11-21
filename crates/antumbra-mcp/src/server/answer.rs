//! The `answer` tool: route a task, then serve it through the covering expert
//! with the user's standing experts composed in.
//!
//! Its own router, joined to the others in `engine.rs`. A standing expert holds
//! the behaviors a user accepted for a scope: everywhere, or one repository.
//! It is never the routed expert. It joins every answer in its scope, and
//! answers alone a task like the ones it was taught when nothing else covers
//! it, or any task when the caller asks for standing experts only.

use super::*;
use antumbra_core::behavior::{self, normalize_scope, State, Status, EVERYWHERE};
use antumbra_core::ports::{Readiness, Serve};
use antumbra_core::{cosine_similarity, Expert};

#[tool_router(router = answer_router, vis = "pub(super)")]
impl McpServer {
    /// Route a task and serve the answer through the covering expert's adapter
    /// (the full recall→route→serve surface). Escalates when nothing covers it
    /// or when no serving engine is configured.
    #[tool(
        name = "answer",
        description = "Answer a task: route it across the shared population and your private experts, then generate a response through the covering expert's adapter, with your standing behaviors (for everywhere and for `repo`) composed in. Escalates if nothing covers it. With `standing_only`, skip the routing and answer through your standing experts alone, to see what your behaviors do on their own."
    )]
    pub(super) async fn answer_until_cancelled(
        &self,
        params: Parameters<AnswerParams>,
        cancelled: tokio_util::sync::CancellationToken,
    ) -> Result<Json<AnswerOut>, ErrorData> {
        // A client that cancels the call, or whose session closes, is not
        // answered: dropping the answer here is what tells the engine.
        tokio::select! {
            answer = self.answer(params) => answer,
            () = cancelled.cancelled() => Err(ErrorData::internal_error(
                "the request was cancelled before it was answered",
                None,
            )),
        }
    }
}

impl McpServer {
    /// [`Self::answer_until_cancelled`] without the cancelling: what the REST
    /// surface calls, where a client that hangs up drops the call itself.
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
        // Standing only: no route, and the task is embedded for nothing.
        let (top, v, why) = if p.standing_only {
            (None, None, Some("no standing expert of yours in scope"))
        } else {
            let v = self.embedder.embed(&p.task).await.map_err(err)?;
            let (routes, why) = self.routed(&v, 1).await.map_err(err)?;
            let top = routes.first().map(|r| ExpertId::new(r.expert_id.clone()));
            (top, Some(v), why)
        };
        // The serving engine holds the adapters on this machine and the ones it
        // has pulled; a route to an expert it can't serve (one placed elsewhere
        // with no source to pull from, or whose pull failed) escalates cleanly
        // rather than surfacing a "no adapter registered" error.
        if let Some(top) = &top {
            self.hold_routed(serve.as_ref(), top).await;
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
        // ones it was taught; anything else is the generalist's. Asked for
        // standing experts only, they answer whatever the task is.
        let standing: Vec<ExpertId> = standing
            .into_iter()
            .filter(|e| {
                p.standing_only
                    || top.is_some()
                    || v.as_ref()
                        .and_then(|v| e.capability_similarity(v))
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
        // A server that has never answered through an expert may not have its
        // base yet, and fetching it (about 3 GB) outlasts any caller. It is
        // fetched in the background and the task escalates until it is here.
        match serve.prepare() {
            Readiness::Ready => {}
            Readiness::Fetching(what) => {
                return Ok(Json(escalation(
                    Some(served_by.as_str().to_string()),
                    format!(
                        "{what} is downloading to this server for its first answer through an \
                         expert; escalate, and this server answers through experts once it is in"
                    ),
                )));
            }
            Readiness::Failed(why) => {
                return Ok(Json(escalation(
                    Some(served_by.as_str().to_string()),
                    format!(
                        "this server could not fetch its base model ({why}); escalate, and the \
                         next answer tries again"
                    ),
                )));
            }
        }
        let mut blend: Vec<ExpertId> = top.into_iter().collect();
        blend.extend(standing.iter().cloned());
        // The workspace is who asks: where the engine answers several
        // workspaces, they take turns.
        let request =
            ActRequest::new(next_id("answer"), p.task, blend).owned_by(self.tenant.as_str());
        let out = if serve.answers_from_its_own_thread() {
            // The engine generates on its own thread and this only waits, so
            // it waits here: when this call is dropped, by a client that hung
            // up or cancelled, the engine hears of it and stops decoding.
            serve.act(request).await.map_err(err)?
        } else {
            // Generation is synchronous compute inside an `async fn`, like a
            // train: it runs off the runtime's workers so other sessions'
            // calls keep moving.
            heavy::spawn_heavy(async move { serve.act(request).await })
                .await
                .map_err(err)?
                .map_err(err)?
        };
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
            .filter(|e| antumbra_core::behavior::is_standing_for(e, &self.tenant, &self.user))
            .collect();
        // The ones asked for that are placed on another machine are pulled
        // first, when this server has a source for them.
        for e in &mine {
            if e.standing_scope()
                .is_some_and(|s| scopes.iter().any(|w| w == s))
            {
                self.hold(serve, e).await;
            }
        }
        let mine: Vec<Expert> = mine
            .into_iter()
            .filter(|e| serve.can_serve(&e.id))
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

impl McpServer {
    /// Every standing expert of the session user's, the closest to `v` first:
    /// how near the task is to what each was taught, whether the task is in
    /// its scope (everywhere, or `repo`), and whether it would answer the
    /// task with no expert routed to it, decided as `answer` decides it.
    /// `route` leaves standing experts out of its routes, so without this a
    /// workspace that has only them is told nothing about any task.
    pub(super) async fn standing_near(
        &self,
        v: &[f32],
        repo: Option<&str>,
    ) -> antumbra_core::Result<Vec<StandingHit>> {
        let repo = normalize_scope(repo);
        let mut hits: Vec<StandingHit> = lifecycle::routable(&self.store)
            .await?
            .iter()
            .filter(|e| behavior::is_standing_for(e, &self.tenant, &self.user))
            .filter_map(|e| {
                let scope = e.standing_scope()?.to_string();
                let similarity = e.capability_similarity(v).unwrap_or(0.0);
                let in_scope = scope == EVERYWHERE || scope == repo;
                Some(StandingHit {
                    expert_id: e.id.as_str().to_string(),
                    answers: in_scope && similarity >= PRIVATE_ROUTE_FLOOR,
                    scope,
                    similarity,
                    in_scope,
                })
            })
            .collect();
        hits.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        Ok(hits)
    }

    /// The session user's behaviors nearest `v`, closest first, at most
    /// [`TAUGHT_SHOWN`]: what they have taught, or proposed, that bears on
    /// the task, and whether each one's expert learned it. A retired one is
    /// left out, as is one stored without an embedding.
    pub(super) async fn taught_near(&self, v: &[f32]) -> antumbra_core::Result<Vec<TaughtHit>> {
        let compartment = behavior::compartment_id(&self.tenant, &self.user);
        let mut hits: Vec<TaughtHit> =
            memory::list_by_compartment(&self.store, &self.tenant, &compartment)
                .await?
                .iter()
                .filter_map(|m| {
                    let state = State::of(&m.evidence)?;
                    if state.status == Status::Retired {
                        return None;
                    }
                    let spec = behavior::spec_of(&m.content)?;
                    Some(TaughtHit {
                        behavior_id: m.id.as_str().to_string(),
                        rule: spec.rule,
                        scope: state.scope,
                        status: state.status.as_str().to_string(),
                        similarity: cosine_similarity(m.embedding.as_deref()?, v),
                        learned: behavior::training(&m.evidence).map(|t| t.learned),
                    })
                })
                .collect();
        hits.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        hits.truncate(TAUGHT_SHOWN);
        Ok(hits)
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
