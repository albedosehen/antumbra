//! Handoffs: work a session leaves for a session on another of the
//! user's machines, announced at session start until it is marked done.
//!
//! Its own router, joined to the others in `engine.rs`, because `server.rs`
//! is at the size rule. The state lives in the memory's evidence
//! (`antumbra_core::handoff`); this is the surface over it.

use super::*;

use antumbra_core::handoff::{self, HandoffState, ANY};
use antumbra_store::repo::device;

/// The most handoffs one answer lists, newest first. The announcement counts
/// every one waiting; this bounds what a caller is handed.
const MAX_LISTED: usize = 50;

#[tool_router(router = handoff_router, vis = "pub(super)")]
impl McpServer {
    /// Leave a handoff for one of your machines, or any.
    #[tool(
        description = "Leave a note for a session on another of your machines. `for_host` names the machine; omit it, or say \"any\", for whichever starts a session next. A session starting there is told it is waiting until one marks it done with complete_handoff. The first line is its title."
    )]
    pub(super) async fn leave_handoff(
        &self,
        Parameters(p): Parameters<LeaveHandoffParams>,
    ) -> Result<Json<LeftHandoffOut>, ErrorData> {
        let content = p.content.trim();
        if content.is_empty() {
            return Err(ErrorData::invalid_params("a handoff needs content", None));
        }
        let for_host = handoff::normalize_host(p.for_host.as_deref().unwrap_or(ANY));
        let from = p
            .from_host
            .as_deref()
            .map(handoff::normalize_host)
            .unwrap_or_else(|| self.device());
        let compartment = self.handoff_compartment().await?;
        let embedding = self.embedder.embed(content).await.map_err(err)?;
        let id = next_id("memory");
        let m = Memory::new(
            id.clone(),
            self.tenant.clone(),
            MemoryNetwork::Bank,
            content,
            1.0,
            Utc::now(),
        )
        .with_embedding(embedding)
        .in_compartment(compartment)
        .by(self.user.clone(), from)
        .with_evidence(vec![handoff::for_evidence(&for_host)])
        // A message, not something to consolidate into an expert.
        .volatile(true);
        memory::upsert(&self.store, &m).await.map_err(err)?;
        if memory::get(&self.store, &self.tenant, &MemoryId::new(id.clone()))
            .await
            .map_err(err)?
            .is_none()
        {
            return Err(ErrorData::internal_error(
                "the handoff did not land in your handoff compartment",
                None,
            ));
        }
        let mut devices: Vec<String> = device::list_for_user(&self.store, &self.tenant, &self.user)
            .await
            .map_err(err)?
            .iter()
            .map(|d| handoff::normalize_host(&d.host))
            .collect();
        devices.sort();
        devices.dedup();
        let registered_device = for_host == ANY || devices.contains(&for_host);
        Ok(Json(LeftHandoffOut {
            id,
            for_host,
            registered_device,
            devices,
        }))
    }

    /// The handoffs waiting for a machine.
    #[tool(
        description = "The handoffs waiting for a machine: those addressed to it or to any machine and not yet marked done, newest first, with the lines a session-start block announces them under. `include_done` also lists the ones already dealt with; `full` returns whole texts."
    )]
    pub(super) async fn handoffs(
        &self,
        Parameters(p): Parameters<HandoffsParams>,
    ) -> Result<Json<HandoffsOut>, ErrorData> {
        let host = p
            .host
            .as_deref()
            .map(handoff::normalize_host)
            .unwrap_or_else(|| self.device());
        let compartment = handoff::compartment_id(&self.tenant, &self.user);
        let all = memory::list_by_compartment(&self.store, &self.tenant, &compartment)
            .await
            .map_err(err)?;
        let waiting = handoff::waiting_for(&all, &host);
        let announcement = handoff::announcement(&waiting, &host, Utc::now());
        let listed: Vec<&Memory> = if p.include_done.unwrap_or(false) {
            let mut mine: Vec<&Memory> = all
                .iter()
                .filter(|m| {
                    HandoffState::of(&m.evidence)
                        .is_some_and(|s| s.for_host == ANY || s.for_host == host)
                })
                .collect();
            mine.sort_by_key(|m| std::cmp::Reverse(m.created_at));
            mine
        } else {
            waiting
        };
        let full = p.full.unwrap_or(false);
        let handoffs = listed
            .into_iter()
            .take(MAX_LISTED)
            .filter_map(|m| {
                let state = HandoffState::of(&m.evidence)?;
                let view = MemoryView::from(m).bounded(full, RECALL_CONTENT_CHARS);
                Some(HandoffView {
                    id: view.id,
                    title: handoff::title(&m.content),
                    content: view.content,
                    content_chars: view.content_chars,
                    truncated: view.truncated,
                    from_host: m.author_host.clone(),
                    for_host: state.for_host,
                    left_at: m.created_at.to_rfc3339(),
                    done_at: state.done.as_ref().map(|(at, _)| at.to_rfc3339()),
                    done_by: state.done.map(|(_, by)| by),
                })
            })
            .collect();
        Ok(Json(HandoffsOut {
            host,
            handoffs,
            announcement,
        }))
    }

    /// Mark a handoff dealt with.
    #[tool(
        description = "Mark a handoff dealt with, so it is no longer announced at session start. It stays readable through handoffs with include_done."
    )]
    pub(super) async fn complete_handoff(
        &self,
        Parameters(p): Parameters<CompleteHandoffParams>,
    ) -> Result<Json<CompletedHandoffOut>, ErrorData> {
        let compartment = handoff::compartment_id(&self.tenant, &self.user);
        let found = memory::get(&self.store, &self.tenant, &MemoryId::new(p.handoff_id))
            .await
            .map_err(err)?
            .filter(|m| m.compartment.as_ref() == Some(&compartment));
        let not_found = Json(CompletedHandoffOut {
            found: false,
            already_done: false,
        });
        let Some(mut m) = found else {
            return Ok(not_found);
        };
        let Some(state) = HandoffState::of(&m.evidence) else {
            return Ok(not_found);
        };
        if state.done.is_some() {
            return Ok(Json(CompletedHandoffOut {
                found: true,
                already_done: true,
            }));
        }
        let by = p
            .host
            .as_deref()
            .map(handoff::normalize_host)
            .unwrap_or_else(|| self.device());
        let now = Utc::now();
        m.evidence.push(handoff::done_evidence(&by, now));
        m.updated_at = now;
        memory::upsert(&self.store, &m).await.map_err(err)?;
        Ok(Json(CompletedHandoffOut {
            found: true,
            already_done: false,
        }))
    }
}

impl McpServer {
    /// The user's handoff compartment, created the first time it is needed.
    async fn handoff_compartment(&self) -> Result<CompartmentId, ErrorData> {
        let id = handoff::compartment_id(&self.tenant, &self.user);
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
                    "handoff",
                    Utc::now(),
                ),
            )
            .await
            .map_err(err)?;
        }
        Ok(id)
    }
}
