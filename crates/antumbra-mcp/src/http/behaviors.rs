//! The behavior keeper (ADR-0027): keeps each user's standing experts in step
//! with the behaviors they accepted.
//!
//! On a node that can train, a pass reads every user's behavior compartment
//! and, for each scope, compares what the user has accepted with what the
//! scope's standing expert holds (`antumbra_core::behavior::stale`). A scope
//! behind its behaviors is trained again (`antumbra_serve::behave`) and the
//! new expert served at once. A scope with nothing left to teach loses its
//! expert.
//!
//! Behaviors are recorded on whichever node the agent talks to and reach this
//! one by sync, so the keeper reads the store rather than waiting on a write.
//! A user whose fabric names another node as their trainer is left to it. A
//! set of behaviors that failed admission is not tried again until it
//! changes: the same set would fail the same way.
//!
//! Database work runs in owner mode under the auth lock, a step at a time;
//! training runs outside it.

// Without models nothing trains, and only the tests read the planning.
#![cfg_attr(not(feature = "models"), allow(dead_code))]

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use antumbra_core::behavior::{self, stale, Spec};
use antumbra_core::{Expert, Memory};

/// What one scope's standing expert needs.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Need {
    Nothing,
    /// Train it on the behaviors with this fingerprint.
    Train(String),
    /// Nothing is left to teach: the expert goes.
    Drop,
}

/// A set of behaviors as taught: each id with its content, so a behavior
/// recorded again with a changed rule or check makes a different set.
pub(super) fn fingerprint(taught: &[(Memory, Spec)]) -> String {
    let mut parts: Vec<(&str, &str)> = taught
        .iter()
        .map(|(m, _)| (m.id.as_str(), m.content.as_str()))
        .collect();
    parts.sort();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    parts.hash(&mut h);
    format!("{:x}", h.finish())
}

/// What a scope needs, given what it should teach, its expert, and the set
/// that last failed admission there.
pub(super) fn need(
    taught: &[(Memory, Spec)],
    expert: Option<&Expert>,
    failed: Option<&String>,
) -> Need {
    if !stale(taught, expert) {
        return Need::Nothing;
    }
    if taught.is_empty() {
        return Need::Drop;
    }
    let print = fingerprint(taught);
    if failed == Some(&print) {
        return Need::Nothing;
    }
    Need::Train(print)
}

/// The sets that failed admission, by `(tenant, user, scope)`.
pub(super) type Failed = HashMap<(String, String, String), String>;

/// One scope of a user's, and what its standing expert needs.
#[derive(Debug)]
pub(super) struct ScopePlan {
    pub scope: String,
    pub taught: Vec<(Memory, Spec)>,
    pub expert: Option<Expert>,
    pub need: Need,
}

/// Each scope of one user and what it needs.
pub(super) fn plan_user(
    memories: &[Memory],
    experts: &[Expert],
    failed: &Failed,
    tenant: &str,
    user: &str,
) -> Vec<ScopePlan> {
    behavior::scopes(memories, experts)
        .into_iter()
        .map(|scope| {
            let taught = behavior::learnable(memories, &scope);
            let expert = experts
                .iter()
                .find(|e| e.standing_scope() == Some(scope.as_str()))
                .cloned();
            let key = (tenant.to_string(), user.to_string(), scope.clone());
            let need = need(&taught, expert.as_ref(), failed.get(&key));
            ScopePlan {
                scope,
                taught,
                expert,
                need,
            }
        })
        .collect()
}

#[cfg(feature = "models")]
pub(super) use keeper::spawn;

#[cfg(not(feature = "models"))]
pub(super) fn spawn(_state: std::sync::Arc<super::HttpState>) {
    eprintln!("antumbra-mcp: standing experts not kept (built without --features models)");
}

#[cfg(feature = "models")]
mod keeper {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use anyhow::Result;

    use antumbra_core::{TenantId, UserId};
    use antumbra_serve::behave;
    use antumbra_store::repo::{compartment, device, expert, memory};

    use super::{plan_user, Failed, Need, ScopePlan};
    use crate::http::HttpState;

    /// The pause between passes.
    const INTERVAL: Duration = Duration::from_secs(600);
    /// The wait before the first, so a restart serves before it trains.
    const FIRST: Duration = Duration::from_secs(120);

    /// What a pass did.
    #[derive(Debug, Default)]
    struct Tally {
        trained: usize,
        refused: usize,
        dropped: usize,
    }

    pub(in crate::http) fn spawn(state: Arc<HttpState>) {
        if state.serve.is_none() {
            eprintln!("antumbra-mcp: standing experts not kept (no serving engine)");
            return;
        }
        if !crate::hardware::role().can_train() {
            eprintln!("antumbra-mcp: standing experts not kept (this node does not train)");
            return;
        }
        eprintln!(
            "antumbra-mcp: standing experts kept here, a pass every {} minutes",
            INTERVAL.as_secs() / 60
        );
        tokio::spawn(async move {
            let mut failed = Failed::new();
            tokio::time::sleep(FIRST).await;
            loop {
                let clock = Instant::now();
                match pass(&state, &mut failed).await {
                    Ok(t) if t.trained + t.refused + t.dropped > 0 => eprintln!(
                        "antumbra-mcp: standing experts, pass in {:.0}s: {} trained, {} refused, {} dropped",
                        clock.elapsed().as_secs_f32(),
                        t.trained,
                        t.refused,
                        t.dropped
                    ),
                    Ok(_) => {}
                    Err(e) => eprintln!("antumbra-mcp: standing experts pass stopped: {e:#}"),
                }
                tokio::time::sleep(INTERVAL).await;
            }
        });
    }

    async fn pass(state: &HttpState, failed: &mut Failed) -> Result<Tally> {
        let compartments = {
            let _guard = state.auth.lock().await;
            state.store.signin_root().await?;
            compartment::list_named(&state.store, antumbra_core::behavior::COMPARTMENT_NAME).await?
        };
        let mut tally = Tally::default();
        for c in compartments {
            user(state, &c.tenant, &c.owner, &c.id, failed, &mut tally).await?;
        }
        Ok(tally)
    }

    async fn user(
        state: &HttpState,
        tenant: &TenantId,
        user: &UserId,
        compartment: &antumbra_core::CompartmentId,
        failed: &mut Failed,
        tally: &mut Tally,
    ) -> Result<()> {
        let (memories, experts, here) = {
            let _guard = state.auth.lock().await;
            state.store.signin_root().await?;
            let memories = memory::list_by_compartment(&state.store, tenant, compartment).await?;
            let experts: Vec<antumbra_core::Expert> = expert::list(&state.store)
                .await?
                .into_iter()
                .filter(|e| e.owner.as_ref() == Some(user) && e.standing_scope().is_some())
                .collect();
            // The fabric's trainer for this user, when it names one, does the work.
            let here = device::genesis_for_user(&state.store, tenant, user)
                .await
                .ok()
                .flatten()
                .is_none_or(|g| g.host == state.host);
            (memories, experts, here)
        };
        if !here {
            return Ok(());
        }
        let scopes = plan_user(&memories, &experts, failed, tenant.as_str(), user.as_str());
        for ScopePlan {
            scope,
            taught,
            expert: current,
            need,
        } in scopes
        {
            let key = (
                tenant.as_str().to_string(),
                user.as_str().to_string(),
                scope.clone(),
            );
            match need {
                Need::Nothing => {}
                Need::Drop => {
                    if let Some(e) = current {
                        let _guard = state.auth.lock().await;
                        state.store.signin_root().await?;
                        expert::delete(&state.store, &e.id).await?;
                        tally.dropped += 1;
                        eprintln!(
                            "antumbra-mcp: standing experts, {} dropped: nothing left to teach",
                            e.id.as_str()
                        );
                    }
                }
                Need::Train(print) => {
                    let plan = behave::Plan {
                        tenant: tenant.clone(),
                        user: user.clone(),
                        scope: scope.clone(),
                        taught,
                    };
                    let embedder = {
                        let _guard = state.auth.lock().await;
                        state.store.signin_root().await?;
                        state.embedder_for(tenant).await
                    };
                    let job = plan.clone();
                    let trained = crate::server::heavy::spawn_heavy(async move {
                        behave::train(&job, embedder.as_ref(), &behave::standing_config()).await
                    })
                    .await??;
                    let report = {
                        let _guard = state.auth.lock().await;
                        state.store.signin_root().await?;
                        behave::mint(&state.store, &plan, trained, &state.host).await?
                    };
                    match report.expert {
                        Some((id, uri)) => {
                            if let Some(serve) = &state.serve {
                                serve.register_expert(&id, &uri);
                            }
                            failed.remove(&key);
                            tally.trained += 1;
                            eprintln!(
                                "antumbra-mcp: standing experts, {} trained on {} behavior(s), now served",
                                id.as_str(),
                                report.behaviors.len()
                            );
                        }
                        None => {
                            failed.insert(key, print);
                            tally.refused += 1;
                            eprintln!(
                                "antumbra-mcp: standing experts, {}:{scope} not admitted: {}",
                                user.as_str(),
                                report.verdict.reasons.join("; ")
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
