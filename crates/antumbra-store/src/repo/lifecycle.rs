//! Expert lifecycle repository (ADR-0022 S-5): the append-only history of every
//! expert's status changes, and the population as the gate may see it.
//!
//! A move is one row, never rewritten, numbered per expert. The number orders
//! an expert's history, and the unique index on it turns two writers racing to
//! move the same expert into one success and one refusal rather than two
//! forks. Each row carries its expert's owner under the expert table's own
//! permissions, so a session sees exactly the moves of the experts it can see:
//! a session that could read an expert but not its moves would take a demoted
//! expert for an active one.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use surql::query::builder::Query;
use surql::query::crud::{create_record, query_records};
use surql::types::operators::eq;

use antumbra_core::{
    current_status, AntumbraError, Expert, ExpertId, ExpertStatus, ExpertTransition, Generation,
    LearnedRouter, Result, TransitionCause, VerifierId,
};

use crate::error::map;
use crate::repo::{expert, router};
use crate::store::Store;

const TABLE: &str = "expert_transition";

#[derive(Serialize, Deserialize)]
struct TransitionRow {
    #[serde(flatten)]
    transition: ExpertTransition,
    /// The move's place in its expert's history, from 0.
    seq: u32,
    // Omitted when None, as on the expert, so a shared expert's moves read as
    // `owner = NONE`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    owner: Option<String>,
}

/// Move expert `id` to `to` for `cause`, if its state machine allows it, and
/// record the move. `generation` is the one that decided it, when a run did.
///
/// Refused when the expert does not exist, when the move is not one its
/// current status allows, and when a deletion's cause is not redundancy with
/// another expert that is itself active: an expert cannot be covered by one
/// the gate no longer routes to, or by itself.
pub async fn transition(
    store: &Store,
    id: &ExpertId,
    to: ExpertStatus,
    cause: TransitionCause,
    generation: Option<Generation>,
) -> Result<ExpertTransition> {
    let Some(subject) = expert::get(store, id).await? else {
        return Err(AntumbraError::rejected(format!("no expert {id}")));
    };
    if let TransitionCause::Redundant { of } = &cause {
        if of == id {
            return Err(AntumbraError::rejected(format!(
                "{id} cannot be redundant with itself"
            )));
        }
        let covering = expert::get(store, of).await?;
        if covering.is_none() || !status_of(store, of).await?.is_routable() {
            return Err(AntumbraError::rejected(format!(
                "{of} does not cover {id}: it is not an active expert"
            )));
        }
    }
    let past = history(store, id).await?;
    let from = current_status(&past);
    from.transition(to, &cause)?;
    let transition = ExpertTransition {
        expert: id.clone(),
        from,
        to,
        cause,
        generation,
        at: Utc::now(),
    };
    let row = TransitionRow {
        transition: transition.clone(),
        seq: u32::try_from(past.len()).unwrap_or(u32::MAX),
        owner: subject.owner.as_ref().map(|u| u.as_str().to_string()),
    };
    create_record(store.client(), TABLE, serde_json::to_value(row)?)
        .await
        .map_err(map)?;
    Ok(transition)
}

/// Every move expert `id` has made, in order.
pub async fn history(store: &Store, id: &ExpertId) -> Result<Vec<ExpertTransition>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("expert", id.as_str()))
        .order_by("seq", "ASC")
        .map_err(map)?;
    let rows: Vec<TransitionRow> = query_records(store.client(), &query).await.map_err(map)?;
    Ok(rows.into_iter().map(|r| r.transition).collect())
}

/// Where expert `id` stands: active until it has moved.
pub async fn status_of(store: &Store, id: &ExpertId) -> Result<ExpertStatus> {
    Ok(current_status(&history(store, id).await?))
}

/// The status of every expert that has moved. One absent from the map has not,
/// and is active.
pub async fn statuses(store: &Store) -> Result<HashMap<ExpertId, ExpertStatus>> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    let rows: Vec<TransitionRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut latest: HashMap<ExpertId, (u32, ExpertStatus)> = HashMap::new();
    for row in rows {
        let entry = latest
            .entry(row.transition.expert)
            .or_insert((row.seq, row.transition.to));
        if row.seq >= entry.0 {
            *entry = (row.seq, row.transition.to);
        }
    }
    Ok(latest.into_iter().map(|(id, (_, s))| (id, s)).collect())
}

/// Every expert with its status.
pub async fn population(store: &Store) -> Result<Vec<(Expert, ExpertStatus)>> {
    let statuses = statuses(store).await?;
    Ok(expert::list(store)
        .await?
        .into_iter()
        .map(|e| {
            let status = statuses.get(&e.id).copied().unwrap_or(ExpertStatus::Active);
            (e, status)
        })
        .collect())
}

/// The experts the gate may route over: the active ones.
pub async fn routable(store: &Store) -> Result<Vec<Expert>> {
    in_states(store, ExpertStatus::is_routable).await
}

/// The experts that are served when named: the active and the dormant.
pub async fn servable(store: &Store) -> Result<Vec<Expert>> {
    in_states(store, ExpertStatus::is_servable).await
}

/// The learned router with every expert the gate may not route to masked
/// out. A router trained before a demotion still holds the demoted expert,
/// and masking it on load means no path routes to it, retrained since or not.
pub async fn load_router(store: &Store) -> Result<Option<LearnedRouter>> {
    let Some(learned) = router::load(store).await? else {
        return Ok(None);
    };
    let statuses = statuses(store).await?;
    Ok(Some(learned.masked(|id| {
        statuses
            .get(id)
            .copied()
            .unwrap_or(ExpertStatus::Active)
            .is_routable()
    })))
}

/// Whether `expert` trained under `verifier`: its card lists it.
pub fn trained_under(expert: &Expert, verifier: &VerifierId) -> bool {
    expert.capability_card["verifiers"]
        .as_array()
        .is_some_and(|ids| ids.iter().any(|v| v.as_str() == Some(verifier.as_str())))
}

/// Archive every expert that trained under `verifier` and is still active or
/// dormant (ADR-0022 S-4), and return them. Archived keeps the weights and
/// the tripwire, and a person can revive one; nothing here deletes.
pub async fn archive_trained_under(store: &Store, verifier: &VerifierId) -> Result<Vec<ExpertId>> {
    let mut archived = Vec::new();
    for (expert, status) in population(store).await? {
        let movable = matches!(status, ExpertStatus::Active | ExpertStatus::Dormant);
        if !movable || !trained_under(&expert, verifier) {
            continue;
        }
        let cause = TransitionCause::Quarantined {
            verifier: verifier.clone(),
        };
        transition(store, &expert.id, ExpertStatus::Archived, cause, None).await?;
        archived.push(expert.id);
    }
    Ok(archived)
}

async fn in_states(store: &Store, keep: fn(ExpertStatus) -> bool) -> Result<Vec<Expert>> {
    Ok(population(store)
        .await?
        .into_iter()
        .filter(|(_, s)| keep(*s))
        .map(|(e, _)| e)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::UserId;

    fn expert(id: &str, owner: Option<&str>) -> Expert {
        let mut v = vec![0.0f32; EMBED_DIM];
        v[0] = 1.0;
        Expert {
            id: ExpertId::new(id),
            name: id.to_string(),
            base_model: "base".into(),
            artifact_uri: "adapters/x.safetensors".into(),
            capability_card: serde_json::json!({}),
            capability_vec: Some(v),
            fitness: 1.0,
            frozen_at: Some(Utc::now()),
            generation: Generation::ZERO,
            owner: owner.map(UserId::new),
            compartment: None,
            placed_on: None,
            created_at: Utc::now(),
        }
    }

    fn operator() -> TransitionCause {
        TransitionCause::Operator { note: None }
    }

    async fn store_with(ids: &[&str]) -> Result<Store> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        for id in ids {
            expert::insert(&s, &expert(id, None)).await?;
        }
        Ok(s)
    }

    fn ids(experts: &[Expert]) -> Vec<&str> {
        let mut ids: Vec<&str> = experts.iter().map(|e| e.id.as_str()).collect();
        ids.sort_unstable();
        ids
    }

    #[tokio::test]
    async fn a_demoted_expert_leaves_routing_and_a_revived_one_returns() -> Result<()> {
        let s = store_with(&["expert:a", "expert:b"]).await?;
        let a = ExpertId::new("expert:a");
        assert_eq!(status_of(&s, &a).await?, ExpertStatus::Active);
        transition(&s, &a, ExpertStatus::Dormant, operator(), None).await?;
        assert_eq!(ids(&routable(&s).await?), ["expert:b"]);
        assert_eq!(ids(&servable(&s).await?), ["expert:a", "expert:b"]);
        transition(
            &s,
            &a,
            ExpertStatus::Archived,
            operator(),
            Some(Generation(3)),
        )
        .await?;
        assert_eq!(ids(&servable(&s).await?), ["expert:b"]);
        transition(&s, &a, ExpertStatus::Active, operator(), None).await?;
        assert_eq!(ids(&routable(&s).await?), ["expert:a", "expert:b"]);

        let moves = history(&s, &a).await?;
        let path: Vec<_> = moves.iter().map(|t| (t.from, t.to)).collect();
        assert_eq!(
            path,
            [
                (ExpertStatus::Active, ExpertStatus::Dormant),
                (ExpertStatus::Dormant, ExpertStatus::Archived),
                (ExpertStatus::Archived, ExpertStatus::Active),
            ]
        );
        assert_eq!(moves[1].generation, Some(Generation(3)));
        // The expert's own record was never touched.
        assert!(expert::get(&s, &a).await?.is_some_and(|e| e.is_frozen()));
        Ok(())
    }

    #[tokio::test]
    async fn a_forbidden_move_is_refused_and_leaves_no_row() -> Result<()> {
        let s = store_with(&["expert:a"]).await?;
        let a = ExpertId::new("expert:a");
        assert!(transition(&s, &a, ExpertStatus::Active, operator(), None)
            .await
            .is_err());
        transition(&s, &a, ExpertStatus::Dormant, operator(), None).await?;
        let refused = transition(&s, &a, ExpertStatus::Deleted, operator(), None)
            .await
            .unwrap_err();
        assert!(refused.is_rejection(), "{refused}");
        assert_eq!(history(&s, &a).await?.len(), 1);
        assert!(transition(
            &s,
            &ExpertId::new("expert:none"),
            ExpertStatus::Dormant,
            operator(),
            None
        )
        .await
        .is_err());
        Ok(())
    }

    #[tokio::test]
    async fn deletion_needs_an_active_expert_that_covers_it() -> Result<()> {
        let s = store_with(&["expert:a", "expert:b"]).await?;
        let (a, b) = (ExpertId::new("expert:a"), ExpertId::new("expert:b"));
        transition(&s, &a, ExpertStatus::Dormant, operator(), None).await?;
        let by = |of: &ExpertId| TransitionCause::Redundant { of: of.clone() };
        assert!(transition(&s, &a, ExpertStatus::Deleted, by(&a), None)
            .await
            .is_err());
        transition(&s, &b, ExpertStatus::Dormant, operator(), None).await?;
        assert!(
            transition(&s, &a, ExpertStatus::Deleted, by(&b), None)
                .await
                .is_err(),
            "a dormant expert covers nothing"
        );
        transition(&s, &b, ExpertStatus::Active, operator(), None).await?;
        transition(&s, &a, ExpertStatus::Deleted, by(&b), None).await?;
        assert_eq!(status_of(&s, &a).await?, ExpertStatus::Deleted);
        assert!(transition(&s, &a, ExpertStatus::Active, operator(), None)
            .await
            .is_err());
        Ok(())
    }

    /// A router trained before a demotion is masked on load: the demoted
    /// expert is neither routed to nor counted toward coverage.
    #[tokio::test]
    async fn a_router_is_loaded_with_its_demoted_experts_masked() -> Result<()> {
        let s = store_with(&["expert:a", "expert:b"]).await?;
        let centroid = |i: usize| {
            let mut c = vec![0.0f32; 2];
            c[i] = 1.0;
            c
        };
        router::save(
            &s,
            &LearnedRouter {
                weights: vec![1.0, 1.0],
                experts: vec![
                    antumbra_core::RouterExpert {
                        id: ExpertId::new("expert:a"),
                        centroid: centroid(0),
                    },
                    antumbra_core::RouterExpert {
                        id: ExpertId::new("expert:b"),
                        centroid: centroid(1),
                    },
                ],
                temperature: 0.1,
                floor: 0.5,
            },
        )
        .await?;
        let a = ExpertId::new("expert:a");
        transition(&s, &a, ExpertStatus::Dormant, operator(), None).await?;
        let loaded = load_router(&s).await?.expect("a router");
        let kept: Vec<&str> = loaded.experts.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(kept, ["expert:b"]);
        assert!(
            !loaded.covers(&[1.0, 0.0]),
            "only the masked expert covered it"
        );
        router::clear(&s).await?;
        router::clear(&s).await?;
        assert!(load_router(&s).await?.is_none());
        Ok(())
    }

    /// The moves of a private expert keep its owner, so the permissions that
    /// hide the expert from other users hide its moves too.
    #[tokio::test]
    async fn a_move_carries_its_experts_owner() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        expert::insert(&s, &expert("expert:mine", Some("user:ada"))).await?;
        let id = ExpertId::new("expert:mine");
        transition(&s, &id, ExpertStatus::Dormant, operator(), None).await?;
        let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
        let rows: Vec<TransitionRow> = query_records(s.client(), &query).await.map_err(map)?;
        assert_eq!(rows[0].owner.as_deref(), Some("user:ada"));
        assert_eq!(rows[0].seq, 0);
        Ok(())
    }
}
