//! Expert lifecycle repository: the append-only history of every
//! expert's status changes, and the population as the gate may see it.
//!
//! A move is one row, never rewritten, numbered per expert. The number orders
//! an expert's history. A move's record is named for its expert and its
//! number, so the same move is the same record on every store that holds it,
//! and two writers racing to make it are one success and one refusal rather
//! than two forks: the second finds the record there. Rows written before
//! that have an engine-assigned record id; [`adopt_ids`] moves them when a
//! store's schema is ensured. Each row carries its expert's owner under the expert table's own
//! permissions, so a session sees exactly the moves of the experts it can see:
//! a session that could read an expert but not its moves would take a demoted
//! expert for an active one.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use serde_json::Value;
use surql::query::builder::Query;
use surql::query::crud::{create_record, delete_record, get_record, query_records, upsert_record};
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::{
    current_status, AntumbraError, Expert, ExpertId, ExpertStatus, ExpertTransition, Generation,
    LearnedRouter, Result, TransitionCause, VerifierId,
};

use crate::error::map;
use crate::repo::{expert, router, sync};
use crate::store::Store;

const TABLE: &str = crate::tables::EXPERT_TRANSITION;

/// What [`adopt_ids`] appends to the `expert` of a row it is moving, so the
/// unique index on an expert's numbers admits the record that takes its
/// place. No expert's id holds a control character.
const MOVING: &str = "\u{1f}moving";

/// The key of a move's record: its expert's id, then its number in that
/// expert's history at a fixed width, so what follows the last colon is
/// always the number, whatever the expert's id holds.
fn record_key(expert: &str, seq: u32) -> String {
    format!("{expert}:{seq:010}")
}

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
    let seq = u32::try_from(past.len()).unwrap_or(u32::MAX);
    let row = TransitionRow {
        transition: transition.clone(),
        seq,
        owner: subject.owner.as_ref().map(|u| u.as_str().to_string()),
    };
    // Created, never put: a second writer making the same move finds the
    // record there and is refused, where a put would replace the first
    // writer's move with its own.
    let at = RecordID::<()>::new(TABLE, record_key(id.as_str(), seq).as_str()).map_err(map)?;
    create_record(store.client(), &at.to_string(), serde_json::to_value(row)?)
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
/// dormant, and return them. Archived keeps the weights and
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

/// A raw row's expert and number, when it has both.
fn named(row: &Value) -> Option<(String, u32)> {
    let expert = row.get("expert")?.as_str()?.to_string();
    let seq = u32::try_from(row.get("seq")?.as_u64()?).ok()?;
    Some((expert, seq))
}

/// Whether a raw `expert_transition` row, as [`sync::list_rows`] returns it,
/// sits in the record its expert and number name. Only such a row is the
/// same record on every store that holds the move, so only such a row can be
/// copied between two: one still under an engine-assigned id, or one
/// [`adopt_ids`] is partway through moving, would be refused by the unique
/// index of a store that has the move, or land beside it.
pub fn is_own_record(row: &Value) -> bool {
    match (sync::row_id(row), named(row)) {
        (Some(at), Some((expert, seq))) => expert::is_home(&at, &record_key(&expert, seq)),
        _ => false,
    }
}

/// Move every move still under an engine-assigned record id to the record its
/// expert and number name, keeping every field: how many were moved. Run when
/// a store's schema is ensured, as the owner. Running it again moves nothing.
///
/// The unique index on an expert's numbers forbids writing the new record
/// while the old one holds the number, so a row is moved in three steps: the
/// old record's expert is marked as moving, the row is written under its own
/// id, and the old record is removed. Between the first and the second the
/// expert's history lacks that move, which is why this runs before a store
/// is served. A move cut short is finished here the next time: a marked
/// record whose move has no record of its own is written, and the marked
/// one removed either way.
pub(crate) async fn adopt_ids(store: &Store) -> Result<usize> {
    let rows = sync::list_rows(store, TABLE).await?;
    let mut moved = 0;
    for mut row in rows {
        let (Some(at), Some((holder, seq))) = (sync::row_id(&row), named(&row)) else {
            continue;
        };
        let moving = holder.strip_suffix(MOVING);
        let expert = moving.unwrap_or(&holder).to_string();
        let key = record_key(&expert, seq);
        if moving.is_none() && expert::is_home(&at, &key) {
            continue;
        }
        let Value::Object(fields) = &mut row else {
            continue;
        };
        fields.remove("id");
        let old = RecordID::<()>::parse(&at).map_err(map)?;
        if moving.is_none() {
            // Out of the number's way, still a whole row.
            fields.insert("expert".into(), Value::String(format!("{expert}{MOVING}")));
            upsert_record(store.client(), &old, Value::Object(fields.clone()))
                .await
                .map_err(map)?;
        }
        let home = RecordID::<()>::new(TABLE, key.as_str()).map_err(map)?;
        if get_record(store.client(), &home)
            .await
            .map_err(map)?
            .is_none()
        {
            fields.insert("expert".into(), Value::String(expert));
            upsert_record(store.client(), &home, Value::Object(fields.clone()))
                .await
                .map_err(map)?;
            moved += 1;
        }
        delete_record(store.client(), &old).await.map_err(map)?;
    }
    Ok(moved)
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
    use crate::testing::mem_store;
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
        let s = mem_store().await?;
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
        let Err(refused) = transition(&s, &a, ExpertStatus::Deleted, operator(), None).await else {
            return Err(antumbra_core::AntumbraError::other(
                "a dormant expert was deleted with nothing covering it",
            ));
        };
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
        let Some(loaded) = load_router(&s).await? else {
            return Err(antumbra_core::AntumbraError::other(
                "the saved router did not load",
            ));
        };
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
        let s = mem_store().await?;
        expert::insert(&s, &expert("expert:mine", Some("user:ada"))).await?;
        let id = ExpertId::new("expert:mine");
        transition(&s, &id, ExpertStatus::Dormant, operator(), None).await?;
        let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
        let rows: Vec<TransitionRow> = query_records(s.client(), &query).await.map_err(map)?;
        assert_eq!(rows[0].owner.as_deref(), Some("user:ada"));
        assert_eq!(rows[0].seq, 0);
        Ok(())
    }

    /// Every record of the table as it is stored: its record id and its row.
    async fn records(s: &Store) -> Result<Vec<(String, Value)>> {
        let mut found = Vec::new();
        for row in sync::list_rows(s, TABLE).await? {
            let Some(at) = sync::row_id(&row) else {
                return Err(AntumbraError::other("a stored row has no record id"));
            };
            found.push((at, row));
        }
        Ok(found)
    }

    /// The row of `expert`'s move number `seq`, from `from` to `to`.
    fn row_of(expert: &str, seq: u32, from: ExpertStatus, to: ExpertStatus) -> Result<Value> {
        Ok(serde_json::to_value(TransitionRow {
            transition: ExpertTransition {
                expert: ExpertId::new(expert),
                from,
                to,
                cause: operator(),
                generation: None,
                at: Utc::now(),
            },
            seq,
            owner: None,
        })?)
    }

    /// Record a move as a store did before a move's record was named for it:
    /// under an id the engine assigns.
    async fn move_as_before(
        s: &Store,
        expert: &str,
        seq: u32,
        from: ExpertStatus,
        to: ExpertStatus,
    ) -> Result<()> {
        create_record(s.client(), TABLE, row_of(expert, seq, from, to)?)
            .await
            .map_err(map)?;
        Ok(())
    }

    #[tokio::test]
    async fn a_move_is_the_record_its_expert_and_number_name() -> Result<()> {
        let id = "expert:ws:a:user:b:behavior:owner/repo";
        let s = store_with(&[id]).await?;
        let subject = ExpertId::new(id);
        transition(&s, &subject, ExpertStatus::Dormant, operator(), None).await?;
        transition(&s, &subject, ExpertStatus::Archived, operator(), None).await?;

        let stored = records(&s).await?;
        assert_eq!(stored.len(), 2);
        for (seq, key) in [
            (0u32, format!("{id}:0000000000")),
            (1, format!("{id}:0000000001")),
        ] {
            assert_eq!(record_key(id, seq), key);
            assert!(
                stored.iter().any(|(at, _)| expert::is_home(at, &key)),
                "no record named {key}: {stored:?}"
            );
        }
        assert!(stored.iter().all(|(_, row)| is_own_record(row)));
        Ok(())
    }

    /// Two writers that both read the same history make the same move. The
    /// second finds the first's record there and is refused, and the history
    /// keeps the first's.
    #[tokio::test]
    async fn the_same_move_made_twice_is_refused() -> Result<()> {
        let s = store_with(&["expert:a"]).await?;
        let a = ExpertId::new("expert:a");
        transition(&s, &a, ExpertStatus::Dormant, operator(), None).await?;

        // The second writer's move, under the number the first one took.
        let at = RecordID::<()>::new(TABLE, record_key("expert:a", 0).as_str()).map_err(map)?;
        let again = row_of("expert:a", 0, ExpertStatus::Active, ExpertStatus::Archived)?;
        let second = create_record(s.client(), &at.to_string(), again).await;
        assert!(second.is_err(), "one move was written twice");

        let moves = history(&s, &a).await?;
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].to, ExpertStatus::Dormant, "the first writer's");
        Ok(())
    }

    #[tokio::test]
    async fn moves_under_engine_assigned_ids_are_moved_to_their_own() -> Result<()> {
        let old = "expert:ws:a:user:b:behavior:owner/repo";
        let s = store_with(&[old, "expert:new"]).await?;
        move_as_before(&s, old, 0, ExpertStatus::Active, ExpertStatus::Dormant).await?;
        move_as_before(&s, old, 1, ExpertStatus::Dormant, ExpertStatus::Archived).await?;
        // And one made since: already where it belongs.
        let new = ExpertId::new("expert:new");
        transition(&s, &new, ExpertStatus::Dormant, operator(), None).await?;
        let before = history(&s, &ExpertId::new(old)).await?;
        assert!(
            records(&s)
                .await?
                .iter()
                .filter(|(_, row)| is_own_record(row))
                .count()
                == 1
        );

        assert_eq!(adopt_ids(&s).await?, 2);

        let stored = records(&s).await?;
        assert_eq!(stored.len(), 3, "the same three moves");
        assert!(
            stored.iter().all(|(_, row)| is_own_record(row)),
            "{stored:?}"
        );
        // Every expert's history and status as they were.
        let after = history(&s, &ExpertId::new(old)).await?;
        assert_eq!(
            serde_json::to_value(&after)?,
            serde_json::to_value(&before)?
        );
        assert_eq!(
            status_of(&s, &ExpertId::new(old)).await?,
            ExpertStatus::Archived
        );
        assert_eq!(status_of(&s, &new).await?, ExpertStatus::Dormant);
        // And the next move takes the next number.
        transition(
            &s,
            &ExpertId::new(old),
            ExpertStatus::Active,
            operator(),
            None,
        )
        .await?;
        assert_eq!(history(&s, &ExpertId::new(old)).await?.len(), 3);

        // Again: nothing to move.
        assert_eq!(adopt_ids(&s).await?, 0);
        assert_eq!(records(&s).await?.len(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn a_move_of_a_move_cut_short_is_finished_the_next_time() -> Result<()> {
        let s = store_with(&["expert:a", "expert:b"]).await?;
        // Cut short after its first step: the old record marked as moving,
        // and no record of the move's own yet.
        let marked = format!("expert:a{MOVING}");
        move_as_before(&s, &marked, 0, ExpertStatus::Active, ExpertStatus::Dormant).await?;
        // Cut short after its second: the move's own record written, and the
        // marked one not yet removed.
        let left = format!("expert:b{MOVING}");
        move_as_before(&s, &left, 0, ExpertStatus::Active, ExpertStatus::Dormant).await?;
        transition(
            &s,
            &ExpertId::new("expert:b"),
            ExpertStatus::Dormant,
            operator(),
            None,
        )
        .await?;
        assert_eq!(
            status_of(&s, &ExpertId::new("expert:a")).await?,
            ExpertStatus::Active,
            "its one move is out of its history until the move is finished"
        );

        assert_eq!(adopt_ids(&s).await?, 1, "one was still to be written");

        let stored = records(&s).await?;
        assert_eq!(stored.len(), 2, "and no marked record is left: {stored:?}");
        assert!(stored.iter().all(|(_, row)| is_own_record(row)));
        for id in ["expert:a", "expert:b"] {
            assert_eq!(
                status_of(&s, &ExpertId::new(id)).await?,
                ExpertStatus::Dormant,
                "{id}"
            );
        }
        Ok(())
    }

    /// What a pass between two stores asks of a move before it copies it.
    #[tokio::test]
    async fn only_a_move_in_its_own_record_is_one_to_copy() -> Result<()> {
        let s = store_with(&["expert:a", "expert:old"]).await?;
        transition(
            &s,
            &ExpertId::new("expert:a"),
            ExpertStatus::Dormant,
            operator(),
            None,
        )
        .await?;
        move_as_before(
            &s,
            "expert:old",
            0,
            ExpertStatus::Active,
            ExpertStatus::Dormant,
        )
        .await?;
        let marked = format!("expert:cut{MOVING}");
        move_as_before(&s, &marked, 0, ExpertStatus::Active, ExpertStatus::Dormant).await?;

        let copied: Vec<String> = records(&s)
            .await?
            .into_iter()
            .filter(|(_, row)| is_own_record(row))
            .filter_map(|(_, row)| named(&row).map(|(expert, _)| expert))
            .collect();
        assert_eq!(copied, vec!["expert:a".to_string()]);

        assert!(!is_own_record(
            &serde_json::json!({ "expert": "expert:a", "seq": 0 })
        ));
        assert!(!is_own_record(
            &serde_json::json!({ "id": "expert_transition:x", "seq": 0 })
        ));
        assert!(!is_own_record(
            &serde_json::json!({ "id": "expert_transition:x", "expert": "x" })
        ));
        Ok(())
    }
}
