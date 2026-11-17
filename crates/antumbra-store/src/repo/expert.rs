//! Expert repository (the umbra population of small frozen experts), with routing-as-retrieval (KNN routing).
//!
//! Built entirely on surql-rs: [`Store::put`] for writes, the `Query` builder
//! + `crud` for typed reads, and `Query::vector_search_indexed` for KNN
//! through `expert_cap_hnsw`. No hand-authored SurrealQL.
//!
//! A row is what two stores can hold a copy of and keep in step:
//!
//! - **One record per expert.** The record id is the expert's id, so writing
//!   an expert that exists replaces it in place.
//! - **A version.** `updated_at`, set here on every write and later than the
//!   one before it.
//! - **A tombstone.** A deleted expert keeps its record, with `deleted_at`
//!   set and its capability vector cleared, until [`purge`] removes it. No
//!   read here returns one.
//!
//! Rows written before that have an engine-assigned record id, with the
//! expert's id only in `key`. [`adopt_ids`] moves them when a store's schema
//! is ensured.

use chrono::{DateTime, Duration, Utc};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{delete_records, query_records, upsert_record};
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::{Expert, ExpertId, Result};

use crate::dto::{parse_dt, ExpertRow};
use crate::error::map;
use crate::knn::search_effort;
use crate::repo::sync;
use crate::store::Store;

const TABLE: &str = crate::tables::EXPERT;

/// What [`adopt_ids`] appends to the `key` of a row it is moving, so the
/// unique index on `key` admits the record that takes its place. No expert's
/// id holds a control character.
const MOVING: &str = "\u{1f}moving";

/// The row for `id` as it is stored, tombstone or not.
async fn stored(store: &Store, id: &str) -> Result<Option<ExpertRow>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("key", id));
    let rows: Vec<ExpertRow> = query_records(store.client(), &query).await.map_err(map)?;
    Ok(rows.into_iter().next())
}

/// The version a write made now carries: the time, moved past `before` when
/// the clock has not, so a row's versions only ever go up and two stores
/// never hold different rows under one version.
fn version_after(before: Option<&str>) -> String {
    let now = Utc::now();
    let later = before
        .and_then(|b| parse_dt(b).ok())
        .filter(|b| *b >= now)
        .map(|b| b + Duration::microseconds(1));
    later.unwrap_or(now).to_rfc3339()
}

/// Write `row` as its expert's one record, as a version later than the one
/// stored.
async fn write(store: &Store, mut row: ExpertRow) -> Result<()> {
    let before = stored(store, &row.key).await?;
    row.updated_at = Some(version_after(
        before.as_ref().and_then(|b| b.updated_at.as_deref()),
    ));
    let key = row.key.clone();
    store.put(TABLE, &key, &row).await
}

/// Write an expert: a new one, or the one of that id replaced in place, a
/// deleted one included.
pub async fn insert(store: &Store, expert: &Expert) -> Result<()> {
    write(store, ExpertRow::from_domain(expert)).await
}

/// Delete an expert by domain id (no-op if absent). Its record stays as a
/// tombstone: `deleted_at` set, the capability vector cleared so it leaves
/// the routing index, and everything else as it was, so a store holding a
/// copy learns it was deleted and by whom it was owned.
pub async fn delete(store: &Store, id: &ExpertId) -> Result<()> {
    let Some(mut row) = stored(store, id.as_str()).await? else {
        return Ok(());
    };
    if row.deleted_at.is_some() {
        return Ok(());
    }
    row.deleted_at = Some(Utc::now().to_rfc3339());
    row.capability_vec = None;
    write(store, row).await
}

/// Remove the tombstones of experts deleted before `older_than`: how many.
/// A live expert is never removed here.
pub async fn purge(store: &Store, older_than: DateTime<Utc>) -> Result<usize> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    let rows: Vec<ExpertRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut purged = 0;
    for row in rows {
        let Some(deleted) = row.deleted_at.as_deref() else {
            continue;
        };
        if parse_dt(deleted).map(|t| t < older_than).unwrap_or(false) {
            delete_records(store.client(), TABLE, Some(&eq("key", row.key.as_str())))
                .await
                .map_err(map)?;
            purged += 1;
        }
    }
    Ok(purged)
}

/// Freeze or thaw an expert in place by domain id: `frozen_at` set (freeze)
/// or cleared (thaw) on the record it has. No-op if the expert is absent.
pub async fn set_frozen(
    store: &Store,
    id: &ExpertId,
    frozen_at: Option<DateTime<Utc>>,
) -> Result<()> {
    if let Some(mut expert) = get(store, id).await? {
        expert.frozen_at = frozen_at;
        insert(store, &expert).await?;
    }
    Ok(())
}

/// Fetch one expert by id. A deleted one is not there.
pub async fn get(store: &Store, id: &ExpertId) -> Result<Option<Expert>> {
    stored(store, id.as_str())
        .await?
        .filter(|row| row.deleted_at.is_none())
        .map(ExpertRow::into_domain)
        .transpose()
}

/// List the whole population, the deleted left out.
pub async fn list(store: &Store) -> Result<Vec<Expert>> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    let rows: Vec<ExpertRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .filter(|row| row.deleted_at.is_none())
        .map(ExpertRow::into_domain)
        .collect()
}

/// Routing-as-retrieval: the `k` experts whose capability vector is
/// nearest to `query`, through `expert_cap_hnsw`.
///
/// The population is global — no tenant pin, no residual filter — so
/// the index returns exactly the `k` the caller asked for and the
/// metric is the index's own cosine. A deleted expert holds no capability
/// vector, so the index does not return it.
pub async fn knn_by_capability(store: &Store, query: &[f32], k: usize) -> Result<Vec<Expert>> {
    let vector: Vec<f64> = query.iter().map(|&x| f64::from(x)).collect();
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .vector_search_indexed("capability_vec", vector, k as i64, search_effort(k))
        .map_err(map)?;
    let rows: Vec<ExpertRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter()
        .filter(|row| row.deleted_at.is_none())
        .map(ExpertRow::into_domain)
        .collect()
}

/// Whether the record id `at` is the one `key` names: the id's part after
/// the table, with the brackets or backticks the engine writes around one
/// that is not a plain word taken off.
fn is_home(at: &str, key: &str) -> bool {
    at.split_once(':').is_some_and(|(_, id)| {
        let id = id.strip_prefix('⟨').unwrap_or(id);
        let id = id.strip_suffix('⟩').unwrap_or(id);
        id.trim_matches('`') == key
    })
}

/// Move every expert still under an engine-assigned record id to the record
/// its own id names, keeping every field: how many were moved. Run when a
/// store's schema is ensured, as the owner. A store never holds both forms
/// after it, and running it again moves nothing.
///
/// The unique index on `key` forbids writing the new record while the old
/// one holds the key, and the two writes cannot be made one. So a row is
/// moved in three steps, and an expert has a whole record at every one of
/// them: the old record's key is marked as moving, the row is written under
/// its own id, and the old record is removed. A move cut short is finished
/// here the next time: a marked record whose expert has no record of its own
/// is moved, and one whose expert has is removed.
pub(crate) async fn adopt_ids(store: &Store) -> Result<usize> {
    let rows = sync::list_rows(store, TABLE).await?;
    let mut moved = 0;
    for mut row in rows {
        let Some(at) = sync::row_id(&row) else {
            continue;
        };
        let Some(key) = row.get("key").and_then(Value::as_str).map(str::to_string) else {
            continue;
        };
        let moving = key.strip_suffix(MOVING);
        let id = moving.unwrap_or(&key).to_string();
        if moving.is_none() && is_home(&at, &id) {
            continue;
        }
        let Value::Object(fields) = &mut row else {
            continue;
        };
        fields.remove("id");
        let marked = format!("{id}{MOVING}");
        if moving.is_none() {
            // Out of the key's way, still a whole row.
            fields.insert("key".into(), Value::String(marked.clone()));
            let old = RecordID::<()>::parse(&at).map_err(map)?;
            upsert_record(store.client(), &old, row.clone())
                .await
                .map_err(map)?;
        }
        if stored(store, &id).await?.is_none() {
            let mut home: ExpertRow = serde_json::from_value(row)?;
            home.key = id.clone();
            // The first version of a row that had none: when it was made.
            if home.updated_at.is_none() {
                home.updated_at = Some(home.created_at.clone());
            }
            store.put(TABLE, &id, &home).await?;
            moved += 1;
        }
        delete_records(store.client(), TABLE, Some(&eq("key", marked.as_str())))
            .await
            .map_err(map)?;
    }
    Ok(moved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::{AntumbraError, Expert, Generation};
    use chrono::Utc;
    use surql::query::crud::create_record;

    fn expert(id: &str, vec: Vec<f32>) -> Expert {
        Expert {
            id: ExpertId::new(id),
            name: id.to_string(),
            base_model: "base".into(),
            artifact_uri: "adapters/x.safetensors".into(),
            capability_card: serde_json::json!({}),
            capability_vec: Some(vec),
            fitness: 1.0,
            frozen_at: Some(Utc::now()),
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: Utc::now(),
        }
    }

    fn unit() -> Vec<f32> {
        let mut v = vec![0.0f32; EMBED_DIM];
        v[0] = 1.0;
        v
    }

    /// Every record of the table as it is stored: its record id and its row.
    async fn records(s: &Store) -> antumbra_core::Result<Vec<(String, ExpertRow)>> {
        let mut found = Vec::new();
        for row in sync::list_rows(s, TABLE).await? {
            let Some(at) = sync::row_id(&row) else {
                return Err(AntumbraError::other("a stored row has no record id"));
            };
            found.push((at, serde_json::from_value(row)?));
        }
        Ok(found)
    }

    /// The one record of `id`, tombstone or not.
    async fn record(s: &Store, id: &str) -> antumbra_core::Result<(String, ExpertRow)> {
        let mut of: Vec<_> = records(s)
            .await?
            .into_iter()
            .filter(|(_, row)| row.key == id)
            .collect();
        match (of.pop(), of.is_empty()) {
            (Some(one), true) => Ok(one),
            (None, _) => Err(AntumbraError::other(format!("{id} has no record"))),
            (Some(_), false) => Err(AntumbraError::other(format!("{id} has two records"))),
        }
    }

    /// The row's version, which every row written here has.
    fn version(row: &ExpertRow) -> antumbra_core::Result<DateTime<Utc>> {
        match row.updated_at.as_deref() {
            Some(at) => parse_dt(at),
            None => Err(AntumbraError::other(format!("{} has no version", row.key))),
        }
    }

    /// Write `e` as a store did before records were named for their expert:
    /// under an id the engine assigns, with no version.
    async fn insert_as_before(s: &Store, e: &Expert) -> antumbra_core::Result<()> {
        let data = serde_json::to_value(ExpertRow::from_domain(e))?;
        create_record(s.client(), TABLE, data).await.map_err(map)?;
        Ok(())
    }

    /// Placement has to survive the store, or a node reads every expert as
    /// servable and routes to adapters it does not hold.
    #[tokio::test]
    async fn placement_survives_the_round_trip() -> antumbra_core::Result<()> {
        let s = crate::testing::mem_store().await?;
        let v = unit();
        let mut placed = expert("expert:placed", v.clone());
        placed.placed_on = Some("the-rig".into());
        insert(&s, &placed).await?;
        insert(&s, &expert("expert:unplaced", v)).await?;

        let found = list(&s).await?;
        let placed_on = |id: &str| {
            found
                .iter()
                .find(|e| e.id.as_str() == id)
                .and_then(|e| e.placed_on.clone())
        };
        assert_eq!(placed_on("expert:placed"), Some("the-rig".to_string()));
        assert_eq!(
            placed_on("expert:unplaced"),
            None,
            "an unplaced expert stays unplaced rather than acquiring a host"
        );
        Ok(())
    }

    #[tokio::test]
    async fn insert_get_list_knn_delete() -> antumbra_core::Result<()> {
        let s = crate::testing::mem_store().await?;
        let v = unit();
        insert(&s, &expert("expert:a", v.clone())).await?;

        let Some(a) = get(&s, &ExpertId::new("expert:a")).await? else {
            return Err(AntumbraError::other("expert:a did not read back"));
        };
        assert_eq!(a.name, "expert:a");
        assert!(get(&s, &ExpertId::new("expert:missing")).await?.is_none());
        assert_eq!(list(&s).await?.len(), 1);

        let near = knn_by_capability(&s, &v, 1).await?;
        assert_eq!(near.len(), 1);
        assert_eq!(near[0].id.as_str(), "expert:a");

        delete(&s, &ExpertId::new("expert:a")).await?;
        assert!(list(&s).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn set_frozen_toggles_in_place_and_ignores_the_absent() -> antumbra_core::Result<()> {
        let s = crate::testing::mem_store().await?;
        // Seed an unfrozen expert (override the helper's frozen default).
        let mut e = expert("expert:f", unit());
        e.frozen_at = None;
        insert(&s, &e).await?;
        let id = ExpertId::new("expert:f");
        let (at, _) = record(&s, "expert:f").await?;

        // Freeze in place: same record (one row), now frozen.
        set_frozen(&s, &id, Some(Utc::now())).await?;
        let Some(frozen) = get(&s, &id).await? else {
            return Err(AntumbraError::other("expert:f went missing on freeze"));
        };
        assert!(frozen.is_frozen());
        assert_eq!(list(&s).await?.len(), 1, "replaced, not duplicated");
        assert_eq!(record(&s, "expert:f").await?.0, at, "the same record");

        // Thaw in place.
        set_frozen(&s, &id, None).await?;
        let Some(thawed) = get(&s, &id).await? else {
            return Err(AntumbraError::other("expert:f went missing on thaw"));
        };
        assert!(!thawed.is_frozen());

        // Absent id is a no-op, not an error.
        set_frozen(&s, &ExpertId::new("expert:ghost"), Some(Utc::now())).await?;
        assert_eq!(list(&s).await?.len(), 1);
        Ok(())
    }

    /// The regression: an expert written again was another record, under
    /// another id, with nothing to say which was the later.
    #[tokio::test]
    async fn an_expert_written_again_is_the_same_record_at_a_later_version(
    ) -> antumbra_core::Result<()> {
        let s = crate::testing::mem_store().await?;
        // An id as a standing expert's is: colons, and a repository's slashes.
        let id = "expert:ws:a:user:b:behavior:github.com/owner/repo";
        let mut e = expert(id, unit());
        insert(&s, &e).await?;
        let (at, first) = record(&s, id).await?;
        assert!(is_home(&at, id), "{at} is named for the expert");

        e.artifact_uri = "adapters/retrained.safetensors".into();
        insert(&s, &e).await?;

        let (again, second) = record(&s, id).await?;
        assert_eq!(again, at);
        assert_eq!(second.artifact_uri, "adapters/retrained.safetensors");
        assert!(version(&second)? > version(&first)?);
        assert_eq!(records(&s).await?.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn a_deleted_expert_leaves_a_tombstone_no_read_returns() -> antumbra_core::Result<()> {
        let s = crate::testing::mem_store().await?;
        let v = unit();
        let id = ExpertId::new("expert:a");
        let mut e = expert("expert:a", v.clone());
        e.owner = Some(antumbra_core::UserId::new("user:b"));
        insert(&s, &e).await?;
        let (at, live) = record(&s, "expert:a").await?;

        delete(&s, &id).await?;

        assert!(get(&s, &id).await?.is_none());
        assert!(list(&s).await?.is_empty());
        assert!(knn_by_capability(&s, &v, 1).await?.is_empty());
        let (still, tombstone) = record(&s, "expert:a").await?;
        assert_eq!(still, at, "the record it had");
        assert!(tombstone.deleted_at.is_some());
        assert_eq!(tombstone.capability_vec, None, "out of the routing index");
        assert_eq!(
            tombstone.owner.as_deref(),
            Some("user:b"),
            "still whose it was"
        );
        assert!(version(&tombstone)? > version(&live)?);

        // Deleting what is deleted, or what was never there, changes nothing.
        delete(&s, &id).await?;
        delete(&s, &ExpertId::new("expert:ghost")).await?;
        assert_eq!(
            version(&record(&s, "expert:a").await?.1)?,
            version(&tombstone)?
        );

        // Written again, it is back, in the record it had, at a later version.
        insert(&s, &e).await?;
        let (back, row) = record(&s, "expert:a").await?;
        assert_eq!(back, at);
        assert_eq!(row.deleted_at, None);
        assert!(version(&row)? > version(&tombstone)?);
        assert_eq!(list(&s).await?.len(), 1);
        assert_eq!(knn_by_capability(&s, &v, 1).await?.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn a_delete_and_an_insert_one_after_the_other_never_share_a_version(
    ) -> antumbra_core::Result<()> {
        let s = crate::testing::mem_store().await?;
        let id = ExpertId::new("expert:a");
        let e = expert("expert:a", unit());
        insert(&s, &e).await?;
        // As a retraining replaces an expert, many times over: each write is
        // later than the one before it, however fast they come.
        let mut last = version(&record(&s, "expert:a").await?.1)?;
        for _ in 0..20 {
            delete(&s, &id).await?;
            let deleted = version(&record(&s, "expert:a").await?.1)?;
            insert(&s, &e).await?;
            let written = version(&record(&s, "expert:a").await?.1)?;
            assert!(deleted > last && written > deleted);
            last = written;
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_tombstone_is_purged_past_its_grace_and_a_live_expert_never(
    ) -> antumbra_core::Result<()> {
        let s = crate::testing::mem_store().await?;
        insert(&s, &expert("expert:kept", unit())).await?;
        insert(&s, &expert("expert:gone", unit())).await?;
        delete(&s, &ExpertId::new("expert:gone")).await?;

        // Within its grace: left.
        assert_eq!(purge(&s, Utc::now() - Duration::hours(1)).await?, 0);
        assert_eq!(records(&s).await?.len(), 2);

        // Past it: removed, and the live expert is not.
        assert_eq!(purge(&s, Utc::now() + Duration::hours(1)).await?, 1);
        let left = records(&s).await?;
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].1.key, "expert:kept");
        Ok(())
    }

    #[tokio::test]
    async fn experts_under_engine_assigned_ids_are_moved_to_their_own() -> antumbra_core::Result<()>
    {
        let s = crate::testing::mem_store().await?;
        let v = unit();
        let mut standing = expert("expert:ws:a:user:b:behavior:owner/repo", v.clone());
        standing.owner = Some(antumbra_core::UserId::new("user:b"));
        standing.placed_on = Some("the-rig".into());
        standing.capability_card = serde_json::json!({ "standing": true, "scope": "owner/repo" });
        let shared = expert("expert:shared", v.clone());
        insert_as_before(&s, &standing).await?;
        insert_as_before(&s, &shared).await?;
        // And one written since: already where it belongs.
        insert(&s, &expert("expert:new", v.clone())).await?;
        let (new_at, new_row) = record(&s, "expert:new").await?;
        let (old_at, _) = record(&s, standing.id.as_str()).await?;
        assert!(!is_home(&old_at, standing.id.as_str()));

        assert_eq!(adopt_ids(&s).await?, 2);

        assert_eq!(records(&s).await?.len(), 3, "the same three experts");
        for before in [&standing, &shared] {
            let (at, row) = record(&s, before.id.as_str()).await?;
            assert!(is_home(&at, before.id.as_str()), "{at}");
            // Every field as it was, and when it was made as its first version.
            let mut want = ExpertRow::from_domain(before);
            want.updated_at = Some(want.created_at.clone());
            assert_eq!(serde_json::to_value(&row)?, serde_json::to_value(&want)?);
        }
        let (at, row) = record(&s, "expert:new").await?;
        assert_eq!(at, new_at);
        assert_eq!(version(&row)?, version(&new_row)?, "left as it was");
        assert_eq!(list(&s).await?.len(), 3);
        assert_eq!(knn_by_capability(&s, &v, 3).await?.len(), 3);

        // Again: nothing to move.
        assert_eq!(adopt_ids(&s).await?, 0);
        assert_eq!(records(&s).await?.len(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn a_move_cut_short_is_finished_the_next_time() -> antumbra_core::Result<()> {
        let s = crate::testing::mem_store().await?;
        // Cut short after its first step: the old record marked as moving,
        // and no record of the expert's own yet.
        let mut marked = expert("expert:a", unit());
        marked.id = ExpertId::new(format!("expert:a{MOVING}"));
        insert_as_before(&s, &marked).await?;
        // Cut short after its second: the expert's own record written, and
        // the marked one not yet removed.
        let mut left = expert("expert:b", unit());
        left.id = ExpertId::new(format!("expert:b{MOVING}"));
        insert_as_before(&s, &left).await?;
        insert(&s, &expert("expert:b", unit())).await?;

        assert_eq!(adopt_ids(&s).await?, 1, "one was still to be written");

        let all = records(&s).await?;
        assert_eq!(all.len(), 2, "and no marked record is left: {all:?}");
        for id in ["expert:a", "expert:b"] {
            let (at, row) = record(&s, id).await?;
            assert!(is_home(&at, id), "{at}");
            assert_eq!(row.name, id);
        }
        assert_eq!(list(&s).await?.len(), 2);
        Ok(())
    }

    #[test]
    fn a_record_id_is_the_expert_s_own_with_or_without_the_engine_s_brackets() {
        let key = "expert:ws:a:user:b:behavior:owner/repo";
        assert!(is_home(&format!("expert:⟨{key}⟩"), key));
        assert!(is_home(&format!("expert:`{key}`"), key));
        assert!(is_home("expert:plain", "plain"));
        assert!(!is_home("expert:8k2jq0xw1v5n", key));
        assert!(!is_home("expert", key));
    }
}
