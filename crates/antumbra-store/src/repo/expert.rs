//! Expert repository (the umbra population). ADR-0001 / ADR-0005 (KNN routing).
//!
//! Built entirely on surql-rs: `crud::create_record` for writes, the `Query`
//! builder + `crud` for typed reads, and `Query::vector_search` for KNN. No
//! hand-authored SurrealQL.

use chrono::{DateTime, Utc};

use surql::query::builder::Query;
use surql::query::crud::{create_record, delete_records, first, query_records};
use surql::query::helpers::VectorDistanceType;
use surql::types::operators::eq;

use antumbra_core::{Expert, ExpertId, Result};

use crate::dto::ExpertRow;
use crate::error::map;
use crate::store::Store;

const TABLE: &str = "expert";

/// Insert an expert. The domain id lives in the `key` column (unique index);
/// SurrealDB assigns the record id.
pub async fn insert(store: &Store, expert: &Expert) -> Result<()> {
    let data = serde_json::to_value(ExpertRow::from_domain(expert))?;
    create_record(store.client(), TABLE, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Remove an expert by domain id (no-op if absent). Used to supersede an
/// expert when a run re-trains under the same name, so the population does not
/// accumulate stale duplicates.
pub async fn delete(store: &Store, id: &ExpertId) -> Result<()> {
    delete_records(store.client(), TABLE, Some(&eq("key", id.as_str())))
        .await
        .map_err(map)?;
    Ok(())
}

/// Freeze or thaw an expert in place by domain id: re-write the record with
/// `frozen_at` set (freeze) or cleared (thaw). The population is keyed by `key`
/// with an engine-assigned record id, so — as when a re-train supersedes an
/// expert — this replaces by delete-then-insert rather than updating in place.
/// No-op if the expert is absent.
pub async fn set_frozen(
    store: &Store,
    id: &ExpertId,
    frozen_at: Option<DateTime<Utc>>,
) -> Result<()> {
    if let Some(mut expert) = get(store, id).await? {
        expert.frozen_at = frozen_at;
        delete(store, id).await?;
        insert(store, &expert).await?;
    }
    Ok(())
}

/// Fetch one expert by id.
pub async fn get(store: &Store, id: &ExpertId) -> Result<Option<Expert>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("key", id.as_str()));
    let row: Option<ExpertRow> = first(store.client(), &query).await.map_err(map)?;
    row.map(ExpertRow::into_domain).transpose()
}

/// List the whole population.
pub async fn list(store: &Store) -> Result<Vec<Expert>> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    let rows: Vec<ExpertRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter().map(ExpertRow::into_domain).collect()
}

/// Routing-as-retrieval (ADR-0005): the `k` experts whose capability vector is
/// nearest to `query` under cosine distance, via surql-rs's vector-search
/// builder.
pub async fn knn_by_capability(store: &Store, query: &[f32], k: usize) -> Result<Vec<Expert>> {
    let vector: Vec<f64> = query.iter().map(|&x| f64::from(x)).collect();
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .vector_search(
            "capability_vec",
            vector,
            k as i64,
            VectorDistanceType::Cosine,
            None,
        )
        .map_err(map)?;
    let rows: Vec<ExpertRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter().map(ExpertRow::into_domain).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::{Expert, Generation};
    use chrono::Utc;

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
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn insert_get_list_knn_delete() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        let mut v = vec![0.0f32; EMBED_DIM];
        v[0] = 1.0;
        insert(&s, &expert("expert:a", v.clone())).await.unwrap();

        assert_eq!(
            get(&s, &ExpertId::new("expert:a"))
                .await
                .unwrap()
                .unwrap()
                .name,
            "expert:a"
        );
        assert!(get(&s, &ExpertId::new("expert:missing"))
            .await
            .unwrap()
            .is_none());
        assert_eq!(list(&s).await.unwrap().len(), 1);

        let near = knn_by_capability(&s, &v, 1).await.unwrap();
        assert_eq!(near.len(), 1);
        assert_eq!(near[0].id.as_str(), "expert:a");

        delete(&s, &ExpertId::new("expert:a")).await.unwrap();
        assert!(list(&s).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn set_frozen_toggles_in_place_and_ignores_the_absent() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        let mut v = vec![0.0f32; EMBED_DIM];
        v[0] = 1.0;
        // Seed an unfrozen expert (override the helper's frozen default).
        let mut e = expert("expert:f", v);
        e.frozen_at = None;
        insert(&s, &e).await.unwrap();
        let id = ExpertId::new("expert:f");

        // Freeze in place: same record (one row), now frozen.
        set_frozen(&s, &id, Some(Utc::now())).await.unwrap();
        let frozen = get(&s, &id).await.unwrap().unwrap();
        assert!(frozen.is_frozen());
        assert_eq!(list(&s).await.unwrap().len(), 1, "replaced, not duplicated");

        // Thaw in place.
        set_frozen(&s, &id, None).await.unwrap();
        assert!(!get(&s, &id).await.unwrap().unwrap().is_frozen());

        // Absent id is a no-op, not an error.
        set_frozen(&s, &ExpertId::new("expert:ghost"), Some(Utc::now()))
            .await
            .unwrap();
        assert_eq!(list(&s).await.unwrap().len(), 1);
    }
}
