//! Learned-router repository (ADR-0009). A singleton per store: the gate's
//! trained metric + centroids, retrained whenever the population changes. No
//! raw SurrealQL; surql-rs `crud` helpers only.

use surql::query::crud::{get_record, upsert_record};
use surql::types::RecordID;

use antumbra_core::{LearnedRouter, Result};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "learned_router";
const KEY: &str = "active";

fn record_id() -> Result<RecordID> {
    RecordID::<()>::new(TABLE, KEY).map_err(map)
}

/// Persist (replace) the active learned router for this population.
pub async fn save(store: &Store, router: &LearnedRouter) -> Result<()> {
    let id = record_id()?;
    let data = serde_json::to_value(router)?;
    upsert_record(store.client(), &id, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Load the active learned router, if one has been trained. Returns `None`
/// (not an error) before the first `gate-train`, when the table does not exist.
pub async fn load(store: &Store) -> Result<Option<LearnedRouter>> {
    let id = record_id()?;
    match get_record(store.client(), &id).await {
        Ok(Some(value)) => Ok(Some(serde_json::from_value(value)?)),
        Ok(None) => Ok(None),
        Err(e) if e.to_string().contains("does not exist") => Ok(None),
        Err(e) => Err(map(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::router::{LearnedRouter, RouterExpert};
    use antumbra_core::ExpertId;

    fn router() -> LearnedRouter {
        LearnedRouter {
            weights: vec![1.0; EMBED_DIM],
            experts: vec![RouterExpert {
                id: ExpertId::new("expert:a"),
                centroid: vec![0.0; EMBED_DIM],
            }],
            temperature: 0.2,
            floor: 0.1,
        }
    }

    #[tokio::test]
    async fn load_before_save_is_none() {
        // The singleton table does not exist until the first save (gate-train);
        // load returns None, not an error.
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        assert!(load(&store).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn save_then_load_round_trips_and_upserts() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        save(&store, &router()).await.unwrap();
        let loaded = load(&store).await.unwrap().expect("a router was saved");
        assert_eq!(loaded.experts.len(), 1);
        assert_eq!(loaded.experts[0].id.as_str(), "expert:a");
        assert_eq!(loaded.temperature, 0.2);
        assert_eq!(loaded.floor, 0.1);

        // Saving again replaces the single active router (upsert, not append).
        let mut next = router();
        next.floor = 0.5;
        save(&store, &next).await.unwrap();
        assert_eq!(load(&store).await.unwrap().unwrap().floor, 0.5);
    }
}
