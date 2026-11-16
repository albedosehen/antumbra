//! Generation-head repository: the durable generational loop checkpoint.
//!
//! A singleton-per-run row, addressed by a stable record id (`RecordID` auto-
//! escapes the run key), upserted via surql-rs's `crud` helpers. No raw
//! SurrealQL.

use surql::query::builder::Query;
use surql::query::crud::query_records;

use antumbra_core::generational::GenerationHead;
use antumbra_core::{Result, RunId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "generation_head";

/// Upsert the loop head; this single write IS the checkpoint.
pub async fn save_head(store: &Store, head: &GenerationHead) -> Result<()> {
    store.put(TABLE, head.run_id.as_str(), head).await
}

/// Load the loop head for a run, if one has been checkpointed.
pub async fn load_head(store: &Store, run_id: &RunId) -> Result<Option<GenerationHead>> {
    store.get_row(TABLE, run_id.as_str()).await
}

/// Every loop head across all runs: the owner/console view (parallels
/// [`crate::repo::memory::all_unscoped`]). The operator console reads as owner.
pub async fn all_heads(store: &Store) -> Result<Vec<GenerationHead>> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    query_records(store.client(), &query).await.map_err(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::generational::LoopState;
    use chrono::Utc;

    #[tokio::test]
    async fn save_then_all_heads_reads_every_run() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        let now = Utc::now();
        let mut a = GenerationHead::new(RunId::new("run:a"), now);
        a.advance_to(LoopState::Explore, now).unwrap();
        save_head(&s, &a).await.unwrap();
        save_head(&s, &GenerationHead::new(RunId::new("run:b"), now))
            .await
            .unwrap();

        let heads = all_heads(&s).await.unwrap();
        assert_eq!(heads.len(), 2, "both run heads");
        // The single-run load still resolves a specific checkpoint.
        let loaded = load_head(&s, &RunId::new("run:a")).await.unwrap().unwrap();
        assert_eq!(loaded.state, LoopState::Explore);
    }

    /// The point write and point read every repository is built on: a row
    /// put under a key reads back as itself, and a key never written reads as
    /// `None` rather than an error.
    #[tokio::test]
    async fn put_then_get_row_round_trips_and_an_absent_key_is_none() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        let now = Utc::now();
        let head = GenerationHead::new(RunId::new("run:a"), now);
        s.put(TABLE, "run:a", &head).await?;

        let back: Option<GenerationHead> = s.get_row(TABLE, "run:a").await?;
        assert_eq!(back.map(|h| h.run_id), Some(RunId::new("run:a")));
        let absent: Option<GenerationHead> = s.get_row(TABLE, "run:never").await?;
        assert!(absent.is_none());
        Ok(())
    }
}
