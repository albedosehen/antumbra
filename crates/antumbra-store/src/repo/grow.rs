//! Grow repository: the region census each contribution
//! measurement leaves, and the grow step's decisions. Both are keyed by run
//! and generation, so a generation measured or decided again after a crash
//! replaces its rows rather than adding second ones.

use surql::query::builder::Query;
use surql::query::crud::{query_records, upsert_record};
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::{Generation, GrowRecord, RegionCensus, Result, RunId};

use crate::error::map;
use crate::store::Store;

const CENSUS: &str = "region_census";
const GROW: &str = "grow";

/// Record one region's census in one generation.
pub async fn upsert_census(store: &Store, census: &RegionCensus) -> Result<()> {
    let key = format!(
        "{}@g{}@{}",
        census.run_id, census.generation.0, census.region
    );
    let id = RecordID::<()>::new(CENSUS, key.as_str()).map_err(map)?;
    upsert_record(store.client(), &id, serde_json::to_value(census)?)
        .await
        .map_err(map)?;
    Ok(())
}

/// The latest census a run took before `generation`: every region of the
/// most recent measured generation. Empty when there is none.
pub async fn census_before(
    store: &Store,
    run_id: &RunId,
    generation: Generation,
) -> Result<Vec<RegionCensus>> {
    let query = Query::new()
        .select(None)
        .from_table(CENSUS)
        .map_err(map)?
        .where_(eq("run_id", run_id.as_str()))
        .order_by("generation", "ASC")
        .map_err(map)?;
    let all: Vec<RegionCensus> = query_records(store.client(), &query).await.map_err(map)?;
    let Some(latest) = all
        .iter()
        .map(|c| c.generation.0)
        .filter(|g| *g < generation.0)
        .max()
    else {
        return Ok(Vec::new());
    };
    let mut census: Vec<RegionCensus> = all
        .into_iter()
        .filter(|c| c.generation.0 == latest)
        .collect();
    census.sort_by(|a, b| a.region.cmp(&b.region));
    Ok(census)
}

/// Record the grow step's decision for a generation.
pub async fn upsert(store: &Store, record: &GrowRecord) -> Result<()> {
    let key = format!("{}@g{}", record.run_id, record.generation.0);
    let id = RecordID::<()>::new(GROW, key.as_str()).map_err(map)?;
    upsert_record(store.client(), &id, serde_json::to_value(record)?)
        .await
        .map_err(map)?;
    Ok(())
}

/// Every decision a run's grow step made, oldest first.
pub async fn history(store: &Store, run_id: &RunId) -> Result<Vec<GrowRecord>> {
    let query = Query::new()
        .select(None)
        .from_table(GROW)
        .map_err(map)?
        .where_(eq("run_id", run_id.as_str()))
        .order_by("generation", "ASC")
        .map_err(map)?;
    query_records(store.client(), &query).await.map_err(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use chrono::Utc;

    fn census(generation: u32, region: &str, acceptability: f32) -> RegionCensus {
        RegionCensus {
            run_id: RunId::new("run:g"),
            generation: Generation(generation),
            region: region.into(),
            tasks: 4,
            acceptability,
            centroid: vec![1.0, 0.0],
            at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn the_latest_census_before_a_generation_is_the_one_read() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        upsert_census(&s, &census(0, "lists", 0.2)).await?;
        upsert_census(&s, &census(0, "dates", 0.9)).await?;
        upsert_census(&s, &census(2, "lists", 0.4)).await?;
        upsert_census(&s, &census(2, "lists", 0.5)).await?;
        let run = RunId::new("run:g");
        assert!(census_before(&s, &run, Generation(0)).await?.is_empty());
        let at1 = census_before(&s, &run, Generation(1)).await?;
        let regions: Vec<&str> = at1.iter().map(|c| c.region.as_str()).collect();
        assert_eq!(regions, ["dates", "lists"]);
        let at3 = census_before(&s, &run, Generation(3)).await?;
        assert_eq!(at3.len(), 1);
        assert_eq!(at3[0].acceptability, 0.5, "measured again, replaced");
        Ok(())
    }

    #[tokio::test]
    async fn decisions_are_kept_per_generation() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        let run = RunId::new("run:g");
        for (g, chosen) in [(0, None), (1, Some("lists")), (1, Some("dates"))] {
            upsert(
                &s,
                &GrowRecord {
                    run_id: run.clone(),
                    generation: Generation(g),
                    census_generation: None,
                    chosen: chosen.map(str::to_string),
                    candidates: Vec::new(),
                    focus: 0,
                    unfiltered: 0,
                    credit: None,
                    warm_from: None,
                    at: Utc::now(),
                },
            )
            .await?;
        }
        let kept = history(&s, &run).await?;
        let chosen: Vec<Option<&str>> = kept.iter().map(|r| r.chosen.as_deref()).collect();
        assert_eq!(chosen, [None, Some("dates")]);
        Ok(())
    }
}
