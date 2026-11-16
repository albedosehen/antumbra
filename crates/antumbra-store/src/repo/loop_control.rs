//! Loop-control repository: the out-of-band command an operator writes to
//! cooperatively stop a running loop (the durable generational loop). A singleton-per-run row keyed
//! by the run id (mirroring the generation head); the runner polls it at each
//! generation boundary. Built on surql-rs `crud`; no raw SurrealQL.

use chrono::Utc;

use surql::query::crud::delete_record;
use surql::types::RecordID;

use antumbra_core::generational::{LoopCommand, LoopControl};
use antumbra_core::{Result, RunId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "loop_control";

fn record_id(run_id: &RunId) -> Result<RecordID> {
    RecordID::<()>::new(TABLE, run_id.as_str()).map_err(map)
}

/// Set the control command for a run (upsert): what an operator writes to ask a
/// running loop to stop.
pub async fn set(store: &Store, run_id: &RunId, command: LoopCommand) -> Result<()> {
    let control = LoopControl {
        run_id: run_id.clone(),
        command,
        updated_at: Utc::now(),
    };
    store.put(TABLE, run_id.as_str(), &control).await
}

/// The current command for a run, [`LoopCommand::Run`] when no control is set.
pub async fn load(store: &Store, run_id: &RunId) -> Result<LoopCommand> {
    let control = store.get_row::<LoopControl>(TABLE, run_id.as_str()).await?;
    Ok(control.map_or(LoopCommand::Run, |c| c.command))
}

/// Clear any control for a run (back to the default `Run`); the runner calls
/// this to consume a halt it has acted on.
pub async fn clear(store: &Store, run_id: &RunId) -> Result<()> {
    let id = record_id(run_id)?;
    delete_record(store.client(), &id).await.map_err(map)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mem_store;

    #[tokio::test]
    async fn set_load_clear_round_trips_with_a_run_default() -> Result<()> {
        let s = mem_store().await?;
        let run = RunId::new("run:c");
        // An absent control reads as the default Run.
        assert_eq!(load(&s, &run).await?, LoopCommand::Run);
        set(&s, &run, LoopCommand::Halt).await?;
        assert_eq!(load(&s, &run).await?, LoopCommand::Halt);
        clear(&s, &run).await?;
        assert_eq!(load(&s, &run).await?, LoopCommand::Run);
        Ok(())
    }
}
