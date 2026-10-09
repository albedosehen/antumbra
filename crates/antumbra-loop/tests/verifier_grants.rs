//! Reward attribution through the loop: what a named verifier
//! granted a run lands on the reward rows and on the expert it graduates,
//! and quarantining the verifier archives that expert.

use chrono::Utc;

use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{
    ExpertStatus, Result, RunId, Tally, TrustPolicy, TrustState, VerifierGrant, VerifierOrigin,
    VerifierRecord, VerifierTier,
};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{expert, lifecycle, reward, verifier};
use antumbra_store::Store;

#[tokio::test]
async fn a_quarantined_verifier_takes_what_it_taught_out_of_the_population() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let record = verifier::propose(
        &store,
        &VerifierRecord::new(
            "strings",
            None,
            VerifierTier::Reducible,
            VerifierOrigin::Synthesized,
            serde_json::json!({ "contains_all": ["x"] }),
            Utc::now(),
        ),
    )
    .await?;
    let sound = Tally {
        repeats: 3,
        good: 5,
        good_passed: 5,
        bad: 29,
        ..Tally::default()
    }
    .judge(&record.id, Utc::now(), &TrustPolicy::default());
    verifier::record_measurement(&store, &record, &sound).await?;

    let trainer = ScriptedTrainer {
        granted_by: vec![VerifierGrant {
            verifier: record.id.clone(),
            passes: 7,
        }],
        ..ScriptedTrainer::graduating()
    };
    let run = RunId::new("run:grants");
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .run_until(&run, 1)
    .await?;
    assert!(reports[0].graduated);

    let granted: Vec<_> = reward::list_by_run(&store, &run)
        .await?
        .into_iter()
        .filter(|s| s.dimension == "granted")
        .collect();
    assert_eq!(granted.len(), 1);
    assert_eq!(granted[0].verifier.as_ref(), Some(&record.id));
    assert_eq!(granted[0].value, 7.0);

    let graduate = expert::list(&store).await?.remove(0);
    assert!(lifecycle::trained_under(&graduate, &record.id));
    let moved = verifier::transition(&store, &record.id, TrustState::Quarantined, None).await?;
    assert_eq!(moved.archived, vec![graduate.id.clone()]);
    assert_eq!(
        lifecycle::status_of(&store, &graduate.id).await?,
        ExpertStatus::Archived
    );
    // Every reward row it produced stays on record.
    assert_eq!(
        reward::list_by_run(&store, &run)
            .await?
            .iter()
            .filter(|s| s.verifier.as_ref() == Some(&record.id))
            .count(),
        1
    );
    Ok(())
}
