//! The verifier namespace against a real in-memory store: a
//! synthesized verifier grants reward only once a sound measurement has
//! trusted it, only for what it checks, and stops the moment it is
//! quarantined or its trust lapses.

use anyhow::Result;
use chrono::{TimeDelta, Utc};

use antumbra_core::ports::TrustedVerifiers;
use antumbra_core::{
    Expert, ExpertId, ExpertStatus, Generation, Tally, TransitionCause, TrustPolicy, TrustState,
    TrustVerdict, VerifierId, VerifierOrigin, VerifierRecord, VerifierTier,
};
use antumbra_store::repo::verifier::{self, Registry};
use antumbra_store::repo::{expert, lifecycle};
use antumbra_store::testing;

fn synthesized(task: Option<&str>) -> VerifierRecord {
    VerifierRecord::new(
        "strings",
        task.map(str::to_string),
        VerifierTier::Reducible,
        VerifierOrigin::Synthesized,
        serde_json::json!({ "contains_all": ["def swap"] }),
        Utc::now(),
    )
}

/// A tally the default policy finds sound: 29 failures, none passed.
fn sound_tally() -> Tally {
    Tally {
        repeats: 3,
        good: 5,
        good_passed: 5,
        bad: 29,
        bad_passed: 0,
        ..Tally::default()
    }
}

#[tokio::test]
async fn proposing_the_same_content_twice_is_one_verifier() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let first = verifier::propose(&store, &synthesized(None)).await?;
    let mut again = synthesized(None);
    again.proposed_by = Some("someone else".into());
    let second = verifier::propose(&store, &again).await?;
    assert_eq!(first.id, second.id);
    assert_eq!(second.proposed_by, None, "the one already there is kept");
    assert_eq!(verifier::list(&store).await?.len(), 1);
    let Some(got) = verifier::get(&store, &first.id).await? else {
        anyhow::bail!("the proposed verifier does not read back");
    };
    assert_eq!(got.spec, first.spec);
    Ok(())
}

#[tokio::test]
async fn a_record_whose_content_does_not_match_its_address_is_refused() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let mut forged = synthesized(None);
    forged.spec = serde_json::json!({ "program": "true" });
    assert!(verifier::propose(&store, &forged).await.is_err());
    Ok(())
}

#[tokio::test]
async fn only_a_sound_measurement_lets_a_synthesized_verifier_grant_reward() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let record = verifier::propose(&store, &synthesized(Some("strings/swap"))).await?;
    let registry = Registry::new(store.clone());
    assert_eq!(
        verifier::state_of(&store, &record).await?,
        TrustState::Proposed
    );
    assert!(registry
        .trusted_spec(&record.id, "strings/swap")
        .await?
        .is_none());
    // A person cannot promote it.
    assert!(
        verifier::transition(&store, &record.id, TrustState::Trusted, None)
            .await
            .is_err()
    );

    let policy = TrustPolicy::default();
    let sound = sound_tally().judge(&record.id, Utc::now(), &policy);
    let Some(moved) = verifier::record_measurement(&store, &record, &sound).await? else {
        anyhow::bail!("a sound measurement did not move the verifier");
    };
    assert_eq!(
        (moved.transition.from, moved.transition.to),
        (TrustState::Proposed, TrustState::Trusted)
    );
    assert_eq!(
        registry.trusted_spec(&record.id, "strings/swap").await?,
        Some(record.spec.clone())
    );
    // Not for a task it does not check.
    assert!(registry
        .trusted_spec(&record.id, "strings/other")
        .await?
        .is_none());

    // A re-measurement that finds false positives quarantines it at once.
    let mut leaky = sound_tally();
    leaky.bad_passed = 3;
    let unsound = leaky.judge(&record.id, Utc::now(), &policy);
    assert!(matches!(
        unsound.verdict,
        TrustVerdict::FalsePositives { .. }
    ));
    let Some(moved) = verifier::record_measurement(&store, &record, &unsound).await? else {
        anyhow::bail!("an unsound measurement did not move the verifier");
    };
    assert_eq!(moved.transition.to, TrustState::Quarantined);
    assert!(registry
        .trusted_spec(&record.id, "strings/swap")
        .await?
        .is_none());
    // A later sound measurement does not undo it; a person may revoke it.
    let again = sound_tally().judge(&record.id, Utc::now(), &policy);
    assert!(verifier::record_measurement(&store, &record, &again)
        .await?
        .is_none());
    verifier::transition(&store, &record.id, TrustState::Revoked, Some("done".into())).await?;
    let history = verifier::history(&store, &record.id).await?;
    let path: Vec<TrustState> = history.iter().map(|t| t.to).collect();
    assert_eq!(
        path,
        vec![
            TrustState::Trusted,
            TrustState::Quarantined,
            TrustState::Revoked
        ]
    );
    assert_eq!(verifier::measurements(&store, &record.id).await?.len(), 3);
    Ok(())
}

#[tokio::test]
async fn a_flaky_proposal_is_revoked_outright() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let record = verifier::propose(&store, &synthesized(None)).await?;
    let mut flaky = sound_tally();
    flaky.flaky = vec!["case-3".into()];
    let m = flaky.judge(&record.id, Utc::now(), &TrustPolicy::default());
    let Some(moved) = verifier::record_measurement(&store, &record, &m).await? else {
        anyhow::bail!("a flaky measurement did not move the verifier");
    };
    assert_eq!(moved.transition.to, TrustState::Revoked);
    Ok(())
}

#[tokio::test]
async fn trust_that_is_not_re_measured_lapses() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let record = verifier::propose(&store, &synthesized(None)).await?;
    let long_ago = Utc::now() - TimeDelta::days(30);
    let m = sound_tally().judge(&record.id, long_ago, &TrustPolicy::default());
    verifier::record_measurement(&store, &record, &m).await?;
    assert_eq!(
        verifier::state_of(&store, &record).await?,
        TrustState::Trusted
    );
    assert!(!verifier::grants(&store, &record, Utc::now()).await?);
    assert!(verifier::grants(&store, &record, long_ago).await?);
    Ok(())
}

#[tokio::test]
async fn an_authored_verifier_grants_by_authorship_until_a_person_moves_it() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let authored = VerifierRecord::new(
        "strings",
        None,
        VerifierTier::Reducible,
        VerifierOrigin::Authored,
        serde_json::json!({ "program": "judge" }),
        Utc::now(),
    );
    let record = verifier::propose(&store, &authored).await?;
    let registry = Registry::new(store.clone());
    assert!(registry
        .trusted_spec(&record.id, "strings/any")
        .await?
        .is_some());
    verifier::transition(&store, &record.id, TrustState::Quarantined, None).await?;
    assert!(registry
        .trusted_spec(&record.id, "strings/any")
        .await?
        .is_none());
    Ok(())
}

fn expert_under(id: &str, verifiers: &[&VerifierId]) -> Expert {
    Expert {
        id: ExpertId::new(id),
        name: id.to_string(),
        base_model: "base".into(),
        artifact_uri: format!("adapters/{id}.safetensors"),
        capability_card: serde_json::json!({
            "verifiers": verifiers.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
        }),
        capability_vec: None,
        fitness: 0.9,
        frozen_at: Some(Utc::now()),
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: Utc::now(),
    }
}

#[tokio::test]
async fn quarantining_a_verifier_archives_what_it_taught_and_nothing_else() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let record = verifier::propose(&store, &synthesized(None)).await?;
    let other = VerifierId::new("verifier:other");
    for e in [
        expert_under("expert:taught", &[&record.id]),
        expert_under("expert:both", &[&other, &record.id]),
        expert_under("expert:dormant", &[&record.id]),
        expert_under("expert:elsewhere", &[&other]),
        expert_under("expert:none", &[]),
    ] {
        expert::insert(&store, &e).await?;
    }
    let dormant = ExpertId::new("expert:dormant");
    let operator = TransitionCause::Operator { note: None };
    lifecycle::transition(&store, &dormant, ExpertStatus::Dormant, operator, None).await?;

    let sound = sound_tally().judge(&record.id, Utc::now(), &TrustPolicy::default());
    let Some(trusted) = verifier::record_measurement(&store, &record, &sound).await? else {
        anyhow::bail!("a sound measurement did not move the verifier");
    };
    assert!(trusted.archived.is_empty(), "trust archives nothing");

    let moved = verifier::transition(&store, &record.id, TrustState::Quarantined, None).await?;
    let mut archived: Vec<&str> = moved.archived.iter().map(|e| e.as_str()).collect();
    archived.sort();
    assert_eq!(archived, ["expert:both", "expert:dormant", "expert:taught"]);
    let statuses = lifecycle::statuses(&store).await?;
    assert_eq!(
        statuses.get(&ExpertId::new("expert:taught")),
        Some(&ExpertStatus::Archived)
    );
    assert_eq!(
        statuses.get(&ExpertId::new("expert:elsewhere")),
        None,
        "still active"
    );
    let history = lifecycle::history(&store, &ExpertId::new("expert:taught")).await?;
    let Some(last) = history.last() else {
        anyhow::bail!("the archived expert has no lifecycle history");
    };
    assert_eq!(
        last.cause,
        TransitionCause::Quarantined {
            verifier: record.id.clone()
        }
    );
    // Revoking it afterwards finds nothing left to archive.
    let revoked = verifier::transition(&store, &record.id, TrustState::Revoked, None).await?;
    assert!(revoked.archived.is_empty());
    Ok(())
}
