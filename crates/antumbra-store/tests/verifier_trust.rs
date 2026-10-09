//! The verifier namespace against a real in-memory store: a
//! synthesized verifier grants reward only once a sound measurement has
//! trusted it, only for what it checks, and stops the moment it is
//! quarantined or its trust lapses.

use chrono::{TimeDelta, Utc};

use antumbra_core::ports::TrustedVerifiers;
use antumbra_core::{
    Expert, ExpertId, ExpertStatus, Generation, Tally, TransitionCause, TrustPolicy, TrustState,
    TrustVerdict, VerifierId, VerifierOrigin, VerifierRecord, VerifierTier,
};
use antumbra_store::repo::verifier::{self, Registry};
use antumbra_store::repo::{expert, lifecycle};
use antumbra_store::Store;

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
async fn proposing_the_same_content_twice_is_one_verifier() {
    let store = Store::connect_memory(4).await.unwrap();
    let first = verifier::propose(&store, &synthesized(None)).await.unwrap();
    let mut again = synthesized(None);
    again.proposed_by = Some("someone else".into());
    let second = verifier::propose(&store, &again).await.unwrap();
    assert_eq!(first.id, second.id);
    assert_eq!(second.proposed_by, None, "the one already there is kept");
    assert_eq!(verifier::list(&store).await.unwrap().len(), 1);
    let got = verifier::get(&store, &first.id).await.unwrap().unwrap();
    assert_eq!(got.spec, first.spec);
}

#[tokio::test]
async fn a_record_whose_content_does_not_match_its_address_is_refused() {
    let store = Store::connect_memory(4).await.unwrap();
    let mut forged = synthesized(None);
    forged.spec = serde_json::json!({ "program": "true" });
    assert!(verifier::propose(&store, &forged).await.is_err());
}

#[tokio::test]
async fn only_a_sound_measurement_lets_a_synthesized_verifier_grant_reward() {
    let store = Store::connect_memory(4).await.unwrap();
    let record = verifier::propose(&store, &synthesized(Some("strings/swap")))
        .await
        .unwrap();
    let registry = Registry::new(store.clone());
    assert_eq!(
        verifier::state_of(&store, &record).await.unwrap(),
        TrustState::Proposed
    );
    assert!(registry
        .trusted_spec(&record.id, "strings/swap")
        .await
        .unwrap()
        .is_none());
    // A person cannot promote it.
    assert!(
        verifier::transition(&store, &record.id, TrustState::Trusted, None)
            .await
            .is_err()
    );

    let policy = TrustPolicy::default();
    let sound = sound_tally().judge(&record.id, Utc::now(), &policy);
    let moved = verifier::record_measurement(&store, &record, &sound)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (moved.transition.from, moved.transition.to),
        (TrustState::Proposed, TrustState::Trusted)
    );
    assert_eq!(
        registry
            .trusted_spec(&record.id, "strings/swap")
            .await
            .unwrap(),
        Some(record.spec.clone())
    );
    // Not for a task it does not check.
    assert!(registry
        .trusted_spec(&record.id, "strings/other")
        .await
        .unwrap()
        .is_none());

    // A re-measurement that finds false positives quarantines it at once.
    let mut leaky = sound_tally();
    leaky.bad_passed = 3;
    let unsound = leaky.judge(&record.id, Utc::now(), &policy);
    assert!(matches!(
        unsound.verdict,
        TrustVerdict::FalsePositives { .. }
    ));
    let moved = verifier::record_measurement(&store, &record, &unsound)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(moved.transition.to, TrustState::Quarantined);
    assert!(registry
        .trusted_spec(&record.id, "strings/swap")
        .await
        .unwrap()
        .is_none());
    // A later sound measurement does not undo it; a person may revoke it.
    let again = sound_tally().judge(&record.id, Utc::now(), &policy);
    assert!(verifier::record_measurement(&store, &record, &again)
        .await
        .unwrap()
        .is_none());
    verifier::transition(&store, &record.id, TrustState::Revoked, Some("done".into()))
        .await
        .unwrap();
    let history = verifier::history(&store, &record.id).await.unwrap();
    let path: Vec<TrustState> = history.iter().map(|t| t.to).collect();
    assert_eq!(
        path,
        vec![
            TrustState::Trusted,
            TrustState::Quarantined,
            TrustState::Revoked
        ]
    );
    assert_eq!(
        verifier::measurements(&store, &record.id)
            .await
            .unwrap()
            .len(),
        3
    );
}

#[tokio::test]
async fn a_flaky_proposal_is_revoked_outright() {
    let store = Store::connect_memory(4).await.unwrap();
    let record = verifier::propose(&store, &synthesized(None)).await.unwrap();
    let mut flaky = sound_tally();
    flaky.flaky = vec!["case-3".into()];
    let m = flaky.judge(&record.id, Utc::now(), &TrustPolicy::default());
    let moved = verifier::record_measurement(&store, &record, &m)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(moved.transition.to, TrustState::Revoked);
}

#[tokio::test]
async fn trust_that_is_not_re_measured_lapses() {
    let store = Store::connect_memory(4).await.unwrap();
    let record = verifier::propose(&store, &synthesized(None)).await.unwrap();
    let long_ago = Utc::now() - TimeDelta::days(30);
    let m = sound_tally().judge(&record.id, long_ago, &TrustPolicy::default());
    verifier::record_measurement(&store, &record, &m)
        .await
        .unwrap();
    assert_eq!(
        verifier::state_of(&store, &record).await.unwrap(),
        TrustState::Trusted
    );
    assert!(!verifier::grants(&store, &record, Utc::now()).await.unwrap());
    assert!(verifier::grants(&store, &record, long_ago).await.unwrap());
}

#[tokio::test]
async fn an_authored_verifier_grants_by_authorship_until_a_person_moves_it() {
    let store = Store::connect_memory(4).await.unwrap();
    let authored = VerifierRecord::new(
        "strings",
        None,
        VerifierTier::Reducible,
        VerifierOrigin::Authored,
        serde_json::json!({ "program": "judge" }),
        Utc::now(),
    );
    let record = verifier::propose(&store, &authored).await.unwrap();
    let registry = Registry::new(store.clone());
    assert!(registry
        .trusted_spec(&record.id, "strings/any")
        .await
        .unwrap()
        .is_some());
    verifier::transition(&store, &record.id, TrustState::Quarantined, None)
        .await
        .unwrap();
    assert!(registry
        .trusted_spec(&record.id, "strings/any")
        .await
        .unwrap()
        .is_none());
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
async fn quarantining_a_verifier_archives_what_it_taught_and_nothing_else() {
    let store = Store::connect_memory(4).await.unwrap();
    let record = verifier::propose(&store, &synthesized(None)).await.unwrap();
    let other = VerifierId::new("verifier:other");
    for e in [
        expert_under("expert:taught", &[&record.id]),
        expert_under("expert:both", &[&other, &record.id]),
        expert_under("expert:dormant", &[&record.id]),
        expert_under("expert:elsewhere", &[&other]),
        expert_under("expert:none", &[]),
    ] {
        expert::insert(&store, &e).await.unwrap();
    }
    let dormant = ExpertId::new("expert:dormant");
    let operator = TransitionCause::Operator { note: None };
    lifecycle::transition(&store, &dormant, ExpertStatus::Dormant, operator, None)
        .await
        .unwrap();

    let sound = sound_tally().judge(&record.id, Utc::now(), &TrustPolicy::default());
    let trusted = verifier::record_measurement(&store, &record, &sound)
        .await
        .unwrap()
        .unwrap();
    assert!(trusted.archived.is_empty(), "trust archives nothing");

    let moved = verifier::transition(&store, &record.id, TrustState::Quarantined, None)
        .await
        .unwrap();
    let mut archived: Vec<&str> = moved.archived.iter().map(|e| e.as_str()).collect();
    archived.sort();
    assert_eq!(archived, ["expert:both", "expert:dormant", "expert:taught"]);
    let statuses = lifecycle::statuses(&store).await.unwrap();
    assert_eq!(
        statuses.get(&ExpertId::new("expert:taught")),
        Some(&ExpertStatus::Archived)
    );
    assert_eq!(
        statuses.get(&ExpertId::new("expert:elsewhere")),
        None,
        "still active"
    );
    let history = lifecycle::history(&store, &ExpertId::new("expert:taught"))
        .await
        .unwrap();
    assert_eq!(
        history.last().unwrap().cause,
        TransitionCause::Quarantined {
            verifier: record.id.clone()
        }
    );
    // Revoking it afterwards finds nothing left to archive.
    let revoked = verifier::transition(&store, &record.id, TrustState::Revoked, None)
        .await
        .unwrap();
    assert!(revoked.archived.is_empty());
}
