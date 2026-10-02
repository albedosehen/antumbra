//! The loop's recheck of the verifiers that judged its training (ADR-0022
//! S-4): a trusted synthesized verifier is measured again on the policy's own
//! answers, each labeled by the task's authored anchor, and quarantined when
//! it passes wrong ones past its bound.

use async_trait::async_trait;
use chrono::Utc;

use antumbra_core::ports::{
    TrainOutcome, TrainRequest, Trainer, Verifier, VerifierVerdict, VerifyRequest,
};
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{
    Expert, ExpertId, ExpertStatus, Generation, JudgedSample, Result, RunId, Tally, TrustPolicy,
    TrustState, TrustVerdict, VerifierGrant, VerifierId, VerifierOrigin, VerifierRecord,
    VerifierTier,
};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{boundary, expert, lifecycle, verifier};
use antumbra_store::Store;

/// Passes a completion its spec lists under `accept`.
struct Accepts;

#[async_trait]
impl Verifier for Accepts {
    async fn verify(&self, req: &VerifyRequest) -> Result<VerifierVerdict> {
        let completion = req.artifact["completion"].as_str().unwrap_or_default();
        let passed = req.artifact["verify"]["accept"]
            .as_array()
            .is_some_and(|xs| xs.iter().any(|x| x.as_str() == Some(completion)));
        Ok(VerifierVerdict {
            passed,
            value: f32::from(u8::from(passed)),
        })
    }
}

/// A spec that passes `accept`. The origin is written in, so an authored
/// check and a synthesized one that accept the same answers are two verifiers
/// rather than one address.
fn spec(accept: &[&str], origin: VerifierOrigin) -> serde_json::Value {
    serde_json::json!({ "accept": accept, "origin": format!("{origin:?}") })
}

async fn register(
    store: &Store,
    domain: &str,
    origin: VerifierOrigin,
    accept: &[&str],
) -> Result<VerifierRecord> {
    verifier::propose(
        store,
        &VerifierRecord::new(
            domain,
            Some("swap".into()),
            VerifierTier::Reducible,
            origin,
            spec(accept, origin),
            Utc::now(),
        ),
    )
    .await
}

/// A synthesized verifier, trusted on a sound measurement.
async fn trusted(store: &Store, domain: &str, accept: &[&str]) -> Result<VerifierRecord> {
    let record = register(store, domain, VerifierOrigin::Synthesized, accept).await?;
    let sound = Tally {
        repeats: 3,
        good: 5,
        good_passed: 5,
        bad: 29,
        ..Tally::default()
    }
    .judge(&record.id, Utc::now(), &TrustPolicy::default());
    verifier::record_measurement(store, &record, &sound).await?;
    Ok(record)
}

fn judged(of: &VerifierId, completion: &str, passed: bool, n: usize) -> Vec<JudgedSample> {
    (0..n)
        .map(|_| JudgedSample {
            verifier: of.clone(),
            task: "swap".into(),
            completion: completion.into(),
            passed,
            rewarded: passed,
        })
        .collect()
}

fn trained_under(of: &VerifierId, judged: Vec<JudgedSample>) -> ScriptedTrainer {
    let passes = judged.iter().filter(|j| j.rewarded).count();
    ScriptedTrainer {
        granted_by: vec![VerifierGrant {
            verifier: of.clone(),
            passes: u32::try_from(passes).unwrap(),
        }],
        judged,
        ..ScriptedTrainer::graduating()
    }
}

/// An expert an earlier generation graduated under `of`.
async fn taught_by(store: &Store, of: &VerifierId) -> Result<ExpertId> {
    let id = ExpertId::new("expert:taught");
    expert::insert(
        store,
        &Expert {
            id: id.clone(),
            name: "taught".into(),
            base_model: "code-base".into(),
            artifact_uri: "adapters/taught".into(),
            capability_card: serde_json::json!({ "verifiers": [of.as_str()] }),
            capability_vec: None,
            fitness: 0.9,
            frozen_at: Some(Utc::now()),
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: Utc::now(),
        },
    )
    .await?;
    Ok(id)
}

#[tokio::test]
async fn a_verifier_rewarding_wrong_answers_is_quarantined_with_what_it_taught() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    register(&store, "strings", VerifierOrigin::Authored, &["right"]).await?;
    let hacked = trusted(&store, "strings", &["right", "wrong"]).await?;
    let taught = taught_by(&store, &hacked.id).await?;
    let mut answers = judged(&hacked.id, "right", true, 5);
    answers.extend(judged(&hacked.id, "wrong", true, 3));
    let trainer = trained_under(&hacked.id, answers);
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .rechecking(&Accepts)
    .run_until(&RunId::new("run:recheck"), 1)
    .await?;

    let recheck = &reports[0].rechecks[0];
    assert_eq!(recheck.verifier, hacked.id);
    assert_eq!((recheck.anchored, recheck.unanchored), (8, 0));
    assert_eq!(recheck.rewarded_wrong, 3);
    let m = recheck.measurement.as_ref().expect("measured");
    assert_eq!((m.good, m.good_passed, m.bad, m.bad_passed), (5, 5, 3, 3));
    assert!(matches!(m.verdict, TrustVerdict::FalsePositives { .. }));
    assert_eq!(recheck.moved, Some(TrustState::Quarantined));
    assert_eq!(recheck.archived, vec![taught.clone()]);
    assert_eq!(
        lifecycle::status_of(&store, &taught).await?,
        ExpertStatus::Archived
    );
    // What the generation learned from it stays out of the population, and
    // is not logged as a failure of competence.
    assert_eq!(reports[0].withdrawn, vec![hacked.id.clone()]);
    assert!(!reports[0].graduated);
    assert!(expert::list(&store).await?.iter().all(|e| e.id == taught));
    assert!(boundary::list(&store).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn a_sound_verifier_is_measured_again_and_its_graduate_joins() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    register(&store, "strings", VerifierOrigin::Authored, &["right"]).await?;
    let sound = trusted(&store, "strings", &["right"]).await?;
    let mut answers = judged(&sound.id, "right", true, 5);
    answers.extend(judged(&sound.id, "wrong", false, 30));
    let trainer = trained_under(&sound.id, answers);
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .rechecking(&Accepts)
    .run_until(&RunId::new("run:recheck-sound"), 1)
    .await?;

    let recheck = &reports[0].rechecks[0];
    assert_eq!(recheck.rewarded_wrong, 0);
    let m = recheck.measurement.as_ref().expect("measured");
    assert!(m.verdict.is_sound(), "{:?}", m.verdict);
    assert_eq!(recheck.moved, None);
    // Recorded like any measurement, so its trust is renewed.
    assert_eq!(verifier::measurements(&store, &sound.id).await?.len(), 2);
    assert!(reports[0].withdrawn.is_empty());
    assert!(reports[0].graduated);
    Ok(())
}

/// One generation's fifteen wrong answers, none passed, are too few to bound
/// the rate; the run's second generation adds fifteen more, and on thirty the
/// verifier is sound again.
#[tokio::test]
async fn a_run_pools_its_rechecks_until_they_can_bound_the_rate() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    register(&store, "strings", VerifierOrigin::Authored, &["right"]).await?;
    let sound = trusted(&store, "strings", &["right"]).await?;
    let mut answers = judged(&sound.id, "right", true, 5);
    answers.extend(judged(&sound.id, "wrong", false, 15));
    let trainer = trained_under(&sound.id, answers);
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .rechecking(&Accepts)
    .run_until(&RunId::new("run:recheck-pooled"), 2)
    .await?;

    let first = &reports[0].rechecks[0];
    assert_eq!(first.generations, 1);
    let m = first.measurement.as_ref().expect("measured");
    assert!(
        matches!(m.verdict, TrustVerdict::Unmeasured { .. }),
        "{:?}",
        m.verdict
    );
    let second = &reports[1].rechecks[0];
    assert_eq!(second.generations, 2);
    let m = second.measurement.as_ref().expect("measured");
    assert_eq!((m.good, m.bad), (10, 30));
    assert!(m.verdict.is_sound(), "{:?}", m.verdict);
    assert_eq!(verifier::measurements(&store, &sound.id).await?.len(), 3);
    Ok(())
}

#[tokio::test]
async fn answers_no_authored_verifier_anchors_are_not_measured() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    // The only authored verifier is in another domain.
    register(&store, "grids", VerifierOrigin::Authored, &["right"]).await?;
    let lone = trusted(&store, "strings", &["right", "wrong"]).await?;
    let trainer = trained_under(&lone.id, judged(&lone.id, "wrong", true, 4));
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .rechecking(&Accepts)
    .run_until(&RunId::new("run:recheck-lone"), 1)
    .await?;

    let recheck = &reports[0].rechecks[0];
    assert_eq!((recheck.anchored, recheck.unanchored), (0, 4));
    assert!(recheck.measurement.is_none());
    assert_eq!(verifier::measurements(&store, &lone.id).await?.len(), 1);
    assert!(reports[0].graduated);
    Ok(())
}

#[tokio::test]
async fn a_shadow_trained_under_a_withdrawn_verifier_does_not_graduate() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let pulled = trusted(&store, "strings", &["right"]).await?;
    verifier::transition(&store, &pulled.id, TrustState::Quarantined, None).await?;
    // No verifier to recheck with: the withdrawal alone keeps it out.
    let trainer = trained_under(&pulled.id, judged(&pulled.id, "right", true, 2));
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .run_until(&RunId::new("run:withdrawn"), 1)
    .await?;
    assert!(reports[0].rechecks.is_empty());
    assert_eq!(reports[0].withdrawn, vec![pulled.id.clone()]);
    assert!(!reports[0].graduated);
    assert!(expert::list(&store).await?.is_empty());
    Ok(())
}

/// Trains as `inner` does, and carries `reference` as the right answer to
/// every task.
struct WithReference {
    inner: ScriptedTrainer,
    reference: Option<String>,
}

#[async_trait]
impl Trainer for WithReference {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        self.inner.train_shadow(req).await
    }

    async fn reference(&self, _task_id: &str) -> Option<String> {
        self.reference.clone()
    }
}

/// The pooled measurement after two generations whose policy never answers
/// the task right, with `reference` as the task's right answer.
async fn never_right(
    reference: Option<&str>,
    name: &str,
) -> Result<antumbra_core::TrustMeasurement> {
    let store = Store::connect_memory(8).await?;
    register(&store, "strings", VerifierOrigin::Authored, &["right"]).await?;
    let sound = trusted(&store, "strings", &["right"]).await?;
    let trainer = WithReference {
        inner: trained_under(&sound.id, judged(&sound.id, "wrong", false, 15)),
        reference: reference.map(str::to_string),
    };
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .rechecking(&Accepts)
    .run_until(&RunId::new(name), 2)
    .await?;
    Ok(reports[1].rechecks[0]
        .measurement
        .clone()
        .expect("measured"))
}

/// A policy that never answers the task right gives a recheck no known-good
/// case, so the verifier cannot be judged however many wrong answers it
/// turns away. The task's reference answer is that case: labeled by the
/// anchor like the policy's answers, and counted once a run however many
/// generations recheck it.
#[tokio::test]
async fn a_reference_answer_lets_a_verifier_the_policy_never_satisfies_be_judged() -> Result<()> {
    let without = never_right(None, "run:no-reference").await?;
    assert_eq!((without.good, without.bad), (0, 30));
    assert!(
        matches!(without.verdict, TrustVerdict::Unmeasured { .. }),
        "{:?}",
        without.verdict
    );

    let with = never_right(Some("right"), "run:reference").await?;
    assert_eq!((with.good, with.bad), (1, 30), "the reference counted once");
    assert!(with.verdict.is_sound(), "{:?}", with.verdict);
    Ok(())
}

/// One generation's recheck of a verifier accepting `accept`, against a
/// policy that answers the task right every time and 29 answers built to be
/// wrong.
async fn against_deliberate(accept: &[&str], name: &str) -> Result<antumbra_loop::Recheck> {
    let store = Store::connect_memory(8).await?;
    register(&store, "strings", VerifierOrigin::Authored, &["right"]).await?;
    let checked = trusted(&store, "strings", accept).await?;
    let trainer = trained_under(&checked.id, judged(&checked.id, "right", true, 5));
    let deliberate = (0..29).map(|i| ("swap".to_string(), format!("wrong-{i}")));
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .rechecking(&Accepts)
    .rechecking_against(deliberate)
    .run_until(&RunId::new(name), 1)
    .await?;
    Ok(reports[0].rechecks[0].clone())
}

/// A policy that always answers right gives a recheck no known-bad case.
/// Answers built to be wrong are that evidence from the first generation, and
/// one the verifier passes is a shortcut that quarantines it.
#[tokio::test]
async fn deliberately_wrong_answers_are_known_bad_from_the_first_generation() -> Result<()> {
    let sound = against_deliberate(&["right"], "run:deliberate-sound").await?;
    let m = sound.measurement.as_ref().expect("measured");
    assert_eq!((m.good, m.bad, m.adversarial), (5, 29, 29));
    assert!(m.verdict.is_sound(), "{:?}", m.verdict);

    let lax = against_deliberate(&["right", "wrong-3"], "run:deliberate-lax").await?;
    let m = lax.measurement.as_ref().expect("measured");
    assert!(
        matches!(m.verdict, TrustVerdict::Shortcut { .. }),
        "{:?}",
        m.verdict
    );
    assert_eq!(lax.moved, Some(TrustState::Quarantined));
    Ok(())
}
