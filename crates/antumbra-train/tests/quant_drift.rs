//! Live measurement of what a quantized base does to an expert: the same
//! adapter over the dense f32 base, a Q8_0 one and a Q4_K one, on the held-out
//! examples of the behaviors it was taught.
//!
//! Two readings per base. Teacher-forced, over every held-out example's own
//! answer: how far each token's log-probability moves from the dense base's,
//! and how often another token is ranked first ([`antumbra_train::quant`]).
//! Generated, over the first few held-out tasks of each behavior: whether the
//! answer still passes the behavior's check, and whether it is the answer the
//! dense base gave.
//!
//! Ignored, and asserts `ANTUMBRA_QUANT_DRIFT` is set. Run it in release:
//!
//!   ANTUMBRA_QUANT_DRIFT=1 ANTUMBRA_QUANT_CORPUS=/path/to/behaviors.json \
//!   ANTUMBRA_CPU_ADAPTER=/path/to/expert.safetensors \
//!     cargo test -p antumbra-train --features models --release --test quant_drift -- --ignored --nocapture
//!
//! The corpus is a JSON array of behaviors, each `{id, scope, rule, must,
//! must_not, examples: [{task, answer}]}`. `ANTUMBRA_QUANT_SCOPE` keeps one
//! scope's behaviors; `ANTUMBRA_QUANT_GENERATE` is how many held-out tasks of
//! each behavior are answered on every base (default 1, 0 for none). The dense
//! base decodes slowly on a CPU, so the generated part is most of the run.

#![cfg(feature = "models")]

use antumbra_core::behavior::Spec;
use antumbra_train::grpo::GrpoLm;
use antumbra_train::models::QwenCausalLm;
use antumbra_train::quant::{Drift, Reading};
use antumbra_train::{RaftConfig, ServeQuant};
use candle_core::Device;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// One held-out example of a behavior.
struct Held {
    behavior: usize,
    prompt: String,
    answer: String,
    /// Whether this one is also answered, not only teacher-forced.
    generate: bool,
}

/// What one base made of every held-out example.
struct Pass {
    form: &'static str,
    readings: Vec<Vec<Reading>>,
    /// The generated answer, for the examples that are answered.
    answers: Vec<Option<String>>,
}

async fn pass(
    form: Option<ServeQuant>,
    adapter: &str,
    held: &[Held],
    cfg: &RaftConfig,
) -> Outcome<Pass> {
    let mut model = QwenCausalLm::load(
        Device::Cpu,
        RaftConfig {
            serve_quant: form,
            ..cfg.clone()
        },
    )?;
    model.load_adapter(adapter)?;
    let (mut readings, mut answers) = (Vec::new(), Vec::new());
    for h in held {
        readings.push(model.teacher_forced(&h.prompt, &h.answer)?);
        answers.push(if h.generate {
            let sample = model
                .sample_group(&h.prompt, 1)
                .await?
                .into_iter()
                .next()
                .ok_or("one sample was asked for and none came back")?;
            Some(sample.completion)
        } else {
            None
        });
    }
    Ok(Pass {
        form: form.map_or("f32", ServeQuant::name),
        readings,
        answers,
    })
}

/// The corpus's behaviors, by id, keeping one scope's when a scope is named.
fn behaviors(corpus: &str, scope: Option<&str>) -> Outcome<Vec<(String, Spec)>> {
    let rows: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(corpus)?)?;
    let mut found = Vec::new();
    for row in rows {
        if scope.is_some_and(|s| row.get("scope").and_then(|v| v.as_str()) != Some(s)) {
            continue;
        }
        let id = row
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("a behavior in the corpus has no id")?
            .to_string();
        found.push((id, serde_json::from_value(row)?));
    }
    Ok(found)
}

#[tokio::test]
#[ignore = "live CPU: set ANTUMBRA_QUANT_DRIFT=1, a corpus and an adapter; run with --release --ignored"]
async fn a_quantized_base_is_measured_against_the_dense_one() -> Outcome<()> {
    assert!(
        std::env::var("ANTUMBRA_QUANT_DRIFT").is_ok(),
        "ANTUMBRA_QUANT_DRIFT=1 must be set to run this ignored live measurement"
    );
    let corpus = std::env::var("ANTUMBRA_QUANT_CORPUS")
        .map_err(|_| "ANTUMBRA_QUANT_CORPUS names the behaviors to measure on")?;
    let adapter = std::env::var("ANTUMBRA_CPU_ADAPTER")
        .map_err(|_| "ANTUMBRA_CPU_ADAPTER names the expert to measure")?;
    let scope = std::env::var("ANTUMBRA_QUANT_SCOPE").ok();
    let per_behavior: usize = match std::env::var("ANTUMBRA_QUANT_GENERATE") {
        Ok(n) => n.parse()?,
        Err(_) => 1,
    };

    let behaviors = behaviors(&corpus, scope.as_deref())?;
    assert!(
        !behaviors.is_empty(),
        "no behaviors in the corpus for that scope"
    );

    // The examples admission held out: ground truth the training never saw.
    let mut held = Vec::new();
    for (b, (id, spec)) in behaviors.iter().enumerate() {
        for (i, task) in antumbra_train::behave::tasks(id, spec)
            .held
            .into_iter()
            .enumerate()
        {
            held.push(Held {
                behavior: b,
                prompt: task.prompt,
                answer: task.completion.unwrap_or_default(),
                generate: i < per_behavior,
            });
        }
    }
    let answered = held.iter().filter(|h| h.generate).count();
    eprintln!(
        "{} behaviors, {} held-out examples, {answered} answered on each base",
        behaviors.len(),
        held.len()
    );

    let cfg = RaftConfig {
        temperature: 0.0,
        max_new_tokens: 64,
        ..RaftConfig::default()
    };
    let dense = pass(None, &adapter, &held, &cfg).await?;
    let follows = |b: usize, answer: &str| behaviors[b].1.follows(answer) == Some(true);
    let passed = |p: &Pass| {
        held.iter()
            .zip(&p.answers)
            .filter(|(h, a)| a.as_deref().is_some_and(|a| follows(h.behavior, a)))
            .count()
    };
    eprintln!(
        "RESULT {}: {} of {answered} answers pass their behavior's check",
        dense.form,
        passed(&dense)
    );

    for form in [ServeQuant::Q8_0, ServeQuant::Q6K, ServeQuant::Q4K] {
        let quant = pass(Some(form), &adapter, &held, &cfg).await?;
        let mut drift = Drift::default();
        let mut flipped = vec![0usize; behaviors.len()];
        for ((h, d), q) in held.iter().zip(&dense.readings).zip(&quant.readings) {
            let mut one = Drift::default();
            one.add(d, q);
            flipped[h.behavior] += one.flips();
            drift.add(d, q);
        }
        let same = dense
            .answers
            .iter()
            .zip(&quant.answers)
            .filter(|(d, q)| d.is_some() && d == q)
            .count();
        eprintln!(
            "RESULT {}: {} positions, mean shift {:.4} nats, largest {:.3}, \
             top-1 agreement {:.2}% ({} flips in {} of {} behaviors); \
             {} of {answered} answers pass, {same} identical to the dense base's",
            quant.form,
            drift.positions(),
            drift.mean_shift(),
            drift.largest_shift(),
            drift.top1_agreement() * 100.0,
            drift.flips(),
            flipped.iter().filter(|n| **n > 0).count(),
            behaviors.len(),
            passed(&quant),
        );
        // Each answer that changed how its check came out, either way.
        for ((h, d), q) in held.iter().zip(&dense.answers).zip(&quant.answers) {
            let (Some(d), Some(q)) = (d, q) else { continue };
            let (was, now) = (follows(h.behavior, d), follows(h.behavior, q));
            if was != now {
                eprintln!(
                    "  {} {}: check {} -> {}",
                    quant.form,
                    behaviors[h.behavior].0,
                    if was { "passed" } else { "failed" },
                    if now { "passed" } else { "failed" },
                );
            }
        }
        assert!(drift.positions() > 0, "nothing was compared");
    }
    Ok(())
}
