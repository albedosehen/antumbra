//! Live CPU check of the engine answering requests together: several
//! requests at once, of two blends, each answered as it is answered alone,
//! and how long the lot takes beside one after another.
//!
//! The two experts are one fresh adapter saved under two names: two blends
//! to the engine, which tells them apart by file. Fresh factors add nothing,
//! so this checks the engine's bookkeeping and its timing, not that two
//! trained experts stay apart, which the model's own tests hold to the bit.
//!
//! Gated twice over, as `cpu_engine` is. Run it in release on a build with
//! no GPU backend:
//!
//!   ANTUMBRA_CPU_SERVE=1 ANTUMBRA_SERVE_QUANT=q4_k \
//!     cargo test -p antumbra-serve --features models --release --test cpu_together -- --ignored --nocapture

#![cfg(feature = "models")]

use std::sync::Arc;
use std::time::Instant;

use antumbra_core::ports::{ActRequest, Serve};
use antumbra_core::ExpertId;
use antumbra_serve::{MultiAdapterServe, RaftConfig, ServeQuant};
use antumbra_train::CausalLm;

const PROMPTS: [&str; 4] = [
    "Write a Rust function that reverses a singly linked list, with a comment on each line.",
    "Explain, step by step, how to rebase a feature branch onto main and resolve a conflict.",
    "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "Describe how a hash map resolves collisions, and what its load factor is for.",
];

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "live CPU: set ANTUMBRA_CPU_SERVE=1, run with --release --ignored"]
async fn the_engine_answers_requests_together_as_it_answers_them_alone(
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(
        std::env::var("ANTUMBRA_CPU_SERVE").is_ok(),
        "ANTUMBRA_CPU_SERVE=1 must be set to run this ignored live CPU test"
    );
    let setting = std::env::var("ANTUMBRA_SERVE_QUANT").unwrap_or_else(|_| "auto".into());
    let quant = ServeQuant::from_setting(&setting)?;
    let cfg = RaftConfig {
        serve_quant: quant,
        ..RaftConfig::for_serving(96, 0.0)
    };

    // A fresh adapter at the trained rank, saved under two names.
    let dir = std::env::temp_dir().join(format!("antumbra-cpu-together-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let files = [dir.join("one.safetensors"), dir.join("two.safetensors")];
    {
        let fresh =
            antumbra_train::models::QwenCausalLm::load(candle_core::Device::Cpu, cfg.clone())?;
        for file in &files {
            fresh.save_adapter(&file.to_string_lossy())?;
        }
    }
    let experts = [ExpertId::new("expert:one"), ExpertId::new("expert:two")];
    let engine = Arc::new(
        MultiAdapterServe::new(&cfg.base_model, cfg.clone())
            .with_adapter(experts[0].clone(), files[0].to_string_lossy())
            .with_adapter(experts[1].clone(), files[1].to_string_lossy()),
    );

    // Each request by itself, one after another: the answers the rest are
    // held to, and the time to beat. The first pays the base's load, and is
    // asked for again after it.
    engine
        .act(ActRequest::new(
            "warm",
            PROMPTS[0],
            vec![experts[0].clone()],
        ))
        .await?;
    let requests: Vec<(usize, usize)> = (0..8).map(|i| (i % PROMPTS.len(), i % 2)).collect();
    let started = Instant::now();
    let mut alone = Vec::new();
    for (i, (prompt, expert)) in requests.iter().enumerate() {
        let answer = engine
            .act(ActRequest::new(
                format!("alone-{i}"),
                PROMPTS[*prompt],
                vec![experts[*expert].clone()],
            ))
            .await?
            .final_output;
        alone.push(answer);
    }
    let one_by_one = started.elapsed().as_secs_f64();

    // The same eight at once.
    let started = Instant::now();
    let mut asked = Vec::new();
    for (i, (prompt, expert)) in requests.iter().enumerate() {
        let (engine, expert) = (engine.clone(), experts[*expert].clone());
        let prompt = PROMPTS[*prompt];
        asked.push(tokio::spawn(async move {
            engine
                .act(ActRequest::new(
                    format!("together-{i}"),
                    prompt,
                    vec![expert],
                ))
                .await
        }));
    }
    let mut together = Vec::new();
    for asked in asked {
        together.push(asked.await??.final_output);
    }
    let at_once = started.elapsed().as_secs_f64();
    for (i, (got, want)) in together.iter().zip(&alone).enumerate() {
        assert_eq!(got, want, "request {i} of eight at once");
    }

    // The eight again, and the callers of every other one hang up a moment
    // in: the four that stay are answered as alone, and sooner than when
    // all eight were answered, since nothing is decoded for the four that
    // left.
    let started = Instant::now();
    let mut asked = Vec::new();
    for (i, (prompt, expert)) in requests.iter().enumerate() {
        let (engine, expert) = (engine.clone(), experts[*expert].clone());
        let prompt = PROMPTS[*prompt];
        asked.push(tokio::spawn(async move {
            engine
                .act(ActRequest::new(format!("some-{i}"), prompt, vec![expert]))
                .await
        }));
    }
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    for asked in asked.iter().skip(1).step_by(2) {
        asked.abort();
    }
    for (i, asked) in asked.into_iter().enumerate().step_by(2) {
        assert_eq!(
            asked.await??.final_output,
            alone[i],
            "request {i}, with four callers gone"
        );
    }
    let with_four_gone = started.elapsed().as_secs_f64();
    assert!(
        with_four_gone < at_once,
        "four of eight answered in {with_four_gone:.2}s, all eight in {at_once:.2}s"
    );
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "RESULT cpu together {}: eight requests of two blends one after another {one_by_one:.2}s, \
         at once {at_once:.2}s ({:.2} times as fast), every answer as alone; with four callers gone \
         after 0.2s the other four in {with_four_gone:.2}s",
        quant.map_or("dense", ServeQuant::name),
        one_by_one / at_once
    );
    Ok(())
}
