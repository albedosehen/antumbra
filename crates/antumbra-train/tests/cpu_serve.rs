//! Live CPU check of the serving path: load the Qwen2.5-Coder base on the CPU,
//! restore an expert trained in bf16 on the GPU, and generate, reporting tokens
//! per second at one expert's rank and at a four-expert blend's.
//!
//! Gated twice over, as the GPU checks are: the file only compiles under
//! `--features models`, and the test is `#[ignore]`d and asserts that
//! `ANTUMBRA_CPU_SERVE` is set. Run it in release, or the numbers mean nothing:
//!
//!   ANTUMBRA_CPU_SERVE=1 ANTUMBRA_CPU_ADAPTER=/path/to/expert.safetensors \
//!     cargo test -p antumbra-train --features models --release --test cpu_serve -- --ignored --nocapture
//!
//! Without `ANTUMBRA_CPU_ADAPTER` the model serves its fresh adapter, which
//! costs the same per token. `ANTUMBRA_CPU_QUANT` (`q8_0` or `q4_k`) loads the
//! base to serve in that form; without it the base is dense f32.
//! `ANTUMBRA_CPU_BASE` overrides the base, which is pulled from the HF hub on
//! first run (~3 GB). The model is built on the CPU even in a `cuda` build:
//! this measures the CPU, not the best device.

#![cfg(feature = "models")]

use std::time::Instant;

use antumbra_train::grpo::GrpoLm;
use antumbra_train::models::QwenCausalLm;
use antumbra_train::RaftConfig;
use candle_core::Device;

const PROMPTS: [&str; 3] = [
    "Write a one-line Rust function that adds two i64 values.",
    "Count the lines in notes.txt. Command only.",
    "Create a git branch for issue #12, \"Add search\". Command only.",
];

/// Greedy generation over every prompt: the tokens produced, the seconds
/// taken, and the first prompt's answer.
async fn generate_all(model: &mut QwenCausalLm) -> (usize, f64, String) {
    let (mut tokens, mut secs, mut first) = (0, 0.0, String::new());
    for p in PROMPTS {
        let start = Instant::now();
        let sample = model
            .sample_group(p, 1)
            .await
            .expect("generates on the CPU")
            .remove(0);
        secs += start.elapsed().as_secs_f64();
        tokens += sample.tokens.len();
        if first.is_empty() {
            first = sample.completion;
        }
    }
    (tokens, secs, first)
}

#[tokio::test]
#[ignore = "live CPU: set ANTUMBRA_CPU_SERVE=1 and run with --release --ignored"]
async fn an_expert_serves_on_the_cpu() {
    assert!(
        std::env::var("ANTUMBRA_CPU_SERVE").is_ok(),
        "ANTUMBRA_CPU_SERVE=1 must be set to run this ignored live CPU test"
    );
    let adapter = std::env::var("ANTUMBRA_CPU_ADAPTER").ok();
    let mut cfg = RaftConfig {
        temperature: 0.0,
        max_new_tokens: 64,
        ..RaftConfig::default()
    };
    if let Ok(base) = std::env::var("ANTUMBRA_CPU_BASE") {
        cfg.base_model = base;
    }
    if let Ok(quant) = std::env::var("ANTUMBRA_CPU_QUANT") {
        cfg.serve_quant = Some(quant.parse().expect("ANTUMBRA_CPU_QUANT is q8_0 or q4_k"));
    }
    let form = cfg
        .serve_quant
        .map_or("f32".to_string(), |q| format!("{q:?}"));
    let threads = std::thread::available_parallelism().map_or(0, |n| n.get());
    eprintln!(
        "base {} ({form}) on the CPU ({threads} threads), adapter {}",
        cfg.base_model,
        adapter.as_deref().unwrap_or("none (fresh)")
    );

    // One expert, at the rank it was trained at, through load_adapter.
    let start = Instant::now();
    let mut one = QwenCausalLm::load(Device::Cpu, cfg.clone()).expect("the base loads");
    if let Some(a) = &adapter {
        one.load_adapter(a)
            .expect("a bf16 expert loads into the f32 model");
    }
    let load_secs = start.elapsed().as_secs_f64();
    let (tokens, secs, answer) = generate_all(&mut one).await;
    drop(one);
    eprintln!("rank {}: {answer:?}", cfg.lora_rank);
    assert!(tokens > 0, "the expert produced no tokens");

    // A four-expert blend's rank: four quarter-weight copies of the expert,
    // through load_blend, which add up to the expert itself.
    let blend_rank = cfg.lora_rank * 4;
    let blend_cfg = RaftConfig {
        lora_rank: blend_rank,
        lora_alpha: cfg.lora_alpha * 4.0,
        ..cfg.clone()
    };
    let mut four = QwenCausalLm::load(Device::Cpu, blend_cfg).expect("the base loads");
    if let Some(a) = &adapter {
        let specs = vec![(a.clone(), 0.25f32); 4];
        four.load_blend(&specs).expect("the blend loads");
    }
    let (blend_tokens, blend_secs, blend_answer) = generate_all(&mut four).await;
    eprintln!("rank {blend_rank}: {blend_answer:?}");
    assert!(blend_tokens > 0, "the blend produced no tokens");

    eprintln!(
        "RESULT cpu {form}, {threads} threads: load {load_secs:.1}s; \
         rank {}: {tokens} tokens in {secs:.1}s = {:.2} tok/s; \
         rank {blend_rank}: {blend_tokens} tokens in {blend_secs:.1}s = {:.2} tok/s",
        cfg.lora_rank,
        tokens as f64 / secs,
        blend_tokens as f64 / blend_secs,
    );
}
