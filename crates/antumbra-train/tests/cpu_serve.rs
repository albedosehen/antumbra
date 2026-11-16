//! Live CPU check of the serving path: load the Qwen2.5-Coder base on the CPU,
//! restore an expert trained in bf16 on the GPU, and generate, reporting the
//! two phases apart (reading the prompt, then each token after the first) for
//! the bare base, at one expert's rank and at a four-expert blend's.
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

use antumbra_train::models::{GenTiming, QwenCausalLm};
use antumbra_train::{CausalLm, RaftConfig};
use candle_core::Device;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

// The last two ask for long answers, so the decode rate is over enough tokens
// to mean something.
const PROMPTS: [&str; 4] = [
    "Write a one-line Rust function that adds two i64 values.",
    "Count the lines in notes.txt. Command only.",
    "Write a Rust function that reverses a singly linked list, with a comment on each line.",
    "Explain, step by step, how to rebase a feature branch onto main and resolve a conflict.",
];

/// Both phases summed over every prompt, and the first prompt's answer.
struct Run {
    timing: GenTiming,
    first: String,
}

impl Run {
    /// Prompt tokens read a second.
    fn prefill(&self) -> f64 {
        self.timing.prompt_tokens as f64 / self.timing.prompt_secs
    }

    /// Tokens decoded a second, after each answer's first.
    fn decode(&self) -> f64 {
        self.timing.decoded as f64 / self.timing.decode_secs
    }
}

/// Greedy generation over every prompt.
async fn generate_all(model: &mut QwenCausalLm) -> Outcome<Run> {
    let (mut timing, mut first) = (GenTiming::default(), String::new());
    for p in PROMPTS {
        let answer = model
            .generate(p, 1)
            .await?
            .into_iter()
            .next()
            .ok_or("one answer was asked for and none came back")?;
        let t = model.last_timing().ok_or("the generation was not timed")?;
        timing.prompt_tokens += t.prompt_tokens;
        timing.prompt_secs += t.prompt_secs;
        timing.decoded += t.decoded;
        timing.decode_secs += t.decode_secs;
        if first.is_empty() {
            first = answer;
        }
    }
    Ok(Run { timing, first })
}

#[tokio::test]
#[ignore = "live CPU: set ANTUMBRA_CPU_SERVE=1 and run with --release --ignored"]
async fn an_expert_serves_on_the_cpu() -> Outcome<()> {
    assert!(
        std::env::var("ANTUMBRA_CPU_SERVE").is_ok(),
        "ANTUMBRA_CPU_SERVE=1 must be set to run this ignored live CPU test"
    );
    let adapter = std::env::var("ANTUMBRA_CPU_ADAPTER").ok();
    let mut cfg = RaftConfig {
        temperature: 0.0,
        max_new_tokens: 128,
        ..RaftConfig::default()
    };
    if let Ok(base) = std::env::var("ANTUMBRA_CPU_BASE") {
        cfg.base_model = base;
    }
    if let Ok(quant) = std::env::var("ANTUMBRA_CPU_QUANT") {
        cfg.serve_quant = Some(quant.parse()?);
    }
    let form = cfg
        .serve_quant
        .map_or("f32".to_string(), |q| format!("{q:?}"));
    eprintln!(
        "base {} ({form}) on the CPU, adapter {}",
        cfg.base_model,
        adapter.as_deref().unwrap_or("none (fresh)")
    );

    // One expert, at the rank it was trained at, through load_adapter: a bf16
    // expert into the f32 model the CPU holds.
    let start = Instant::now();
    let mut one = QwenCausalLm::load(Device::Cpu, cfg.clone())?;
    if let Some(a) = &adapter {
        one.load_adapter(a)?;
    }
    let load_secs = start.elapsed().as_secs_f64();
    let own = generate_all(&mut one).await?;
    eprintln!("rank {}: {:?}", cfg.lora_rank, own.first);
    assert!(own.timing.decoded > 0, "the expert decoded no tokens");

    // The same model with the adapter's delta off: what the base alone costs
    // a token, and so what an adapter adds to it.
    one.set_adapter(false);
    let bare = generate_all(&mut one).await?;
    drop(one);
    assert!(bare.timing.decoded > 0, "the bare base decoded no tokens");

    // A four-expert blend's rank: four quarter-weight copies of the expert,
    // through load_blend, which add up to the expert itself.
    let blend_rank = cfg.lora_rank * 4;
    let blend_cfg = RaftConfig {
        lora_rank: blend_rank,
        lora_alpha: cfg.lora_alpha * 4.0,
        ..cfg.clone()
    };
    let mut four = QwenCausalLm::load(Device::Cpu, blend_cfg)?;
    if let Some(a) = &adapter {
        let specs = vec![(a.clone(), 0.25f32); 4];
        four.load_blend(&specs)?;
    }
    let blend = generate_all(&mut four).await?;
    eprintln!("rank {blend_rank}: {:?}", blend.first);
    assert!(blend.timing.decoded > 0, "the blend decoded no tokens");

    eprintln!(
        "RESULT cpu {form}: load {load_secs:.1}s; \
         bare base: prefill {:.1} tok/s, decode {:.2} tok/s over {}; \
         rank {}: prefill {:.1} tok/s over {} tokens, decode {:.2} tok/s over {}; \
         rank {blend_rank}: prefill {:.1} tok/s, decode {:.2} tok/s over {}",
        bare.prefill(),
        bare.decode(),
        bare.timing.decoded,
        cfg.lora_rank,
        own.prefill(),
        own.timing.prompt_tokens,
        own.decode(),
        own.timing.decoded,
        blend.prefill(),
        blend.decode(),
        blend.timing.decoded,
    );
    Ok(())
}
