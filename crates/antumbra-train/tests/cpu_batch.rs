//! Live CPU check of several requests decoded side by side: load the
//! Qwen2.5-Coder base on the CPU, answer the same prompts alone and then 2,
//! 4, 8 and 16 at once, and report what the machine decodes a second and
//! what each request sees. Each answer decoded beside others must be the
//! answer decoded alone.
//!
//! Gated twice over, as `cpu_serve` is. Run it in release:
//!
//!   ANTUMBRA_CPU_BATCH=1 ANTUMBRA_CPU_QUANT=q4_k \
//!     cargo test -p antumbra-train --features models --release --test cpu_batch -- --ignored --nocapture
//!
//! `ANTUMBRA_CPU_ADAPTER` serves an expert; without it the model serves its
//! fresh adapter, which costs the same per token.

#![cfg(feature = "models")]

use antumbra_train::models::QwenCausalLm;
use antumbra_train::{CausalLm, RaftConfig};
use candle_core::Device;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const PROMPTS: [&str; 4] = [
    "Write a Rust function that reverses a singly linked list, with a comment on each line.",
    "Explain, step by step, how to rebase a feature branch onto main and resolve a conflict.",
    "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "Describe how a hash map resolves collisions, and what its load factor is for.",
];

#[tokio::test]
#[ignore = "live CPU: set ANTUMBRA_CPU_BATCH=1 and run with --release --ignored"]
async fn requests_decoded_together_answer_as_alone() -> Outcome<()> {
    assert!(
        std::env::var("ANTUMBRA_CPU_BATCH").is_ok(),
        "ANTUMBRA_CPU_BATCH=1 must be set to run this ignored live CPU test"
    );
    let mut cfg = RaftConfig {
        temperature: 0.0,
        max_new_tokens: 128,
        ..RaftConfig::default()
    };
    if let Ok(quant) = std::env::var("ANTUMBRA_CPU_QUANT") {
        cfg.serve_quant = Some(quant.parse()?);
    }
    let form = cfg
        .serve_quant
        .map_or("f32".to_string(), |q| format!("{q:?}"));
    let mut model = QwenCausalLm::load(Device::Cpu, cfg)?;
    if let Ok(adapter) = std::env::var("ANTUMBRA_CPU_ADAPTER") {
        model.load_adapter(&adapter)?;
    }
    // A first forward pays for work no later one does.
    model.generate(PROMPTS[0], 1).await?;

    // Each prompt alone: the answers every batch is held to.
    let mut alone = Vec::new();
    for prompt in PROMPTS {
        let answer = model
            .generate(prompt, 1)
            .await?
            .into_iter()
            .next()
            .ok_or("one answer was asked for and none came back")?;
        alone.push(answer);
    }

    let mut report = format!("RESULT cpu batch {form}:");
    // `ANTUMBRA_CPU_BATCH_SIZES=1,2,3` answers those many at once instead.
    let sizes: Vec<usize> = match std::env::var("ANTUMBRA_CPU_BATCH_SIZES") {
        Ok(sizes) => sizes
            .split(',')
            .map(|n| n.trim().parse())
            .collect::<Result<_, _>>()?,
        Err(_) => vec![1, 2, 4, 8, 16],
    };
    for together in sizes {
        let prompts: Vec<&str> = (0..together).map(|i| PROMPTS[i % PROMPTS.len()]).collect();
        let answers = model.generate_each(&prompts, 0)?;
        for (i, answer) in answers.iter().enumerate() {
            assert_eq!(
                answer,
                &alone[i % PROMPTS.len()],
                "request {i} of {together} at once"
            );
        }
        let t = model.last_timing().ok_or("the run was not timed")?;
        let node = t.decoded as f64 / t.decode_secs;
        report.push_str(&format!(
            " {together} at once: {node:.1} tok/s decoded ({:.1} each) over {};",
            node / together as f64,
            t.decoded
        ));
    }
    eprintln!("{report}");
    Ok(())
}
