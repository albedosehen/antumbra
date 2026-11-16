//! Live CPU check of the resident engine as a node with no GPU runs it: the
//! serving config a CPU build gets (a Q8_0 base unless `ANTUMBRA_SERVE_QUANT`
//! says otherwise), an expert trained in bf16 on the GPU hot-swapped onto it,
//! one answer, then the same expert as two half-weight copies.
//!
//! Gated twice over, as the GPU check is: the file only compiles under
//! `--features models`, and the test is `#[ignore]`d and asserts that
//! `ANTUMBRA_CPU_SERVE` is set. Run it in release on a build with no GPU
//! backend:
//!
//!   ANTUMBRA_CPU_SERVE=1 ANTUMBRA_CPU_ADAPTER=/path/to/expert.safetensors \
//!     cargo test -p antumbra-serve --features models --release --test cpu_engine -- --ignored --nocapture
//!
//! `ANTUMBRA_CPU_BASE` overrides the base, pulled from the HF hub on first run
//! (~3 GB).

#![cfg(feature = "models")]

use std::time::Instant;

use antumbra_core::ports::{ActRequest, Serve};
use antumbra_core::ExpertId;
use antumbra_serve::{MultiAdapterServe, RaftConfig, ServeQuant};

const PROMPT: &str = "Write a one-line Rust function that adds two i64 values.";

// MultiAdapterServe generates under block_in_place: a multi-threaded runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "live CPU: set ANTUMBRA_CPU_SERVE=1 and ANTUMBRA_CPU_ADAPTER, run with --release --ignored"]
async fn the_engine_serves_an_expert_on_a_node_with_no_gpu() {
    assert!(
        std::env::var("ANTUMBRA_CPU_SERVE").is_ok(),
        "ANTUMBRA_CPU_SERVE=1 must be set to run this ignored live CPU test"
    );
    let adapter = std::env::var("ANTUMBRA_CPU_ADAPTER")
        .expect("ANTUMBRA_CPU_ADAPTER names the expert to serve");
    let setting = std::env::var("ANTUMBRA_SERVE_QUANT").unwrap_or_else(|_| "auto".into());
    let quant = ServeQuant::from_setting(&setting).expect("a serving quantization");
    let cfg = RaftConfig {
        serve_quant: quant,
        ..RaftConfig::for_serving(48, 0.0)
    };
    let base = std::env::var("ANTUMBRA_CPU_BASE").unwrap_or_else(|_| cfg.base_model.clone());
    eprintln!(
        "base {base}, held {}",
        quant.map_or("dense", ServeQuant::name)
    );

    let expert = ExpertId::new("expert:cpu-probe");
    let engine = MultiAdapterServe::new(&base, cfg).with_adapter(expert.clone(), &adapter);

    // The first answer pays the base load; the second reuses the resident base
    // and swaps only the factors.
    let start = Instant::now();
    let one = engine
        .act(ActRequest::new("one", PROMPT, vec![expert.clone()]))
        .await
        .expect("serves one expert on the CPU")
        .final_output;
    let first = start.elapsed().as_secs_f64();
    assert!(!one.trim().is_empty(), "the expert produced no answer");

    let mut halves = ActRequest::new("halves", PROMPT, vec![expert.clone(), expert.clone()]);
    halves.weights = vec![0.5, 0.5];
    let start = Instant::now();
    let two = engine
        .act(halves)
        .await
        .expect("serves a blend on the CPU")
        .final_output;
    let second = start.elapsed().as_secs_f64();

    eprintln!("one expert: {one:?}\ntwo halves: {two:?}");
    eprintln!("RESULT first answer {first:.1}s (base load included), second {second:.1}s");
    assert_eq!(two, one, "two half-weight copies answer as the expert does");
}
