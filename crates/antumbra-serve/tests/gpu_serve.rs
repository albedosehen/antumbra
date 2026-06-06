//! Live GPU end-to-end validation of the real resident `MultiAdapterServe` (R-4):
//! load the shared Qwen2.5-Coder-1.5B-Instruct base once, hot-swap a trained LoRA
//! adapter onto it, and run a real generation -- the path the MCP `answer` tool
//! drives, proven only with the `EchoServe` fake until now.
//!
//! Gated twice over so the normal suite never touches a GPU or downloads weights:
//! the file only compiles under `--features models`, and the body skips unless
//! `ANTUMBRA_GPU_SERVE` is set. Reproduce on the 3090 Ti (CUDA 13.x toolkit):
//!
//!   # in a VS Dev Shell, with the CUDA bin on PATH:
//!   $env:CUDARC_CUDA_VERSION = "13000"   # 13.0 bindings link against a 13.3 runtime
//!   $env:ANTUMBRA_GPU_SERVE  = "1"
//!   cargo test -p antumbra-serve --features cuda --test gpu_serve -- --nocapture
//!
//! Optional: `ANTUMBRA_GPU_ADAPTER` (default adapters/run_train_g0.safetensors),
//! `ANTUMBRA_GPU_BASE` (default Qwen/Qwen2.5-Coder-1.5B-Instruct). The base is
//! pulled from the HF hub on first run (~3 GB).

#![cfg(feature = "models")]

use antumbra_core::ports::{ActRequest, Serve};
use antumbra_core::ExpertId;
use antumbra_serve::{MultiAdapterServe, RaftConfig};

// MultiAdapterServe runs candle generation under block_in_place, so it needs a
// multi-threaded runtime (a replacement worker takes over while a request blocks).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_adapter_serve_generates_on_gpu() {
    if std::env::var("ANTUMBRA_GPU_SERVE").is_err() {
        eprintln!("skipped: set ANTUMBRA_GPU_SERVE=1 to run the live GPU serve validation");
        return;
    }
    let base = std::env::var("ANTUMBRA_GPU_BASE")
        .unwrap_or_else(|_| "Qwen/Qwen2.5-Coder-1.5B-Instruct".into());
    // Default to the workspace-root adapters dir; `cargo test` runs with the
    // crate dir as CWD, so anchor on CARGO_MANIFEST_DIR (crates/antumbra-serve).
    let adapter = std::env::var("ANTUMBRA_GPU_ADAPTER").unwrap_or_else(|_| {
        format!(
            "{}/../../adapters/run_train_g0.safetensors",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    assert!(
        std::path::Path::new(&adapter).exists(),
        "adapter file must exist (set ANTUMBRA_GPU_ADAPTER): {adapter}"
    );

    // Serve the learned mode: greedy + repetition penalty + n-gram block + nucleus.
    let cfg = RaftConfig::for_serving(48, 0.0);
    let expert = ExpertId::new("expert:gpu-probe");
    let mut engine = MultiAdapterServe::new(&base, cfg);
    engine.register(expert.clone(), &adapter);
    assert_eq!(engine.len(), 1, "one adapter registered");

    // First act(): loads the base resident, hot-swaps the adapter, generates.
    let out = engine
        .act(ActRequest {
            task_id: "gpu-probe".into(),
            prompt: "Write a one-line Rust function that adds two i64 values.".into(),
            adapters: vec![expert.clone()],
        })
        .await
        .expect("real GPU generation succeeds");

    assert!(
        !out.final_output.trim().is_empty(),
        "the resident base + adapter produced output"
    );
    assert_eq!(
        out.steps.len(),
        1,
        "v0 serves the top expert as a single step"
    );

    // A second act() to the same expert reuses the resident factors (no reload):
    // proves the hot-path is the O(adapter) swap, not an O(base) reload.
    let again = engine
        .act(ActRequest {
            task_id: "gpu-probe-2".into(),
            prompt: "Now write one that multiplies them.".into(),
            adapters: vec![expert],
        })
        .await
        .expect("second generation on the resident base succeeds");
    assert!(!again.final_output.trim().is_empty());

    eprintln!(
        "RESULT: PASS - MultiAdapterServe generated on GPU (base={base})\n--- first ---\n{}\n--- second ---\n{}",
        out.final_output, again.final_output
    );
}
