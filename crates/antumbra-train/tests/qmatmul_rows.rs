//! Live timing of candle's quantized matmul on the shapes the Qwen2.5-Coder
//! 1.5B base multiplies by, one row at a time (a decode step) and a prompt's
//! worth of rows at once (prefill), with no model around it.
//!
//! A decoded token is 28 layers of seven projections and one head: 197 calls.
//! Summing what each call costs here, and setting that beside what a token
//! costs in the whole model, says how much of a token is candle's kernels and
//! how much is the loop around them.
//!
//! Ignored, and asserts `ANTUMBRA_CPU_SERVE` is set. Run it in release:
//!
//!   ANTUMBRA_CPU_SERVE=1 cargo test -p antumbra-train --features models --release \
//!     --test qmatmul_rows -- --ignored --nocapture
//!
//! The weights are random: the kernels' speed does not depend on their values.

#![cfg(feature = "models")]

use std::time::Instant;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const HIDDEN: usize = 1536;
const KV: usize = 256;
const FFN: usize = 8960;
const VOCAB: usize = 151_936;
const LAYERS: f64 = 28.0;
/// Rows in the prefill reading: a short chat prompt.
const PROMPT_ROWS: usize = 24;

/// A random `(out, in)` weight held as `dtype`, ready to multiply.
fn weight(out: usize, inn: usize, dtype: GgmlDType) -> Outcome<QMatMul> {
    let dense = Tensor::randn(0f32, 0.02, (out, inn), &Device::Cpu)?;
    let q = QTensor::quantize_onto(&dense, dtype, &Device::Cpu)?;
    Ok(QMatMul::from_qtensor(q)?)
}

/// Milliseconds one call takes on `rows` rows, the mean of `reps` after a
/// warm-up call.
fn call_ms(w: &QMatMul, inn: usize, rows: usize, reps: usize) -> Outcome<f64> {
    let x = Tensor::randn(0f32, 1.0, (1, rows, inn), &Device::Cpu)?;
    w.forward(&x)?;
    let start = Instant::now();
    for _ in 0..reps {
        w.forward(&x)?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / reps as f64)
}

#[test]
#[ignore = "live CPU: set ANTUMBRA_CPU_SERVE=1 and run with --release --ignored"]
fn the_quantized_kernels_are_timed_on_the_base_shapes() -> Outcome<()> {
    assert!(
        std::env::var("ANTUMBRA_CPU_SERVE").is_ok(),
        "ANTUMBRA_CPU_SERVE=1 must be set to run this ignored live CPU test"
    );
    // (name, out, in, how many of them a layer has)
    let shapes = [
        ("q/o", HIDDEN, HIDDEN, 2.0),
        ("k/v", KV, HIDDEN, 2.0),
        ("gate/up", FFN, HIDDEN, 2.0),
        ("down", HIDDEN, FFN, 1.0),
    ];
    for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0] {
        let (mut decode, mut prefill) = (0.0, 0.0);
        for (name, out, inn, per_layer) in shapes {
            let w = weight(out, inn, dtype)?;
            let one = call_ms(&w, inn, 1, 60)?;
            let many = call_ms(&w, inn, PROMPT_ROWS, 20)?;
            eprintln!(
                "{dtype:?} {name} ({out}x{inn}): {one:.3} ms a row alone, \
                 {:.3} ms a row in {PROMPT_ROWS}",
                many / PROMPT_ROWS as f64
            );
            decode += LAYERS * per_layer * one;
            prefill += LAYERS * per_layer * many / PROMPT_ROWS as f64;
        }
        // The head is held as Q8_0 whatever the body is, and reads one row in
        // either phase: only the last position's logits are taken.
        let head = call_ms(&weight(VOCAB, HIDDEN, GgmlDType::Q8_0)?, HIDDEN, 1, 20)?;
        eprintln!("{dtype:?} head ({VOCAB}x{HIDDEN}, Q8_0): {head:.3} ms a row");
        eprintln!(
            "RESULT {dtype:?}: a decoded token is {:.1} ms of kernels ({:.1} body + {head:.1} head), \
             {:.1} tok/s if nothing else cost anything; a prompt token is {:.2} ms of body",
            decode + head,
            decode,
            1000.0 / (decode + head),
            prefill,
        );
    }
    Ok(())
}
