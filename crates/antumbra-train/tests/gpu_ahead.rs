//! Live check that what is read ahead on the best device decides nothing:
//! two models that read ahead answer the same prompts side by side, one
//! that has answered nothing before and one that has answered every prompt
//! already, so that each is guessed for differently. Their answers are
//! asserted to be the same. A third that does not read ahead answers them
//! too, for its rate and for how many of its answers are worded otherwise.
//!
//! On a CUDA device a model that reads ahead decodes through passes of one
//! width whatever is guessed, which is what makes this hold there; it rests
//! on the device's library computing a row the same way whatever is beside
//! it, which the model asks of the device before it reads that way. This is
//! the check of the whole of it on a real base. A model that does not read
//! ahead decodes a row a pass, which that library rounds another way, so
//! its answers are not asserted to be the readers'.
//!
//! Gated twice over, as the other live checks are: the file only compiles
//! under `--features models`, and the test is `#[ignore]`d and asserts that
//! `ANTUMBRA_GPU_AHEAD` is set. On a GPU host, in release:
//!
//!   ANTUMBRA_GPU_AHEAD=1 [ANTUMBRA_GPU_AHEAD_ADAPTER=/path/to/expert.safetensors] \
//!     cargo test -p antumbra-train --features cuda --release --test gpu_ahead -- --ignored --nocapture
//!
//! Without a GPU backend it runs on the CPU from the dense base, where
//! nothing is read ahead and the three decode alike.

#![cfg(feature = "models")]

use antumbra_train::models::QwenCausalLm;
use antumbra_train::{CausalLm, RaftConfig};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const PROMPTS: [&str; 8] = [
    "Write a one-line Rust function that adds two i64 values.",
    "Count the lines in notes.txt. Command only.",
    "Write a Rust function that reverses a singly linked list, with a comment on each line.",
    "Explain, step by step, how to rebase a feature branch onto main and resolve a conflict.",
    "Write a one-line Rust function that adds two u32 values.",
    "Count the words in notes.txt. Command only.",
    "Show disk usage on this machine. Command only.",
    "Create a git branch for issue #12, \"Add search\". Command only.",
];

/// What one model wrote for every prompt, and what it took.
#[derive(Default)]
struct Run {
    answers: Vec<String>,
    decoded: usize,
    passes: usize,
    decode_secs: f64,
}

impl Run {
    async fn of(model: &mut QwenCausalLm) -> Outcome<Self> {
        let mut run = Self::default();
        for prompt in PROMPTS {
            let answer = model
                .generate(prompt, 1)
                .await?
                .into_iter()
                .next()
                .ok_or("one answer was asked for and none came back")?;
            let timing = model.last_timing().ok_or("the generation was not timed")?;
            run.decoded += timing.decoded;
            run.passes += timing.passes;
            run.decode_secs += timing.decode_secs;
            run.answers.push(answer);
        }
        Ok(run)
    }

    fn told(&self, what: &str) {
        eprintln!(
            "{what}: {} tokens in {} passes ({:.2} a pass), {:.1} tok/s, {:.2} ms a pass",
            self.decoded,
            self.passes,
            self.decoded as f64 / self.passes.max(1) as f64,
            self.decoded as f64 / self.decode_secs,
            self.decode_secs * 1e3 / self.passes.max(1) as f64,
        );
    }
}

#[tokio::test]
#[ignore = "live GPU: set ANTUMBRA_GPU_AHEAD=1 and run with --release --ignored"]
async fn what_is_read_ahead_on_the_best_device_decides_nothing() -> Outcome<()> {
    assert!(
        std::env::var("ANTUMBRA_GPU_AHEAD").is_ok(),
        "ANTUMBRA_GPU_AHEAD=1 must be set to run this ignored live test"
    );
    let adapter = std::env::var("ANTUMBRA_GPU_AHEAD_ADAPTER").ok();
    let load = |ahead: bool| -> Outcome<QwenCausalLm> {
        let cfg = RaftConfig {
            serve_quant: None,
            read_ahead: ahead,
            ..RaftConfig::for_serving(96, 0.0)
        };
        let mut model = QwenCausalLm::load(antumbra_train::device::best_device()?, cfg)?;
        match &adapter {
            Some(adapter) => model.load_adapter(adapter)?,
            None => model.set_adapter(false),
        }
        Ok(model)
    };
    let (mut alone, mut cold, mut warm) = (load(false)?, load(true)?, load(true)?);
    // The warm reader has answered every prompt once: asked again, it is
    // guessed for from the whole of what it wrote.
    let first = Run::of(&mut warm).await?;
    let alone = Run::of(&mut alone).await?;
    let cold = Run::of(&mut cold).await?;
    let again = Run::of(&mut warm).await?;
    alone.told("not read ahead");
    cold.told("read ahead, nothing remembered");
    again.told("read ahead, every prompt answered before");
    let worded_otherwise = alone
        .answers
        .iter()
        .zip(&cold.answers)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "{worded_otherwise} of {} answers not read ahead are worded otherwise",
        PROMPTS.len()
    );
    assert_eq!(
        first.answers, cold.answers,
        "two readers that had answered nothing wrote different answers"
    );
    assert_eq!(
        again.answers, cold.answers,
        "a reader that had answered the prompt before wrote another answer"
    );
    // Asked again, a reader that reads ahead here takes fewer passes than
    // tokens; one that does not takes as many.
    assert!(again.passes <= again.decoded);
    Ok(())
}
