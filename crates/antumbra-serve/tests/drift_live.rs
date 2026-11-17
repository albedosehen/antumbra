//! Live check of the measurement an admitted training ends with: a real
//! expert read through the base it was trained over and through that base
//! quantized, as `behave::train` reads the one it has just admitted.
//!
//! Gated twice over, as the other live checks are: the file only compiles
//! under `--features models`, and the test is `#[ignore]`d and needs an
//! adapter named. It loads the base once for the reference and once for each
//! form, so run it in release:
//!
//!   ANTUMBRA_DRIFT_ADAPTER=/path/to/expert.safetensors \
//!     cargo test -p antumbra-serve --features models --release --test drift_live -- --ignored --nocapture
//!
//! `ANTUMBRA_BEHAVE_DRIFT` names the forms, as the server's setting does
//! (`auto` is Q8_0). The examples here are a handful of this file's own: the
//! check is of the path, and what a form does to a user's expert is read off
//! that expert's card.

#![cfg(feature = "models")]

use antumbra_core::behavior::Recipe;
use antumbra_serve::behave::{drift, standing_config};
use antumbra_serve::ServeQuant;
use antumbra_train::{CandleModelLoader, CorpusTask, ModelLoader};

/// Tasks with the answer a behavior would expect, as held-out examples carry.
const EXAMPLES: [(&str, &str); 4] = [
    (
        "Write a one-line Rust function that adds two i64 values.",
        "fn add(a: i64, b: i64) -> i64 { a + b }",
    ),
    (
        "Name the command that installs a package with bun.",
        "bun add <package>",
    ),
    (
        "Write a Rust expression that parses the string s as a u32, or zero.",
        "s.parse::<u32>().unwrap_or(0)",
    ),
    (
        "Say in one sentence what a unit test is.",
        "A unit test checks one small piece of code on its own.",
    ),
];

fn held() -> Vec<CorpusTask> {
    EXAMPLES
        .iter()
        .enumerate()
        .map(|(n, (prompt, answer))| CorpusTask {
            id: format!("example#{n}"),
            prompt: prompt.to_string(),
            verify: serde_json::Value::Null,
            completion: Some(answer.to_string()),
            skill: None,
            scope: None,
            impossible: false,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "live: set ANTUMBRA_DRIFT_ADAPTER to an expert's adapter, run with --release --ignored"]
async fn an_admitted_expert_is_measured_on_each_form_asked_for(
) -> Result<(), Box<dyn std::error::Error>> {
    let adapter = std::env::var("ANTUMBRA_DRIFT_ADAPTER")
        .map_err(|_| "ANTUMBRA_DRIFT_ADAPTER names the expert to measure")?;
    let setting = std::env::var("ANTUMBRA_BEHAVE_DRIFT").unwrap_or_else(|_| "auto".into());
    let forms = ServeQuant::drift_from_setting(&setting)?;
    // The expert as a training holds it when it is admitted: the dense base
    // with its adapter, at the rank standing experts are trained at.
    let cfg = standing_config(&Recipe::default());
    let loader = CandleModelLoader::new(cfg.clone());
    let model = ModelLoader::load(&loader, &cfg.base_model, Some(adapter.as_str())).await?;

    let measured = drift::measure(model, &cfg, &adapter, &held(), &forms).await;

    eprintln!("{}", drift::describe(&measured));
    assert_eq!(
        measured.iter().map(|(form, _)| *form).collect::<Vec<_>>(),
        forms,
        "every form asked for was measured"
    );
    for (form, found) in &measured {
        assert!(
            found.positions() > 0,
            "{}: no position was compared",
            form.name()
        );
        assert!(found.mean_shift().is_finite() && found.largest_shift() >= found.mean_shift());
        assert!(found.flips() <= found.positions());
    }
    Ok(())
}
