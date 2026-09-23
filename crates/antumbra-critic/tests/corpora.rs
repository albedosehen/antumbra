//! The shipped corpora, judged by the verifier the trainer actually uses.
//!
//! The generator checks every workbench task through a Python mirror of this
//! verifier. These tests close the gap between the mirror and the real thing:
//! the fence extraction, the environment variable and the exit status all come
//! from `CommandVerifier` here. They need a real Python, so they are ignored by
//! default:
//!
//!     ANTUMBRA_PYTHON=<python> cargo test -p antumbra-critic --test corpora -- --ignored

use std::path::{Path, PathBuf};

use antumbra_core::ports::{Verifier, VerifyRequest};
use antumbra_core::{Result, RunId};
use antumbra_critic::CommandVerifier;
use serde_json::{json, Value};
use tokio::task::JoinSet;

/// Ways to exit 0 without defining anything. Every one of them passed every
/// Python verifier in corpora/ before those verifiers moved to the workbench
/// judge, and none may pass any verifier now.
const EXITS: [&str; 4] = [
    "raise SystemExit",
    "import os\nos._exit(0)\n",
    "```python\nimport sys\nsys.exit(0)\n```",
    "import atexit, os\natexit.register(lambda: os._exit(0))\n",
];

fn corpora() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpora")
}

fn read(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| antumbra_core::AntumbraError::other(format!("{}: {e}", path.display())))?;
    Ok(serde_json::from_str(&text)?)
}

/// Every `verify` object in a corpus file, wherever it sits: on a task, or on a
/// scope corpus's contexts.
fn verifiers(corpus: &Value, found: &mut Vec<Value>) {
    match corpus {
        Value::Array(items) => items.iter().for_each(|item| verifiers(item, found)),
        Value::Object(map) => {
            for (key, value) in map {
                if key == "verify" {
                    found.push(value.clone());
                } else {
                    verifiers(value, found);
                }
            }
        }
        _ => {}
    }
}

/// Run every (verify, completion) pair through the real verifier, a batch at a
/// time, returning each pair's label with whether it passed.
async fn judge(pairs: Vec<(String, Value, String)>) -> Result<Vec<(String, bool)>> {
    let mut out = Vec::with_capacity(pairs.len());
    for batch in pairs.chunks(32) {
        let mut set = JoinSet::new();
        for (label, verify, completion) in batch.iter().cloned() {
            set.spawn(async move {
                let req = VerifyRequest {
                    run_id: RunId::new("corpora"),
                    step_idx: 0,
                    dimension: "exec".into(),
                    artifact: json!({ "verify": verify, "completion": completion }),
                };
                CommandVerifier
                    .verify(&req)
                    .await
                    .map(|v| (label, v.passed))
            });
        }
        while let Some(joined) = set.join_next().await {
            out.push(joined.map_err(|e| antumbra_core::AntumbraError::other(e.to_string()))??);
        }
    }
    Ok(out)
}

#[tokio::test]
#[ignore = "needs a real Python (ANTUMBRA_PYTHON); runs every workbench task, about a minute"]
async fn every_workbench_reference_passes_and_no_exit_does() -> Result<()> {
    let tasks = read(&corpora().join("workbench/all.json"))?;
    let tasks = tasks.as_array().cloned().unwrap_or_default();
    assert!(
        tasks.len() > 300,
        "the workbench corpus is missing or truncated"
    );

    let mut pairs = Vec::new();
    for task in &tasks {
        let id = task["id"].as_str().unwrap_or("?").to_string();
        if let Some(reference) = task["completion"].as_str() {
            pairs.push((
                format!("{id}: reference"),
                task["verify"].clone(),
                reference.to_string(),
            ));
        } else {
            assert_eq!(
                task["impossible"],
                json!(true),
                "{id} has no reference and is not marked impossible"
            );
        }
        for exit in EXITS {
            pairs.push((
                format!("{id}: {exit:?}"),
                task["verify"].clone(),
                exit.to_string(),
            ));
        }
    }

    let results = judge(pairs).await?;
    let wrong: Vec<&String> = results
        .iter()
        .filter(|(label, passed)| *passed != label.ends_with(": reference"))
        .map(|(label, _)| label)
        .collect();
    assert!(
        wrong.is_empty(),
        "{} verdict(s) wrong, first: {:?}",
        wrong.len(),
        wrong.first()
    );
    Ok(())
}

/// A guard for the next corpus as much as for these: a verifier that executes
/// the candidate in its own process can be passed by exiting, and this fails
/// the moment one appears anywhere in corpora/.
#[tokio::test]
#[ignore = "needs a real Python (ANTUMBRA_PYTHON)"]
async fn no_shipped_python_verifier_can_be_passed_by_exiting() -> Result<()> {
    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(corpora())
        .map_err(|e| antumbra_core::AntumbraError::other(e.to_string()))?
    {
        let path = entry
            .map_err(|e| antumbra_core::AntumbraError::other(e.to_string()))?
            .path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let mut found = Vec::new();
        verifiers(&read(&path)?, &mut found);
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();
        for verify in found
            .into_iter()
            .filter(|v| v["program"] == json!("python"))
        {
            for exit in EXITS {
                pairs.push((
                    format!("{name}: {exit:?}"),
                    verify.clone(),
                    exit.to_string(),
                ));
            }
        }
    }
    assert!(
        !pairs.is_empty(),
        "no Python verifiers found; the corpora path is wrong"
    );

    let passed: Vec<String> = judge(pairs)
        .await?
        .into_iter()
        .filter(|(_, passed)| *passed)
        .map(|(label, _)| label)
        .collect();
    assert!(passed.is_empty(), "passed by exiting: {passed:?}");
    Ok(())
}
