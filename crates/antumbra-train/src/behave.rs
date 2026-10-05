//! Training the user's behaviours into a standing expert (ADR-0027): the part
//! that needs no model. Tasks from each behaviour's worked examples, a quarter
//! of them held out; the shipped replay prompts and controls; and the rule an
//! expert is admitted by.
//!
//! The configuration is validation 2's. On kuskokwim, ten examples per
//! behaviour with the base model's own answers to 214 everyday prompts as
//! replay taught four of the user's behaviours (0% to 89-100% on held-out
//! phrasings) and kept unrelated tasks at the base model's level (32/40
//! against 34/40). Without the replay the expert applied its behaviours where
//! they do not belong. Command prompts alone as replay still let it turn code
//! and prose asks into commands (16/20 against the base's 19/20), so the
//! replay and the controls also carry code, prose and config.

use serde_json::{json, Value};

use antumbra_core::behaviour::{Example, Spec};

use crate::eval::TaskResult;
use crate::model::CorpusTask;

/// Each behaviour's held-out tasks must pass at least this often.
pub const MIN_RATE: f32 = 0.6;
/// And at least this much more often than the base model's.
pub const MIN_GAIN: f32 = 0.3;
/// The controls may fall this far below the base model's rate, and no further:
/// four of the forty, the noise validation 2 saw between runs.
pub const CONTROL_SLACK: f32 = 0.1;
/// Every this-many-th example is held out to admit the expert by.
const HOLD_EVERY: usize = 4;

/// The check, in the language the trainer's verifier runs. The patterns ride
/// as arguments, so no pattern is ever spliced into code.
const CHECK: &str = "import json,os,re,sys\n\
c=os.environ.get('ANTUMBRA_COMPLETION','')\n\
must=json.loads(sys.argv[1])\n\
must_not=json.loads(sys.argv[2])\n\
sys.exit(0 if all(re.search(p,c) for p in must) and not any(re.search(p,c) for p in must_not) else 1)";

const REPLAY: &str = include_str!("../../../corpora/behave-replay.json");
const CONTROLS: &str = include_str!("../../../corpora/behave-controls.json");

/// The verifier spec that applies `spec`'s check to a completion.
pub fn verify(spec: &Spec) -> Value {
    json!({
        "program": "python",
        "extract_code": false,
        "args": [
            "-c",
            CHECK,
            serde_json::to_string(&spec.must).unwrap_or_default(),
            serde_json::to_string(&spec.must_not).unwrap_or_default(),
        ],
    })
}

/// A verifier that accepts anything: a replay answer is the base model's own,
/// kept as it is, right or wrong.
pub fn always() -> Value {
    json!({ "program": "python", "extract_code": false, "args": ["-c", "import sys; sys.exit(0)"] })
}

/// The examples to train on and the ones held out: every fourth, so a
/// behaviour with ten examples is admitted on two it never saw. Below four
/// examples nothing is held out, and the behaviour cannot be admitted.
pub fn split(examples: &[Example]) -> (Vec<&Example>, Vec<&Example>) {
    let mut train = Vec::new();
    let mut held = Vec::new();
    for (i, e) in examples.iter().enumerate() {
        if examples.len() >= HOLD_EVERY && i % HOLD_EVERY == HOLD_EVERY - 1 {
            held.push(e);
        } else {
            train.push(e);
        }
    }
    (train, held)
}

/// A behaviour's training and held-out tasks. Ids are `{id}#t{i}` and
/// `{id}#h{i}`; every task's skill is the behaviour's id.
pub struct BehaviourTasks {
    pub train: Vec<CorpusTask>,
    pub held: Vec<CorpusTask>,
}

pub fn tasks(id: &str, spec: &Spec) -> BehaviourTasks {
    let check = verify(spec);
    let task = |tag: &str, i: usize, e: &Example| {
        let mut t = CorpusTask::new(format!("{id}#{tag}{i}"), e.task.clone())
            .with_verify(check.clone())
            .with_completion(e.answer.clone());
        t.skill = Some(id.to_string());
        t
    };
    let (train, held) = split(&spec.examples);
    BehaviourTasks {
        train: train
            .iter()
            .enumerate()
            .map(|(i, e)| task("t", i, e))
            .collect(),
        held: held
            .iter()
            .enumerate()
            .map(|(i, e)| task("h", i, e))
            .collect(),
    }
}

/// A shipped corpus's rows: `(id, prompt, verify)`.
fn rows(text: &str) -> Vec<(String, String, Option<Value>)> {
    let parsed: Vec<Value> = serde_json::from_str(text).unwrap_or_default();
    parsed
        .into_iter()
        .filter_map(|r| {
            let id = r.get("id")?.as_str()?.to_string();
            let prompt = r.get("prompt")?.as_str()?.to_string();
            Some((id, prompt, r.get("verify").cloned()))
        })
        .collect()
}

/// The everyday prompts the base model answers for replay, none of them
/// governed by a behaviour: local git, gh, shell, docker and package commands,
/// and code, prose and config in several languages.
pub fn replay_prompts() -> Vec<CorpusTask> {
    rows(REPLAY)
        .into_iter()
        .map(|(id, prompt, _)| CorpusTask::new(id, prompt).with_verify(always()))
        .collect()
}

/// Replay tasks from the base model's own answers, keyed by prompt id.
pub fn replay(prompts: &[CorpusTask], answers: &[(String, String)]) -> Vec<CorpusTask> {
    prompts
        .iter()
        .filter_map(|p| {
            let (_, answer) = answers.iter().find(|(id, _)| *id == p.id)?;
            let answer = answer.trim();
            (!answer.is_empty()).then(|| p.clone().with_completion(answer))
        })
        .collect()
}

/// Tasks no behaviour governs, each with its own check, to measure what an
/// expert costs elsewhere, in families: `control-cmd-` and `control-code-`.
pub fn controls() -> Vec<CorpusTask> {
    rows(CONTROLS)
        .into_iter()
        .filter_map(|(id, prompt, verify)| Some(CorpusTask::new(id, prompt).with_verify(verify?)))
        .collect()
}

/// One behaviour's held-out pass rates, base and expert.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourScore {
    pub id: String,
    pub base: f32,
    pub expert: f32,
    pub admitted: bool,
}

/// One family of controls' pass rates, base and expert: `control-cmd` for
/// commands, `control-code` for code and prose.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlScore {
    pub family: String,
    pub base: f32,
    pub expert: f32,
}

/// Whether an expert is admitted, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub admitted: bool,
    pub behaviours: Vec<BehaviourScore>,
    /// Each family of controls, in the order the corpus lists them.
    pub controls: Vec<ControlScore>,
    pub reasons: Vec<String>,
}

/// A control's family: its id up to the last `-`, so `control-code-3` is in
/// `control-code`.
fn family(id: &str) -> Option<&str> {
    id.starts_with("control-")
        .then(|| id.rsplit_once('-').map(|(f, _)| f))
        .flatten()
}

fn rate(results: &[TaskResult], prefix: &str) -> Option<f32> {
    let (passed, total) = results
        .iter()
        .filter(|r| r.id.starts_with(prefix))
        .fold((0, 0), |(p, t), r| (p + r.passed, t + r.total));
    (total > 0).then(|| passed as f32 / total as f32)
}

/// Admit an expert when every behaviour it was taught passes its held-out
/// tasks at least [`MIN_RATE`], each at least [`MIN_GAIN`] above the base model
/// unless the base already passed it that often, at least one clearly rose, and
/// each family of controls falls no more than [`CONTROL_SLACK`] below the
/// base's. Each family is held on its own, so a fall in code answers cannot
/// hide inside a larger pool of commands.
pub fn admit(behaviours: &[String], base: &[TaskResult], expert: &[TaskResult]) -> Verdict {
    let mut reasons = Vec::new();
    let scores: Vec<BehaviourScore> = behaviours
        .iter()
        .map(|id| {
            let prefix = format!("{id}#h");
            match (rate(base, &prefix), rate(expert, &prefix)) {
                (Some(b), Some(e)) => {
                    // A behaviour the base already follows needs holding, not
                    // raising: requiring a gain would let it block the rest.
                    let admitted = e >= MIN_RATE && (e - b >= MIN_GAIN || b >= MIN_RATE);
                    if !admitted {
                        reasons.push(format!(
                            "{id}: held out {e:.2} against the base's {b:.2}; needs {MIN_RATE}, and {MIN_GAIN} more unless the base had it"
                        ));
                    }
                    BehaviourScore { id: id.clone(), base: b, expert: e, admitted }
                }
                _ => {
                    reasons.push(format!("{id}: no held-out tasks to admit it by"));
                    BehaviourScore { id: id.clone(), base: 0.0, expert: 0.0, admitted: false }
                }
            }
        })
        .collect();
    let mut families: Vec<&str> = Vec::new();
    for f in base.iter().filter_map(|r| family(&r.id)) {
        if !families.contains(&f) {
            families.push(f);
        }
    }
    let controls: Vec<ControlScore> = families
        .iter()
        .map(|f| {
            let prefix = format!("{f}-");
            ControlScore {
                family: f.to_string(),
                base: rate(base, &prefix).unwrap_or(0.0),
                expert: rate(expert, &prefix).unwrap_or(0.0),
            }
        })
        .collect();
    for c in &controls {
        if c.expert < c.base - CONTROL_SLACK {
            reasons.push(format!(
                "{} fell from {:.2} to {:.2}: the expert applies its behaviours where they do not belong",
                c.family, c.base, c.expert
            ));
        }
    }
    if !scores.is_empty() && !scores.iter().any(|s| s.expert - s.base >= MIN_GAIN) {
        reasons.push("no behaviour rose: the base model already follows them all".to_string());
    }
    Verdict {
        admitted: reasons.is_empty() && !scores.is_empty(),
        behaviours: scores,
        controls,
        reasons,
    }
}

#[cfg(test)]
mod tests;
