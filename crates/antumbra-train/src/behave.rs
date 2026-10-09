//! Training the user's behaviors into a standing expert (ADR-0027): the part
//! that needs no model. Tasks from each behavior's worked examples, a quarter
//! of them held out; the shipped replay prompts and controls; and the rule an
//! expert is admitted by.
//!
//! The configuration is validation 2's. On kuskokwim, ten examples per
//! behavior with the base model's own answers to 214 everyday prompts as
//! replay taught four of the user's behaviors (0% to 89-100% on held-out
//! phrasings) and kept unrelated tasks at the base model's level (32/40
//! against 34/40). Without the replay the expert applied its behaviors where
//! they do not belong. Command prompts alone as replay still let it turn code
//! and prose asks into commands (16/20 against the base's 19/20), so the
//! replay and the controls also carry code, prose and config.

use serde_json::{json, Value};

use antumbra_core::behavior::{Example, Spec};

use crate::eval::TaskResult;
use crate::model::CorpusTask;

/// Each behavior's held-out tasks must pass at least this often.
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
/// behavior with ten examples is admitted on two it never saw. Below four
/// examples nothing is held out, and the behavior cannot be admitted.
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

/// A behavior's training and held-out tasks. Ids are `{id}#t{i}` and
/// `{id}#h{i}`; every task's skill is the behavior's id.
pub struct BehaviorTasks {
    pub train: Vec<CorpusTask>,
    pub held: Vec<CorpusTask>,
}

pub fn tasks(id: &str, spec: &Spec) -> BehaviorTasks {
    let check = verify(spec);
    let task = |tag: &str, i: usize, e: &Example| {
        let mut t = CorpusTask::new(format!("{id}#{tag}{i}"), e.task.clone())
            .with_verify(check.clone())
            .with_completion(e.answer.clone());
        t.skill = Some(id.to_string());
        t
    };
    let (train, held) = split(&spec.examples);
    BehaviorTasks {
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
/// governed by a behavior: local git, gh, shell, docker and package commands,
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

/// Tasks no behavior governs, each with its own check, to measure what an
/// expert costs elsewhere, in families: `control-cmd-` and `control-code-`.
pub fn controls() -> Vec<CorpusTask> {
    rows(CONTROLS)
        .into_iter()
        .filter_map(|(id, prompt, verify)| Some(CorpusTask::new(id, prompt).with_verify(verify?)))
        .collect()
}

/// One behavior's held-out pass rates, base and expert.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviorScore {
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
    /// Each behavior taught; `admitted` on one says it was learned, and so is
    /// in the expert when the expert is admitted.
    pub behaviors: Vec<BehaviorScore>,
    /// Each family of controls, in the order the corpus lists them.
    pub controls: Vec<ControlScore>,
    /// Why the expert was refused. Empty when it was admitted.
    pub reasons: Vec<String>,
}

impl Verdict {
    /// The behaviors learned, by id: what an admitted expert holds.
    pub fn learned(&self) -> Vec<&str> {
        self.behaviors
            .iter()
            .filter(|s| s.admitted)
            .map(|s| s.id.as_str())
            .collect()
    }

    /// Each behavior not learned, with its rates and the bar it missed.
    pub fn missed(&self) -> Vec<String> {
        self.behaviors
            .iter()
            .filter(|s| !s.admitted)
            .map(|s| {
                format!(
                    "{}: held out {:.2} against the base's {:.2}; needs {MIN_RATE}, and {MIN_GAIN} more unless the base had it",
                    s.id, s.expert, s.base
                )
            })
            .collect()
    }
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

/// Judge an expert behavior by behavior. One is learned when it passes its
/// held-out tasks at least [`MIN_RATE`], and at least [`MIN_GAIN`] above the
/// base model unless the base already passed it that often.
///
/// The expert is admitted, holding the behaviors it learned, when at least one
/// of them clearly rose, no behavior it missed fell more than
/// [`CONTROL_SLACK`] below the base model, and each family of controls falls
/// no more than that below the base's either. So a behavior it did not learn
/// does not keep the rest out, but one it made worse does: the expert must
/// answer every task it was taught at least as well as the base model would.
/// Each family is held on its own, so a fall in code answers cannot hide
/// inside a larger pool of commands.
pub fn admit(behaviors: &[String], base: &[TaskResult], expert: &[TaskResult]) -> Verdict {
    let mut reasons = Vec::new();
    let scores: Vec<BehaviorScore> = behaviors
        .iter()
        .map(|id| {
            match (
                rate(base, &format!("{id}#h")),
                rate(expert, &format!("{id}#h")),
            ) {
                // A behavior the base already follows needs holding, not raising.
                (Some(b), Some(e)) => BehaviorScore {
                    id: id.clone(),
                    base: b,
                    expert: e,
                    admitted: e >= MIN_RATE && (e - b >= MIN_GAIN || b >= MIN_RATE),
                },
                // No held-out tasks: nothing to say it was learned.
                _ => BehaviorScore {
                    id: id.clone(),
                    base: 0.0,
                    expert: 0.0,
                    admitted: false,
                },
            }
        })
        .collect();
    for s in scores.iter().filter(|s| s.expert < s.base - CONTROL_SLACK) {
        reasons.push(format!(
            "{} fell from {:.2} to {:.2} held out: the expert answers it worse than the base model",
            s.id, s.base, s.expert
        ));
    }
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
                "{} fell from {:.2} to {:.2}: the expert applies its behaviors where they do not belong",
                c.family, c.base, c.expert
            ));
        }
    }
    if !scores.is_empty()
        && !scores
            .iter()
            .any(|s| s.admitted && s.expert - s.base >= MIN_GAIN)
    {
        reasons.push("no behavior was learned beyond what the base model already does".to_string());
    }
    Verdict {
        admitted: reasons.is_empty() && !scores.is_empty(),
        behaviors: scores,
        controls,
        reasons,
    }
}

#[cfg(test)]
mod tests;
