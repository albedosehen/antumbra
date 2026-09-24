//! A critic as a shadow (ADR-0022 S-2): the base model with an adapter of its
//! own, taught to read a task and a completion and answer whether the
//! completion does what the task asks.
//!
//! Every label it learns from here is a fresh verifier outcome, so the
//! exogenous floor holds with room to spare. It is read the way the record
//! asks it to be measured: calibration and agreement per slice, and its rank
//! correlation with the verifier. It reaches training only through
//! [`crate::grpo::CriticShaping`], whose arithmetic bounds what it can do
//! there, and it never enters graduation.

use async_trait::async_trait;
use tokio::sync::Mutex;

use antumbra_core::critic::{calibration_by_slice, spearman, Isotonic, Scored, SliceCalibration};
use antumbra_core::ports::{ActOutput, Critic, CriticScore};
use antumbra_core::Result;

use crate::model::{CausalLm, SftExample};

/// The critic's answer when a completion does what its task asks.
pub const YES: &str = "yes";
/// Its answer when it does not.
pub const NO: &str = "no";

/// What the critic is asked about one completion.
pub fn critic_prompt(task: &str, completion: &str) -> String {
    format!(
        "Here is a task and a candidate answer.\n\nTask:\n{task}\n\nAnswer:\n{completion}\n\n\
         Does the answer do exactly what the task asks? Reply {YES} or {NO}."
    )
}

/// Probabilities from logits, stable against large ones.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// A verifier outcome to learn from or be measured on.
#[derive(Debug, Clone, PartialEq)]
pub struct Labeled {
    /// The task's prompt.
    pub prompt: String,
    pub completion: String,
    /// The verifier's verdict.
    pub passed: bool,
    /// What it is reported under: a skill, a task family.
    pub slice: String,
}

/// The labeled completions as training examples, balanced: the smaller class
/// is repeated, in order, until it matches the larger, so a critic trained
/// where most completions fail does not learn to say no to everything.
pub fn examples(labeled: &[Labeled]) -> Vec<SftExample> {
    let example = |l: &Labeled| SftExample {
        prompt: critic_prompt(&l.prompt, &l.completion),
        completion: if l.passed { YES } else { NO }.to_string(),
    };
    let (pass, fail): (Vec<&Labeled>, Vec<&Labeled>) = labeled.iter().partition(|l| l.passed);
    let (small, large) = if pass.len() <= fail.len() {
        (pass, fail)
    } else {
        (fail, pass)
    };
    let mut out: Vec<SftExample> = large.iter().map(|l| example(l)).collect();
    if !small.is_empty() {
        out.extend(small.iter().cycle().take(large.len()).map(|l| example(l)));
    }
    out
}

/// Teach `model` the verifier's verdicts for `rounds` passes over the
/// balanced examples. Returns the loss of each pass.
pub async fn train_critic(
    model: &mut (dyn CausalLm + Send),
    labeled: &[Labeled],
    rounds: usize,
) -> Result<Vec<f32>> {
    let batch = examples(labeled);
    let mut losses = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        losses.push(model.sft_step(&batch).await?);
    }
    Ok(losses)
}

/// The critic's probability that `completion` does what `prompt` asks.
pub async fn judge(
    model: &mut (dyn CausalLm + Send),
    prompt: &str,
    completion: &str,
) -> Result<f32> {
    let probs = model
        .choose(&critic_prompt(prompt, completion), &[YES, NO])
        .await?;
    Ok(probs.first().copied().unwrap_or(0.0))
}

/// How a critic reads against the verifier.
#[derive(Debug, Clone, PartialEq)]
pub struct CriticReading {
    pub n: usize,
    /// Every slice, calibration and agreement each.
    pub slices: Vec<SliceCalibration>,
    /// The same over everything, for reference; the slices are what counts.
    pub overall: Option<SliceCalibration>,
    /// Spearman's correlation between its scores and the verifier's verdicts.
    /// The influence `CriticShaping` gives it scales by this, inside a group.
    pub correlation: Option<f32>,
    /// Every completion's score, in the order given, for recalibration.
    pub scored: Vec<Scored>,
}

/// A reading of already-scored completions.
pub fn read(scored: Vec<Scored>) -> CriticReading {
    let all: Vec<Scored> = scored
        .iter()
        .map(|s| Scored {
            slice: "all".into(),
            ..s.clone()
        })
        .collect();
    let predicted: Vec<f32> = scored.iter().map(|s| s.predicted).collect();
    let verdicts: Vec<f32> = scored
        .iter()
        .map(|s| if s.passed { 1.0 } else { 0.0 })
        .collect();
    CriticReading {
        n: scored.len(),
        slices: calibration_by_slice(&scored, 10),
        overall: calibration_by_slice(&all, 10).into_iter().next(),
        correlation: spearman(&predicted, &verdicts),
        scored,
    }
}

/// Fit a recalibration on one reading and apply it to another: the critic's
/// scores on `evaluate` mapped through what `fit` showed they stand for.
/// `None` when `fit` is empty.
pub fn recalibrate(fit: &CriticReading, evaluate: &CriticReading) -> Option<CriticReading> {
    let scores: Vec<f32> = fit.scored.iter().map(|s| s.predicted).collect();
    let passed: Vec<bool> = fit.scored.iter().map(|s| s.passed).collect();
    let map = Isotonic::fit(&scores, &passed)?;
    Some(read(
        evaluate
            .scored
            .iter()
            .map(|s| Scored {
                predicted: map.apply(s.predicted),
                ..s.clone()
            })
            .collect(),
    ))
}

/// Measure `model` as a critic on labeled completions it did not learn from.
pub async fn measure_critic(
    model: &mut (dyn CausalLm + Send),
    labeled: &[Labeled],
) -> Result<CriticReading> {
    let mut scored = Vec::with_capacity(labeled.len());
    for l in labeled {
        scored.push(Scored {
            slice: l.slice.clone(),
            predicted: judge(model, &l.prompt, &l.completion).await?,
            passed: l.passed,
        });
    }
    Ok(read(scored))
}

/// A trained critic behind the [`Critic`] port. It reads the task, so it
/// scores through [`Critic::score`]; handed a trace with no task, it scores
/// nothing rather than guess.
pub struct ModelCritic {
    model: Mutex<Box<dyn CausalLm + Send>>,
}

impl ModelCritic {
    pub fn new(model: Box<dyn CausalLm + Send>) -> Self {
        ModelCritic {
            model: Mutex::new(model),
        }
    }
}

#[async_trait]
impl Critic for ModelCritic {
    async fn densify(&self, _output: &ActOutput) -> Result<Vec<CriticScore>> {
        Ok(Vec::new())
    }

    async fn score(&self, prompt: &str, completion: &str) -> Result<Option<f32>> {
        let mut model = self.model.lock().await;
        Ok(Some(judge(model.as_mut(), prompt, completion).await?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labeled(completion: &str, passed: bool, slice: &str) -> Labeled {
        Labeled {
            prompt: "task".into(),
            completion: completion.into(),
            passed,
            slice: slice.into(),
        }
    }

    /// Answers yes to a completion that says PASS, and learns nothing.
    struct Reader {
        trained_on: usize,
    }

    #[async_trait]
    impl CausalLm for Reader {
        async fn generate(&mut self, _prompt: &str, _n: usize) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
        async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32> {
            self.trained_on += batch.len();
            Ok(0.5)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
        async fn choose(&mut self, prompt: &str, choices: &[&str]) -> Result<Vec<f32>> {
            assert_eq!(choices, [YES, NO]);
            let yes = if prompt.contains("Answer:\nPASS") {
                0.9
            } else {
                0.2
            };
            Ok(vec![yes, 1.0 - yes])
        }
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_logits() {
        let p = softmax(&[1000.0, 999.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[0] > p[1]);
    }

    #[test]
    fn examples_are_balanced_and_carry_the_verifiers_answer() {
        let data = vec![
            labeled("a", true, "s"),
            labeled("b", false, "s"),
            labeled("c", false, "s"),
            labeled("d", false, "s"),
        ];
        let ex = examples(&data);
        assert_eq!(ex.len(), 6);
        assert_eq!(ex.iter().filter(|e| e.completion == YES).count(), 3);
        assert!(ex[0].prompt.contains("Answer:\nb"));
        assert!(examples(&[labeled("x", false, "s")]).len() == 1);
    }

    #[tokio::test]
    async fn a_critic_is_trained_on_verdicts_and_read_per_slice() {
        let mut model = Reader { trained_on: 0 };
        let data = vec![
            labeled("PASS 1", true, "strings"),
            labeled("FAIL 1", false, "strings"),
            labeled("PASS 2", false, "dates"),
            labeled("FAIL 2", false, "dates"),
        ];
        let losses = train_critic(&mut model, &data, 2).await.unwrap();
        assert_eq!(losses, vec![0.5, 0.5]);
        assert_eq!(model.trained_on, 2 * 6);
        let reading = measure_critic(&mut model, &data).await.unwrap();
        assert_eq!(reading.n, 4);
        let dates = reading.slices.iter().find(|s| s.slice == "dates").unwrap();
        let strings = reading
            .slices
            .iter()
            .find(|s| s.slice == "strings")
            .unwrap();
        // It says yes to a PASS that failed: the dates slice shows it, the
        // strings slice does not.
        assert_eq!(dates.agreement, 0.5);
        assert_eq!(strings.agreement, 1.0);
        assert!(reading.correlation.unwrap() > 0.0);
    }

    #[test]
    fn a_recalibration_fitted_on_one_reading_corrects_another() {
        let s = |predicted: f32, passed: bool| Scored {
            slice: "s".into(),
            predicted,
            passed,
        };
        // Confident about everything, right about half.
        let overconfident = |n: usize| {
            read(
                (0..n)
                    .map(|i| s(0.9 + (i % 5) as f32 * 0.01, i % 2 == 0))
                    .collect(),
            )
        };
        let (fit, evaluate) = (overconfident(40), overconfident(40));
        let raw = evaluate.overall.clone().unwrap().ece;
        let fixed = recalibrate(&fit, &evaluate).unwrap().overall.unwrap().ece;
        assert!(fixed < raw, "{fixed} !< {raw}");
        assert!(recalibrate(&read(Vec::new()), &evaluate).is_none());
    }

    #[tokio::test]
    async fn behind_the_port_it_scores_with_the_task_and_nothing_without_it() {
        let critic = ModelCritic::new(Box::new(Reader { trained_on: 0 }));
        assert_eq!(critic.score("task", "PASS").await.unwrap(), Some(0.9));
        let trace = ActOutput {
            steps: Vec::new(),
            final_output: "PASS".into(),
        };
        assert!(critic.densify(&trace).await.unwrap().is_empty());
    }
}
