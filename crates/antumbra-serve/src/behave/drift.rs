//! What a quantized base does to an expert, measured when it is admitted.
//!
//! An expert is trained over the dense base and, on a server with no GPU,
//! served over a quantized copy of it. [`measure`] reads the expert's
//! held-out examples through the model as it was trained and through each
//! quantized form of the base with the same adapter, and compares the two
//! position by position ([`Drift`]).
//!
//! It is a record, kept on the expert's card. Nothing is refused on it, and
//! a measurement that fails changes nothing about the expert it was of.

use antumbra_core::Result;
use antumbra_train::models::QwenCausalLm;
use antumbra_train::quant::{Drift, Reading};
use antumbra_train::{CandleModelLoader, CorpusTask, ModelLoader, RaftConfig, ServeQuant};

/// The key on an expert's card that holds its drift, by form.
pub const CARD_DRIFT: &str = "drift";

/// The drift measured on each form, in the order asked for.
pub type Measured = Vec<(ServeQuant, Drift)>;

/// What was measured as a card holds it: each form's name to its numbers.
pub fn to_card(measured: &[(ServeQuant, Drift)]) -> serde_json::Value {
    serde_json::Value::Object(
        measured
            .iter()
            .map(|(form, drift)| (form.name().to_string(), drift.to_card()))
            .collect(),
    )
}

/// What was measured as a log line says it: `on a q8_0 base 3.6% of 2943
/// positions rank another token first, mean shift 0.10 nats`, one clause a
/// form. Empty when nothing was measured.
pub fn describe(measured: &[(ServeQuant, Drift)]) -> String {
    measured
        .iter()
        .map(|(form, drift)| {
            let flipped = 100.0 * (1.0 - drift.top1_agreement());
            format!(
                "on a {} base {flipped:.1}% of {} positions rank another token first, mean shift \
                 {:.2} nats",
                form.name(),
                drift.positions(),
                drift.mean_shift()
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The held-out examples a drift is read over: the ones that carry the answer
/// their behavior expects. A control has none and is left out.
fn texts(held: &[CorpusTask]) -> Vec<(&str, &str)> {
    held.iter()
        .filter_map(|task| {
            task.completion
                .as_deref()
                .map(|answer| (task.prompt.as_str(), answer))
        })
        .collect()
}

fn readings(model: &mut QwenCausalLm, texts: &[(&str, &str)]) -> Result<Vec<Vec<Reading>>> {
    texts
        .iter()
        .map(|(prompt, answer)| model.teacher_forced(prompt, answer))
        .collect()
}

/// Measure what each of `forms` does to the expert `model` holds, over the
/// `held` examples that carry an answer.
///
/// `model` is the expert as it was trained: its readings are the reference,
/// and it is let go before another base is loaded. Each form is then the
/// base of `cfg` quantized that way with the adapter at `adapter_uri`.
///
/// Never an error. A form that cannot be measured is said so and left out,
/// and the forms measured before it are kept.
pub async fn measure(
    mut model: QwenCausalLm,
    cfg: &RaftConfig,
    adapter_uri: &str,
    held: &[CorpusTask],
    forms: &[ServeQuant],
) -> Measured {
    let texts = texts(held);
    if forms.is_empty() || texts.is_empty() {
        return Vec::new();
    }
    let reference = match readings(&mut model, &texts) {
        Ok(reference) => reference,
        Err(e) => {
            eprintln!(
                "antumbra: an expert's drift was not measured, the reference pass failed: {e}"
            );
            return Vec::new();
        }
    };
    drop(model);
    let mut measured = Vec::new();
    for &form in forms {
        match on(form, cfg, adapter_uri, &texts, &reference).await {
            Ok(drift) => measured.push((form, drift)),
            Err(e) => eprintln!(
                "antumbra: an expert's drift on a {} base was not measured: {e}",
                form.name()
            ),
        }
    }
    measured
}

/// The drift of one form against the `reference` readings.
async fn on(
    form: ServeQuant,
    cfg: &RaftConfig,
    adapter_uri: &str,
    texts: &[(&str, &str)],
    reference: &[Vec<Reading>],
) -> Result<Drift> {
    let cfg = RaftConfig {
        serve_quant: Some(form),
        quantize_base: false,
        ..cfg.clone()
    };
    let loader = CandleModelLoader::new(cfg.clone());
    let mut model = ModelLoader::load(&loader, &cfg.base_model, Some(adapter_uri)).await?;
    let mut drift = Drift::default();
    for (reference, other) in reference.iter().zip(readings(&mut model, texts)?) {
        drift.add(reference, &other);
    }
    Ok(drift)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(logprob: f32, top1: u32) -> Reading {
        Reading { logprob, top1 }
    }

    /// Two positions, one of them ranking another token first, a quarter of a
    /// nat apart on average.
    fn two_positions() -> Drift {
        let mut drift = Drift::default();
        drift.add(
            &[reading(-1.0, 1), reading(-2.0, 2)],
            &[reading(-1.5, 1), reading(-2.0, 3)],
        );
        drift
    }

    #[test]
    fn a_card_holds_each_form_s_numbers_under_its_name() {
        let measured = vec![
            (ServeQuant::Q8_0, Drift::default()),
            (ServeQuant::Q4K, two_positions()),
        ];
        assert_eq!(
            to_card(&measured),
            serde_json::json!({
                "q8_0": { "positions": 0, "mean_shift": 0.0, "largest_shift": 0.0, "flips": 0 },
                "q4_k": { "positions": 2, "mean_shift": 0.25, "largest_shift": 0.5, "flips": 1 },
            })
        );
        assert_eq!(to_card(&[]), serde_json::json!({}));
    }

    #[test]
    fn a_log_line_says_what_each_form_did() {
        assert_eq!(
            describe(&[(ServeQuant::Q4K, two_positions())]),
            "on a q4_k base 50.0% of 2 positions rank another token first, mean shift 0.25 nats"
        );
        assert_eq!(describe(&[]), "");
    }

    #[test]
    fn a_drift_is_read_over_the_examples_that_carry_an_answer() {
        let task = |id: &str, completion: Option<&str>| CorpusTask {
            id: id.into(),
            prompt: format!("the task of {id}"),
            verify: serde_json::Value::Null,
            completion: completion.map(str::to_string),
            skill: None,
            scope: None,
            impossible: false,
        };
        let held = [task("a", Some("its answer")), task("control", None)];
        assert_eq!(texts(&held), [("the task of a", "its answer")]);
    }
}
