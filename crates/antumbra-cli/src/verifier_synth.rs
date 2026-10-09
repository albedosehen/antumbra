//! `antumbra verifier synthesize`: the model proposes the inputs
//! of a differential check, the reducible tier.
//!
//! For each task the model is asked for inputs that would tell a correct
//! function from a wrong one, and nothing else. The expected outputs come from
//! the task's authored reference, run by `corpora/workbench/synthesize.py`
//! through the same child the judge runs a candidate in, and the judge's
//! equality decides. So the model only wires the check up, something frozen
//! decides it, and what the trust protocol then measures is whether the
//! model's inputs catch wrong functions as well as the authored ones do.

#[cfg(any(feature = "models", test))]
use std::collections::BTreeSet;

#[cfg(any(feature = "models", test))]
use antumbra_core::canonical_json;

#[cfg(any(feature = "models", test))]
/// The function a task asks for: the first backticked call in its prompt,
/// as in "Write a Python function `swap_letters(s)`".
pub fn function_name(prompt: &str) -> Option<&str> {
    let mut rest = prompt;
    while let Some(tick) = rest.find('`') {
        let after = &rest[tick + 1..];
        let ident = after
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(after.len());
        if ident > 0 && after[ident..].starts_with('(') {
            return Some(&after[..ident]);
        }
        rest = after;
    }
    None
}

#[cfg(any(feature = "models", test))]
/// What the model is asked, after the task's own prompt.
pub fn inputs_prompt(task_prompt: &str, function: &str, count: usize) -> String {
    format!(
        "{task_prompt}\nDo not write the function. Write {count} test inputs for `{function}` that \
         would tell a correct implementation from a wrong one: ordinary cases, edge cases, and \
         every case the description singles out. Reply with one JSON array in a ```json block. \
         Each element is the JSON array of the arguments for one call, so for `f(a, b)` write \
         [[1, 2], [3, 4]] and for `f(xs)` write [[[1, 2]], [[]]]."
    )
}

#[cfg(any(feature = "models", test))]
/// The inputs one answer proposes: the JSON array in its first fenced block,
/// or in the answer itself when it has none. Anything else proposes nothing.
pub fn parse_inputs(answer: &str) -> Vec<serde_json::Value> {
    let body = antumbra_critic::verifiers::extract_code_block(answer);
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(serde_json::Value::Array(items)) => items,
        _ => Vec::new(),
    }
}

#[cfg(any(feature = "models", test))]
/// Every distinct input across the answers, in the order first proposed, up
/// to `max`.
pub fn merge_inputs(answers: &[Vec<serde_json::Value>], max: usize) -> Vec<serde_json::Value> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for input in answers.iter().flatten() {
        if out.len() >= max {
            break;
        }
        if seen.insert(canonical_json(input)) {
            out.push(input.clone());
        }
    }
    out
}

/// What `synthesize` was given.
pub struct SynthesizeArgs {
    pub corpus: String,
    pub skill: Option<String>,
    pub task: Option<String>,
    pub inputs: usize,
    pub draws: usize,
    pub max_new_tokens: usize,
    pub base_model: Option<String>,
    pub out: String,
}

pub async fn synthesize(args: SynthesizeArgs) -> anyhow::Result<()> {
    #[cfg(feature = "models")]
    {
        use anyhow::Context;

        use antumbra_train::{CandleModelLoader, CausalLm, ModelLoader, RaftConfig};

        let tasks: Vec<serde_json::Value> = serde_json::from_str(
            &std::fs::read_to_string(&args.corpus)
                .with_context(|| format!("reading {}", args.corpus))?,
        )?;
        let cfg = RaftConfig {
            max_new_tokens: args.max_new_tokens,
            ..RaftConfig::default()
        };
        let base_model = args.base_model.unwrap_or_else(|| cfg.base_model.clone());
        let loader = CandleModelLoader::new(cfg);
        let mut model = ModelLoader::load(&loader, &base_model, None).await?;
        let mut proposals = Vec::new();
        for task in &tasks {
            let (Some(id), Some(prompt)) = (task["id"].as_str(), task["prompt"].as_str()) else {
                continue;
            };
            let skill = task["skill"].as_str().unwrap_or("default");
            let wanted = args.task.as_deref().is_none_or(|t| t == id)
                && args.skill.as_deref().is_none_or(|s| s == skill);
            // An impossible task has no reference to compute outputs from.
            if !wanted || task["impossible"].as_bool().unwrap_or(false) {
                continue;
            }
            let Some(function) = function_name(prompt) else {
                eprintln!("{id}: no function named in the prompt; skipped");
                continue;
            };
            let asked = inputs_prompt(prompt, function, args.inputs);
            let answers: Vec<Vec<serde_json::Value>> = model
                .generate(&asked, args.draws)
                .await?
                .iter()
                .map(|a| parse_inputs(a))
                .collect();
            let inputs = merge_inputs(&answers, args.inputs * args.draws.max(1));
            println!("{id}: {} input(s) proposed", inputs.len());
            proposals.push(serde_json::json!({
                "task": id,
                "domain": skill,
                "function": function,
                "inputs": inputs,
                "proposed_by": base_model,
            }));
        }
        std::fs::write(&args.out, serde_json::to_vec_pretty(&proposals)?)
            .with_context(|| format!("writing {}", args.out))?;
        println!("{} proposal(s) -> {}", proposals.len(), args.out);
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let SynthesizeArgs {
            corpus,
            skill,
            task,
            inputs,
            draws,
            max_new_tokens,
            base_model,
            out,
        } = args;
        let _ = (
            corpus,
            skill,
            task,
            inputs,
            draws,
            max_new_tokens,
            base_model,
            out,
        );
        anyhow::bail!(
            "`verifier synthesize` requires building with --features models (candle + a GPU)"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_function_is_the_first_backticked_call() {
        let p = "# Write a Python function `swap_letters(s)` that returns `s` with `i` swapped.";
        assert_eq!(function_name(p), Some("swap_letters"));
        assert_eq!(function_name("uses `x` then `f_2(a, b)`"), Some("f_2"));
        assert_eq!(function_name("no call `here`"), None);
        assert_eq!(function_name("`(` alone"), None);
    }

    #[test]
    fn inputs_come_from_the_first_json_block() {
        let answer = "Here are inputs:\n```json\n[[\"ab\"], [\"\"], [\"in\"]]\n```\nMore text.";
        assert_eq!(parse_inputs(answer).len(), 3);
        assert_eq!(parse_inputs("[[1, 2], [3, 4]]").len(), 2);
        assert!(parse_inputs("```json\n{\"a\": 1}\n```").is_empty());
        assert!(parse_inputs("```json\n[[1, 2],\n```").is_empty());
        assert!(parse_inputs("no json at all").is_empty());
    }

    #[test]
    fn merged_inputs_are_distinct_in_first_proposed_order_and_capped() {
        let a = vec![serde_json::json!([1]), serde_json::json!([2])];
        let b = vec![
            serde_json::json!([2]),
            serde_json::json!([3]),
            serde_json::json!([4]),
        ];
        let merged = merge_inputs(&[a.clone(), b.clone()], 10);
        assert_eq!(
            merged,
            vec![
                serde_json::json!([1]),
                serde_json::json!([2]),
                serde_json::json!([3]),
                serde_json::json!([4])
            ]
        );
        assert_eq!(merge_inputs(&[a, b], 3).len(), 3);
    }

    #[test]
    fn the_prompt_names_the_function_and_the_count() {
        let asked = inputs_prompt("# Write `f(x)`.", "f", 12);
        assert!(asked.starts_with("# Write `f(x)`.\n"));
        assert!(asked.contains("12 test inputs for `f`"));
        assert!(asked.contains("```json"));
    }
}
