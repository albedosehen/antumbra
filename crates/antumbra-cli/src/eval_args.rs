//! What `antumbra eval` is given, in one place for the parser and the
//! command both.

use clap::Args;

#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Path to the JSON corpus of verifiable tasks ({id,prompt,verify}).
    #[arg(long)]
    pub corpus: String,
    /// Saved adapter to load over the base before scoring. Omit to score
    /// the bare base: the prior floor.
    #[arg(long)]
    pub adapter: Option<String>,
    /// Base model to score. Defaults to the one training loads
    /// (Qwen2.5-Coder-1.5B-Instruct), so an eval measures the model a run
    /// would start from.
    #[arg(long)]
    pub base_model: Option<String>,
    /// Completions sampled per task (the pass-rate denominator is tasks x K).
    #[arg(long, default_value_t = 8)]
    pub samples: usize,
    /// Tokens generated per completion. The training default, so a function
    /// long enough to pass training is not cut short here.
    #[arg(long, default_value_t = 256)]
    pub max_new_tokens: usize,
    /// Write every task's result (passes out of samples) to this JSON file.
    #[arg(long)]
    pub report: Option<String>,
    /// Write every sampled completion, as `{task, completion, passed}`, to
    /// this JSON file: the policy's own answers, for labeling as cases a
    /// verifier is measured on (`antumbra verifier cases`).
    #[arg(long)]
    pub completions: Option<String>,
    /// Seed the draws. Unseeded, every eval of the same model draws the same
    /// completions, so a second, independent sample needs a seed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Sampling temperature. Defaults to training's, so the eval sees the
    /// draws a run would; 0 is greedy.
    #[arg(long)]
    pub temperature: Option<f64>,
    /// Nucleus cutoff. Defaults to training's (1.0, off).
    #[arg(long)]
    pub top_p: Option<f64>,
    /// Compute precision on the GPU: f32, bf16 or f16. Defaults to
    /// training's (bf16). Comparing f32 with bf16 on the same draws is how
    /// to tell a numerics problem from a model that cannot do the task.
    #[arg(long)]
    pub dtype: Option<String>,
}
