//! The model seam the RAFT loop drives. A [`CausalLm`] generates candidate
//! completions and trains its LoRA adapter on verified winners. The candle
//! Qwen2.5-Coder + LoRA realization (MT-1) is GPU-validated; the RAFT control
//! flow is tested against a fake, mirroring how the rest of Antumbra seams off
//! heavy dependencies.

use async_trait::async_trait;

use antumbra_core::{AntumbraError, Result, TrainingRecipe};

/// A verifiable task drawn from the corpus (your repo): a prompt to complete,
/// plus the per-task `verify` spec passed through to the verifier. For
/// `CommandVerifier` that is `{ "program": ..., "args": [...], "cwd": ... }`;
/// `Null` means the task carries no verification.
#[derive(Debug, Clone)]
pub struct CorpusTask {
    pub id: String,
    pub prompt: String,
    pub verify: serde_json::Value,
    /// A supplied, verifier-checked correction to internalize (the capture
    /// intake into the composed model). `None` for RAFT tasks, which discover their own.
    pub completion: Option<String>,
    /// The skill group this task belongs to: many tasks can share one skill, and
    /// a grown expert is a specialist for a *skill*, not a single task. Defaults
    /// to the task id (see [`CorpusTask::skill`]).
    pub skill: Option<String>,
    /// When a correction also asserts *where* it applies (the governing feature
    /// and the contrastive context pair), capture promotes the verified
    /// correction to an actionable boundary of competence. `None` for a plain
    /// correction or a RAFT task.
    pub scope: Option<TaskScope>,
    /// A task whose specification cannot be satisfied. It is never
    /// learned from and never counted in fitness; under a holdout it is
    /// measured, and a pass fails the whole generation, because a pass here is
    /// proof of a shortcut rather than a near miss.
    pub impossible: bool,
}

/// The contrastive scope a correction carries: the context where the behavior is
/// incorrect (C) and the nearest one where it is acceptable (C'), plus the
/// governing feature -- supplied, or `None` to infer it from the one key that
/// differs between C and C'. A *verified* correction bearing this becomes a
/// [`antumbra_core::BoundaryFinding`] in the capture outcome.
#[derive(Debug, Clone)]
pub struct TaskScope {
    pub governing_feature: Option<String>,
    pub fail_context: serde_json::Value,
    pub near_ok_context: serde_json::Value,
}

impl CorpusTask {
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            verify: serde_json::Value::Null,
            completion: None,
            skill: None,
            scope: None,
            impossible: false,
        }
    }

    /// Mark this task as unsatisfiable by construction.
    pub fn impossible(mut self) -> Self {
        self.impossible = true;
        self
    }

    pub fn with_verify(mut self, verify: serde_json::Value) -> Self {
        self.verify = verify;
        self
    }

    pub fn with_completion(mut self, completion: impl Into<String>) -> Self {
        self.completion = Some(completion.into());
        self
    }

    /// Attach the contrastive scope with an explicit governing feature: turns
    /// this correction, once verified, into an actionable boundary.
    pub fn with_scope(
        mut self,
        governing_feature: impl Into<String>,
        fail_context: serde_json::Value,
        near_ok_context: serde_json::Value,
    ) -> Self {
        self.scope = Some(TaskScope {
            governing_feature: Some(governing_feature.into()),
            fail_context,
            near_ok_context,
        });
        self
    }

    /// Attach a contrastive scope whose governing feature is *inferred* from the
    /// one context key that differs between C and C' (the counterfactual boundary). If zero or
    /// several keys differ the feature cannot be named, and capture emits no
    /// boundary for it.
    pub fn with_inferred_scope(
        mut self,
        fail_context: serde_json::Value,
        near_ok_context: serde_json::Value,
    ) -> Self {
        self.scope = Some(TaskScope {
            governing_feature: None,
            fail_context,
            near_ok_context,
        });
        self
    }

    /// The skill group this task belongs to (its declared `skill`, or its id).
    pub fn skill(&self) -> String {
        self.skill.clone().unwrap_or_else(|| self.id.clone())
    }
}

/// One supervised example: train the adapter to produce `completion` for `prompt`.
#[derive(Debug, Clone)]
pub struct SftExample {
    pub prompt: String,
    pub completion: String,
}

#[async_trait]
pub trait CausalLm {
    /// Sample `n_samples` completions for a prompt (the RAFT rollouts).
    /// `&mut` because decoding advances the model's KV cache.
    async fn generate(&mut self, prompt: &str, n_samples: usize) -> Result<Vec<String>>;
    /// One supervised fine-tuning step over verified winners; returns the loss.
    async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32>;
    /// Persist the trained LoRA adapter (safetensors).
    fn save_adapter(&self, path: &str) -> Result<()>;

    /// Draw every later sample from `seed`, so a measurement can be repeated
    /// exactly or taken again under a seed its draws have never used. The
    /// default refuses: a model that cannot seed its draws must say so, or a
    /// "new seed" re-measurement would quietly reuse whatever stream the model
    /// was already on.
    fn seed_draws(&mut self, seed: u64) -> Result<()> {
        let _ = seed;
        Err(AntumbraError::Unimplemented("seeded draws"))
    }

    /// How likely, under the adapter, `prompt` is answered by each of
    /// `choices`, each read by its first token and normalized over the
    /// choices: how a critic reads its own verdict. The default refuses.
    async fn choose(&mut self, prompt: &str, choices: &[&str]) -> Result<Vec<f32>> {
        let _ = (prompt, choices);
        Err(AntumbraError::Unimplemented("scoring choices"))
    }
}

/// Builds a fresh [`CausalLm`] for a shadow: the shared base plus a new LoRA
/// adapter (optionally warm-started from a parent expert's adapter). The candle
/// Qwen2.5-Coder realization is the GPU-validated impl (MT-1).
#[async_trait]
pub trait ModelLoader: Send + Sync {
    type Model: CausalLm + Send;

    /// Build the model to train under `recipe`, or under the loader's own
    /// configuration when `None`. Required, so every loader decides what a
    /// recipe means for it: one that took a recipe and quietly trained under
    /// something else would make the recipe echoed back to the loop a lie.
    async fn load_trained(
        &self,
        base_model: &str,
        parent_adapter: Option<&str>,
        recipe: Option<&TrainingRecipe>,
    ) -> Result<Self::Model>;

    /// Build the model under the loader's own configuration: inference, and any
    /// training that searches nothing.
    async fn load(&self, base_model: &str, parent_adapter: Option<&str>) -> Result<Self::Model> {
        self.load_trained(base_model, parent_adapter, None).await
    }
}

/// Resolves corpus task ids (from a `TrainRequest`) to verifiable prompts.
pub trait Corpus: Send + Sync {
    fn tasks(&self, task_ids: &[String]) -> Vec<CorpusTask>;
}
