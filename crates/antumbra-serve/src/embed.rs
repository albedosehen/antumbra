//! Real candle BERT sentence embedder (ADR-0005 routing-as-retrieval).
//!
//! all-MiniLM-L6-v2 produces 384-d normalized sentence vectors — the
//! capability/context space the gate routes in. Runs on CPU: the model is tiny
//! (~23M params) and we keep the GPU free for the trainer and server. This is
//! the real implementation behind the [`Embedder`] port that, until now, only
//! had the byte-histogram fake.

use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use antumbra_core::ports::Embedder;
use antumbra_core::{AntumbraError, Result};

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MAX_TOKENS: usize = 512;

fn err(context: &str, e: impl std::fmt::Display) -> AntumbraError {
    AntumbraError::other(format!("{context}: {e}"))
}

/// A loaded all-MiniLM-L6-v2 model + tokenizer, ready to embed text.
pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
}

impl BertEmbedder {
    /// Download (or load from the hf-hub cache) all-MiniLM-L6-v2 and build the
    /// model on CPU. Network on first call; cached thereafter.
    pub fn load() -> Result<Self> {
        let device = Device::Cpu;
        let api = Api::new().map_err(|e| err("hf-hub api", e))?;
        let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));
        let config_path = repo.get("config.json").map_err(|e| err("get config", e))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| err("get tokenizer", e))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| err("get weights", e))?;

        let config_str = std::fs::read_to_string(config_path).map_err(|e| err("read config", e))?;
        let config: Config =
            serde_json::from_str(&config_str).map_err(|e| err("parse config", e))?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| err("load tokenizer", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
                .map_err(|e| err("load weights", e))?
        };
        let dim = config.hidden_size;
        let model = BertModel::load(vb, &config).map_err(|e| err("build bert", e))?;
        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
        })
    }

    /// Tokenize, run BERT, mean-pool over tokens, and L2-normalize.
    fn encode_pooled(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| err("tokenize", e))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        ids.truncate(MAX_TOKENS);
        let n_tokens = ids.len().max(1);

        let token_ids = Tensor::new(ids.as_slice(), &self.device)
            .map_err(|e| err("token tensor", e))?
            .unsqueeze(0)
            .map_err(|e| err("unsqueeze", e))?;
        let token_type_ids = token_ids.zeros_like().map_err(|e| err("type ids", e))?;

        let hidden = self
            .model
            .forward(&token_ids, &token_type_ids, None)
            .map_err(|e| err("bert forward", e))?;
        let pooled = (hidden.sum(1).map_err(|e| err("sum", e))? / n_tokens as f64)
            .map_err(|e| err("mean", e))?;
        let norm = pooled
            .sqr()
            .map_err(|e| err("sqr", e))?
            .sum_keepdim(1)
            .map_err(|e| err("norm sum", e))?
            .sqrt()
            .map_err(|e| err("sqrt", e))?;
        let normalized = pooled
            .broadcast_div(&norm)
            .map_err(|e| err("normalize", e))?;
        normalized
            .squeeze(0)
            .map_err(|e| err("squeeze", e))?
            .to_vec1::<f32>()
            .map_err(|e| err("to_vec", e))
    }
}

#[async_trait]
impl Embedder for BertEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_pooled(text)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }

    // Downloads ~90 MB of weights, so it is opt-in: `cargo test -p antumbra-serve
    // --features models -- --ignored`. Confirms the embedding space is semantic:
    // an arithmetic query sits closer to the arithmetic card than the string one.
    #[tokio::test]
    #[ignore = "downloads model weights"]
    async fn embeddings_are_semantic() {
        let embedder = BertEmbedder::load().expect("load model");
        assert_eq!(embedder.dim(), 384);
        let query = embedder.embed("sum two integers together").await.unwrap();
        let arith = embedder
            .embed("integer arithmetic and addition")
            .await
            .unwrap();
        let string = embedder
            .embed("reverse and format text strings")
            .await
            .unwrap();
        assert!(
            cosine(&query, &arith) > cosine(&query, &string),
            "arith {} should beat string {}",
            cosine(&query, &arith),
            cosine(&query, &string)
        );
    }
}
