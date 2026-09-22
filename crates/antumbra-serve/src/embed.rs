//! Real candle BERT sentence embedder (routing-as-retrieval).
//!
//! all-MiniLM-L6-v2 produces 384-d normalized sentence vectors: the
//! capability/context space the gate routes in. Runs on CPU: the model is tiny
//! (~23M params) and we keep the GPU free for the trainer and server. This is
//! the real implementation behind the [`Embedder`] port that, until now, only
//! had the byte-histogram fake.

use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use std::path::PathBuf;

use hf_hub::HFClientSync;
use tokenizers::Tokenizer;

use antumbra_core::ports::Embedder;
use antumbra_core::{AntumbraError, Result};

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MAX_TOKENS: usize = 512;

/// The sentence model to load: [`MODEL_ID`], or whatever `ANTUMBRA_EMBED_MODEL`
/// names. Read once per call rather than cached, so a test can compare two.
fn model_id() -> String {
    std::env::var("ANTUMBRA_EMBED_MODEL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| MODEL_ID.to_string())
}

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
    /// Download (or load from the hf-hub cache) the configured sentence model and
    /// build it on CPU. Network on first call; cached thereafter.
    ///
    /// The model is [`MODEL_ID`] unless `ANTUMBRA_EMBED_MODEL` names another. The
    /// override exists because the default is a SYMMETRIC similarity model being
    /// used for ASYMMETRIC retrieval -- short queries against long passages -- and
    /// that mismatch, not any bad data, is what makes a 66-character stub score
    /// 0.774 against "banana bread recipe" while a 1900-character memory scores
    /// 0.05-0.17 against a query about its own contents. Swapping in a model
    /// trained for query-to-passage retrieval is a one-constant change; whether it
    /// actually flattens that curve is an empirical question, and this is the knob
    /// that lets it be answered without a code edit. Changing it invalidates every
    /// stored vector, so a swap means re-embedding the corpus.
    pub fn load() -> Result<Self> {
        Self::load_model(&model_id())
    }

    /// [`load`](Self::load) against an explicitly named hf-hub model, so two can be
    /// compared in one process.
    pub fn load_model(model: &str) -> Result<Self> {
        let device = Device::Cpu;
        // hf-hub 1.0's blocking client wraps an async runtime and `block_on`s it,
        // which panics if called from within an existing tokio runtime (the CLI and
        // MCP load the embedder from async). Run the downloads on a dedicated thread
        // that has no ambient runtime.
        let owned = model.to_string();
        let (config_path, tokenizer_path, weights_path) =
            std::thread::scope(|s| s.spawn(move || Self::fetch_files(&owned)).join())
                .map_err(|_| err("hf-hub", "download thread panicked"))??;

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

    /// Fetch the model's config/tokenizer/weights from the hf-hub (cached after the
    /// first call). Runs the hf-hub 1.0 blocking client, so it must be called off
    /// any tokio runtime (see [`load`](Self::load)).
    fn fetch_files(model_id: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
        let client = HFClientSync::new().map_err(|e| err("hf-hub client", e))?;
        let (owner, name) = model_id.split_once('/').unwrap_or(("", model_id));
        let repo = client.model(owner, name);
        let config = repo
            .download_file()
            .filename("config.json")
            .send()
            .map_err(|e| err("get config", e))?;
        let tokenizer = repo
            .download_file()
            .filename("tokenizer.json")
            .send()
            .map_err(|e| err("get tokenizer", e))?;
        let weights = repo
            .download_file()
            .filename("model.safetensors")
            .send()
            .map_err(|e| err("get weights", e))?;
        Ok((config, tokenizer, weights))
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

    /// Does a model trained for query-to-passage retrieval flatten the length
    /// curve? Prints, for each model, the similarity of a NONSENSE query to texts
    /// of increasing length, and of a TOPICAL query to the one text it is about.
    ///
    /// The defect this measures: under the default symmetric model, similarity
    /// tracks LENGTH rather than relevance -- measured on the live store at 66
    /// chars 0.774, ~130 0.655, ~180-250 0.42-0.48, ~300-330 0.31-0.40, 1000-1900
    /// 0.05-0.17, all against a query about banana bread. Short stubs win every
    /// query and long memories never surface, which is why a session bootstrap
    /// returned twelve memories about other projects.
    ///
    /// What to look for: `nonsense` should NOT fall monotonically with length, and
    /// `topical` should beat every `nonsense` score. Under all-MiniLM-L6-v2 it does
    /// not. If a swap fixes it, the fix is one constant plus a re-embed; if it does
    /// not, chunk-and-max-pool is the fallback and this test says so before anyone
    /// builds it.
    #[tokio::test]
    #[ignore = "downloads two models (~180 MB)"]
    async fn length_curve_across_models() {
        let short = "[Project] data-core-hub-parent - Part of WTS-Paradigm organization";
        let medium = "The sparse leg of hybrid recall must order by relevance. Without ORDER BY \
                      the engine returns matches in record order and the limit truncates to an \
                      arbitrary subset of them.";
        let long = format!(
            "Antumbra recall fuses a dense HNSW leg and a BM25 lexical leg with reciprocal \
             rank fusion. {} The lexical leg is what rescues identifiers and error codes that \
             a 384-dimensional vector silently drops, and it carried recall alone while the \
             dense leg ranked by length. {}",
            "Each leg pulls a candidate pool wider than the caller's k so fusion has room to \
             reorder before truncating. ".repeat(4),
            "Provenance is stored as a git evidence entry and judged at recall time. ".repeat(6)
        );
        let nonsense = "banana bread recipe";
        let topical = "reciprocal rank fusion dense and lexical legs";

        for model in [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        ] {
            let e = match BertEmbedder::load_model(model) {
                Ok(e) => e,
                Err(err) => {
                    println!("{model}: could not load ({err}) -- skipped");
                    continue;
                }
            };
            let q_non = e.embed(nonsense).await.expect("embed nonsense");
            let q_top = e.embed(topical).await.expect("embed topical");
            println!("\n=== {model} ===");
            for (label, text) in [
                ("short  ", short),
                ("medium ", medium),
                ("long   ", long.as_str()),
            ] {
                let v = e.embed(text).await.expect("embed passage");
                println!(
                    "  {label} len={:<5} nonsense={:.3}  topical={:.3}",
                    text.len(),
                    cosine(&q_non, &v),
                    cosine(&q_top, &v)
                );
            }
        }
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
