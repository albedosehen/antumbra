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
        // A `--features cuda` build pinned this to the CPU, so the one candle
        // model on every recall path never touched the card the feature exists
        // to use. `cuda_if_available` is the whole fix: it returns a CUDA device
        // only in a build that has the backend AND on a host with a working
        // card, and falls back otherwise, so a CPU host and a default build are
        // unaffected. MiniLM is 22M parameters and costs about 100MB of VRAM,
        // which matters on this deployment because the GPU has another tenant.
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
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

    /// The real-corpus validation the three-text probe could not provide: does
    /// calibrated ranking beat raw cosine on ACTUAL memories, scored the way recall
    /// scores them?
    ///
    /// Protocol. `ANTUMBRA_CALIB_SAMPLE` points at a JSON array of real memory
    /// texts spanning the length range. For each one a query is cut from ~60%
    /// THROUGH it -- deliberately not the head, because the head is what dominates
    /// a mean-pooled vector, and a query answerable from the head would flatter the
    /// uncalibrated score. The correct answer is known by construction: the memory
    /// the snippet came from. Every memory is then ranked against that query twice,
    /// by raw cosine and by z-score against its own baseline, and the two are
    /// scored on top-1 accuracy and MRR.
    ///
    /// This is the measurement that decides whether the two-float schema change is
    /// worth making. It is skipped, not failed, when the sample file is absent.
    #[tokio::test]
    #[ignore = "needs ANTUMBRA_CALIB_SAMPLE and downloads model weights"]
    async fn calibrated_ranking_beats_raw_cosine_on_the_real_corpus() {
        let Ok(path) = std::env::var("ANTUMBRA_CALIB_SAMPLE") else {
            println!("ANTUMBRA_CALIB_SAMPLE unset -- skipped");
            return;
        };
        let raw = std::fs::read_to_string(&path).expect("read sample");
        let texts: Vec<String> = serde_json::from_str(&raw).expect("parse sample");
        assert!(texts.len() >= 5, "need a few memories to rank against");

        let baseline_queries = [
            "banana bread recipe",
            "the weather in Reykjavik on a Tuesday",
            "how to repot a fiddle leaf fig",
            "tax deadlines for sole traders",
            "which strings to use on a fretless bass",
            "the offside rule explained simply",
        ];

        let e = BertEmbedder::load().expect("load model");
        let mut vecs = Vec::new();
        let mut base = Vec::new(); // (mean, sd) per memory
        let mut probes = Vec::new();
        for q in baseline_queries {
            probes.push(e.embed(q).await.expect("probe"));
        }
        for t in &texts {
            let v = e.embed(t).await.expect("memory");
            let sims: Vec<f32> = probes.iter().map(|p| cosine(p, &v)).collect();
            let n = sims.len() as f32;
            let mean = sims.iter().sum::<f32>() / n;
            let sd = (sims.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n)
                .sqrt()
                .max(1e-6);
            base.push((mean, sd));
            vecs.push(v);
        }

        // A query cut from ~60% through each memory, on word boundaries.
        let query_of = |t: &str| -> String {
            let words: Vec<&str> = t.split_whitespace().collect();
            let start = (words.len() as f32 * 0.6) as usize;
            words[start..(start + 12).min(words.len())].join(" ")
        };

        let (mut raw_top1, mut cal_top1) = (0usize, 0usize);
        let (mut raw_mrr, mut cal_mrr) = (0.0f32, 0.0f32);
        for (i, t) in texts.iter().enumerate() {
            let q = query_of(t);
            if q.split_whitespace().count() < 6 {
                continue;
            }
            let qv = e.embed(&q).await.expect("query");
            let mut by_raw: Vec<(usize, f32)> = vecs
                .iter()
                .enumerate()
                .map(|(j, v)| (j, cosine(&qv, v)))
                .collect();
            let mut by_cal: Vec<(usize, f32)> = by_raw
                .iter()
                .map(|(j, s)| (*j, (s - base[*j].0) / base[*j].1))
                .collect();
            by_raw.sort_by(|a, b| b.1.total_cmp(&a.1));
            by_cal.sort_by(|a, b| b.1.total_cmp(&a.1));
            let rank_of =
                |v: &[(usize, f32)]| v.iter().position(|(j, _)| *j == i).map_or(v.len(), |p| p) + 1;
            let (rr, rc) = (rank_of(&by_raw), rank_of(&by_cal));
            if rr == 1 {
                raw_top1 += 1;
            }
            if rc == 1 {
                cal_top1 += 1;
            }
            raw_mrr += 1.0 / rr as f32;
            cal_mrr += 1.0 / rc as f32;
        }
        let n = texts.len() as f32;
        println!(
            "\n  {} memories, query cut from 60% through each",
            texts.len()
        );
        println!(
            "  RAW COSINE   top1={raw_top1}/{}  MRR={:.3}",
            texts.len(),
            raw_mrr / n
        );
        println!(
            "  CALIBRATED   top1={cal_top1}/{}  MRR={:.3}",
            texts.len(),
            cal_mrr / n
        );
    }

    /// Is the length bias PREDICTABLE enough to subtract?
    ///
    /// Every fix measured so far tries to change the vectors (a different model, a
    /// different pooling). This asks a cheaper question: the raw cosine is useless
    /// ACROSS lengths, but if a text's similarity to an ARBITRARY query is a stable
    /// function of its length, then that expectation can be estimated per text and
    /// subtracted, and what remains is topicality. That is a calibration, not a
    /// model change -- no re-embedding, no schema, and it applies to the short
    /// stubs too, which is the half chunking cannot reach.
    ///
    /// Method: for each text, score it against several MUTUALLY UNRELATED queries
    /// to get a baseline mean and spread, then express the TOPICAL query's score as
    /// a z-score against that text's own baseline. If calibration works, the long
    /// passage's z must beat the stub's z, even though its raw cosine is far lower.
    #[tokio::test]
    #[ignore = "downloads model weights"]
    async fn is_the_length_bias_predictable_enough_to_subtract() {
        let stub = "[Project] data-core-hub-parent - Part of WTS-Paradigm organization";
        let medium = "The sparse leg of hybrid recall must order by relevance. Without ORDER BY \
                      the engine returns matches in record order and the limit truncates to an \
                      arbitrary subset of them.";
        let long = format!(
            "Antumbra recall fuses a dense HNSW leg and a BM25 lexical leg with reciprocal \
             rank fusion. {} The lexical leg is what rescues identifiers and error codes that \
             a 384-dimensional vector silently drops. {}",
            "Each leg pulls a candidate pool wider than the caller's k so fusion has room to \
             reorder before truncating. "
                .repeat(4),
            "Provenance is stored as a git evidence entry and judged at recall time. ".repeat(6)
        );
        // Deliberately unrelated to each other AND to every text, so their spread
        // estimates "what this text scores against an arbitrary query".
        let baseline_queries = [
            "banana bread recipe",
            "the weather in Reykjavik on a Tuesday",
            "kubernetes ingress TLS renewal",
            "how to repot a fiddle leaf fig",
            "tax deadlines for sole traders",
            "which strings to use on a fretless bass",
        ];
        let topical = "reciprocal rank fusion dense and lexical retrieval legs";

        let e = BertEmbedder::load().expect("load model");
        let q_top = e.embed(topical).await.expect("topical");
        let mut baselines = Vec::new();
        for q in baseline_queries {
            baselines.push(e.embed(q).await.expect("baseline query"));
        }

        println!("\n  text      len    raw_topical  baseline_mean  sd      z-score");
        for (label, text) in [
            ("stub  ", stub),
            ("medium", medium),
            ("long  ", long.as_str()),
        ] {
            let v = e.embed(text).await.expect("passage");
            let sims: Vec<f32> = baselines.iter().map(|b| cosine(b, &v)).collect();
            let n = sims.len() as f32;
            let mean = sims.iter().sum::<f32>() / n;
            let sd = (sims.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n)
                .sqrt()
                .max(1e-6);
            let raw = cosine(&q_top, &v);
            println!(
                "  {label}  {:<5}  {raw:.3}        {mean:.3}          {sd:.3}   {:+.2}",
                text.len(),
                (raw - mean) / sd
            );
        }
        println!("\n  calibration works if `long` has the highest z despite the lowest raw score.");
    }

    /// Does chunk-and-max-pool lift a long passage over a short off-topic stub,
    /// and at what chunk size?
    ///
    /// The bar is concrete: the 66-char stub scores ~0.774 against a NONSENSE
    /// query, and a long topical passage scores ~0.194 as one vector. For chunking
    /// to fix recall, the best-matching CHUNK of that passage has to beat the
    /// stub's nonsense score, or the stub still wins every query.
    ///
    /// Chunk size is the variable, and the answer is not free: `document_chunk`
    /// uses DEFAULT_CHUNK_CHARS = 1200, which is itself deep in the diluted regime,
    /// so memory chunking cannot simply reuse the document constant. This prints
    /// the curve so the size is chosen from evidence rather than inherited.
    #[tokio::test]
    #[ignore = "downloads model weights"]
    async fn chunk_and_max_pool_across_chunk_sizes() {
        let long = format!(
            "Antumbra recall fuses a dense HNSW leg and a BM25 lexical leg with reciprocal \
             rank fusion. {} The lexical leg is what rescues identifiers and error codes that \
             a 384-dimensional vector silently drops, and it carried recall alone while the \
             dense leg ranked by length. {}",
            "Each leg pulls a candidate pool wider than the caller's k so fusion has room to \
             reorder before truncating. "
                .repeat(4),
            "Provenance is stored as a git evidence entry and judged at recall time. ".repeat(6)
        );
        let e = BertEmbedder::load().expect("load model");
        let q_top = e
            .embed("reciprocal rank fusion dense and lexical legs")
            .await
            .expect("q");
        let q_non = e.embed("banana bread recipe").await.expect("q");

        // The score to beat: the short stub against a query about nothing.
        let stub = e
            .embed("[Project] data-core-hub-parent - Part of WTS-Paradigm organization")
            .await
            .expect("stub");
        let bar = cosine(&q_non, &stub);
        println!("\nbar to beat (66-char stub vs nonsense query): {bar:.3}");
        println!("whole passage as one vector: len={}", long.len());

        for size in [120usize, 200, 300, 600, 1200] {
            let chunks: Vec<&str> = long
                .as_bytes()
                .chunks(size)
                .filter_map(|c| std::str::from_utf8(c).ok())
                .collect();
            let (mut best_top, mut best_non) = (f32::MIN, f32::MIN);
            for c in &chunks {
                let v = e.embed(c).await.expect("chunk");
                best_top = best_top.max(cosine(&q_top, &v));
                best_non = best_non.max(cosine(&q_non, &v));
            }
            println!(
                "  chunk={size:<5} n={:<3} max_topical={best_top:.3}  max_nonsense={best_non:.3}  \
                 beats_stub={}",
                chunks.len(),
                best_top > bar
            );
        }
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
             reorder before truncating. "
                .repeat(4),
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
