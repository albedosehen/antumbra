//! A pair encoder for ADR-0024's D-2 relevance floor: one transformer that
//! reads the query and the memory together, fine-tuned on the constructed
//! labels.
//!
//! **Why this and not another head.** The frozen-encoder heads in
//! [`decision_probe`](crate::decision_probe) all read the two texts through
//! vectors computed apart, and none came near the 0.785 F1 control: 0.519
//! joined, 0.520 separate, 0.613 at the best chunk size. What is left is
//! cross-attention over the pair, which is what this is: every query token can
//! attend to every memory token, in every layer.
//!
//! **Why it reads past 512 tokens.** The control is `BAAI/bge-reranker-base`,
//! which the deployed server truncates at 512 tokens, and `scripts/d2-labels.sh`
//! cuts the query from about 60% of the way through memories that average
//! about 3,000 characters, so on the longer ones the passage the query was cut
//! from is gone before the control reads it. ModernBERT was trained to 8,192
//! tokens; [`FineTune::max_tokens`] sets how much of the memory this reads,
//! 1,024 by default.
//!
//! **What it is built from.** The [`ModernBert`] backbone and the checkpoint's
//! own prediction head (dense, GELU, norm), both loaded from the pretrained
//! weights through [`from_checkpoint`], and a fresh two-way classifier on the
//! pooled output. The training step updates all of it.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Shape, Tensor, Var, D};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{
    linear, linear_no_bias, Activation, AdamW, Init, LayerNorm, Linear, Module, Optimizer,
    ParamsAdamW, VarBuilder, VarMap,
};
use candle_transformers::models::modernbert::{Config, ModernBert};
use hf_hub::HFClientSync;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use antumbra_core::{AntumbraError, Result};

use crate::decision_probe::{score_at_half, LabeledPair, Scored};

/// The encoder ADR-0024 names, at the size that fits a training pass on one
/// 24GB card alongside nothing else.
pub const DEFAULT_MODEL: &str = "answerdotai/ModernBERT-base";

/// Where the trainable classifier lives in the variable map: the one part of
/// the model the checkpoint does not supply.
const CLASSIFIER: &str = "classifier";

/// How a fine-tuning run is set up.
#[derive(Debug, Clone, PartialEq)]
pub struct FineTune {
    /// The hf-hub model to start from.
    pub model: String,
    /// How many tokens of each pair it reads, query included.
    pub max_tokens: usize,
    pub epochs: usize,
    /// Pairs per optimizer step. Attention is materialized over the whole
    /// sequence, so memory grows with `batch * max_tokens^2`; 4 pairs of 1,024
    /// tokens is what one card holds for the base model.
    pub batch: usize,
    /// Peak learning rate, reached after a tenth of the steps and decayed
    /// linearly to zero.
    pub lr: f64,
    /// Seeds the order the training pairs are visited in.
    pub seed: u64,
}

impl Default for FineTune {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            max_tokens: 1024,
            epochs: 3,
            batch: 4,
            lr: 3e-5,
            seed: 0,
        }
    }
}

fn err(context: &str, e: impl std::fmt::Display) -> AntumbraError {
    AntumbraError::other(format!("{context}: {e}"))
}

/// How the backbone's per-token output becomes one vector for the head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// The first token's output.
    Cls,
    /// The mean over the tokens the mask keeps. ModernBERT's own default for
    /// classification.
    Mean,
}

/// A variable builder that serves a checkpoint's tensors as trainable
/// variables and creates only the classifier, which the checkpoint does not
/// have.
///
/// A plain [`VarMap`] builder creates every name it is asked for, and that is
/// wrong for ModernBERT: its first layer has no attention norm, and candle
/// builds that norm only when the weight exists (`.ok()` on the load). Under a
/// `VarMap` the weight always "exists", so the first layer would gain a fresh
/// norm that re-normalizes the embeddings and discards their norm's learned
/// scale: no longer the pretrained model. Refusing every name the checkpoint
/// lacks keeps that norm absent, and turns a real mismatch in names into an
/// error when the model is built rather than a silent random start.
struct Checkpoint {
    tensors: HashMap<String, Tensor>,
    varmap: VarMap,
}

impl SimpleBackend for Checkpoint {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        init: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        if name.starts_with(&format!("{CLASSIFIER}.")) {
            return self.varmap.get(shape, name, init, dtype, dev);
        }
        let Some(stored) = self.tensors.get(name) else {
            candle_core::bail!("{name} is not in the checkpoint")
        };
        if stored.shape() != &shape {
            candle_core::bail!(
                "{name}: the checkpoint has {:?}, the model wants {shape:?}",
                stored.shape()
            )
        }
        let mut vars = self.varmap.data().lock().expect("var map lock");
        if let Some(var) = vars.get(name) {
            return Ok(var.as_tensor().clone());
        }
        let var = Var::from_tensor(&stored.to_dtype(dtype)?.to_device(dev)?)?;
        let tensor = var.as_tensor().clone();
        vars.insert(name.to_string(), var);
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> candle_core::Result<Tensor> {
        candle_core::bail!("{name}: a checkpoint builder needs a shape")
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// A builder for [`PairModel::new`] over a checkpoint's tensors: every
/// variable it creates lands in `varmap`, trainable, holding the checkpoint's
/// values, except the classifier, which starts fresh. See [`Checkpoint`].
pub fn from_checkpoint(
    tensors: HashMap<String, Tensor>,
    varmap: &VarMap,
    device: &Device,
) -> VarBuilder<'static> {
    let backend: Box<dyn SimpleBackend> = Box::new(Checkpoint {
        tensors,
        varmap: varmap.clone(),
    });
    VarBuilder::from_backend(backend, DType::F32, device.clone())
}

/// The backbone and the classifier on it, apart from tokenization, so it can
/// be built and trained from token ids alone.
pub struct PairModel {
    encoder: ModernBert,
    dense: Linear,
    norm: LayerNorm,
    classifier: Linear,
    pooling: Pooling,
}

impl PairModel {
    /// Build every variable in `vb`: from a checkpoint with [`from_checkpoint`],
    /// or all fresh from a [`VarMap`] builder.
    pub fn new(vb: VarBuilder, config: &Config, pooling: Pooling) -> candle_core::Result<Self> {
        let encoder = ModernBert::load(vb.clone(), config)?;
        let hidden = config.hidden_size;
        // The checkpoint's prediction head: dense without bias, then a norm
        // without bias, as the MLM objective trained them.
        let dense = linear_no_bias(hidden, hidden, vb.pp("head.dense"))?;
        let norm_weight = vb
            .pp("head.norm")
            .get_with_hints(hidden, "weight", Init::Const(1.0))?;
        let norm = LayerNorm::new_no_bias(norm_weight, config.layer_norm_eps);
        let classifier = linear(hidden, 2, vb.pp(CLASSIFIER))?;
        Ok(Self {
            encoder,
            dense,
            norm,
            classifier,
            pooling,
        })
    }

    /// Two logits per pair, column 1 for "the memory answers the query".
    pub fn logits(&self, ids: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let hidden = self.encoder.forward(ids, mask)?;
        let pooled = match self.pooling {
            Pooling::Cls => hidden.i((.., 0, ..))?.contiguous()?,
            Pooling::Mean => {
                let keep = mask.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;
                let summed = hidden.broadcast_mul(&keep)?.sum(1)?;
                summed.broadcast_div(&keep.sum(1)?)?
            }
        };
        let x = self.dense.forward(&pooled)?;
        let x = Activation::Gelu.forward(&x)?;
        let x = self.norm.forward(&x)?;
        self.classifier.forward(&x)
    }
}

/// Tokenize pairs as one sequence each, `[CLS] query [SEP] memory [SEP]`,
/// cut to `max_tokens` from the memory's end and padded to the longest in the
/// batch. Returns the ids and the attention mask.
pub fn encode_pairs(
    tokenizer: &Tokenizer,
    pairs: &[&LabeledPair],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let inputs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|p| (p.query.as_str(), p.memory.as_str()))
        .collect();
    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| err("tokenize", e))?;
    let width = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
    let mut ids = Vec::with_capacity(pairs.len() * width);
    let mut mask = Vec::with_capacity(pairs.len() * width);
    for e in &encodings {
        ids.extend_from_slice(e.get_ids());
        mask.extend_from_slice(e.get_attention_mask());
    }
    let shape = (pairs.len(), width);
    Ok((
        Tensor::from_vec(ids, shape, device).map_err(|e| err("ids", e))?,
        Tensor::from_vec(mask, shape, device).map_err(|e| err("mask", e))?,
    ))
}

/// Set a tokenizer to cut each pair at `max_tokens`, taking from the longer
/// side (always the memory here), and pad a batch to its longest pair.
pub fn configure_tokenizer(
    tokenizer: &mut Tokenizer,
    max_tokens: usize,
    pad_id: u32,
) -> Result<()> {
    let pad_token = tokenizer
        .id_to_token(pad_id)
        .ok_or_else(|| err("tokenizer", format!("no token for pad id {pad_id}")))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            direction: TruncationDirection::Right,
            max_length: max_tokens,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }))
        .map_err(|e| err("truncation", e))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id,
        pad_type_id: 0,
        pad_token,
    }));
    Ok(())
}

/// The learning rate at `step` of `total`: a linear warmup over the first
/// tenth, then a linear decay to zero.
pub fn scheduled_lr(peak: f64, step: usize, total: usize) -> f64 {
    let warmup = (total / 10).max(1);
    if step < warmup {
        peak * (step + 1) as f64 / warmup as f64
    } else {
        let left = total.saturating_sub(step) as f64;
        peak * left / (total - warmup).max(1) as f64
    }
}

/// `0..n` in an order fixed by `seed`: a Fisher-Yates shuffle over splitmix64.
pub fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

fn fetch(model: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let client = HFClientSync::new().map_err(|e| err("hf-hub client", e))?;
    let (owner, name) = model.split_once('/').unwrap_or(("", model));
    let repo = client.model(owner, name);
    let get = |file: &str| {
        repo.download_file()
            .filename(file)
            .send()
            .map_err(|e| err(&format!("get {file}"), e))
    };
    Ok((
        get("config.json")?,
        get("tokenizer.json")?,
        get("model.safetensors")?,
    ))
}

/// Fine-tune [`FineTune::model`] on `train` and score it on `test`, at the
/// same cut and with the same arithmetic as the frozen-encoder heads, so the
/// numbers compare. `progress` receives a line per epoch.
pub fn fine_tune_and_score(
    train: &[LabeledPair],
    test: &[LabeledPair],
    cfg: &FineTune,
    progress: &mut dyn FnMut(String),
) -> Result<Scored> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    // hf-hub's blocking client must not run inside a tokio runtime.
    let model = cfg.model.clone();
    let (config_path, tokenizer_path, weights_path) =
        std::thread::scope(|s| s.spawn(move || fetch(&model)).join())
            .map_err(|_| err("hf-hub", "download thread panicked"))??;
    let raw = std::fs::read_to_string(config_path).map_err(|e| err("read config", e))?;
    let config: Config = serde_json::from_str(&raw).map_err(|e| err("parse config", e))?;
    let pooling = match serde_json::from_str::<serde_json::Value>(&raw)
        .ok()
        .and_then(|v| v.get("classifier_pooling")?.as_str().map(str::to_string))
        .as_deref()
    {
        Some("cls") => Pooling::Cls,
        _ => Pooling::Mean,
    };
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| err("tokenizer", e))?;
    configure_tokenizer(&mut tokenizer, cfg.max_tokens, config.pad_token_id)?;

    let tensors =
        candle_core::safetensors::load(&weights_path, &device).map_err(|e| err("weights", e))?;
    let varmap = VarMap::new();
    let pair = PairModel::new(from_checkpoint(tensors, &varmap, &device), &config, pooling)
        .map_err(|e| err("build", e))?;
    let filled = varmap
        .data()
        .lock()
        .map_err(|_| err("var map", "lock poisoned"))?
        .keys()
        .filter(|name| !name.starts_with(&format!("{CLASSIFIER}.")))
        .count();
    progress(format!(
        "{}: {filled} pretrained tensors loaded, pooling {pooling:?}, {} tokens, batch {}, lr {}, {} epoch(s), on {:?}",
        cfg.model, cfg.max_tokens, cfg.batch, cfg.lr, cfg.epochs, device
    ));

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: cfg.lr,
            weight_decay: 0.01,
            ..Default::default()
        },
    )
    .map_err(|e| err("optimizer", e))?;
    let batch = cfg.batch.max(1);
    let total = cfg.epochs * train.len().div_ceil(batch);
    let mut step = 0;
    for epoch in 0..cfg.epochs {
        let order = shuffled(train.len(), cfg.seed.wrapping_add(epoch as u64));
        let mut loss_sum = 0f64;
        let mut batches = 0usize;
        for chunk in order.chunks(batch) {
            let pairs: Vec<&LabeledPair> = chunk.iter().map(|&i| &train[i]).collect();
            let (ids, mask) = encode_pairs(&tokenizer, &pairs, &device)?;
            let labels: Vec<u32> = pairs.iter().map(|p| u32::from(p.relevant)).collect();
            let labels = Tensor::new(labels.as_slice(), &device).map_err(|e| err("labels", e))?;
            let logits = pair.logits(&ids, &mask).map_err(|e| err("forward", e))?;
            let loss =
                candle_nn::loss::cross_entropy(&logits, &labels).map_err(|e| err("loss", e))?;
            opt.set_learning_rate(scheduled_lr(cfg.lr, step, total));
            opt.backward_step(&loss).map_err(|e| err("step", e))?;
            loss_sum += f64::from(loss.to_scalar::<f32>().map_err(|e| err("loss value", e))?);
            batches += 1;
            step += 1;
        }
        progress(format!(
            "epoch {}: mean training loss {:.4} over {batches} step(s)",
            epoch + 1,
            loss_sum / batches.max(1) as f64
        ));
    }

    let mut probs = Vec::with_capacity(test.len());
    for chunk in test.chunks(batch) {
        let pairs: Vec<&LabeledPair> = chunk.iter().collect();
        let (ids, mask) = encode_pairs(&tokenizer, &pairs, &device)?;
        let p = candle_nn::ops::softmax(
            &pair
                .logits(&ids, &mask)
                .map_err(|e| err("forward", e))?
                .detach(),
            D::Minus1,
        )
        .and_then(|p| p.i((.., 1)))
        .and_then(|p| p.to_vec1::<f32>())
        .map_err(|e| err("probs", e))?;
        probs.extend(p);
    }
    Ok(score_at_half(&probs, test.iter().map(|p| p.relevant)))
}

#[cfg(test)]
mod tests;
