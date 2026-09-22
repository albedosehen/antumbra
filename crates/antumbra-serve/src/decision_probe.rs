//! Training and scoring the typed decision head against ADR-0024's D-2 bar.
//!
//! This is where the two halves meet: [`BertEmbedder`](crate::BertEmbedder)
//! encodes a `(query, memory)` pair into one pooled vector, and
//! [`DecisionHead`](antumbra_train::decision::DecisionHead) turns that into a
//! calibrated probability that the memory answers the query.
//!
//! **The encoder is frozen and small on purpose.** ADR-0024 names ModernBERT-
//! large, 400M parameters, referenced from elsewhere rather than vendored. Before
//! paying for that, the question worth answering is whether the encoder already
//! in this stack — all-MiniLM-L6-v2, 22M, already loaded, already tested —
//! carries enough pair signal for a two-layer head to beat a threshold over a
//! cross-encoder. If it does, the result is far better than needing the large
//! one. If it does not, that is a cheap finding rather than an expensive one,
//! and it is the argument for the bigger encoder rather than an assumption of
//! it.
//!
//! The bar is 0.782 F1, measured over this same construction by
//! `scripts/d2-relevance-baseline.sh`. A bar measured on one set and beaten on
//! another is not beaten, so the labels come from `scripts/d2-labels.sh`, which
//! builds pairs identically.

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::Deserialize;

use antumbra_core::ports::Embedder;
use antumbra_core::Result;
use antumbra_train::decision::DecisionHead;

/// One labelled pair, as `scripts/d2-labels.sh` writes it.
#[derive(Debug, Clone, Deserialize)]
pub struct LabelledPair {
    pub query: String,
    pub memory: String,
    /// True when the query was cut from inside this memory — the deterministic
    /// verifier ADR-0024's rule requires, rather than a human or model judgement.
    pub relevant: bool,
}

/// What a run of [`train_and_score`] found on its held-out half.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

/// How a pair is turned into features for the head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pairing {
    /// One encoder pass over both texts joined, mean-pooled. Measured at 0.519
    /// F1 against a 0.782 control — chance is 0.500.
    Joined,
    /// Encode each side separately and hand the head `[u, v, |u-v|, u*v]`, the
    /// standard bi-encoder recipe for sentence-pair classification.
    ///
    /// Added to test the explanation `Joined`'s result was first given: that
    /// mean-pooling a concatenation destroys the relationship, and supplying the
    /// interaction terms explicitly would recover it. **It does not.** 0.548
    /// accuracy and 0.520 F1, three points of accuracy and nothing in F1.
    ///
    /// So the pairing is not the binding constraint. A frozen MiniLM vector of a
    /// long memory is a mean-pool of up to 512 tokens into 384 dimensions, and
    /// the twelve-word span a query was cut from does not survive it. No
    /// function of `u` and `v` recovers what neither vector contains. The routes
    /// left are cross-attention over the pair, or chunking so an indexed unit is
    /// short enough that its vector still describes it.
    Separate,
}

/// How a pair is presented to the encoder.
///
/// Both sides in one string, so the pooled vector describes the PAIR rather than
/// either half. A bi-encoder scoring them separately is what recall already does
/// and what ranks by length; the whole reason a head can judge relevance is that
/// it reads them jointly.
fn pair_text(query: &str, memory: &str) -> String {
    format!("query: {query}\npassage: {memory}")
}

/// Encode every pair into one flat `(n * dim)` buffer, with its one-hot label.
///
/// Column 0 is "relevant", so a caller reads `probs[0]` as the probability the
/// floor compares against — the same column the relevance floor reads.
async fn encode(
    embedder: &dyn Embedder,
    pairs: &[LabelledPair],
    dim: usize,
    how: Pairing,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut xs = Vec::with_capacity(pairs.len() * feature_dim(dim, how));
    let mut ys = Vec::with_capacity(pairs.len() * 2);
    for p in pairs {
        match how {
            Pairing::Joined => {
                xs.extend_from_slice(&embedder.embed(&pair_text(&p.query, &p.memory)).await?);
            }
            Pairing::Separate => {
                let u = embedder.embed(&p.query).await?;
                let v = embedder.embed(&p.memory).await?;
                xs.extend_from_slice(&u);
                xs.extend_from_slice(&v);
                xs.extend(u.iter().zip(&v).map(|(a, b)| (a - b).abs()));
                xs.extend(u.iter().zip(&v).map(|(a, b)| a * b));
            }
        }
        ys.extend_from_slice(if p.relevant { &[1.0, 0.0] } else { &[0.0, 1.0] });
    }
    Ok((xs, ys))
}

/// How wide the head's input is for a given pairing: `u, v, |u-v|, u*v`.
fn feature_dim(dim: usize, how: Pairing) -> usize {
    match how {
        Pairing::Joined => dim,
        Pairing::Separate => dim * 4,
    }
}

/// Train a head on `train` and score it on `test`.
///
/// The split is the caller's, so a caller can hold out by memory rather than by
/// pair if it wants to — pairs built from the same memory share its text, and
/// splitting them across the boundary would let the head recognise the passage
/// rather than judge the match.
pub async fn train_and_score(
    embedder: &dyn Embedder,
    train: &[LabelledPair],
    test: &[LabelledPair],
    epochs: usize,
    how: Pairing,
) -> Result<Scored> {
    let dev = Device::Cpu;
    let dim = embedder.dim();
    let width = feature_dim(dim, how);

    let (train_x, train_y) = encode(embedder, train, dim, how).await?;
    let (test_x, _test_y) = encode(embedder, test, dim, how).await?;
    let map_err = |e: candle_core::Error| antumbra_core::AntumbraError::other(e.to_string());

    let tx = Tensor::from_vec(train_x, (train.len(), width), &dev).map_err(map_err)?;
    let ty = Tensor::from_vec(train_y, (train.len(), 2), &dev).map_err(map_err)?;
    let ex = Tensor::from_vec(test_x, (test.len(), width), &dev).map_err(map_err)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let head = DecisionHead::new(width, dim, 2, vb).map_err(map_err)?;
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 1e-3,
            ..Default::default()
        },
    )
    .map_err(map_err)?;
    for _ in 0..epochs {
        let loss = head.loss(&tx, &ty).map_err(map_err)?;
        opt.backward_step(&loss).map_err(map_err)?;
    }

    let probs = head
        .probs(&ex)
        .map_err(map_err)?
        .to_vec2::<f32>()
        .map_err(map_err)?;
    let (mut tp, mut fp, mut fnn, mut tn) = (0f32, 0f32, 0f32, 0f32);
    for (row, want) in probs.iter().zip(test.iter().map(|p| p.relevant)) {
        // Half is where a calibrated probability says "more likely than not",
        // and it is the floor recall defaults to. Scoring at the same point the
        // system will actually use is the only honest comparison.
        match (row[0] >= 0.5, want) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fnn += 1.0,
            (false, false) => tn += 1.0,
        }
    }
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fnn > 0.0 { tp / (tp + fnn) } else { 0.0 };
    Ok(Scored {
        accuracy: (tp + tn) / test.len().max(1) as f32,
        precision,
        recall,
        f1: if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BertEmbedder;

    /// ADR-0024 Validation 2 for D-2: does a head trained on constructed labels
    /// beat the control?
    ///
    /// The control is 0.782 F1, a threshold over the deployed cross-encoder,
    /// measured on pairs built the same way (`scripts/d2-relevance-baseline.sh`).
    /// This trains against `scripts/d2-labels.sh` output and prints both, so the
    /// record's bar is compared rather than asserted.
    ///
    /// Ignored: it needs the label file and downloads the encoder. Run with
    ///   ANTUMBRA_D2_LABELS=/path/to/d2-labels.json \
    ///   cargo test -p antumbra-serve --features models --lib -- --ignored --nocapture d2_head
    #[tokio::test]
    #[ignore = "needs ANTUMBRA_D2_LABELS and downloads model weights"]
    async fn d2_head_against_the_control() {
        let Ok(path) = std::env::var("ANTUMBRA_D2_LABELS") else {
            println!("ANTUMBRA_D2_LABELS unset -- skipped");
            return;
        };
        let raw = std::fs::read_to_string(&path).expect("read labels");
        let pairs: Vec<LabelledPair> = serde_json::from_str(&raw).expect("parse labels");
        assert!(pairs.len() >= 20, "need a real set, got {}", pairs.len());

        // Split by MEMORY, not by pair. Each memory contributes a positive and a
        // negative built from the same text, so splitting by pair would put the
        // same passage on both sides and let the head recognise it rather than
        // judge the match -- the result would be a measurement of leakage.
        let mut seen: Vec<&str> = Vec::new();
        for p in &pairs {
            if !seen.contains(&p.memory.as_str()) {
                seen.push(&p.memory);
            }
        }
        let cut = seen.len() / 2;
        let held: Vec<&str> = seen[cut..].to_vec();
        let (test, train): (Vec<_>, Vec<_>) = pairs
            .iter()
            .cloned()
            .partition(|p| held.contains(&p.memory.as_str()));

        println!(
            "\n  {} pairs: {} train / {} held out, split by memory",
            pairs.len(),
            train.len(),
            test.len()
        );
        let e = BertEmbedder::load().expect("load encoder");
        for (label, how) in [
            ("joined, mean-pooled ", Pairing::Joined),
            ("separate [u,v,|u-v|,u*v]", Pairing::Separate),
        ] {
            let got = train_and_score(&e, &train, &test, 400, how)
                .await
                .expect("train");
            println!(
                "  {label}  acc={:.3} prec={:.3} rec={:.3} F1={:.3}  {}",
                got.accuracy,
                got.precision,
                got.recall,
                got.f1,
                if got.f1 > 0.782 { "BEATS 0.782" } else { "" }
            );
        }
        println!("  CONTROL (cross-encoder threshold)                       F1=0.782");
    }
}
