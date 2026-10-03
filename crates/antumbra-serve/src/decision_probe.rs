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
//! The bar is 0.785 F1 — accuracy 0.802, precision 0.862, recall 0.720 — a
//! threshold over the deployed cross-encoder, measured by
//! `scripts/d2-relevance-baseline.sh`, which reads the very label file this
//! trains on. It reads rather than rebuilds for a reason: the two scripts used
//! to construct pairs separately and identically, and were identically wrong.
//!
//! **A label set that `grep` can answer measures nothing.** The first version of
//! these labels left the query verbatim inside its positive memory, so a
//! `contains` check with no model scored F1 1.000 and a chunk sweep appeared to
//! beat the control at 0.940. The span is now excised, `contains` scores 0.000,
//! and the harness prints that line on every run so the degeneracy cannot come
//! back quietly. Read every score against it, not against the control.

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
    /// F1 against a 0.785 control — chance is 0.500.
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
    /// Chunk the memory, embed each chunk, and use the chunk that best matches
    /// the query as `v` — then the same `[u, v, |u-v|, u*v]` features.
    ///
    /// This tests the surviving explanation directly and cheaply. If the twelve-
    /// word span a query was cut from is lost because it is mean-pooled with
    /// four thousand characters of other text, then a chunk SHORT ENOUGH to be
    /// mostly that span should carry it, and the head should recover. No new
    /// model and no fine-tuning: the same frozen encoder, applied to shorter
    /// units.
    ///
    /// **Read the history here before trusting any chunking number, because the
    /// first set of them was an artefact and nearly became a decision.**
    ///
    /// Swept on the original label set, this appeared to beat the control
    /// outright: 0.940 F1 at 150 characters, 0.862 at 200, 0.730 at 300, against
    /// a 0.782 bar. The curve was MONOTONIC in smaller chunks, which was the
    /// tell. `scripts/d2-labels.sh` cut a twelve-word query out of a memory and
    /// left it there, so a short enough chunk simply WAS the query, and the task
    /// was substring detection wearing relevance as a costume. A `contains`
    /// check with no model scored 1.000 on that set.
    ///
    /// With the span excised — same memories, same hard negatives, same split —
    /// the shortcut dies (`contains` scores 0.000) and so does most of the gain:
    ///
    /// | chunk | F1 (clean) | F1 (degenerate) |
    /// |---|---|---|
    /// | 150 | 0.594 | 0.940 |
    /// | 200 | **0.613** | 0.862 |
    /// | 300 | 0.525 | 0.730 |
    /// | 450 | 0.540 | 0.599 |
    /// | 600 | 0.520 | 0.658 |
    ///
    /// Two things survive. Chunking still helps, 0.519 to 0.613, which is real
    /// but modest. And the clean curve PEAKS IN THE MIDDLE rather than running
    /// to the smallest chunk, which is what an independent measurement of
    /// chunk-and-max-pool found for ranking (a topical/nonsense gap of 0.149 at
    /// 300 characters against 0.083 unchunked and 0.091 at 120, because very
    /// small chunks raise every score). Two unrelated measurements agreeing on
    /// the shape is the reason to believe this one.
    ///
    /// So chunking is NECESSARY AND NOT SUFFICIENT against a 0.785 control. That
    /// was the conclusion before the artefact was found and it is the conclusion
    /// after, which is luck rather than vindication: for one run the evidence
    /// said the opposite and said it loudly.
    BestChunk { chunk_chars: usize },
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
    let mut cache: std::collections::HashMap<String, Vec<Vec<f32>>> =
        std::collections::HashMap::new();
    for p in pairs {
        match how {
            Pairing::Joined => {
                xs.extend_from_slice(&embedder.embed(&pair_text(&p.query, &p.memory)).await?);
            }
            Pairing::Separate => {
                let u = embedder.embed(&p.query).await?;
                let v = embedder.embed(&p.memory).await?;
                push_pair(&mut xs, &u, &v);
            }
            Pairing::BestChunk { chunk_chars } => {
                let u = embedder.embed(&p.query).await?;
                // Embedding every chunk of every memory is the expensive part,
                // and each memory appears in two pairs, so cache by text.
                let chunks = match cache.get(&p.memory) {
                    Some(c) => c.clone(),
                    None => {
                        let mut c = Vec::new();
                        for t in antumbra_core::chunk_text(&p.memory, chunk_chars, chunk_chars / 4)
                        {
                            c.push(embedder.embed(&t).await?);
                        }
                        cache.insert(p.memory.clone(), c.clone());
                        c
                    }
                };
                // The best-matching chunk stands for the memory. If the span the
                // query came from survives anywhere, it survives here.
                let v = chunks
                    .iter()
                    .max_by(|a, b| {
                        antumbra_core::cosine_similarity(&u, a)
                            .total_cmp(&antumbra_core::cosine_similarity(&u, b))
                    })
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; dim]);
                push_pair(&mut xs, &u, &v);
            }
        }
        ys.extend_from_slice(if p.relevant { &[1.0, 0.0] } else { &[0.0, 1.0] });
    }
    Ok((xs, ys))
}

/// `[u, v, |u-v|, u*v]`: the two vectors and the interaction terms between
/// them, which is where a bi-encoder puts the comparison a cross-encoder gets
/// from attention.
fn push_pair(xs: &mut Vec<f32>, u: &[f32], v: &[f32]) {
    xs.extend_from_slice(u);
    xs.extend_from_slice(v);
    xs.extend(u.iter().zip(v).map(|(a, b)| (a - b).abs()));
    xs.extend(u.iter().zip(v).map(|(a, b)| a * b));
}

/// How wide the head's input is for a given pairing: `u, v, |u-v|, u*v`.
fn feature_dim(dim: usize, how: Pairing) -> usize {
    match how {
        Pairing::Joined => dim,
        Pairing::Separate | Pairing::BestChunk { .. } => dim * 4,
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
    // The head is small, but the ENCODER pass in front of it is not, and the
    // chunked pairing multiplies it: MiniLM pads a batch to a fixed shape, so a
    // 300-character chunk costs nearly what a 4000-character memory costs. An
    // 800-pair three-way run did not finish in 161 minutes on CPU, which is why
    // the recorded figures came from a 120-pair sample and had to be read as an
    // ordering. `cuda_if_available` falls back to CPU on a host without a card
    // and without the feature, so this is the same code either way.
    let dev = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
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
    let relevant: Vec<f32> = probs.iter().map(|row| row[0]).collect();
    Ok(score_at_half(&relevant, test.iter().map(|p| p.relevant)))
}

/// Score probabilities that each memory answers its query against the labels,
/// at one half.
///
/// Half is where a calibrated probability says "more likely than not", and it
/// is the floor recall defaults to. Scoring at the same point the system will
/// actually use is the only fair comparison, and every head is scored here so
/// their numbers compare.
pub fn score_at_half(relevant: &[f32], labels: impl Iterator<Item = bool>) -> Scored {
    let (mut tp, mut fp, mut fnn, mut tn) = (0f32, 0f32, 0f32, 0f32);
    let mut n = 0usize;
    for (p, want) in relevant.iter().zip(labels) {
        n += 1;
        match (*p >= 0.5, want) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fnn += 1.0,
            (false, false) => tn += 1.0,
        }
    }
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fnn > 0.0 { tp / (tp + fnn) } else { 0.0 };
    Scored {
        accuracy: (tp + tn) / n.max(1) as f32,
        precision,
        recall,
        f1: if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        },
    }
}

/// Split pairs into a training half and a held-out half by MEMORY, not by pair.
///
/// Each memory contributes a positive and a negative built from the same text,
/// so splitting by pair would put the same passage on both sides and let a
/// model recognise it rather than judge the match: the result would be a
/// measurement of leakage. The first half of the memories, in first-seen order,
/// trains; the rest is held out. Returns `(train, test)`.
pub fn split_by_memory(pairs: &[LabelledPair]) -> (Vec<LabelledPair>, Vec<LabelledPair>) {
    let mut seen: Vec<&str> = Vec::new();
    for p in pairs {
        if !seen.contains(&p.memory.as_str()) {
            seen.push(&p.memory);
        }
    }
    let held: Vec<&str> = seen[seen.len() / 2..].to_vec();
    let (test, train): (Vec<_>, Vec<_>) = pairs
        .iter()
        .cloned()
        .partition(|p| held.contains(&p.memory.as_str()));
    (train, test)
}

/// The control that decides whether any score on the label set means anything:
/// one `contains` call, no model, no training, no GPU.
///
/// `scripts/d2-labels.sh` builds a query by taking twelve words from ~60%
/// through a memory. If this scores near 1.000, the labelled task is substring
/// provenance rather than relevance, and a method scores well on it exactly
/// insofar as it detects near-exact overlap. The span is now excised, so it
/// should score near zero; every run prints it so a degenerate label file
/// cannot come back quietly.
pub fn verbatim_containment(test: &[LabelledPair]) -> Scored {
    fn flat(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }
    let hits: Vec<f32> = test
        .iter()
        .map(|p| f32::from(u8::from(flat(&p.memory).contains(flat(&p.query).trim()))))
        .collect();
    score_at_half(&hits, test.iter().map(|p| p.relevant))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BertEmbedder;

    /// ADR-0024 Validation 2 for D-2: does a head trained on constructed labels
    /// beat the control?
    ///
    /// The control is 0.785 F1, a threshold over the deployed cross-encoder,
    /// measured by `scripts/d2-relevance-baseline.sh` over the same label file
    /// this reads. It prints the head, the control and the no-model `contains`
    /// check together, so the bar is compared rather than asserted and the
    /// benchmark's own validity is visible in the same output.
    ///
    /// The cross-encoder scored 0.782 on the degenerate set and 0.785 on the
    /// clean one, which is worth knowing: it never used the verbatim shortcut,
    /// while the frozen-encoder head fell from 0.940 to 0.613 because that was
    /// all it had been doing.
    ///
    /// Ignored: it needs the label file and downloads the encoder. Run with
    ///   ANTUMBRA_D2_LABELS=/path/to/d2-labels.json \
    ///   cargo test -p antumbra-serve --features models --lib -- --ignored --nocapture d2_head
    ///
    /// SIZE THE LABEL FILE BEFORE RUNNING IT. `BestChunk` is not cheaper per
    /// unit than the passes above it: MiniLM pads a batch to a fixed shape, so a
    /// 300-character chunk costs nearly what a 4000-character memory costs, and
    /// chunking multiplies the number of passes rather than shrinking them. On
    /// CPU an 800-pair three-way run did not finish in 161 minutes and was
    /// abandoned; 120 pairs completes. The numbers above came from the small
    /// set, which is why they are read as an ordering.
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

        let (train, test) = split_by_memory(&pairs);

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
            // 300 was carried over from a DIFFERENT measurement -- chunk-and-max-
            // pool for ranking -- so it is a borrowed constant rather than one
            // tuned for this question. Sweeping it costs seconds on a GPU and
            // settles whether the remaining gap to the control is a chunk-size
            // choice or something the frozen encoder cannot do.
            (
                "best chunk, 150 chars   ",
                Pairing::BestChunk { chunk_chars: 150 },
            ),
            (
                "best chunk, 200 chars   ",
                Pairing::BestChunk { chunk_chars: 200 },
            ),
            (
                "best chunk, 300 chars   ",
                Pairing::BestChunk { chunk_chars: 300 },
            ),
            (
                "best chunk, 450 chars   ",
                Pairing::BestChunk { chunk_chars: 450 },
            ),
            (
                "best chunk, 600 chars   ",
                Pairing::BestChunk { chunk_chars: 600 },
            ),
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
                if got.f1 > 0.785 { "BEATS 0.785" } else { "" }
            );
        }
        println!("  CONTROL (cross-encoder threshold, span-excised set)      F1=0.785");
        let triv = verbatim_containment(&test);
        println!(
            "  NO MODEL: does the memory contain the query verbatim?  acc={:.3} prec={:.3} rec={:.3} F1={:.3}",
            triv.accuracy, triv.precision, triv.recall, triv.f1
        );
        println!(
            "  If that last line is near 1.000 the benchmark is DEGENERATE: the labels are\n  \
             cut verbatim from the memory, so a short enough chunk IS the query and the task\n  \
             collapses from judging relevance into detecting a substring. Read every score\n  \
             above against this line, not against the control."
        );
    }
}
