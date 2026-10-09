use super::*;

use std::collections::HashMap;

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;

use crate::decision_probe::{split_by_memory, verbatim_containment};

/// A ModernBERT small enough to build and train on a CPU in a test: two
/// layers, one of them local, sixteen wide.
fn tiny_config() -> Config {
    serde_json::from_value(serde_json::json!({
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 0,
        "global_attn_every_n_layers": 2,
        "global_rope_theta": 10000.0,
        "local_attention": 8,
        "local_rope_theta": 10000.0
    }))
    .expect("tiny config")
}

fn tiny_model(varmap: &VarMap, pooling: Pooling) -> PairModel {
    let vb = VarBuilder::from_varmap(varmap, DType::F32, &Device::Cpu);
    PairModel::new(vb, &tiny_config(), pooling).expect("build tiny model")
}

/// Rows of token ids, all the same width, with a full mask.
fn rows(ids: &[[u32; 6]]) -> (Tensor, Tensor) {
    let flat: Vec<u32> = ids.iter().flatten().copied().collect();
    let shape = (ids.len(), 6);
    (
        Tensor::from_vec(flat, shape, &Device::Cpu).unwrap(),
        Tensor::ones(shape, DType::U32, &Device::Cpu).unwrap(),
    )
}

#[test]
fn the_model_gives_two_logits_per_pair_under_either_pooling() {
    for pooling in [Pooling::Cls, Pooling::Mean] {
        let varmap = VarMap::new();
        let model = tiny_model(&varmap, pooling);
        let (ids, mask) = rows(&[[1, 5, 2, 7, 8, 2], [1, 6, 2, 9, 10, 2], [1, 5, 2, 3, 4, 2]]);
        let logits = model.logits(&ids, &mask).unwrap();
        assert_eq!(logits.dims(), &[3, 2], "{pooling:?}");
    }
}

/// The training step reaches every part of the model: on a signal only the
/// query token carries, the loss falls and the pairs separate.
#[test]
fn training_learns_a_signal_carried_by_the_query() {
    let varmap = VarMap::new();
    let model = tiny_model(&varmap, Pooling::Mean);
    // Token 5 after [CLS] means "relevant", token 6 means not; the memory
    // tokens are shared noise.
    let data = [
        ([1, 5, 2, 7, 8, 2], 1u32),
        ([1, 6, 2, 7, 8, 2], 0),
        ([1, 5, 2, 9, 10, 2], 1),
        ([1, 6, 2, 9, 10, 2], 0),
        ([1, 5, 2, 11, 12, 2], 1),
        ([1, 6, 2, 11, 12, 2], 0),
    ];
    let ids: Vec<[u32; 6]> = data.iter().map(|(r, _)| *r).collect();
    let (ids, mask) = rows(&ids);
    let labels: Vec<u32> = data.iter().map(|(_, l)| *l).collect();
    let labels = Tensor::new(labels.as_slice(), &Device::Cpu).unwrap();
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 1e-2,
            ..Default::default()
        },
    )
    .unwrap();
    let loss_of = |m: &PairModel| {
        candle_nn::loss::cross_entropy(&m.logits(&ids, &mask).unwrap(), &labels).unwrap()
    };
    let first = loss_of(&model).to_scalar::<f32>().unwrap();
    for _ in 0..60 {
        opt.backward_step(&loss_of(&model)).unwrap();
    }
    let last = loss_of(&model).to_scalar::<f32>().unwrap();
    assert!(last < first * 0.25, "loss {first} -> {last}");
    let probs = candle_nn::ops::softmax(&model.logits(&ids, &mask).unwrap(), D::Minus1)
        .unwrap()
        .i((.., 1))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let scored = score_at_half(&probs, data.iter().map(|(_, l)| *l == 1));
    assert_eq!(scored.accuracy, 1.0, "{probs:?}");
}

/// A checkpoint of a tiny model's own variables, minus the classifier and
/// minus any names in `drop`.
fn checkpoint_of(drop: &[&str]) -> HashMap<String, Tensor> {
    let source = VarMap::new();
    let _ = tiny_model(&source, Pooling::Mean);
    let tensors = source
        .data()
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| !name.starts_with("classifier.") && !drop.contains(&name.as_str()))
        .map(|(name, var)| (name.clone(), var.as_tensor().copy().unwrap()))
        .collect();
    tensors
}

fn build_from(checkpoint: HashMap<String, Tensor>) -> candle_core::Result<(VarMap, PairModel)> {
    let varmap = VarMap::new();
    let model = PairModel::new(
        from_checkpoint(checkpoint, &varmap, &Device::Cpu),
        &tiny_config(),
        Pooling::Mean,
    )?;
    Ok((varmap, model))
}

/// Built from a checkpoint, every variable but the classifier holds the
/// checkpoint's value, and every one of them trains.
#[test]
fn a_checkpoint_supplies_every_variable_but_the_classifier() {
    let checkpoint = checkpoint_of(&[]);
    let (varmap, _) = build_from(checkpoint.clone()).unwrap();
    // Counted before the lock below: `all_vars` takes the same lock.
    let trainable = varmap.all_vars().len();
    let vars = varmap.data().lock().unwrap();
    let fresh: Vec<&String> = vars
        .keys()
        .filter(|n| !checkpoint.contains_key(*n))
        .collect();
    let mut fresh: Vec<&str> = fresh.iter().map(|s| s.as_str()).collect();
    fresh.sort_unstable();
    assert_eq!(fresh, ["classifier.bias", "classifier.weight"]);
    let name = "model.final_norm.weight";
    assert_eq!(
        vars[name].as_tensor().to_vec1::<f32>().unwrap(),
        checkpoint[name].to_vec1::<f32>().unwrap()
    );
    assert_eq!(trainable, vars.len());
}

/// ModernBERT's first layer has no attention-norm weight, so candle leaves
/// that norm out. Built from such a checkpoint, the norm stays out rather than
/// starting fresh, which would re-normalize the embeddings and stop being the
/// pretrained model.
#[test]
fn a_norm_the_checkpoint_lacks_is_left_out_not_started_fresh() {
    let norm = "model.layers.0.attn_norm.weight";
    let (varmap, model) = build_from(checkpoint_of(&[norm])).unwrap();
    assert!(!varmap.data().lock().unwrap().contains_key(norm));
    let (ids, mask) = rows(&[[1, 5, 2, 7, 8, 2]]);
    assert_eq!(model.logits(&ids, &mask).unwrap().dims(), &[1, 2]);
}

/// A variable the model needs and the checkpoint lacks is an error when the
/// model is built, not a silent random start.
#[test]
fn a_required_tensor_the_checkpoint_lacks_is_an_error() {
    let refused = build_from(checkpoint_of(&["model.final_norm.weight"]))
        .err()
        .expect("missing tensor refused")
        .to_string();
    assert!(refused.contains("not in the checkpoint"), "{refused}");
}

/// A word-level tokenizer with the pair template ModernBERT's uses.
fn tiny_tokenizer() -> Tokenizer {
    let words = [
        "[PAD]", "[CLS]", "[SEP]", "[UNK]", "where", "is", "the", "key", "under", "mat", "a", "b",
    ];
    let vocab: HashMap<String, u32> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.to_string(), i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    let template = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .unwrap()
        .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
        .unwrap()
        .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
        .build()
        .unwrap();
    tokenizer.with_post_processor(Some(template));
    tokenizer
}

fn pair(query: &str, memory: &str, relevant: bool) -> LabeledPair {
    LabeledPair {
        query: query.into(),
        memory: memory.into(),
        relevant,
    }
}

/// A pair reads as one sequence, the memory is what gets cut, and a batch is
/// padded to its longest pair with the padding masked out.
#[test]
fn pairs_are_one_sequence_cut_from_the_memory_and_padded() {
    let mut tokenizer = tiny_tokenizer();
    // 14 tokens: three special, the 4-token query whole, 7 of the memory's 8.
    configure_tokenizer(&mut tokenizer, 14, 0).unwrap();
    let long = pair("where is the key", "under the mat under a b a b", true);
    let short = pair("key", "mat", false);
    let (ids, mask) = encode_pairs(&tokenizer, &[&long, &short], &Device::Cpu).unwrap();
    let ids = ids.to_vec2::<u32>().unwrap();
    let mask = mask.to_vec2::<u32>().unwrap();
    // [CLS] where is the key [SEP] under the mat under a b a [SEP]: the query
    // whole, the memory's last token cut.
    assert_eq!(ids[0], vec![1, 4, 5, 6, 7, 2, 8, 6, 9, 8, 10, 11, 10, 2]);
    assert_eq!(mask[0], vec![1; 14]);
    // [CLS] key [SEP] mat [SEP], then padding the mask leaves out.
    assert_eq!(ids[1], vec![1, 7, 2, 9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(mask[1], vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn the_learning_rate_warms_up_then_decays_to_zero() {
    let peak = 1e-3;
    assert!(scheduled_lr(peak, 0, 100) < peak);
    assert_eq!(scheduled_lr(peak, 9, 100), peak);
    assert!(scheduled_lr(peak, 50, 100) < peak);
    assert!(scheduled_lr(peak, 99, 100) < scheduled_lr(peak, 50, 100));
    assert!(scheduled_lr(peak, 100, 100).abs() < 1e-12);
}

#[test]
fn the_shuffle_is_a_permutation_fixed_by_its_seed() {
    let a = shuffled(50, 7);
    let mut sorted = a.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..50).collect::<Vec<_>>());
    assert_eq!(a, shuffled(50, 7));
    assert_ne!(a, shuffled(50, 8));
}

/// ADR-0024 Validation 2 for D-2, with a pair encoder: does a fine-tuned
/// cross-attention model beat the control?
///
/// The control is 0.785 F1, a threshold over the deployed cross-encoder, and
/// the floor that ships maps that cross-encoder's score through a fitted
/// logistic, 0.797. The frozen-encoder heads reached 0.613 at best. This reads
/// the same label file, splits it the same way and scores it at the same cut.
///
/// Ignored: it needs the label file, downloads the encoder and trains it, which
/// wants a GPU. Run it in the `d2-probe` image with the filter `d2_pair`:
///   ANTUMBRA_D2_LABELS=/path/to/d2-labels.json \
///   cargo test -p antumbra-serve --features models --lib -- --ignored --nocapture d2_pair
/// Knobs: ANTUMBRA_D2_ENCODER (default answerdotai/ModernBERT-base),
/// ANTUMBRA_D2_MAX_TOKENS (1024), ANTUMBRA_D2_EPOCHS (3), ANTUMBRA_D2_BATCH (4),
/// ANTUMBRA_D2_LR (3e-5), ANTUMBRA_D2_SEED (0).
#[test]
#[ignore = "needs ANTUMBRA_D2_LABELS, downloads and fine-tunes an encoder"]
fn d2_pair_encoder_against_the_control() {
    let Ok(path) = std::env::var("ANTUMBRA_D2_LABELS") else {
        println!("ANTUMBRA_D2_LABELS unset -- skipped");
        return;
    };
    let raw = std::fs::read_to_string(&path).expect("read labels");
    let pairs: Vec<LabeledPair> = serde_json::from_str(&raw).expect("parse labels");
    assert!(pairs.len() >= 20, "need a real set, got {}", pairs.len());
    let (train, test) = split_by_memory(&pairs);

    fn knob<T: std::str::FromStr>(name: &str, default: T) -> T {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    }
    let base = FineTune::default();
    let cfg = FineTune {
        model: knob("ANTUMBRA_D2_ENCODER", base.model.clone()),
        max_tokens: knob("ANTUMBRA_D2_MAX_TOKENS", base.max_tokens),
        epochs: knob("ANTUMBRA_D2_EPOCHS", base.epochs),
        batch: knob("ANTUMBRA_D2_BATCH", base.batch),
        lr: knob("ANTUMBRA_D2_LR", base.lr),
        seed: knob("ANTUMBRA_D2_SEED", base.seed),
    };
    println!(
        "\n  {} pairs: {} train / {} held out, split by memory",
        pairs.len(),
        train.len(),
        test.len()
    );
    let started = std::time::Instant::now();
    let got = fine_tune_and_score(&train, &test, &cfg, &mut |line| println!("  {line}"))
        .expect("fine-tune");
    println!(
        "  PAIR ENCODER  acc={:.3} prec={:.3} rec={:.3} F1={:.3}  in {:.0}s  {}",
        got.accuracy,
        got.precision,
        got.recall,
        got.f1,
        started.elapsed().as_secs_f64(),
        if got.f1 > 0.797 {
            "BEATS the shipped floor (0.797) and the control (0.785)"
        } else if got.f1 > 0.785 {
            "beats the control (0.785), not the shipped floor (0.797)"
        } else {
            ""
        }
    );
    println!("  CONTROL (cross-encoder threshold, span-excised set)  F1=0.785");
    println!("  SHIPPED FLOOR (logistic over the cross-encoder)      F1=0.797");
    let triv = verbatim_containment(&test);
    println!(
        "  NO MODEL: does the memory contain the query verbatim?  acc={:.3} F1={:.3}",
        triv.accuracy, triv.f1
    );
}
