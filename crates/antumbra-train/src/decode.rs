//! Decoding helpers that shape raw model output into a clean completion.
//!
//! A base code model, prompted to complete a function, keeps generating past
//! the function (the next definition, a blank-line paragraph, fences). RAFT
//! verifies *and* trains on whatever `generate` returns, so trimming the
//! run-on here keeps the two consistent: the model is reinforced toward the
//! clean completion that actually passed.

/// Default stop markers for code completion: a new top-level definition, a
/// fence, or a `__main__` guard end the useful completion.
pub const DEFAULT_STOPS: &[&str] = &["```", "\ndef ", "\nclass ", "\nif __name__", "\n\n\n"];

use std::collections::HashSet;

use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;

/// How each generation step turns logits into a token. Pure logits math, so the
/// whole policy is unit-tested on CPU (no candle) and reused by the candle
/// sampler. Grounded in the neural-text-degeneration literature: a repetition
/// penalty (Keskar et al, 1909.05858) and nucleus sampling (Holtzman et al,
/// 1904.09751) are what keep greedy/likelihood decoding from looping.
pub struct DecodePolicy {
    /// `<= 0` = greedy/argmax; otherwise softmax temperature.
    pub temperature: f64,
    /// Nucleus cutoff; `>= 1.0` disables truncation. Ignored when greedy.
    pub top_p: f64,
    /// `> 1.0` divides an already-generated token's logit (once per token).
    pub repetition_penalty: f64,
    /// Block tokens that would complete an `n`-gram already in the continuation.
    pub no_repeat_ngram_size: usize,
}

/// Apply the policy to one step's `logits` (given the tokens generated so far,
/// the continuation only, *not* the prompt, so legitimately echoing prompt
/// tokens is never penalized) and pick the next token.
pub fn pick_token(
    mut logits: Vec<f32>,
    generated: &[u32],
    policy: &DecodePolicy,
    rng: &mut StdRng,
) -> u32 {
    // Repetition penalty: at most once per distinct generated token.
    if policy.repetition_penalty > 1.0 {
        let p = policy.repetition_penalty as f32;
        let seen: HashSet<u32> = generated.iter().copied().collect();
        for t in seen {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 { *l / p } else { *l * p };
            }
        }
    }

    // No-repeat n-gram: ban any token that would repeat an n-gram already seen.
    // `n < 2` is a no-op: a 1-gram block would forbid re-emitting *every* token
    // already produced, which is degenerate, not repetition avoidance.
    let n = policy.no_repeat_ngram_size;
    if n >= 2 && generated.len() >= n {
        let prefix = &generated[generated.len() + 1 - n..]; // last n-1 tokens
        for i in 0..=generated.len() - n {
            if &generated[i..i + n - 1] == prefix {
                if let Some(l) = logits.get_mut(generated[i + n - 1] as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Greedy: deterministic, but the penalties above already broke loops.
    if policy.temperature <= 0.0 {
        return argmax(&logits);
    }

    // Temperature softmax (shifted for numerical stability).
    let temp = policy.temperature as f32;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| ((l - max) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(&logits);
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    if policy.top_p < 1.0 {
        nucleus_filter(&mut probs, policy.top_p as f32);
    }

    match WeightedIndex::new(&probs) {
        Ok(dist) => dist.sample(rng) as u32,
        Err(_) => argmax(&logits),
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}

/// Keep only the nucleus: the smallest set of highest-probability tokens whose
/// cumulative mass reaches `top_p`; zero the rest.
fn nucleus_filter(probs: &mut [f32], top_p: f32) {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f32;
    let mut keep = vec![false; probs.len()];
    for &i in &order {
        keep[i] = true;
        cum += probs[i];
        if cum >= top_p {
            break;
        }
    }
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep[i] {
            *p = 0.0;
        }
    }
}

/// Truncate `text` at the earliest occurrence of any stop marker, trimming
/// trailing whitespace. Returns the whole (trimmed) text if no marker is found.
pub fn truncate_at_stops(text: &str, stops: &[&str]) -> String {
    let mut cut = text.len();
    for stop in stops {
        if let Some(idx) = text.find(stop) {
            cut = cut.min(idx);
        }
    }
    text[..cut].trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_at_a_second_definition() {
        let raw = "def add(a, b):\n    return a + b\ndef other():\n    pass\n";
        assert_eq!(
            truncate_at_stops(raw, DEFAULT_STOPS),
            "def add(a, b):\n    return a + b"
        );
    }

    #[test]
    fn stops_at_a_fence() {
        let raw = "def add(a, b):\n    return a + b\n```\nsome prose";
        assert_eq!(
            truncate_at_stops(raw, DEFAULT_STOPS),
            "def add(a, b):\n    return a + b"
        );
    }

    #[test]
    fn keeps_clean_single_function() {
        let raw = "def reverse(s):\n    return s[::-1]";
        assert_eq!(truncate_at_stops(raw, DEFAULT_STOPS), raw);
    }

    #[test]
    fn does_not_cut_indented_nested_def() {
        // a nested (indented) def is "\n    def", not "\ndef", so it survives.
        let raw = "def outer():\n    def inner():\n        return 1\n    return inner";
        assert_eq!(truncate_at_stops(raw, DEFAULT_STOPS), raw);
    }

    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(7)
    }

    fn greedy(temp: f64, rep: f64, ngram: usize) -> DecodePolicy {
        DecodePolicy {
            temperature: temp,
            top_p: 1.0,
            repetition_penalty: rep,
            no_repeat_ngram_size: ngram,
        }
    }

    #[test]
    fn greedy_picks_argmax() {
        // token 2 has the highest logit.
        let logits = vec![0.1, 0.2, 5.0, 0.3];
        let t = pick_token(logits, &[], &greedy(0.0, 1.0, 0), &mut rng());
        assert_eq!(t, 2);
    }

    #[test]
    fn repetition_penalty_demotes_a_repeat() {
        // Greedy would pick token 0 (highest), but it was already generated, so a
        // penalty divides its logit below token 1 -> the loop is broken.
        let logits = vec![3.0, 2.5, 0.1];
        let plain = pick_token(logits.clone(), &[0], &greedy(0.0, 1.0, 0), &mut rng());
        assert_eq!(plain, 0, "without penalty greedy repeats token 0");
        let penalized = pick_token(logits, &[0], &greedy(0.0, 2.0, 0), &mut rng());
        assert_eq!(penalized, 1, "penalty pushes the repeat below token 1");
    }

    #[test]
    fn no_repeat_ngram_blocks_the_loop() {
        // Generated ...a,b ; the only time "a" appeared it was followed by "b".
        // A bigram block (n=2) bans "b" after "a", forcing a different token even
        // though "b" has the top logit.
        let logits = vec![0.0, 0.0, 9.0, 1.0]; // token 2 = "b" highest, token 3 next
        let generated = [2u32, 3, 2]; // last token is 2 ("a"=2 here), 2 was once followed by 3
                                      // After "...2,3,2", the bigram prefix is [2]; 2 was followed by 3 before.
        let t = pick_token(logits, &generated, &greedy(0.0, 1.0, 2), &mut rng());
        assert_ne!(t, 3, "the n-gram that would repeat 2->3 is blocked");
    }

    #[test]
    fn no_repeat_ngram_of_one_is_a_no_op() {
        // A 1-gram block would forbid re-emitting any already-generated token,
        // which is degenerate; n < 2 must leave the logits untouched. Token 2 is
        // the argmax and was already generated, so it is still picked.
        let logits = vec![0.1, 0.2, 5.0, 0.3];
        let t = pick_token(logits, &[2u32], &greedy(0.0, 1.0, 1), &mut rng());
        assert_eq!(t, 2, "a 1-gram block must not ban the generated argmax");
    }

    #[test]
    fn nucleus_truncates_the_tail() {
        // One dominant token (mass ~0.99) and a long tail; top_p 0.9 keeps only the
        // head, so sampling is deterministic on it.
        let mut logits = vec![0.0f32; 100];
        logits[42] = 20.0; // overwhelmingly likely
        let p = DecodePolicy {
            temperature: 1.0,
            top_p: 0.9,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
        };
        for _ in 0..10 {
            assert_eq!(pick_token(logits.clone(), &[], &p, &mut rng()), 42);
        }
    }
}
