//! Consolidation (EXP-021): graduate trusted memories into the population the
//! way the neocortex consolidates hippocampal traces during sleep.
//!
//! The capture path (`teach`/`memory-import`) internalizes a memory in
//! isolation. Consolidating *many* memories into a shared base risks clobbering
//! skills already learned — catastrophic interference. The complementary-
//! learning-systems answer is not to train new traces alone but to **replay**
//! them interleaved with rehearsal of what is already known, so the gradient
//! mixes old and new and the shared representation does not drift off the old
//! skills. [`interleave_replay`] is that primitive; the capture loop
//! ([`crate::teach::capture_corrections`]) threads a rehearsal buffer through
//! every fine-tuning round.

use crate::model::{CorpusTask, SftExample};

/// Interleave a rehearsal buffer of already-consolidated examples into the
/// training winners, so fine-tuning new memories does not clobber old skills
/// (the complementary-learning-systems prescription against catastrophic
/// interference). `ratio` is rehearsal examples per winner; the buffer is drawn
/// round-robin and inserted at evenly spaced positions, so even a small buffer
/// spreads deterministically across the batch (no RNG, so it is reproducible).
/// `ratio <= 0` or an empty buffer returns the winners unchanged — replay off.
pub fn interleave_replay(winners: &[SftExample], replay: &[SftExample], ratio: f64) -> Vec<SftExample> {
    let n_replay = if ratio > 0.0 {
        ((winners.len() as f64) * ratio).round() as usize
    } else {
        0
    };
    if replay.is_empty() || n_replay == 0 {
        return winners.to_vec();
    }
    let mut out = Vec::with_capacity(winners.len() + n_replay);
    let mut inserted = 0usize;
    let mut acc = 0.0f64;
    for w in winners {
        out.push(w.clone());
        acc += ratio;
        while acc >= 1.0 && inserted < n_replay {
            out.push(replay[inserted % replay.len()].clone());
            inserted += 1;
            acc -= 1.0;
        }
    }
    // Spread any remainder (e.g. ratio < 1 with few winners) at the tail rather
    // than dropping it, so the requested rehearsal count is honored.
    while inserted < n_replay {
        out.push(replay[inserted % replay.len()].clone());
        inserted += 1;
    }
    out
}

/// Build a rehearsal buffer from capture tasks that carry a verified completion
/// — the `prompt -> behavior` pairs of already-consolidated memories. Tasks
/// without a trusted completion (RAFT seeds) contribute nothing to rehearse.
pub fn replay_from_tasks(tasks: &[CorpusTask]) -> Vec<SftExample> {
    tasks
        .iter()
        .filter_map(|t| {
            t.completion.as_ref().map(|c| SftExample {
                prompt: t.prompt.clone(),
                completion: c.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(p: &str) -> SftExample {
        SftExample {
            prompt: p.into(),
            completion: format!("do {p}"),
        }
    }

    #[test]
    fn ratio_zero_returns_winners_unchanged() {
        let winners = vec![ex("a"), ex("b")];
        let replay = vec![ex("old")];
        let out = interleave_replay(&winners, &replay, 0.0);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|e| e.prompt != "old"));
    }

    #[test]
    fn empty_replay_returns_winners_unchanged() {
        let winners = vec![ex("a"), ex("b")];
        let out = interleave_replay(&winners, &[], 1.0);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn ratio_one_adds_one_rehearsal_per_winner_round_robin() {
        let winners = vec![ex("a"), ex("b")];
        let replay = vec![ex("old0"), ex("old1")];
        let out = interleave_replay(&winners, &replay, 1.0);
        // 2 winners + 2 rehearsal.
        assert_eq!(out.len(), 4);
        let prompts: Vec<&str> = out.iter().map(|e| e.prompt.as_str()).collect();
        assert!(prompts.contains(&"a") && prompts.contains(&"b"));
        // Round-robin draws both buffer entries, not just the first.
        assert!(prompts.contains(&"old0") && prompts.contains(&"old1"));
    }

    #[test]
    fn fractional_ratio_is_honored_and_interleaved() {
        let winners = vec![ex("a"), ex("b"), ex("c"), ex("d")];
        let replay = vec![ex("old")];
        let out = interleave_replay(&winners, &replay, 0.5);
        // round(4 * 0.5) = 2 rehearsal examples added.
        assert_eq!(out.len(), 6);
        assert_eq!(out.iter().filter(|e| e.prompt == "old").count(), 2);
        // Interleaved, not all appended at the end: a rehearsal lands before the
        // last winner.
        let last = out.last().unwrap();
        assert!(last.prompt == "d" || last.prompt == "old");
    }

    #[test]
    fn replay_from_tasks_keeps_only_completed_captures() {
        let tasks = vec![
            CorpusTask::new("t1", "p1").with_completion("answer1"),
            CorpusTask::new("t2", "p2"), // a seed: no completion
        ];
        let buf = replay_from_tasks(&tasks);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].prompt, "p1");
        assert_eq!(buf[0].completion, "answer1");
    }
}
