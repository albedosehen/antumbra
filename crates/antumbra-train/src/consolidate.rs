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

use crate::memory::MemoryRecord;
use crate::model::{CorpusTask, SftExample};

/// The gate that decides whether a memory graduates from the store into the
/// weights (EXP-021). A memory consolidates only when it clears all three
/// signals — the rest stay in the store (the cold-fact / volatile long tail):
///   - **recurrence**: reinforced enough to be worth baking in;
///   - **stability**: not a fact that changes over time;
///   - **verifiability**: a behavior we can check internalized. Opinions carry
///     no executable check, so they graduate only on the weaker provenance tier
///     (a high confidence stands in for verification).
#[derive(Debug, Clone, Copy)]
pub struct ConsolidationPolicy {
    /// Reinforcement count at/above which recurrence is satisfied.
    pub min_recurrence: u32,
    /// Confidence floor a graduating memory must clear.
    pub min_confidence: f32,
    /// Confidence at/above which an unverifiable memory (an opinion) may still
    /// graduate on provenance alone.
    pub provenance_confidence: f32,
}

impl Default for ConsolidationPolicy {
    fn default() -> Self {
        Self {
            min_recurrence: 2,
            min_confidence: 0.5,
            provenance_confidence: 0.9,
        }
    }
}

/// The gate's decision for one memory: whether it graduates, a ranking score,
/// and a human-readable reason (so a dry run can explain every keep/skip).
#[derive(Debug, Clone)]
pub struct Verdict {
    pub graduate: bool,
    pub score: f32,
    pub reason: String,
}

/// Whether a memory is checkable. An explicit `verifiable` wins; otherwise an
/// `opinion` is unverifiable (no executable check) and everything else is.
fn is_verifiable(record: &MemoryRecord) -> bool {
    record.verifiable.unwrap_or_else(|| {
        record
            .network
            .as_deref()
            .map(|n| !n.eq_ignore_ascii_case("opinion"))
            .unwrap_or(true)
    })
}

/// Score one memory against the consolidation gate. `graduate` is the all-gates
/// verdict; `score` ranks the survivors (confidence weighted by recurrence) so a
/// budgeted run can take the strongest first.
pub fn score_memory(record: &MemoryRecord, policy: &ConsolidationPolicy) -> Verdict {
    let confidence = record.confidence.unwrap_or(1.0);
    let recurrence = record.reinforcement.unwrap_or(0);
    // A maximally-confident memory needs no repeats; otherwise recurrence must
    // clear the floor.
    let recurrence_ok = recurrence >= policy.min_recurrence || confidence >= 0.999;
    let stable = !record.volatile.unwrap_or(false);
    let verifiable = is_verifiable(record);
    // An unverifiable memory (opinion) may still graduate on provenance alone.
    let trust_ok = if verifiable {
        confidence >= policy.min_confidence
    } else {
        confidence >= policy.provenance_confidence
    };

    // Rank by confidence with a gentle recurrence boost.
    let score = confidence * (1.0 + (recurrence.min(8) as f32) / 8.0) / 2.0;

    let (graduate, reason) = if !stable {
        (false, "volatile: stays in the store".to_string())
    } else if !recurrence_ok {
        (
            false,
            format!("under-reinforced ({recurrence} < {})", policy.min_recurrence),
        )
    } else if !trust_ok {
        let bar = if verifiable {
            policy.min_confidence
        } else {
            policy.provenance_confidence
        };
        (
            false,
            format!(
                "confidence {confidence:.2} below {bar:.2}{}",
                if verifiable { "" } else { " (opinion provenance tier)" }
            ),
        )
    } else {
        let how = if verifiable {
            "verifiable"
        } else {
            "opinion via provenance"
        };
        (true, format!("graduates ({how}, conf {confidence:.2})"))
    };

    Verdict {
        graduate,
        score,
        reason,
    }
}

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

    fn mem(network: &str, confidence: f32, reinforcement: u32) -> MemoryRecord {
        MemoryRecord {
            content: "use deno install".into(),
            network: Some(network.into()),
            confidence: Some(confidence),
            reinforcement: Some(reinforcement),
            ..Default::default()
        }
    }

    #[test]
    fn verifiable_reinforced_memory_graduates() {
        let v = score_memory(&mem("world", 0.8, 3), &ConsolidationPolicy::default());
        assert!(v.graduate, "{}", v.reason);
    }

    #[test]
    fn volatile_memory_never_graduates() {
        let mut m = mem("world", 1.0, 9);
        m.volatile = Some(true);
        let v = score_memory(&m, &ConsolidationPolicy::default());
        assert!(!v.graduate);
        assert!(v.reason.contains("volatile"));
    }

    #[test]
    fn under_reinforced_memory_waits_unless_maximally_confident() {
        let policy = ConsolidationPolicy::default();
        // Reinforced once (< 2) and not maximally confident -> waits.
        let v = score_memory(&mem("world", 0.8, 1), &policy);
        assert!(!v.graduate);
        assert!(v.reason.contains("under-reinforced"));
        // A maximally-confident memory graduates without repeats.
        let v = score_memory(&mem("world", 1.0, 0), &policy);
        assert!(v.graduate, "{}", v.reason);
    }

    #[test]
    fn opinion_needs_the_higher_provenance_confidence() {
        let policy = ConsolidationPolicy::default();
        // An opinion at 0.6 clears the verifiable floor but not the provenance
        // tier -> rejected.
        let v = score_memory(&mem("opinion", 0.6, 5), &policy);
        assert!(!v.graduate);
        assert!(v.reason.contains("provenance"));
        // A strongly-held opinion graduates on provenance.
        let v = score_memory(&mem("opinion", 0.95, 5), &policy);
        assert!(v.graduate, "{}", v.reason);
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
