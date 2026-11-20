//! What a model is likely to write next, guessed from what it has written
//! before.
//!
//! A model that decodes greedily on a CPU reads its weights once for every
//! token, and the reading is most of the time. The same reading can carry
//! further rows for a part of what it costs, so tokens guessed ahead can
//! ride along with the one in hand: where the model's own choice after a
//! guess's row is the next guess, that token is decided too, without a
//! reading of its own. A wrong guess costs its row and nothing else: what
//! is written is what the model would have written a token at a time.
//!
//! So a guess only has to be cheap and often right. Here it is the token
//! that has followed the same run of tokens before: in the request's own
//! prompt, in what the model has written so far, and in what it wrote for
//! earlier requests. An expert answers the tasks it was
//! taught in the words it was taught, and says the same things again. And
//! where the very prompt was answered before, the guess is what was written
//! after it then, whole: greedy decoding answers a prompt the same way
//! every time, until the expert behind it changes.
//!
//! Pure bookkeeping over token ids, with no model in it.

use std::collections::{HashMap, VecDeque};

/// The fewest and the most tokens a guess is looked up by. A token is
/// remembered after every run of these lengths that ends before it, and
/// looked up by the longest run that is known. One token is run enough to
/// offer a guess by: over one expert's held-out tasks a guess by the token
/// before it alone was right about one time in six, by a run of two one
/// time in three, of three one time in two, and of four seven times in
/// ten. Whether a guess is worth carrying is for whoever carries it to
/// weigh, by the run it comes with ([`Guess::run`]).
const LEAST: usize = 1;
const MOST: usize = 4;

/// Runs a memory holds before it lets its older half go: the runs before
/// some sixteen thousand tokens, in about two megabytes, twice over.
const ROOM: usize = 1 << 16;

/// Prompts a memory keeps the whole answer to: the latest of them.
const ANSWERS: usize = 256;

/// How far a run's token is held to: each time it follows the run again it
/// is held one firmer, up to this, and each time another token follows it
/// is held one looser. Another token takes its place only when it is held
/// to no more.
const HELD_MOST: u8 = 3;

/// The token that follows a run, and how firmly it is held.
#[derive(Clone, Copy)]
struct Held {
    token: u32,
    firm: u8,
}

/// The token that has followed each run of tokens, and what was written
/// whole after the latest prompts. A run's token is the one that followed it
/// first, until another has followed it more often than it since: an expert
/// that says a thing one way nine times and another way once is guessed to
/// say it the first way the next time. Replayed over one expert's held-out
/// tasks that took 1378 passes where believing whatever followed a run last
/// took 1392.
#[derive(Default)]
pub struct Drafts {
    /// The runs seen lately, and those seen before them: when the first is
    /// full it becomes the second, and what the second held is forgotten.
    now: HashMap<u64, Held>,
    before: HashMap<u64, Held>,
    /// What was last written after a prompt, by the prompt as one word; and
    /// the prompts in the order they were answered, the oldest first.
    said: HashMap<u64, Vec<u32>>,
    answered: VecDeque<u64>,
}

/// A run of tokens as one word. Two runs may share a word, once in a great
/// many: a guess looked up by the wrong run is a wrong guess, and costs what
/// any wrong guess costs. The word starts from the run's length mixed into a
/// fixed one, so that no run of small tokens brings it to nothing, where
/// runs that begin alike would meet.
fn key(run: &[u32]) -> u64 {
    let mut word = 0x243F_6A88_85A3_08D3 ^ (run.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for token in run {
        word = (word ^ u64::from(*token)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        word ^= word >> 32;
    }
    word
}

impl Drafts {
    /// Nothing remembered.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many runs are remembered.
    pub fn len(&self) -> usize {
        self.now.len() + self.before.len()
    }

    pub fn is_empty(&self) -> bool {
        self.now.is_empty() && self.before.is_empty() && self.said.is_empty()
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        self.now.clear();
        self.before.clear();
        self.said.clear();
        self.answered.clear();
    }

    /// Keep whole what was written after `prompt`, in place of what was
    /// kept for it before. The prompt itself is kept only as one word; of
    /// more prompts than a memory keeps, the one answered longest ago is let
    /// go.
    pub fn keep(&mut self, prompt: &[u32], written: &[u32]) {
        let prompt = key(prompt);
        if self.said.insert(prompt, written.to_vec()).is_none() {
            self.answered.push_back(prompt);
            if self.answered.len() > ANSWERS {
                if let Some(oldest) = self.answered.pop_front() {
                    self.said.remove(&oldest);
                }
            }
        }
    }

    /// What was last written after `prompt`, if that is kept. Two prompts
    /// may share a word, so this is a guess like any other.
    pub fn said(&self, prompt: &[u32]) -> Option<&[u32]> {
        self.said.get(&key(prompt)).map(Vec::as_slice)
    }

    /// Remember each token of `tokens` from `from` on by the runs that end
    /// before it. The tokens before `from` are only ever what a run is made
    /// of: a prompt's own tokens are not remembered as what follows.
    pub fn learn(&mut self, tokens: &[u32], from: usize) {
        for at in from.max(LEAST)..tokens.len() {
            for run in LEAST..=MOST.min(at) {
                let (key, token) = (key(&tokens[at - run..at]), tokens[at]);
                let held = match self.held(key) {
                    Some(held) if held.token == token => Held {
                        token,
                        firm: (held.firm + 1).min(HELD_MOST),
                    },
                    Some(held) if held.firm > 0 => Held {
                        token: held.token,
                        firm: held.firm - 1,
                    },
                    _ => Held { token, firm: 0 },
                };
                if self.now.len() >= ROOM {
                    self.before = std::mem::take(&mut self.now);
                }
                self.now.insert(key, held);
            }
        }
    }

    /// What is held for the run with this word, lately or before.
    fn held(&self, key: u64) -> Option<Held> {
        self.now
            .get(&key)
            .or_else(|| self.before.get(&key))
            .copied()
    }

    /// The token remembered after the longest run that ends `tail`, and how
    /// long that run is.
    fn after(&self, tail: &[u32]) -> Option<(usize, u32)> {
        (LEAST..=MOST.min(tail.len())).rev().find_map(|run| {
            self.held(key(&tail[tail.len() - run..]))
                .map(|held| (run, held.token))
        })
    }
}

/// A token guessed, and how many tokens the run it was looked up by was:
/// the longer the run, the likelier the guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Guess {
    pub token: u32,
    pub run: usize,
}

/// Up to `most` tokens guessed to follow `context`, into `out`: each the
/// token one of `memories` remembers after the longest run it knows, the
/// guesses before it taken as written. Of two memories that know a run of
/// the same length the earlier in `memories` is believed. The guessing stops
/// at the first token nothing is remembered for, and never offers `end`.
pub fn guess(memories: &[&Drafts], context: &[u32], most: usize, end: u32, out: &mut Vec<Guess>) {
    out.clear();
    let mut tail = [0u32; MOST];
    let held = context.len().min(MOST);
    tail[MOST - held..].copy_from_slice(&context[context.len() - held..]);
    let mut known = held;
    while out.len() < most {
        let run = &tail[MOST - known..];
        let found = memories
            .iter()
            .filter_map(|memory| memory.after(run))
            // The longest run, and of equals the first memory's.
            .reduce(|best, next| if next.0 > best.0 { next } else { best });
        let Some((run, token)) = found else { break };
        if token == end {
            break;
        }
        out.push(Guess { token, run });
        tail.copy_within(1.., 0);
        tail[MOST - 1] = token;
        known = (known + 1).min(MOST);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Outcome = Result<(), Box<dyn std::error::Error>>;

    const END: u32 = 999;

    fn guessed(memories: &[&Drafts], context: &[u32], most: usize) -> Vec<u32> {
        let mut out = vec![Guess { token: 7, run: 7 }; 3];
        guess(memories, context, most, END, &mut out);
        out.iter().map(|guess| guess.token).collect()
    }

    /// What followed a run is guessed after it again, as far as the runs go
    /// on, by the longest run that is known, and no further than asked.
    #[test]
    fn what_followed_a_run_is_guessed_after_it() -> Outcome {
        let mut memory = Drafts::new();
        assert!(memory.is_empty());
        memory.learn(&[1, 2, 3, 4, 5, 6], 0);
        assert!(!memory.is_empty());
        assert_eq!(
            memory.len(),
            14,
            "every run of one to four tokens before a token"
        );
        assert_eq!(guessed(&[&memory], &[9, 1, 2], 8), [3, 4, 5, 6]);
        // Each by the longest run that was known when it was guessed: the
        // two tokens before the first, and one more with every guess taken
        // as written, up to the longest a guess is looked up by.
        let mut runs = Vec::new();
        guess(&[&memory], &[9, 1, 2], 8, END, &mut runs);
        assert_eq!(
            runs.iter().map(|guess| guess.run).collect::<Vec<_>>(),
            [2, 3, 4, 4]
        );
        assert_eq!(
            guessed(&[&memory], &[1, 2], 2),
            [3, 4],
            "no more than asked"
        );
        assert_eq!(
            guessed(&[&memory], &[2], 8),
            [3, 4, 5, 6],
            "one token is run enough"
        );
        assert_eq!(
            guessed(&[&memory], &[8, 9], 8),
            [] as [u32; 0],
            "a run never seen"
        );
        assert_eq!(guessed(&[&memory], &[], 8), [] as [u32; 0]);
        assert_eq!(guessed(&[&memory], &[1, 2], 0), [] as [u32; 0]);

        // The longer run decides: after 7 2 3 came 8, after 2 3 alone came 4.
        memory.learn(&[7, 2, 3, 8], 0);
        memory.learn(&[1, 2, 3, 4], 0);
        assert_eq!(guessed(&[&memory], &[7, 2, 3], 1), [8]);
        assert_eq!(guessed(&[&memory], &[5, 2, 3], 1), [4]);
        // What a run was last followed by is what is remembered.
        memory.learn(&[5, 2, 3, 6], 0);
        assert_eq!(guessed(&[&memory], &[0, 2, 3], 1), [6]);
        Ok(())
    }

    /// Tokens before `from` are what runs are made of and are not themselves
    /// remembered, and the end is never offered.
    #[test]
    fn a_prompt_is_not_remembered_and_the_end_is_not_guessed() -> Outcome {
        let mut memory = Drafts::new();
        memory.learn(&[1, 2, 3, 4, 5, END], 3);
        assert_eq!(
            guessed(&[&memory], &[1, 2], 4),
            [] as [u32; 0],
            "3 was the prompt's"
        );
        assert_eq!(
            guessed(&[&memory], &[2, 3], 4),
            [4, 5],
            "and stops before the end"
        );
        memory.clear();
        assert!(memory.is_empty());
        Ok(())
    }

    /// Of several memories the longest run is believed, and of runs of one
    /// length the first memory's.
    #[test]
    fn several_memories_are_asked_and_the_longest_run_believed() -> Outcome {
        let (mut near, mut far) = (Drafts::new(), Drafts::new());
        near.learn(&[2, 3, 10], 0);
        far.learn(&[1, 2, 3, 20, 21], 0);
        assert_eq!(
            guessed(&[&near, &far], &[1, 2, 3], 1),
            [20],
            "three tokens beat two"
        );
        assert_eq!(
            guessed(&[&near, &far], &[9, 2, 3], 1),
            [10],
            "the first of equals"
        );
        assert_eq!(guessed(&[&far, &near], &[9, 2, 3], 1), [20]);
        assert_eq!(guessed(&[&near, &far], &[1, 2, 3], 3), [20, 21]);
        Ok(())
    }

    /// What was written after a prompt is kept whole, the latest for each
    /// prompt, for as many prompts as a memory keeps.
    #[test]
    fn what_was_written_after_a_prompt_is_kept_whole() -> Outcome {
        let mut memory = Drafts::new();
        assert_eq!(memory.said(&[1, 2, 3]), None);
        memory.keep(&[1, 2, 3], &[7, 8, 9]);
        memory.keep(&[1, 2], &[5]);
        assert!(!memory.is_empty());
        assert_eq!(memory.said(&[1, 2, 3]), Some(&[7u32, 8, 9][..]));
        assert_eq!(memory.said(&[1, 2]), Some(&[5u32][..]));
        assert_eq!(memory.said(&[3, 2, 1]), None, "another prompt");
        memory.keep(&[1, 2, 3], &[4]);
        assert_eq!(memory.said(&[1, 2, 3]), Some(&[4u32][..]), "the latest");
        // More prompts than are kept: the oldest go, the latest stay.
        for prompt in 0..ANSWERS as u32 {
            memory.keep(&[100, prompt], &[prompt]);
        }
        assert_eq!(memory.said(&[1, 2]), None);
        assert_eq!(memory.said(&[1, 2, 3]), None);
        assert_eq!(memory.said(&[100, 0]), Some(&[0u32][..]));
        let last = ANSWERS as u32 - 1;
        assert_eq!(memory.said(&[100, last]), Some(&[last][..]));
        memory.clear();
        assert!(memory.is_empty() && memory.said(&[100, last]).is_none());
        Ok(())
    }

    /// Runs that are not the same run do not share a word, over every run
    /// of one to three small tokens: the tokens a short guess is made by.
    #[test]
    fn short_runs_of_small_tokens_have_words_of_their_own() -> Outcome {
        let mut words = std::collections::HashSet::new();
        let mut runs = 0;
        for a in 0..24u32 {
            runs += 1;
            assert!(words.insert(key(&[a])), "{a}");
            for b in 0..24u32 {
                runs += 1;
                assert!(words.insert(key(&[a, b])), "{a} {b}");
                for c in 0..24u32 {
                    runs += 1;
                    assert!(words.insert(key(&[a, b, c])), "{a} {b} {c}");
                }
            }
        }
        assert_eq!(words.len(), runs);
        Ok(())
    }

    /// A run's token is held until another has followed the run more often
    /// than it since: once is not enough to unseat a token seen three times,
    /// and three times is.
    #[test]
    fn a_runs_token_is_held_until_it_is_outvoted() -> Outcome {
        let mut memory = Drafts::new();
        for _ in 0..3 {
            memory.learn(&[1, 2, 3], 2);
        }
        memory.learn(&[1, 2, 9], 2);
        assert_eq!(guessed(&[&memory], &[1, 2], 1), [3], "said otherwise once");
        memory.learn(&[1, 2, 9], 2);
        assert_eq!(guessed(&[&memory], &[1, 2], 1), [3], "and twice");
        memory.learn(&[1, 2, 9], 2);
        assert_eq!(guessed(&[&memory], &[1, 2], 1), [9], "as often as not");
        // A token seen once gives way to the next: nothing was held to.
        memory.learn(&[5, 6, 7], 2);
        memory.learn(&[5, 6, 8], 2);
        assert_eq!(guessed(&[&memory], &[5, 6], 1), [8]);
        // And however often a token follows, it is not held past unseating.
        for _ in 0..50 {
            memory.learn(&[4, 4, 1], 2);
        }
        for _ in 0..usize::from(HELD_MOST) + 1 {
            memory.learn(&[4, 4, 2], 2);
        }
        assert_eq!(guessed(&[&memory], &[4, 4], 1), [2]);
        Ok(())
    }

    /// A memory that is full lets its older half go and keeps what it saw
    /// lately.
    #[test]
    fn a_full_memory_forgets_its_older_half() -> Outcome {
        let mut memory = Drafts::new();
        memory.learn(&[1, 2, 3], 0);
        // Enough runs to fill it twice over, none of them 1 2.
        let filler: Vec<u32> = (0..ROOM as u32).map(|i| 1000 + i).collect();
        memory.learn(&filler, 0);
        memory.learn(&filler, 0);
        assert!(memory.len() <= 2 * ROOM);
        assert_eq!(guessed(&[&memory], &[1, 2], 1), [] as [u32; 0], "long gone");
        let last = filler.len() - 1;
        assert_eq!(
            guessed(&[&memory], &filler[last - 2..last], 1),
            [filler[last]],
            "what it saw last is kept"
        );
        Ok(())
    }
}
