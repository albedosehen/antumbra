//! Greedy decoding several tokens a pass.
//!
//! A decoded token on a CPU is a reading of every weight, and the reading is
//! most of the time: a pass that carries a second row takes about a tenth
//! longer than a pass of one on the machine this was measured on, and less
//! on one that runs at its full clock. So a pass carries the token in hand
//! and, after it, the tokens guessed to follow ([`crate::draft`]). The model
//! gives its own first token after every row ([`QwenLora::step_ahead`]).
//! After the token in hand that is the next token, whatever was guessed.
//! Where it is also the first guess, the first guess's row was read on the
//! right text, so the model's token after that row is decided too, and so on
//! down the guesses until one is not what the model chose. The rows after
//! that one were read on text that was never written: the cache is cut back
//! to before them, and the next pass starts from the model's own token.
//!
//! What is written is what a token at a time writes, token for token: every
//! token kept is the model's first after rows it would have read alone, and
//! rows read together are computed as each would be alone, to the bit. A
//! guess decides nothing; a wrong one costs its row.
//!
//! What a row costs is the machine's, and how often a guess is right is the
//! text's, so neither is written down here: both are counted as the model
//! decodes ([`Odds`]). A guess rides along while the chance that it and
//! every guess before it are right, by the time a token has been taking, is
//! at least the time a further row has been adding. Where rows are cheap
//! beside the reading, a pass carries guesses that are right one time in
//! six; where a row costs a third of a pass, as on a machine whose memory
//! outruns its arithmetic, only text it has all but written before.
//!
//! On one laptop CPU clocking half its rating, with a model that reads a
//! token a pass answering the same tasks beside it, one expert's 78 held-out
//! tasks at Q6_K decoded 1.26 times as fast once the model had answered the
//! tasks the expert was taught, 1.17 times with nothing remembered, and
//! 3.9 to 4.1 times when the same tasks were asked again; 1.15 and 3.1
//! times at Q4_K, 1.41 and 5.0 at Q8_0, 1.25 and 3.9 through a blend of
//! three experts. Every answer was the answer a token at a time gave.

use std::time::{Duration, Instant};

use candle_core::{bail, Result};

use super::{GenTiming, QwenLora, Stepped};
use crate::draft::{guess, Drafts};

/// Guesses a pass carries at most: sixteen rows with the token in hand.
const CARRY_MOST: usize = 15;

/// The kinds of guess that are counted apart: the first of a pass by the
/// run of tokens it was looked up by, one to four; a later one by its run;
/// and one taken from the answer kept whole for this very prompt.
const KINDS: usize = 9;
const SAID: u8 = 8;

/// How often each kind was right over one expert's held-out tasks, after
/// the tasks it was taught had been answered: what is believed of a kind
/// before it has been tried here. A later guess that rests on one token
/// alone follows a guess, not the text, and was all but never right.
const BELIEVED: [f32; KINDS] = [0.18, 0.35, 0.55, 0.70, 0.03, 0.45, 0.55, 0.75, 0.95];
/// The tries that belief stands for beside the ones counted.
const BELIEF: f32 = 16.0;
/// Tries at which a kind's counts are halved, so that an hour ago does not
/// outweigh the last minute.
const TRIES: f32 = 1024.0;

/// What a further row is taken to add to a pass of one, until passes of two
/// sizes have been timed: a tenth.
const ROW_BELIEVED: f64 = 0.1;
/// The least a further row is ever taken to add, of a pass of one: a clock
/// that read a wide pass short does not make rows free.
const ROW_LEAST: f64 = 0.005;
/// How far a timed pass moves what passes are taken to cost.
const MOVES: f64 = 0.2;
/// How far a pass moves the time a token is taken to take.
const TOKEN_MOVES: f64 = 1.0 / 32.0;
/// The chance a guess needs before any pass has been timed.
const FIRST: f32 = 0.25;

/// The kind of the guess at place `at` of a pass's guesses, looked up by a
/// run of `run` tokens.
fn kind(at: usize, run: usize) -> u8 {
    let run = run.clamp(1, 4) as u8 - 1;
    match at {
        0 => run,
        _ => 4 + run,
    }
}

/// What guesses have been worth to this model on this machine: how often
/// each kind was the model's own token, what a pass costs by its rows, and
/// the time a token has been taking. Kept from one generation to the next.
#[derive(Debug, Clone, Default)]
pub(in crate::models::qwen) struct Odds {
    /// Each kind's tries and how many were right.
    seen: [[f32; 2]; KINDS],
    /// Seconds a pass of one row is taken to cost, and each further row to
    /// add: nothing until a pass has been timed.
    one: f64,
    more: f64,
    /// Seconds and tokens to a pass, lately.
    secs: f64,
    tokens: f64,
    /// Every guess offered is carried, whatever it is worth.
    #[cfg(test)]
    eager: bool,
}

impl Odds {
    /// Odds under which every guess offered is carried: a test's, to read
    /// passes of every width.
    #[cfg(test)]
    pub(in crate::models::qwen) fn eager() -> Self {
        Self {
            eager: true,
            ..Self::default()
        }
    }

    /// How likely a guess of `kind` is the model's own token, the guesses
    /// before it being right: what was counted, from what is believed.
    fn likely(&self, kind: u8) -> f32 {
        let kind = usize::from(kind).min(KINDS - 1);
        let [tried, right] = self.seen[kind];
        (right + BELIEVED[kind] * BELIEF) / (tried + BELIEF)
    }

    /// How many of the guesses of these kinds, in their order, are worth
    /// their rows: each while the chance that it and all before it are
    /// right, by the time a token takes, is at least what a row adds.
    fn carry(&self, kinds: &[u8]) -> usize {
        #[cfg(test)]
        if self.eager {
            return kinds.len();
        }
        let mut likely = 1f32;
        kinds
            .iter()
            .take_while(|kind| {
                likely *= self.likely(**kind);
                match self.tokens > 0.0 {
                    true => f64::from(likely) * self.secs / self.tokens >= self.more,
                    false => likely >= FIRST,
                }
            })
            .count()
    }

    /// A pass carried guesses of these kinds and the first `kept` were the
    /// model's own. The one after them was tried and was wrong; the ones
    /// after that were read on text that was never written, and say nothing.
    fn saw(&mut self, kinds: &[u8], kept: usize) {
        for (at, kind) in kinds.iter().enumerate().take(kept + 1) {
            let seen = &mut self.seen[usize::from(*kind).min(KINDS - 1)];
            seen[0] += 1.0;
            if at < kept {
                seen[1] += 1.0;
            }
            if seen[0] >= TRIES {
                *seen = [seen[0] / 2.0, seen[1] / 2.0];
            }
        }
    }

    /// A pass of `rows` rows decided `tokens` tokens and took `took`.
    fn passed(&mut self, rows: usize, tokens: usize, took: Duration) {
        let (secs, further) = (took.as_secs_f64(), rows.saturating_sub(1) as f64);
        if secs <= 0.0 || rows == 0 {
            return;
        }
        let secs = match self.one > 0.0 {
            // The cost of a pass as a line through its rows, moved toward
            // this pass by what the line got wrong of it: passes of one size
            // move the line to them and leave its slope, and passes of two
            // sizes turn it. A pass the machine was called away in the
            // middle of is not what passes cost, so a pass counts for no
            // more than half again what the line took it for, and no less
            // than half.
            true => {
                let taken = self.one + self.more * further;
                let secs = secs.clamp(0.5 * taken, 1.5 * taken);
                let off = MOVES * (secs - taken) / (1.0 + further * further);
                self.one = (self.one + off).max(taken * 0.05);
                self.more = (self.more + off * further).max(self.one * ROW_LEAST);
                secs
            }
            false => {
                self.one = secs / (1.0 + ROW_BELIEVED * further);
                self.more = ROW_BELIEVED * self.one;
                secs
            }
        };
        match self.tokens > 0.0 {
            true => {
                self.secs += TOKEN_MOVES * (secs - self.secs);
                self.tokens += TOKEN_MOVES * (tokens as f64 - self.tokens);
            }
            false => (self.secs, self.tokens) = (secs, tokens.max(1) as f64),
        }
    }
}

/// What a generation is decoded ahead with.
pub(in crate::models::qwen) struct Ahead<'a> {
    /// The token that ends a generation, and the most tokens it may write.
    pub eos: u32,
    pub most: usize,
    /// What this generation's own prompt and tokens are remembered in: let
    /// go and filled here.
    pub near: &'a mut Drafts,
    /// What earlier generations wrote.
    pub far: &'a Drafts,
    /// What guesses have been worth, counted on here.
    pub odds: &'a mut Odds,
}

/// What a generation that was decoded ahead took.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::models::qwen) struct Passes {
    /// Passes after the prompt's.
    pub passes: usize,
    /// Tokens guessed, and guesses that were the model's own token.
    pub guessed: usize,
    pub kept: usize,
}

/// Write the generation that follows the prompt in `tokens` onto the end of
/// it, greedily, several tokens a pass: what one token a pass writes, token
/// for token, stopping where it stops (the end token, a loop, the most
/// tokens allowed). `None`, with nothing read or written, when `model` does
/// not read ahead: the caller decodes a token at a time.
pub(in crate::models::qwen) fn decode(
    model: &mut QwenLora,
    tokens: &mut Vec<u32>,
    prompt_len: usize,
    ahead: Ahead,
    timing: &mut GenTiming,
) -> Result<Option<Passes>> {
    let Ahead {
        eos,
        most,
        near,
        far,
        odds,
    } = ahead;
    if !model.reads_ahead() {
        return Ok(None);
    }
    let mut told = Passes::default();
    if most == 0 {
        return Ok(Some(told));
    }
    // The prompt, and the first token after it.
    let began = Instant::now();
    let mut next = match model.step(tokens, 0, true)? {
        Some(Stepped::First(token)) => token,
        Some(Stepped::Logits(_)) => bail!("a head that reads ahead gave logits for one token"),
        None => return Ok(None),
    };
    timing.record(0, tokens.len(), began.elapsed());
    // What was written after this very prompt before, if that is kept: the
    // generation is guessed from it for as long as it goes the same way.
    let said = far.said(&tokens[..prompt_len]).unwrap_or(&[]);
    near.clear();
    let mut learned = 0;
    let (mut offered, mut kinds, mut rows) = (Vec::new(), Vec::new(), Vec::new());
    loop {
        // `next` follows everything in `tokens`.
        if next == eos {
            break;
        }
        tokens.push(next);
        let written = tokens.len() - prompt_len;
        if crate::decode::loops(&tokens[prompt_len..]) || written >= most {
            break;
        }
        // One pass: the token just written, and what is guessed to follow
        // it, no further than the generation may go and no more than the
        // guesses are worth.
        near.learn(tokens, learned);
        learned = tokens.len();
        let at = tokens.len() - 1;
        let room = (most - written - 1).min(CARRY_MOST);
        kinds.clear();
        rows.clear();
        rows.push(tokens[at]);
        match said.len() > written && said[..written] == tokens[prompt_len..] {
            // The same prompt and the same words so far: what came next
            // then.
            true => {
                kinds.resize(room.min(said.len() - written), SAID);
                kinds.truncate(odds.carry(&kinds));
                rows.extend_from_slice(&said[written..written + kinds.len()]);
            }
            false => {
                guess(&[&*near, far], tokens, room, eos, &mut offered);
                kinds.extend(
                    offered
                        .iter()
                        .enumerate()
                        .map(|(at, guess)| kind(at, guess.run)),
                );
                kinds.truncate(odds.carry(&kinds));
                rows.extend(offered[..kinds.len()].iter().map(|guess| guess.token));
            }
        }
        let began = Instant::now();
        let Some(firsts) = model.step_ahead(&rows, at)? else {
            bail!("a model that read its prompt ahead did not read a token ahead")
        };
        // The model's token after the token in hand, then after each guess
        // that was the model's own token.
        let guesses = &rows[1..];
        let (mut kept, mut over) = (0, false);
        next = firsts[0];
        while kept < guesses.len() && guesses[kept] == next {
            tokens.push(next);
            kept += 1;
            if crate::decode::loops(&tokens[prompt_len..]) || tokens.len() - prompt_len >= most {
                over = true;
                break;
            }
            next = firsts[kept];
        }
        let (took, decided) = (began.elapsed(), kept + usize::from(!over));
        timing.passed(decided, took);
        odds.saw(&kinds, kept);
        odds.passed(rows.len(), decided, took);
        told.passes += 1;
        told.guessed += guesses.len();
        told.kept += kept;
        if over {
            break;
        }
        // The rows after the first guess that was not the model's token
        // were read on text that was never written.
        model.truncate(at + 1 + kept);
    }
    Ok(Some(told))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pass of `rows` rows that took `millis` and decided `tokens`.
    fn timed(odds: &mut Odds, rows: usize, tokens: usize, millis: f64) {
        odds.passed(rows, tokens, Duration::from_secs_f64(millis / 1e3));
    }

    /// Guesses looked up by the longest run, the first and four after it.
    const SURE: [u8; 5] = [3, 7, 7, 7, 7];

    /// Before any pass is timed a guess is carried on what is believed of
    /// its kind: a run of the longest runs for as long as they are likely
    /// to all be right, one token's say-so not at all.
    #[test]
    fn untimed_a_guess_rides_on_what_is_believed_of_it() {
        let odds = Odds::default();
        assert_eq!(odds.carry(&[]), 0);
        // 0.70, then by 0.75 each: 0.525, 0.39, 0.30, 0.22.
        assert_eq!(odds.carry(&SURE), 4);
        assert_eq!(odds.carry(&[kind(0, 1)]), 0);
        assert_eq!(odds.carry(&[kind(0, 3), kind(1, 1), kind(2, 4)]), 1);
        assert_eq!(odds.carry(&[SAID; 15]), 15);
        assert_eq!(
            (kind(0, 1), kind(0, 9), kind(3, 2), kind(1, 0)),
            (0, 3, 5, 4)
        );
    }

    /// Where a further row adds little to a pass, guesses that are seldom
    /// right are worth carrying; where it adds a third, only sure ones are.
    #[test]
    fn a_guess_is_worth_what_its_row_costs_here() {
        // A row adds a fortieth of a pass.
        let mut cheap = Odds::default();
        for _ in 0..64 {
            timed(&mut cheap, 1, 1, 36.0);
            timed(&mut cheap, 4, 2, 38.7);
        }
        assert!((cheap.one - 0.036).abs() < 0.001 && (cheap.more - 0.0009).abs() < 0.0003);
        let long = [3, 7, 7, 7, 7, 7, 7, 7];
        assert_eq!(cheap.carry(&[kind(0, 1)]), 1, "one time in six pays");
        assert_eq!(cheap.carry(&long), 8);
        // A row adds a third of a pass.
        let mut dear = Odds::default();
        for _ in 0..64 {
            timed(&mut dear, 1, 1, 5.0);
            timed(&mut dear, 4, 2, 9.65);
        }
        assert!((dear.more - 0.00155).abs() < 0.0004, "{}", dear.more);
        assert_eq!(dear.carry(&[kind(0, 1)]), 0);
        assert!(dear.carry(&long) < 5, "{}", dear.carry(&long));
        // Passes of one width alone leave the slope as it was believed.
        let mut one = Odds::default();
        for _ in 0..64 {
            timed(&mut one, 2, 1, 40.0);
        }
        assert!((one.one + one.more - 0.040).abs() < 0.001);
        assert!((one.more / one.one - ROW_BELIEVED).abs() < 0.01);
        // And nothing timed in no time moves nothing.
        one.passed(3, 1, Duration::ZERO);
        one.passed(0, 0, Duration::from_millis(5));
        assert!((one.one + one.more - 0.040).abs() < 0.001);
    }

    /// What is counted moves what is believed: a kind that keeps being wrong
    /// stops riding, one that keeps being right rides further, and only the
    /// guesses that were tried are counted.
    #[test]
    fn what_guesses_did_is_counted() {
        let mut odds = Odds::default();
        for _ in 0..32 {
            timed(&mut odds, 1, 1, 36.0);
            timed(&mut odds, 2, 1, 40.0);
        }
        let before = odds.carry(&SURE);
        assert!(before >= 2);
        // The first of five was wrong, four hundred times: the four after
        // it were never tried.
        for _ in 0..400 {
            odds.saw(&SURE, 0);
        }
        assert!(odds.likely(3) < 0.05 && odds.likely(7) == BELIEVED[7]);
        assert_eq!(odds.carry(&SURE), 0);
        // All five right, many times over.
        for _ in 0..4000 {
            odds.saw(&SURE, 5);
        }
        assert!(odds.likely(3) > 0.85 && odds.likely(7) > 0.95);
        assert!(odds.carry(&SURE) == 5 && odds.carry(&[7; 15]) > before);
        // Counts are halved before they outweigh what comes next.
        for _ in 0..5000 {
            odds.saw(&[SAID], 1);
        }
        assert!(odds.seen[usize::from(SAID)][0] < TRIES);
        // A kind past the last is counted as the last.
        odds.saw(&[200], 0);
        assert!(odds.likely(200) < 1.0);
    }
}
