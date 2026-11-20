//! Which requests a resident model answers in its next pass.
//!
//! A model that reads rows as plain values decodes one token for each of
//! several requests in one pass over its weights, each request with its own
//! blend of experts, and takes a request in between two passes. This is the
//! bookkeeping around that: requests wait in the order they came; a request
//! is taken in as soon as its blend is kept by the model or can be, and there
//! is room; and each is answered the moment its last token is decoded,
//! whatever the others are doing.
//!
//! A blend is kept in one of a few slots. A slot is given to another blend
//! only when no request is being answered with it, the one least recently
//! asked for first, so a request never reads factors that changed under it.
//!
//! A model that does not read rows that way (a GPU build, a dense base, a
//! CPU with no tiles) answers one request at a time as it always has: the
//! first blend it is asked to keep says so, and from then on every request is
//! answered alone, in order.
//!
//! Nothing here knows candle or a runtime: the model is a trait, and a
//! request's answer goes to whatever its reply is.

use std::collections::{HashMap, VecDeque};

use antumbra_core::{AntumbraError, Result};

/// A blend of experts as a request names it: each adapter's file and weight.
pub(crate) type Specs = Vec<(String, f32)>;

/// What tells one blend from another: each adapter's file and the bits of
/// its weight. The file, not the expert: a retrained expert keeps its id and
/// is served from another file, and must not pass for what is kept.
pub(crate) type BlendKey = Vec<(String, u32)>;

pub(crate) fn blend_key(specs: &[(String, f32)]) -> BlendKey {
    specs
        .iter()
        .map(|(path, weight)| (path.clone(), weight.to_bits()))
        .collect()
}

/// Where a request's answer goes.
pub(crate) type Reply = Box<dyn FnOnce(Result<String>) + Send>;

/// One request: its prompt, the blend to answer it with, and where the
/// answer goes.
pub(crate) struct Job {
    pub(crate) prompt: String,
    pub(crate) specs: Specs,
    pub(crate) reply: Reply,
}

/// The resident model, as the schedule drives it.
pub(crate) trait Model {
    /// Keep `specs` as blend `blend` (one or more). `false` when this model
    /// keeps no blends and answers one request at a time.
    fn keep(&mut self, blend: u16, specs: &[(String, f32)]) -> Result<bool>;
    /// Take a request in, to be answered with kept blend `blend`: its place,
    /// or nothing when there is no room for another.
    fn admit(&mut self, prompt: &str, blend: u16) -> Result<Option<usize>>;
    /// One pass for every request taken in: the places of those that ended
    /// in it, each with its answer.
    fn advance(&mut self) -> Result<Vec<(usize, String)>>;
    /// Let go of every request taken in, after a pass that failed.
    fn reset(&mut self);
    /// Answer one request by itself, with `specs` loaded.
    fn alone(&mut self, prompt: &str, specs: &[(String, f32)]) -> Result<String>;
}

/// One kept blend.
#[derive(Default)]
struct Slot {
    key: Option<BlendKey>,
    /// Requests being answered with it.
    users: usize,
    /// When it was last asked for.
    last: u64,
}

/// The requests waiting and being answered, and the blends kept for them.
pub(crate) struct Schedule {
    slots: Vec<Slot>,
    waiting: VecDeque<Job>,
    /// By place: where the answer goes, and the slot the request reads.
    answering: HashMap<usize, (Reply, usize)>,
    clock: u64,
    /// Whether the model keeps blends: not known until it is first asked to.
    keeps: Option<bool>,
}

impl Schedule {
    /// A schedule over a model that keeps up to `slots` blends at once.
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            slots: (0..slots.max(1)).map(|_| Slot::default()).collect(),
            waiting: VecDeque::new(),
            answering: HashMap::new(),
            clock: 0,
            keeps: None,
        }
    }

    pub(crate) fn push(&mut self, job: Job) {
        self.waiting.push_back(job);
    }

    /// Whether nothing waits and nothing is being answered.
    pub(crate) fn idle(&self) -> bool {
        self.waiting.is_empty() && self.answering.is_empty()
    }

    /// The slot that keeps `key`, or one that may be given to it: an empty
    /// one, else the one least recently asked for among those no request is
    /// being answered with. The second value says the slot already keeps it.
    fn slot_for(&self, key: &BlendKey) -> Option<(usize, bool)> {
        if let Some(at) = self.slots.iter().position(|s| s.key.as_ref() == Some(key)) {
            return Some((at, true));
        }
        if let Some(at) = self.slots.iter().position(|s| s.key.is_none()) {
            return Some((at, false));
        }
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.users == 0)
            .min_by_key(|(_, s)| s.last)
            .map(|(at, _)| (at, false))
    }

    /// Take in every waiting request that can be taken in now, then one pass
    /// for those being answered. Each request that ends, or fails, is
    /// answered here.
    pub(crate) fn turn(&mut self, model: &mut impl Model) {
        if self.keeps == Some(false) {
            self.one_at_a_time(model);
            return;
        }
        let mut still = VecDeque::new();
        let mut full = false;
        while let Some(job) = self.waiting.pop_front() {
            if full {
                still.push_back(job);
                continue;
            }
            let key = blend_key(&job.specs);
            let Some((at, kept)) = self.slot_for(&key) else {
                // Every slot is in use by another blend: it waits its turn.
                still.push_back(job);
                continue;
            };
            let blend = (at + 1) as u16;
            if !kept {
                self.slots[at].key = None;
                match model.keep(blend, &job.specs) {
                    Ok(true) => {
                        self.keeps = Some(true);
                        self.slots[at].key = Some(key);
                    }
                    Ok(false) if self.answering.is_empty() => {
                        // This model answers one at a time: this request,
                        // and every one after it.
                        self.keeps = Some(false);
                        still.push_back(job);
                        still.extend(self.waiting.drain(..));
                        self.waiting = still;
                        self.one_at_a_time(model);
                        return;
                    }
                    Ok(false) => {
                        (job.reply)(Err(AntumbraError::other(
                            "the resident model stopped keeping blends while requests were being answered",
                        )));
                        continue;
                    }
                    Err(err) => {
                        (job.reply)(Err(err));
                        continue;
                    }
                }
            }
            match model.admit(&job.prompt, blend) {
                Ok(Some(place)) => {
                    self.clock += 1;
                    self.slots[at].last = self.clock;
                    self.slots[at].users += 1;
                    self.answering.insert(place, (job.reply, at));
                }
                Ok(None) => {
                    // No room for another: it and the rest wait for a place.
                    full = true;
                    still.push_back(job);
                }
                Err(err) => (job.reply)(Err(err)),
            }
        }
        self.waiting = still;
        if self.answering.is_empty() {
            // Nothing could be taken in and nothing is being answered:
            // nothing would ever change, so the waiting are told.
            for job in self.waiting.drain(..) {
                (job.reply)(Err(AntumbraError::other(
                    "the resident model had no room for a request while answering none",
                )));
            }
            return;
        }
        match model.advance() {
            Ok(ended) => {
                for (place, answer) in ended {
                    if let Some((reply, at)) = self.answering.remove(&place) {
                        self.slots[at].users = self.slots[at].users.saturating_sub(1);
                        reply(Ok(answer));
                    }
                }
            }
            Err(err) => {
                // What the model holds of these requests is not to be
                // trusted after a pass that failed: each is told, and let go.
                let why = err.to_string();
                for (_, (reply, at)) in self.answering.drain() {
                    self.slots[at].users = self.slots[at].users.saturating_sub(1);
                    reply(Err(AntumbraError::other(format!(
                        "a pass over the resident model failed: {why}"
                    ))));
                }
                model.reset();
            }
        }
    }

    /// Every waiting request answered by itself, in order.
    fn one_at_a_time(&mut self, model: &mut impl Model) {
        while let Some(job) = self.waiting.pop_front() {
            (job.reply)(model.alone(&job.prompt, &job.specs));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, PoisonError};

    /// A model that answers a prompt with its blend's name and the prompt,
    /// after as many passes as the prompt has letters, and records what it
    /// was asked to do.
    #[derive(Default)]
    struct Fake {
        keeps: bool,
        room: usize,
        kept: HashMap<u16, String>,
        places: HashMap<usize, (String, String, usize)>,
        log: Vec<String>,
        fail_next_pass: bool,
    }

    impl Fake {
        fn keeping(room: usize) -> Self {
            Self {
                keeps: true,
                room,
                ..Self::default()
            }
        }
    }

    fn name(specs: &[(String, f32)]) -> String {
        specs
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<Vec<_>>()
            .join("+")
    }

    impl Model for Fake {
        fn keep(&mut self, blend: u16, specs: &[(String, f32)]) -> Result<bool> {
            if !self.keeps {
                return Ok(false);
            }
            if name(specs) == "broken" {
                return Err(AntumbraError::other("no such adapter"));
            }
            self.log.push(format!("keep {blend} {}", name(specs)));
            self.kept.insert(blend, name(specs));
            Ok(true)
        }

        fn admit(&mut self, prompt: &str, blend: u16) -> Result<Option<usize>> {
            if self.places.len() >= self.room {
                return Ok(None);
            }
            let Some(kept) = self.kept.get(&blend) else {
                return Err(AntumbraError::other("a blend that was not kept"));
            };
            let Some(place) = (0..self.room).find(|p| !self.places.contains_key(p)) else {
                return Ok(None);
            };
            self.places
                .insert(place, (kept.clone(), prompt.to_string(), prompt.len()));
            Ok(Some(place))
        }

        fn advance(&mut self) -> Result<Vec<(usize, String)>> {
            if self.fail_next_pass {
                self.fail_next_pass = false;
                return Err(AntumbraError::other("out of memory"));
            }
            self.log.push(format!("pass {}", self.places.len()));
            let mut ended = Vec::new();
            for (place, (blend, prompt, left)) in &mut self.places {
                *left -= 1;
                if *left == 0 {
                    ended.push((*place, format!("{blend}: {prompt}")));
                }
            }
            for (place, _) in &ended {
                self.places.remove(place);
            }
            Ok(ended)
        }

        fn reset(&mut self) {
            self.places.clear();
            self.log.push("reset".into());
        }

        fn alone(&mut self, prompt: &str, specs: &[(String, f32)]) -> Result<String> {
            self.log.push(format!("alone {}", name(specs)));
            Ok(format!("{}: {prompt}", name(specs)))
        }
    }

    type Answers = Arc<Mutex<Vec<(String, std::result::Result<String, String>)>>>;

    fn job(answers: &Answers, prompt: &str, blend: &str) -> Job {
        let (answers, id) = (answers.clone(), prompt.to_string());
        Job {
            prompt: prompt.to_string(),
            specs: vec![(blend.to_string(), 1.0)],
            reply: Box::new(move |answer| {
                answers
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push((id, answer.map_err(|e| e.to_string())));
            }),
        }
    }

    fn answered(answers: &Answers) -> Vec<(String, std::result::Result<String, String>)> {
        answers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn run(schedule: &mut Schedule, model: &mut Fake) {
        for _ in 0..100 {
            if schedule.idle() {
                return;
            }
            schedule.turn(model);
        }
    }

    /// Requests of different blends are answered in the same passes, each
    /// with its own blend, and each as soon as its own last token: the short
    /// one first, though it came last.
    #[test]
    fn requests_of_different_blends_share_their_passes() -> Result<()> {
        let answers = Answers::default();
        let (mut schedule, mut model) = (Schedule::new(4), Fake::keeping(8));
        schedule.push(job(&answers, "a long one", "rust"));
        schedule.push(job(&answers, "middling", "git"));
        schedule.push(job(&answers, "hi", "rust"));
        run(&mut schedule, &mut model);
        let got = answered(&answers);
        let order: Vec<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(order, ["hi", "middling", "a long one"]);
        assert_eq!(got[0].1, Ok("rust: hi".to_string()));
        assert_eq!(got[1].1, Ok("git: middling".to_string()));
        // Two blends were kept, once each, and the first pass held all three.
        let keeps = model.log.iter().filter(|l| l.starts_with("keep")).count();
        assert_eq!(keeps, 2, "{:?}", model.log);
        assert!(model.log.contains(&"pass 3".to_string()), "{:?}", model.log);
        Ok(())
    }

    /// A request that comes while others are being answered joins at the
    /// next pass, and does not wait for them to end.
    #[test]
    fn a_request_joins_at_the_next_pass() -> Result<()> {
        let answers = Answers::default();
        let (mut schedule, mut model) = (Schedule::new(4), Fake::keeping(8));
        schedule.push(job(&answers, "a request of many letters", "rust"));
        schedule.turn(&mut model);
        schedule.turn(&mut model);
        schedule.push(job(&answers, "now", "git"));
        run(&mut schedule, &mut model);
        let got = answered(&answers);
        assert_eq!(
            got[0].0, "now",
            "the later, shorter request is answered first"
        );
        assert!(model.log.contains(&"pass 2".to_string()), "{:?}", model.log);
        Ok(())
    }

    /// A slot goes to another blend only when no request is being answered
    /// with it, the least recently asked for first; a blend with no slot to
    /// take waits, and is answered once one is free.
    #[test]
    fn a_blend_in_use_keeps_its_slot() -> Result<()> {
        let answers = Answers::default();
        let (mut schedule, mut model) = (Schedule::new(2), Fake::keeping(8));
        schedule.push(job(&answers, "first, and long", "a"));
        schedule.push(job(&answers, "second, long too", "b"));
        schedule.push(job(&answers, "third", "c"));
        schedule.turn(&mut model);
        assert!(
            !model.log.iter().any(|l| l.ends_with(" c")),
            "c has no slot while a and b are read: {:?}",
            model.log
        );
        run(&mut schedule, &mut model);
        let got = answered(&answers);
        assert_eq!(got.len(), 3);
        assert!(got.iter().all(|(_, a)| a.is_ok()), "{got:?}");
        assert_eq!(got[2], ("third".to_string(), Ok("c: third".to_string())));
        // c took the slot of the blend that ended first, a's.
        assert!(
            model.log.contains(&"keep 1 c".to_string()),
            "{:?}",
            model.log
        );
        Ok(())
    }

    /// With no room for another, a request waits for a place, and those
    /// being answered go on.
    #[test]
    fn a_request_waits_for_a_place() -> Result<()> {
        let answers = Answers::default();
        let (mut schedule, mut model) = (Schedule::new(2), Fake::keeping(2));
        for prompt in ["one", "two", "three", "four"] {
            schedule.push(job(&answers, prompt, "rust"));
        }
        run(&mut schedule, &mut model);
        let got = answered(&answers);
        assert_eq!(got.len(), 4);
        assert!(got.iter().all(|(_, a)| a.is_ok()), "{got:?}");
        assert!(
            !model.log.iter().any(|l| l == "pass 3" || l == "pass 4"),
            "never more than its room at once: {:?}",
            model.log
        );
        Ok(())
    }

    /// A model that keeps no blends answers every request alone, in the
    /// order they came.
    #[test]
    fn a_model_that_keeps_no_blends_answers_one_at_a_time() -> Result<()> {
        let answers = Answers::default();
        let (mut schedule, mut model) = (Schedule::new(4), Fake::default());
        schedule.push(job(&answers, "one", "rust"));
        schedule.push(job(&answers, "two", "git"));
        run(&mut schedule, &mut model);
        schedule.push(job(&answers, "three", "rust"));
        run(&mut schedule, &mut model);
        let got = answered(&answers);
        let order: Vec<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(order, ["one", "two", "three"]);
        assert_eq!(model.log, ["alone rust", "alone git", "alone rust"]);
        Ok(())
    }

    /// A blend that cannot be kept fails its own request and no other, and
    /// a pass that fails tells every request in it and leaves the schedule
    /// ready for the next.
    #[test]
    fn a_failure_is_told_to_those_it_is_theirs() -> Result<()> {
        let answers = Answers::default();
        let (mut schedule, mut model) = (Schedule::new(4), Fake::keeping(8));
        schedule.push(job(&answers, "fine", "rust"));
        schedule.push(job(&answers, "lost", "broken"));
        run(&mut schedule, &mut model);
        let got = answered(&answers);
        assert_eq!(got[0].0, "lost");
        assert!(got[0].1.is_err());
        assert_eq!(got[1], ("fine".to_string(), Ok("rust: fine".to_string())));

        let answers = Answers::default();
        schedule.push(job(&answers, "doomed", "rust"));
        schedule.push(job(&answers, "also doomed", "git"));
        model.fail_next_pass = true;
        run(&mut schedule, &mut model);
        let got = answered(&answers);
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|(_, a)| a.is_err()), "{got:?}");
        assert!(schedule.idle());
        assert!(model.log.contains(&"reset".to_string()));

        let answers = Answers::default();
        schedule.push(job(&answers, "after", "rust"));
        run(&mut schedule, &mut model);
        assert_eq!(
            answered(&answers),
            [("after".to_string(), Ok("rust: after".to_string()))]
        );
        Ok(())
    }
}
