//! Behaviors: how the user wants an agent to act on a class of
//! tasks, stated as a rule with a check a program can apply.
//!
//! Experts learn behavior, never facts. A behavior trains from tasks it
//! governs and a check that decides each answer, so it is recorded with both,
//! by the agent that learned it, rather than dug out of free text afterwards.
//!
//! A behavior is a memory in the user's own `behavior` compartment, as a
//! handoff is in its own. Its content leads with the rule in prose, which is
//! what recall and the agent read, followed by a fenced `behavior` block with
//! what training reads: the patterns an answer must and must not match, worked
//! examples, and violating answers. Evidence entries carry its status, its
//! scope, and the behavior it supersedes.

use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ids::{CompartmentId, TenantId, UserId};
use crate::{Expert, Memory};

/// The scope of a behavior that applies in every repository.
pub const EVERYWHERE: &str = "everywhere";
/// The fewest worked examples a behavior is recorded with. Four, because
/// training holds one example in three out to admit the expert by
/// (`antumbra_train::behave::split`), and three would leave two to learn
/// from and one to judge by. Ten, varied in phrasing and case, hold three out
/// and teach more reliably.
pub const MIN_EXAMPLES: usize = 4;

/// The name of a user's behavior compartment, and the last part of its id.
/// It keeps the spelling it was first stored with: the compartments already
/// in a store are found by it.
pub const COMPARTMENT_NAME: &str = "behaviour";

const STATUS: &str = "behavior-status:";
const SCOPE: &str = "behavior-scope:";
const SUPERSEDES: &str = "behavior-supersedes:";
const EXPERT: &str = "behavior-expert:";
const TRAINING: &str = "behavior-training:";
/// A served expert measured on examples the behavior gained after the
/// expert was trained, one entry per audit; see [`Audit`].
const AUDIT: &str = "behavior-audit:";
/// On a standing expert's card: each taught behavior's examples, as
/// [`example_print`]s, and the later examples the expert was audited on.
pub const CARD_EXAMPLES: &str = "examples";
pub const CARD_AUDITED: &str = "audited";
/// On a standing expert's card: each taught behavior's held-out scores at
/// admission (`base`, `expert`, `held`, `worse`, `learned`), the decision an
/// audit later checks.
pub const CARD_SCORES: &str = "scores";
/// A memory the behavior was drawn from, one entry each. Read by the
/// candidate pass, which leaves cited memories out; not part of the set a
/// training is fingerprinted by, since the fingerprint hashes the content.
const SOURCE: &str = "behavior-source:";
/// Refusals noted while one behavior not learned refused its whole scope.
/// Dropped when a behavior is next trained, and never read: under today's
/// rule such a set may be admitted.
const REFUSED: &str = "behavior-refused:";
const FENCE: &str = "```behavior";

// What behaviors were written with before the US spelling, still read so
// those already stored keep their state; nothing is written with them.
const LEGACY_STATUS: &str = "behaviour-status:";
const LEGACY_SCOPE: &str = "behaviour-scope:";
const LEGACY_SUPERSEDES: &str = "behaviour-supersedes:";
const LEGACY_EXPERT: &str = "behaviour-expert:";
const LEGACY_FENCE: &str = "```behaviour";
const LEGACY_CARD_KEY: &str = "behaviours";

/// An evidence entry's value under `prefix`, or under its earlier spelling.
fn value_of<'a>(entry: &'a str, prefix: &str, legacy: &str) -> Option<&'a str> {
    entry
        .strip_prefix(prefix)
        .or_else(|| entry.strip_prefix(legacy))
}

/// A task the behavior governs and an answer that follows it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Example {
    /// The task, as a person would ask it.
    pub task: String,
    /// An answer that follows the behavior.
    pub answer: String,
}

/// What training reads: the check and the evidence that it discriminates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spec {
    /// The rule in a sentence.
    pub rule: String,
    /// Patterns (regular expressions) every following answer matches.
    #[serde(default)]
    pub must: Vec<String>,
    /// Patterns no following answer matches.
    #[serde(default)]
    pub must_not: Vec<String>,
    /// Tasks with answers that follow the behavior.
    pub examples: Vec<Example>,
    /// Answers that break the behavior; the check must refuse each.
    #[serde(default)]
    pub violations: Vec<String>,
}

/// Where a behavior stands. The system proposes and the user accepts; a
/// trained behavior is in an expert; a retired one trains nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Proposed,
    Accepted,
    Trained,
    Retired,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Proposed => "proposed",
            Status::Accepted => "accepted",
            Status::Trained => "trained",
            Status::Retired => "retired",
        }
    }

    pub fn parse(s: &str) -> Option<Status> {
        match s.trim().to_lowercase().as_str() {
            "proposed" => Some(Status::Proposed),
            "accepted" => Some(Status::Accepted),
            "trained" => Some(Status::Trained),
            "retired" => Some(Status::Retired),
            _ => None,
        }
    }
}

fn compile(patterns: &[String], problems: &mut Vec<String>) -> Vec<Regex> {
    patterns
        .iter()
        .filter_map(|p| match Regex::new(p) {
            Ok(r) => Some(r),
            Err(e) => {
                problems.push(format!("the pattern `{p}` does not compile: {e}"));
                None
            }
        })
        .collect()
}

fn follows(must: &[Regex], must_not: &[Regex], answer: &str) -> bool {
    must.iter().all(|r| r.is_match(answer).unwrap_or(false))
        && !must_not.iter().any(|r| r.is_match(answer).unwrap_or(false))
}

impl Spec {
    /// Whether `answer` follows the behavior: every `must` pattern matches and
    /// no `must_not` pattern does. `None` when a pattern does not compile.
    pub fn follows(&self, answer: &str) -> Option<bool> {
        let mut problems = Vec::new();
        let must = compile(&self.must, &mut problems);
        let must_not = compile(&self.must_not, &mut problems);
        problems
            .is_empty()
            .then(|| follows(&must, &must_not, answer))
    }

    /// Everything that keeps this behavior from training; empty when it can.
    ///
    /// A check needs a `must` pattern, not only `must_not` ones: a check that
    /// only refuses passes an empty or evasive answer, and training would then
    /// learn to evade. It is shown to discriminate before it is trusted: every
    /// example's answer passes and every violation fails.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.rule.trim().is_empty() {
            problems.push("the rule is empty".to_string());
        }
        if self.must.is_empty() {
            problems.push(
                "the check needs at least one `must` pattern: `must_not` alone passes an empty answer"
                    .to_string(),
            );
        }
        let must = compile(&self.must, &mut problems);
        let must_not = compile(&self.must_not, &mut problems);
        if self.examples.len() < MIN_EXAMPLES {
            problems.push(format!(
                "{} example(s); a behavior needs at least {MIN_EXAMPLES}, since training holds one in three out to admit its expert by",
                self.examples.len()
            ));
        }
        if self.violations.is_empty() {
            problems.push(
                "no violating answer, so nothing shows the check refuses what breaks the rule"
                    .to_string(),
            );
        }
        if problems.iter().any(|p| p.contains("does not compile")) {
            return problems;
        }
        for e in &self.examples {
            if e.task.trim().is_empty() || e.answer.trim().is_empty() {
                problems.push("an example has an empty task or answer".to_string());
            } else if !follows(&must, &must_not, &e.answer) {
                problems.push(format!(
                    "the check refuses the example answer `{}`",
                    e.answer
                ));
            }
        }
        for v in &self.violations {
            if follows(&must, &must_not, v) {
                problems.push(format!("the check passes the violating answer `{v}`"));
            }
        }
        problems
    }
}

/// The user's behavior compartment, one per `(tenant, user)`.
pub fn compartment_id(tenant: &TenantId, user: &UserId) -> CompartmentId {
    CompartmentId::new(format!(
        "comp:{}:{}:{COMPARTMENT_NAME}",
        tenant.as_str(),
        user.as_str()
    ))
}

/// Whether `e` is one of `user`'s standing experts in `tenant`. The owner
/// alone does not say: every hosted workspace's first user has the same id,
/// so the compartment it was trained from, which names the workspace,
/// decides. Every standing expert has carried its compartment.
pub fn is_standing_for(e: &Expert, tenant: &TenantId, user: &UserId) -> bool {
    e.standing_scope().is_some()
        && e.owner.as_ref() == Some(user)
        && e.compartment.as_ref() == Some(&compartment_id(tenant, user))
}

/// A behavior as a memory's content: the rule first, for recall and the
/// agent, then the fenced block training reads.
pub fn content(spec: &Spec) -> String {
    let json = serde_json::to_string_pretty(spec).unwrap_or_default();
    format!("{}\n\n{FENCE}\n{json}\n```\n", spec.rule.trim())
}

/// The spec in a behavior memory's content, if it holds one: in a
/// `behavior` block, or a `behaviour` block written before the US spelling.
///
/// The block is the last thing in the content ([`content`] writes it after
/// the rule), so the fence that closes it is the last one: an example answer
/// that is itself a fenced code block (a Mermaid diagram, a Rust snippet)
/// carries fences of its own inside the JSON, and the first of those is not
/// the end of the spec.
pub fn spec_of(content: &str) -> Option<Spec> {
    let start = content
        .find(FENCE)
        .map(|i| i + FENCE.len())
        .or_else(|| content.find(LEGACY_FENCE).map(|i| i + LEGACY_FENCE.len()))?;
    let rest = &content[start..];
    let end = rest.rfind("```")?;
    serde_json::from_str(rest[..end].trim()).ok()
}

/// A scope as behaviors compare them: a repository slug, or [`EVERYWHERE`].
pub fn normalize_scope(scope: Option<&str>) -> String {
    match scope.map(str::trim) {
        None | Some("") => EVERYWHERE.to_string(),
        Some(s) => s.to_lowercase(),
    }
}

pub fn status_evidence(status: Status) -> String {
    format!("{STATUS}{}", status.as_str())
}

pub fn scope_evidence(scope: &str) -> String {
    format!("{SCOPE}{scope}")
}

pub fn supersedes_evidence(id: &str) -> String {
    format!("{SUPERSEDES}{id}")
}

/// Set a behavior's status in its evidence, replacing the one it had.
pub fn set_status(evidence: &mut Vec<String>, status: Status) {
    evidence.retain(|e| value_of(e, STATUS, LEGACY_STATUS).is_none());
    evidence.push(status_evidence(status));
}

/// Mark a behavior trained into `expert`, replacing the expert it was
/// trained into before, if any.
pub fn mark_trained(evidence: &mut Vec<String>, expert: &str) {
    set_status(evidence, Status::Trained);
    evidence.retain(|e| value_of(e, EXPERT, LEGACY_EXPERT).is_none());
    evidence.push(format!("{EXPERT}{expert}"));
}

/// Mark a behavior its scope's new expert did not learn: accepted, and in no
/// expert, even if an earlier one held it.
pub fn mark_untrained(evidence: &mut Vec<String>) {
    set_status(evidence, Status::Accepted);
    evidence.retain(|e| value_of(e, EXPERT, LEGACY_EXPERT).is_none());
}

/// A set of behaviors as taught: each id with its content, so one recorded
/// again with a changed rule or check makes a different set. It is stored
/// with each training and on the expert, so it is the same in every build.
pub fn fingerprint(taught: &[(Memory, Spec)]) -> String {
    let mut parts: Vec<(&str, &str)> = taught
        .iter()
        .map(|(m, _)| (m.id.as_str(), m.content.as_str()))
        .collect();
    parts.sort();
    let mut h = Sha256::new();
    for (id, content) in parts {
        for part in [id, content] {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part.as_bytes());
        }
    }
    h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// How the last training of a behavior's scope went for it: the set it was
/// taught in, its held-out tasks, and whether the scope's expert was admitted.
#[derive(Debug, Clone, PartialEq)]
pub struct Training {
    /// The [`fingerprint`] of the set taught.
    pub set: String,
    /// The base model's held-out pass rate.
    pub base: f32,
    /// The trained expert's.
    pub expert: f32,
    /// Whether this behavior met the bar, and so is in the expert when one
    /// was admitted.
    pub learned: bool,
    /// Whether the scope's expert was admitted. It is refused when its
    /// controls fell or no behavior rose above the base model, never for a
    /// behavior it did not learn.
    pub admitted: bool,
    /// How many held-out tasks the rates are over. Notes written before this
    /// was kept read as none.
    pub held: usize,
    /// How likely, from the held-out counts, the expert's true pass rate on
    /// this behavior is below the base model's. Admission refuses the set
    /// when this passes the recipe's confidence; 0.5 says the counts said
    /// nothing either way. The decision an [`Audit`] later checks.
    pub worse: f32,
}

impl Training {
    /// How it went, for the user.
    pub fn describe(&self) -> String {
        let mut rates = format!(
            "held-out {:.2} against the base model's {:.2}",
            self.expert, self.base
        );
        if self.worse > 0.5 {
            rates.push_str(&format!(", {:.0}% likely worse", self.worse * 100.0));
        }
        const MORE: &str = "more examples in varied phrasing may teach it";
        match (self.learned, self.admitted) {
            (true, true) => format!("learned ({rates}); its scope's expert serves it"),
            (true, false) => format!(
                "learned ({rates}), but its scope's expert was refused: its controls fell, or no behavior rose above the base model"
            ),
            (false, true) => format!(
                "not learned ({rates}); its scope's expert serves the others without it, and {MORE}"
            ),
            (false, false) => {
                format!("not learned ({rates}), and its scope's expert was refused; {MORE}")
            }
        }
    }
}

/// Note how the last training of its scope went, replacing the note before.
/// A set whose expert was refused is not trained again, even after a
/// restart, until it changes.
pub fn mark_training(evidence: &mut Vec<String>, training: &Training) {
    // Refusals noted while one behavior not learned refused its whole scope
    // go too: under today's rule such a set may be admitted.
    evidence.retain(|e| !e.starts_with(TRAINING) && !e.starts_with(REFUSED));
    evidence.push(format!(
        "{TRAINING}{} base={:.2} expert={:.2} learned={} admitted={} held={} worse={:.2}",
        training.set,
        training.base,
        training.expert,
        training.learned,
        training.admitted,
        training.held,
        training.worse
    ));
}

/// The last training noted in a behavior's evidence.
pub fn training(evidence: &[String]) -> Option<Training> {
    let entry = evidence
        .iter()
        .rev()
        .find_map(|e| e.strip_prefix(TRAINING))?;
    let mut words = entry.split_whitespace();
    let set = words.next()?.to_string();
    let mut field = |name: &str| words.next()?.strip_prefix(name).map(str::to_string);
    let base = field("base=")?.parse().ok()?;
    let expert = field("expert=")?.parse().ok()?;
    let learned = field("learned=")?.parse().ok()?;
    let admitted = field("admitted=")?.parse().ok()?;
    // Notes written before the counts were kept end here.
    let held = field("held=").and_then(|v| v.parse().ok()).unwrap_or(0);
    let worse = field("worse=").and_then(|v| v.parse().ok()).unwrap_or(0.5);
    Some(Training {
        set,
        base,
        expert,
        learned,
        admitted,
        held,
        worse,
    })
}

fn ln_factorial(n: usize) -> f64 {
    (1..=n).map(|k| (k as f64).ln()).sum()
}

/// `ln B(a, b)` for positive integer parameters.
fn ln_beta(a: usize, b: usize) -> f64 {
    ln_factorial(a - 1) + ln_factorial(b - 1) - ln_factorial(a + b - 1)
}

/// `P(X > Y)` for `X ~ Beta(a1, b1)` and `Y ~ Beta(a2, b2)` with positive
/// integer parameters, in closed form: the sum over the first shape of the
/// ratio of beta functions (Cook, 2005).
pub fn prob_beta_greater(a1: usize, b1: usize, a2: usize, b2: usize) -> f64 {
    (0..a1)
        .map(|i| {
            (ln_beta(a2 + i, b1 + b2)
                - ((b1 + i) as f64).ln()
                - ln_beta(1 + i, b1)
                - ln_beta(a2, b2))
            .exp()
        })
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// How likely it is that the expert's true pass rate on a behavior is below
/// the base model's, given `base_passed` and `expert_passed` of `held` tasks
/// each. Each rate is a `Beta(1 + passed, 1 + failed)` from a flat prior, so
/// the answer carries the evidence: 3 of 3 against 2 of 3 is about 0.74, one
/// task lost of a few; 3 of 3 against 0 of 3 is above 0.98; 18 of 18 against
/// 15 of 18 is above 0.9. With no held-out task there is nothing to say,
/// and the answer is 0.5.
pub fn prob_worse(base_passed: usize, expert_passed: usize, held: usize) -> f64 {
    if held == 0 {
        return 0.5;
    }
    let base_passed = base_passed.min(held);
    let expert_passed = expert_passed.min(held);
    prob_beta_greater(
        1 + base_passed,
        1 + held - base_passed,
        1 + expert_passed,
        1 + held - expert_passed,
    )
}

/// On a standing expert's card, how likely admission found the expert worse
/// than the base model on the behavior under any of `ids`: its own, or the
/// one it supersedes. `None` for a card from before scores were kept on it.
pub fn card_worse(card: &serde_json::Value, ids: &[&str]) -> Option<f32> {
    ids.iter()
        .find_map(|id| card.get(CARD_SCORES)?.get(id)?.get("worse")?.as_f64())
        .map(|w| w as f32)
}

/// A served expert measured, beside the base model, on examples a behavior
/// gained after the expert was trained: tasks the training never saw, written
/// by the user, so what they say about the expert comes from outside the loop
/// that admitted it. The keeper runs one before it retrains a scope whose set
/// changed, and it changes nothing about admission: it is the evidence the
/// recipe's regression confidence is tuned from ([`tuning`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Audit {
    /// The [`fingerprint`] of the set the expert was trained on, which ties
    /// the audit to the [`Training`] that admitted it.
    pub set: String,
    /// How many later examples were measured.
    pub fresh: usize,
    /// How many of them the base model passed.
    pub base: usize,
    /// How many the expert passed.
    pub expert: usize,
    /// How likely admission found the expert worse than the base model on
    /// this behavior, from the card: the bet the audit checks. Above 0.5, the
    /// set was admitted although the behavior had slipped. `None` when the
    /// card was minted before scores were kept on it.
    pub worse: Option<f32>,
}

impl Audit {
    /// How it went, for the user.
    pub fn describe(&self) -> String {
        format!(
            "audited on {} later example{}: the expert passed {}, the base model {}",
            self.fresh,
            if self.fresh == 1 { "" } else { "s" },
            self.expert,
            self.base
        )
    }
}

/// Note an audit of the scope's expert on this behavior's later examples.
/// Audits keep: each is of different examples, and together they are the
/// evidence on the expert's admission.
pub fn mark_audit(evidence: &mut Vec<String>, audit: &Audit) {
    let mut note = format!(
        "{AUDIT}{} fresh={} base={} expert={}",
        audit.set, audit.fresh, audit.base, audit.expert
    );
    if let Some(w) = audit.worse {
        note.push_str(&format!(" worse={w:.2}"));
    }
    evidence.push(note);
}

/// Every audit noted in a behavior's evidence, oldest first.
pub fn audits(evidence: &[String]) -> Vec<Audit> {
    evidence
        .iter()
        .filter_map(|e| {
            let mut words = e.strip_prefix(AUDIT)?.split_whitespace();
            let set = words.next()?.to_string();
            let mut field = |name: &str| words.next()?.strip_prefix(name).map(str::to_string);
            let fresh = field("fresh=")?.parse().ok()?;
            let base = field("base=")?.parse().ok()?;
            let expert = field("expert=")?.parse().ok()?;
            // Notes from before the bet was kept end here.
            let worse = field("worse=").and_then(|v| v.parse().ok());
            Some(Audit {
                set,
                fresh,
                base,
                expert,
                worse,
            })
        })
        .collect()
}

/// What names an example on a standing expert's card: the first eight bytes
/// of a hash over its task and answer, as hex.
pub fn example_print(e: &Example) -> String {
    let mut h = Sha256::new();
    for part in [e.task.as_str(), e.answer.as_str()] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part.as_bytes());
    }
    h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The examples `taught` has that the expert whose `card` this is was neither
/// trained on nor audited on, by behavior: the fresh slice an audit measures.
/// A behavior recorded again is the one it supersedes, continued: its
/// examples are read against what the card knows under either id, since
/// recording again is how a behavior gains examples. A behavior the card never
/// taught, under its own id or the one it supersedes, is left out, since the
/// expert made no decision about it; one with nothing new is left out too.
pub fn fresh_examples(
    taught: &[(Memory, Spec)],
    card: &serde_json::Value,
) -> Vec<(String, Vec<Example>)> {
    let known = |key: &str, id: &str| -> Option<Vec<&str>> {
        Some(
            card.get(key)?
                .get(id)?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str())
                .collect(),
        )
    };
    taught
        .iter()
        .filter_map(|(m, spec)| {
            let id = m.id.as_str();
            let earlier = State::of(&m.evidence).and_then(|s| s.supersedes);
            let ids: Vec<&str> = std::iter::once(id).chain(earlier.as_deref()).collect();
            let trained_on = ids.iter().find_map(|i| known(CARD_EXAMPLES, i))?;
            let audited: Vec<&str> = ids
                .iter()
                .filter_map(|i| known(CARD_AUDITED, i))
                .flatten()
                .collect();
            let fresh: Vec<Example> = spec
                .examples
                .iter()
                .filter(|e| {
                    let print = example_print(e);
                    !trained_on.contains(&print.as_str()) && !audited.contains(&print.as_str())
                })
                .cloned()
                .collect();
            (!fresh.is_empty()).then(|| (id.to_string(), fresh))
        })
        .collect()
}

/// Whether this exact set was trained and its expert refused: every behavior
/// in it notes that training under its [`fingerprint`]. The same set would be
/// refused the same way.
pub fn refused(taught: &[(Memory, Spec)]) -> bool {
    let print = fingerprint(taught);
    !taught.is_empty()
        && taught
            .iter()
            .all(|(m, _)| training(&m.evidence).is_some_and(|t| t.set == print && !t.admitted))
}

/// The training every standing expert on a node gets: the adapter's rank, the
/// epochs over the examples and the replay, and the learning rate. One per
/// server, read from its flags; the keeper trains every scope with it and the
/// serving engine is built at its rank. The default is what nine behaviors
/// were all learned with; 32 taught together at it were not (five learned,
/// two regressed), so these are the knobs for measuring where capacity stops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recipe {
    /// The LoRA rank. The engine serves a blend of three adapters, each
    /// padded to this rank, so lowering it below an expert already minted on
    /// the node means retraining that expert.
    pub rank: usize,
    /// Epochs over the examples and the replay.
    pub rounds: usize,
    pub learning_rate: f64,
    /// How sure admission has to be that the expert answers a behavior it
    /// missed worse than the base model before that refuses the set: the
    /// probability, from the held-out counts, above which a behavior counts
    /// as made worse. Two or three held-out tasks carry little evidence, so
    /// one lost task stays below it; many tasks lost, or many held out,
    /// cross it. The knob the loop tunes as outcomes come in.
    pub regression_confidence: f64,
}

impl Default for Recipe {
    fn default() -> Self {
        Self {
            rank: 16,
            rounds: 3,
            learning_rate: 1.5e-4,
            regression_confidence: 0.9,
        }
    }
}

impl Recipe {
    /// A recipe as a server's flags give it, refused with the reason when a
    /// value could train nothing or everything into noise.
    pub fn new(
        rank: usize,
        rounds: usize,
        learning_rate: f64,
        regression_confidence: f64,
    ) -> Result<Self, String> {
        if !(1..=256).contains(&rank) {
            return Err(format!("the rank is {rank}; it is between 1 and 256"));
        }
        if !(1..=50).contains(&rounds) {
            return Err(format!("{rounds} rounds; between 1 and 50"));
        }
        if !(learning_rate > 0.0 && learning_rate <= 1e-2) {
            return Err(format!(
                "the learning rate is {learning_rate}; it is above 0 and at most 1e-2"
            ));
        }
        if !(regression_confidence > 0.5 && regression_confidence < 1.0) {
            return Err(format!(
                "the regression confidence is {regression_confidence}; it is above 0.5 and below 1, since at 0.5 a coin toss refuses and at 1 nothing does"
            ));
        }
        Ok(Self {
            rank,
            rounds,
            learning_rate,
            regression_confidence,
        })
    }

    /// `rank 16, 3 rounds, 1.5e-4, refusing at 0.90`.
    pub fn describe(&self) -> String {
        format!(
            "rank {}, {} round{}, {:e}, refusing at {:.2}",
            self.rank,
            self.rounds,
            if self.rounds == 1 { "" } else { "s" },
            self.learning_rate,
            self.regression_confidence
        )
    }
}

/// A behavior's state, read back from its evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    pub status: Status,
    pub scope: String,
    pub supersedes: Option<String>,
    /// The expert it was last trained into.
    pub expert: Option<String>,
    /// The memories it was drawn from, when it came from the store.
    pub sources: Vec<String>,
}

impl State {
    /// `None` when the evidence carries no behavior status: not a behavior.
    pub fn of(evidence: &[String]) -> Option<State> {
        let field = |prefix: &str, legacy: &str| {
            evidence
                .iter()
                .rev()
                .find_map(|e| value_of(e, prefix, legacy).map(str::to_string))
        };
        Some(State {
            status: Status::parse(&field(STATUS, LEGACY_STATUS)?)?,
            scope: field(SCOPE, LEGACY_SCOPE).unwrap_or_else(|| EVERYWHERE.to_string()),
            supersedes: field(SUPERSEDES, LEGACY_SUPERSEDES),
            expert: field(EXPERT, LEGACY_EXPERT),
            sources: sources(evidence),
        })
    }
}

/// The evidence entry citing `memory_id` as a memory a behavior was drawn from.
pub fn source_evidence(memory_id: &str) -> String {
    format!("{SOURCE}{memory_id}")
}

/// The memories a behavior cites as what it was drawn from, in the order
/// they were written.
pub fn sources(evidence: &[String]) -> Vec<String> {
    evidence
        .iter()
        .filter_map(|e| e.strip_prefix(SOURCE))
        .map(str::to_string)
        .collect()
}

/// What the standing expert for `scope` is taught: the behaviors accepted,
/// or trained before and still in force, each with its spec. Retired and
/// proposed ones are left out, and so is one that could not be recorded
/// today ([`Spec::problems`]), such as one recorded with three examples before
/// four were required: teaching it ends in refusal every time, after the GPU
/// time is spent. `list_behaviors` names what each one lacks.
pub fn learnable(memories: &[Memory], scope: &str) -> Vec<(Memory, Spec)> {
    memories
        .iter()
        .filter_map(|m| {
            let state = State::of(&m.evidence)?;
            let wanted = matches!(state.status, Status::Accepted | Status::Trained);
            (wanted && state.scope == scope)
                .then(|| Some((m.clone(), spec_of(&m.content)?)))
                .flatten()
        })
        .filter(|(_, spec)| spec.problems().is_empty())
        .collect()
}

/// The scopes a user's behaviors or standing experts are in, everywhere
/// first: each one an expert may have to be trained, retrained, or dropped
/// for.
pub fn scopes(memories: &[Memory], experts: &[Expert]) -> Vec<String> {
    let mut scopes = vec![EVERYWHERE.to_string()];
    let found = memories
        .iter()
        .filter_map(|m| State::of(&m.evidence).map(|s| s.scope))
        .chain(
            experts
                .iter()
                .filter_map(|e| e.standing_scope().map(str::to_string)),
        );
    for s in found {
        if !scopes.contains(&s) {
            scopes.push(s);
        }
    }
    scopes
}

/// The behaviors a standing expert was taught, from its card.
pub fn taught_by(expert: &Expert) -> Vec<String> {
    let card = &expert.capability_card;
    card.get("behaviors")
        .or_else(|| card.get(LEGACY_CARD_KEY))
        .and_then(|v| v.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The [`fingerprint`] of the set a standing expert was trained on, when its
/// card records one. It may hold fewer: only those it learned.
pub fn trained_set(expert: &Expert) -> Option<&str> {
    expert.capability_card.get("set").and_then(|v| v.as_str())
}

/// Whether a scope's standing expert is behind its behaviors: missing while
/// there is something to teach, or trained on a set other than the one in
/// force (a behavior added, retired, or recorded again). An expert that did
/// not learn every behavior of its set is in step with that set: the ones it
/// missed are accepted, not trained, and wait for the set to change.
///
/// An expert whose card records no set is judged by what it holds: behind
/// when it holds a different set, or a behavior was accepted since.
pub fn stale(taught: &[(Memory, Spec)], expert: Option<&Expert>) -> bool {
    let Some(expert) = expert else {
        return !taught.is_empty();
    };
    if let Some(set) = trained_set(expert) {
        return taught.is_empty() || set != fingerprint(taught);
    }
    let accepted_since = taught
        .iter()
        .any(|(m, _)| State::of(&m.evidence).is_some_and(|s| s.status == Status::Accepted));
    let mut held = taught_by(expert);
    held.sort();
    let mut wanted: Vec<String> = taught
        .iter()
        .map(|(m, _)| m.id.as_str().to_string())
        .collect();
    wanted.sort();
    accepted_since || held != wanted
}

pub mod candidate;
pub mod tuning;

#[cfg(test)]
mod tests;
