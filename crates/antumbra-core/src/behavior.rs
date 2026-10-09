//! Behaviors (ADR-0027): how the user wants an agent to act on a class of
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
/// training holds one example in four out to admit the expert by
/// (`antumbra_train::behave::split`): with three, nothing is held out, the
/// expert can never be admitted, and every training of it is spent for
/// nothing.
pub const MIN_EXAMPLES: usize = 4;

/// The name of a user's behavior compartment, and the last part of its id.
/// It keeps the spelling it was first stored with: the compartments already
/// in a store are found by it.
pub const COMPARTMENT_NAME: &str = "behaviour";

const STATUS: &str = "behavior-status:";
const SCOPE: &str = "behavior-scope:";
const SUPERSEDES: &str = "behavior-supersedes:";
const EXPERT: &str = "behavior-expert:";
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
                "{} example(s); a behavior needs at least {MIN_EXAMPLES}, since training holds one in four out to admit its expert by",
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

/// A behavior as a memory's content: the rule first, for recall and the
/// agent, then the fenced block training reads.
pub fn content(spec: &Spec) -> String {
    let json = serde_json::to_string_pretty(spec).unwrap_or_default();
    format!("{}\n\n{FENCE}\n{json}\n```\n", spec.rule.trim())
}

/// The spec in a behavior memory's content, if it holds one: in a
/// `behavior` block, or a `behaviour` block written before the US spelling.
pub fn spec_of(content: &str) -> Option<Spec> {
    let start = content
        .find(FENCE)
        .map(|i| i + FENCE.len())
        .or_else(|| content.find(LEGACY_FENCE).map(|i| i + LEGACY_FENCE.len()))?;
    let rest = &content[start..];
    let end = rest.find("```")?;
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
    evidence.retain(|e| value_of(e, EXPERT, LEGACY_EXPERT).is_none() && !e.starts_with(REFUSED));
    evidence.push(format!("{EXPERT}{expert}"));
}

/// A set of behaviors as taught: each id with its content, so one recorded
/// again with a changed rule or check makes a different set. It is stored
/// with a refusal, so it is the same in every build.
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

/// The last training of a behavior's scope that minted no expert: the set it
/// was taught in, and how this behavior's held-out tasks went.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    /// The [`fingerprint`] of the set taught.
    pub set: String,
    /// The base model's held-out pass rate.
    pub base: f32,
    /// The trained expert's.
    pub expert: f32,
    /// Whether this behavior met the bar itself, so the expert was refused
    /// for another behavior in its scope or for the controls.
    pub learned: bool,
}

impl Refusal {
    /// How it went, for the user.
    pub fn describe(&self) -> String {
        let rates = format!(
            "held-out {:.2} against the base model's {:.2}",
            self.expert, self.base
        );
        if self.learned {
            format!("learned ({rates}), but its scope's expert was refused for another behavior or the controls")
        } else {
            format!("not learned ({rates}), so its scope's expert was refused; more examples in varied phrasing may teach it")
        }
    }
}

/// Note that `set` was trained and refused, replacing any earlier refusal.
/// The keeper does not train the same set again, even after a restart.
pub fn mark_refused(evidence: &mut Vec<String>, refusal: &Refusal) {
    evidence.retain(|e| !e.starts_with(REFUSED));
    evidence.push(format!(
        "{REFUSED}{} base={:.2} expert={:.2} learned={}",
        refusal.set, refusal.base, refusal.expert, refusal.learned
    ));
}

/// The last refusal noted in a behavior's evidence.
pub fn refusal(evidence: &[String]) -> Option<Refusal> {
    let entry = evidence
        .iter()
        .rev()
        .find_map(|e| e.strip_prefix(REFUSED))?;
    let mut words = entry.split_whitespace();
    let set = words.next()?.to_string();
    let mut field = |name: &str| words.next()?.strip_prefix(name).map(str::to_string);
    Some(Refusal {
        set,
        base: field("base=")?.parse().ok()?,
        expert: field("expert=")?.parse().ok()?,
        learned: field("learned=")?.parse().ok()?,
    })
}

/// Whether this exact set was trained and refused: every behavior in it
/// carries a refusal under its [`fingerprint`]. The same set would fail the
/// same way.
pub fn refused(taught: &[(Memory, Spec)]) -> bool {
    let print = fingerprint(taught);
    !taught.is_empty()
        && taught
            .iter()
            .all(|(m, _)| refusal(&m.evidence).is_some_and(|r| r.set == print))
}

/// A behavior's state, read back from its evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    pub status: Status,
    pub scope: String,
    pub supersedes: Option<String>,
    /// The expert it was last trained into.
    pub expert: Option<String>,
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
        })
    }
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

/// Whether a scope's standing expert is behind its behaviors: missing while
/// there is something to teach, holding any while nothing is left, holding a
/// different set, or with a behavior accepted since (a new one, or one
/// recorded again with a changed rule or check).
pub fn stale(taught: &[(Memory, Spec)], expert: Option<&Expert>) -> bool {
    let Some(expert) = expert else {
        return !taught.is_empty();
    };
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

#[cfg(test)]
mod tests;
