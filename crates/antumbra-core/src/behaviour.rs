//! Behaviours (ADR-0027): how the user wants an agent to act on a class of
//! tasks, stated as a rule with a check a program can apply.
//!
//! Experts learn behaviour, never facts. A behaviour trains from tasks it
//! governs and a check that decides each answer, so it is recorded with both,
//! by the agent that learned it, rather than dug out of free text afterwards.
//!
//! A behaviour is a memory in the user's own `behaviour` compartment, as a
//! handoff is in its own. Its content leads with the rule in prose, which is
//! what recall and the agent read, followed by a fenced `behaviour` block with
//! what training reads: the patterns an answer must and must not match, worked
//! examples, and violating answers. Evidence entries carry its status, its
//! scope, and the behaviour it supersedes.

use fancy_regex::Regex;
use serde::{Deserialize, Serialize};

use crate::ids::{CompartmentId, TenantId, UserId};

/// The scope of a behaviour that applies in every repository.
pub const EVERYWHERE: &str = "everywhere";
/// The fewest worked examples a behaviour is recorded with.
pub const MIN_EXAMPLES: usize = 3;

const STATUS: &str = "behaviour-status:";
const SCOPE: &str = "behaviour-scope:";
const SUPERSEDES: &str = "behaviour-supersedes:";
const FENCE: &str = "```behaviour";

/// A task the behaviour governs and an answer that follows it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Example {
    /// The task, as a person would ask it.
    pub task: String,
    /// An answer that follows the behaviour.
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
    /// Tasks with answers that follow the behaviour.
    pub examples: Vec<Example>,
    /// Answers that break the behaviour; the check must refuse each.
    #[serde(default)]
    pub violations: Vec<String>,
}

/// Where a behaviour stands. The system proposes and the user accepts; a
/// trained behaviour is in an expert; a retired one trains nothing.
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
    /// Whether `answer` follows the behaviour: every `must` pattern matches and
    /// no `must_not` pattern does. `None` when a pattern does not compile.
    pub fn follows(&self, answer: &str) -> Option<bool> {
        let mut problems = Vec::new();
        let must = compile(&self.must, &mut problems);
        let must_not = compile(&self.must_not, &mut problems);
        problems
            .is_empty()
            .then(|| follows(&must, &must_not, answer))
    }

    /// Everything that keeps this behaviour from training; empty when it can.
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
                "{} example(s); a behaviour needs at least {MIN_EXAMPLES}",
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

/// The user's behaviour compartment, one per `(tenant, user)`.
pub fn compartment_id(tenant: &TenantId, user: &UserId) -> CompartmentId {
    CompartmentId::new(format!(
        "comp:{}:{}:behaviour",
        tenant.as_str(),
        user.as_str()
    ))
}

/// A behaviour as a memory's content: the rule first, for recall and the
/// agent, then the fenced block training reads.
pub fn content(spec: &Spec) -> String {
    let json = serde_json::to_string_pretty(spec).unwrap_or_default();
    format!("{}\n\n{FENCE}\n{json}\n```\n", spec.rule.trim())
}

/// The spec in a behaviour memory's content, if it holds one.
pub fn spec_of(content: &str) -> Option<Spec> {
    let start = content.find(FENCE)? + FENCE.len();
    let rest = &content[start..];
    let end = rest.find("```")?;
    serde_json::from_str(rest[..end].trim()).ok()
}

/// A scope as behaviours compare them: a repository slug, or [`EVERYWHERE`].
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

/// Set a behaviour's status in its evidence, replacing the one it had.
pub fn set_status(evidence: &mut Vec<String>, status: Status) {
    evidence.retain(|e| !e.starts_with(STATUS));
    evidence.push(status_evidence(status));
}

/// A behaviour's state, read back from its evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    pub status: Status,
    pub scope: String,
    pub supersedes: Option<String>,
}

impl State {
    /// `None` when the evidence carries no behaviour status: not a behaviour.
    pub fn of(evidence: &[String]) -> Option<State> {
        let field = |prefix: &str| {
            evidence
                .iter()
                .rev()
                .find_map(|e| e.strip_prefix(prefix).map(str::to_string))
        };
        Some(State {
            status: Status::parse(&field(STATUS)?)?,
            scope: field(SCOPE).unwrap_or_else(|| EVERYWHERE.to_string()),
            supersedes: field(SUPERSEDES),
        })
    }
}

#[cfg(test)]
mod tests;
