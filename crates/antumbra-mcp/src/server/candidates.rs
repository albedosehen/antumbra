//! Candidates for behaviors among the memories already stored: a page of the
//! ones that may state how to act, for the agent to judge and record.
//!
//! Its own router, joined to the others in `engine.rs`. What is picked and
//! what is passed over is `antumbra_core::behavior::candidate`; this reads
//! the store in the order memories were written, from a cursor, and leaves
//! out what a behavior already cites, so a pass over a store converges and a
//! later pass starts where the last one stopped.

use std::collections::HashSet;

use chrono::{DateTime, Utc};

use super::*;
use antumbra_core::behavior::{self, candidate};

/// How many memories one call reads when the caller does not say.
const DEFAULT_SCAN: u32 = 200;
/// The most one call reads. A page costs its own rows, and the excerpts of
/// the candidates among them come back in one answer.
const MAX_SCAN: u32 = 500;

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct CandidatesParams {
    /// Read memories written after this time (RFC 3339): the `scanned_through`
    /// of the last call, to continue. Omit to start from the oldest.
    #[serde(default)]
    pub(super) after: Option<String>,
    /// How many memories to read, oldest first (default 200, at most 500).
    /// Only the candidates among them come back, so a page holds fewer.
    #[serde(default)]
    pub(super) scan: Option<u32>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct CandidateView {
    /// The memory's id: cite it in `record_behavior`'s `sources`.
    pub(super) id: String,
    /// When it was written (RFC 3339).
    pub(super) created_at: String,
    /// `world`, `bank` or `opinion`.
    pub(super) network: String,
    /// The repository its git anchor names, when it has one: the likely
    /// scope of a behavior drawn from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) repo: Option<String>,
    /// The first 400 characters of its content.
    pub(super) excerpt: String,
    /// Its sentences that open like a rule, with a negation or an imperative
    /// verb: up to three, each at most 160 characters. The rule is often an
    /// aside deep inside a long note, past the excerpt.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) hits: Vec<String>,
    /// Why it was picked, up to eight: `opinion`; `tag:<word>` for a leading
    /// `[feedback]`-style tag; `says:<phrase>` for a phrase that records a
    /// correction or a preference; `opens:<word>` for each word its rule-shaped
    /// sentences open with.
    pub(super) reasons: Vec<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct CandidatesOut {
    /// The memories among those read that may state a behavior, oldest
    /// first: at most `scan` of them, and none when nothing read was worth
    /// a look (`scanned` says how many were read).
    pub(super) candidates: Vec<CandidateView>,
    /// How many memories were read for them.
    pub(super) scanned: u32,
    /// When the last memory read was written: pass it as `after` to
    /// continue. Absent when nothing was read, which is the end of the store.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) scanned_through: Option<String>,
    /// Whether memories remain after these.
    pub(super) more: bool,
}

#[tool_router(router = candidates_router, vis = "pub(super)")]
impl McpServer {
    /// Memories that may state a behavior, for the agent to judge.
    #[tool(
        description = "Memories already stored that may state a behavior, oldest first: every opinion memory, any memory opening with a tag such as [feedback] or [convention], any memory with a phrase that records a correction or a preference, and any memory with a short sentence that opens like a rule (never, always, do not, use, ...), which comes back in `hits`. Behaviors, handoffs, volatile memories and memories a behavior already cites are left out. Read each one. When it states how an agent should act on a class of tasks, and a program could check an answer, call record_behavior with the rule, a check, examples and violations, citing the memory's id in `sources` (with any memories that restate it, found with recall_memories), so it is not picked again; otherwise pass it over. Pass `scanned_through` back as `after` to continue; `more` false is the end of the store."
    )]
    pub(super) async fn behavior_candidates(
        &self,
        Parameters(p): Parameters<CandidatesParams>,
    ) -> Result<Json<CandidatesOut>, ErrorData> {
        let after = match p.after.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(s) => Some(
                DateTime::parse_from_rfc3339(s)
                    .map_err(|e| {
                        ErrorData::invalid_params(
                            format!("after is not an RFC 3339 time: {e}"),
                            None,
                        )
                    })?
                    .with_timezone(&Utc),
            ),
            None => None,
        };
        let scan = p.scan.unwrap_or(DEFAULT_SCAN).clamp(1, MAX_SCAN);
        let compartment = behavior::compartment_id(&self.tenant, &self.user);
        let cited: HashSet<String> =
            memory::list_by_compartment(&self.store, &self.tenant, &compartment)
                .await
                .map_err(err)?
                .iter()
                .flat_map(|m| behavior::sources(&m.evidence))
                .collect();
        let mut rows = memory::since(&self.store, &self.tenant, after, scan + 1)
            .await
            .map_err(err)?;
        let more = rows.len() > scan as usize;
        rows.truncate(scan as usize);
        let scanned = rows.len() as u32;
        let scanned_through = rows.last().map(|m| m.created_at.to_rfc3339());
        let candidates = rows
            .iter()
            .filter_map(|m| candidate::candidate(m, &cited))
            .map(|c| CandidateView {
                id: c.id,
                created_at: c.created_at.to_rfc3339(),
                network: c.network.as_str().to_string(),
                repo: c.repo,
                excerpt: c.excerpt,
                hits: c.hits,
                reasons: c.reasons,
            })
            .collect();
        Ok(Json(CandidatesOut {
            candidates,
            scanned,
            scanned_through,
            more,
        }))
    }
}
