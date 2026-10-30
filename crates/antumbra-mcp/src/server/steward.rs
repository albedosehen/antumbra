//! The system's own hand on its memories: what it watches, and what it does
//! about what it sees, so nobody has to reinforce or penalize by hand.
//!
//! Three signals, each counted in `memory_use` (never on the memory row):
//!
//! - **Shown.** Recall counts every memory it returns, and remembers which a
//!   session was shown (hooks pass the client's session id) or, failing that,
//!   which this user's agent on this machine pulled up itself.
//! - **Used or passed over.** When a turn ends, the client's Stop hook sends
//!   the turn's last answer (`observe_turn`). The memories that session was
//!   shown are judged against it, by the relevance head where one is configured
//!   and by cosine where not: each one the answer drew on counts as used, the
//!   rest as passed over. What an agent recalled for itself is tied to no one
//!   session, so it can only count as used, never as passed over.
//! - **Restated.** A new memory is linked to the existing ones it comes close
//!   to, and one close enough to be the same knowledge counts as recurring.
//!
//! Strength is read from those counts at every recall ([`antumbra_core::usage`]),
//! with the workspace's knobs, which an operator turns from the console. An
//! agent's own judgment (`reinforce_memory`, `penalize_memory`) is counted
//! here too, beside the system's.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use antumbra_core::{strength, MemoryPolicy};
use antumbra_store::repo::memory_policy;
use antumbra_store::repo::memory_use::{self, Seen};

use super::*;

/// How long a workspace's knobs are read from memory before the store again.
/// A console change clears its workspace's entry at once, so this bounds only a
/// change made through another server.
const POLICY_TTL: Duration = Duration::from_secs(30);
/// How long a session's shown memories wait for the turn's end.
const SESSION_SHOWN_TTL: Duration = Duration::from_secs(6 * 3600);
/// How long what an agent recalled for itself waits.
const AGENT_SHOWN_TTL: Duration = Duration::from_secs(30 * 60);
/// The most memories remembered as shown to one session or agent at once.
const MAX_SHOWN: usize = 64;
/// The most sessions and agents remembered at once.
const MAX_WATCHED: usize = 4096;
/// The most recent turns remembered, so a turn is judged once.
const MAX_TURNS: usize = 4096;
/// How much of a turn's answer is judged.
const ANSWER_CHARS: usize = 4000;
/// How much of each memory the relevance head reads when judging a turn.
const MEMORY_CHARS: usize = 2000;

struct Shown {
    ids: Vec<MemoryId>,
    at: Instant,
    ttl: Duration,
}

/// What the server keeps between calls to govern memories: the knobs it has
/// read, what each session and agent was shown, and the turns already judged.
/// One per server process, shared by every identity's tools.
#[derive(Default)]
pub(crate) struct Steward {
    policies: Mutex<HashMap<String, (MemoryPolicy, Instant)>>,
    shown: Mutex<HashMap<String, Shown>>,
    turns: Mutex<(VecDeque<String>, HashSet<String>)>,
}

impl Steward {
    /// `tenant`'s knobs, from memory while fresh. A store that cannot be read
    /// runs the call on the defaults, said on stderr, rather than failing it.
    async fn policy(&self, store: &Store, tenant: &TenantId) -> MemoryPolicy {
        if let Some((p, at)) = self.lock_policies().get(tenant.as_str()) {
            if at.elapsed() < POLICY_TTL {
                return p.clone();
            }
        }
        let policy = memory_policy::get(store, tenant).await.unwrap_or_else(|e| {
            eprintln!("antumbra-mcp: memory knobs unreadable for {tenant}, using defaults: {e}");
            MemoryPolicy::default()
        });
        self.lock_policies().insert(
            tenant.as_str().to_string(),
            (policy.clone(), Instant::now()),
        );
        policy
    }

    /// Forget `tenant`'s cached knobs, so the next read takes the new ones.
    pub(crate) fn forget_policy(&self, tenant: &str) {
        self.lock_policies().remove(tenant);
    }

    fn remember_shown(&self, key: String, ids: &[MemoryId], ttl: Duration) {
        let mut shown = self.lock_shown();
        if shown.len() >= MAX_WATCHED {
            shown.retain(|_, s| s.at.elapsed() < s.ttl);
            if shown.len() >= MAX_WATCHED {
                if let Some(oldest) = shown
                    .iter()
                    .min_by_key(|(_, s)| s.at)
                    .map(|(k, _)| k.clone())
                {
                    shown.remove(&oldest);
                }
            }
        }
        let entry = shown.entry(key).or_insert_with(|| Shown {
            ids: Vec::new(),
            at: Instant::now(),
            ttl,
        });
        if entry.at.elapsed() >= entry.ttl {
            entry.ids.clear();
        }
        for id in ids {
            if !entry.ids.contains(id) {
                entry.ids.push(id.clone());
            }
        }
        let over = entry.ids.len().saturating_sub(MAX_SHOWN);
        entry.ids.drain(..over);
        entry.at = Instant::now();
    }

    fn take_shown(&self, key: &str) -> Vec<MemoryId> {
        match self.lock_shown().remove(key) {
            Some(s) if s.at.elapsed() < s.ttl => s.ids,
            _ => Vec::new(),
        }
    }

    /// `true` the first time `turn` is seen.
    fn first_time(&self, turn: String) -> bool {
        let mut guard = self.turns.lock().unwrap_or_else(|p| p.into_inner());
        let (order, seen) = &mut *guard;
        if !seen.insert(turn.clone()) {
            return false;
        }
        order.push_back(turn);
        if order.len() > MAX_TURNS {
            if let Some(old) = order.pop_front() {
                seen.remove(&old);
            }
        }
        true
    }

    fn lock_policies(&self) -> std::sync::MutexGuard<'_, HashMap<String, (MemoryPolicy, Instant)>> {
        self.policies.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn lock_shown(&self) -> std::sync::MutexGuard<'_, HashMap<String, Shown>> {
        self.shown.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ObserveParams {
    /// The client session the turn belongs to, as its recalls named it.
    pub(super) session: Option<String>,
    /// The turn's id, so a turn reported twice is judged once.
    pub(super) turn: Option<String>,
    /// The turn's last answer.
    #[serde(default)]
    pub(super) text: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ObservedOut {
    /// Memories the session was shown and the answer was judged against.
    pub(super) judged: u32,
    /// Of those, and of what the agent recalled for itself, the ones the
    /// answer drew on.
    pub(super) used: u32,
    /// The turn had been reported already, and was not judged again.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) repeated: bool,
}

/// The first `n` characters of `s`.
fn prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

impl McpServer {
    /// Share `steward` with this server, so what one identity's session was
    /// shown can be judged when another connection reports the turn.
    #[must_use]
    pub(crate) fn with_steward(mut self, steward: Arc<Steward>) -> Self {
        self.steward = steward;
        self
    }

    /// This workspace's knobs.
    pub(super) async fn policy(&self) -> MemoryPolicy {
        self.steward.policy(&self.store, &self.tenant).await
    }

    /// `hits`, in relevance order, moved by strength as far as the knobs let
    /// it, with each one's strength. A hit keeps its place unless one near it
    /// stands enough stronger: relevance is read from the position, `1 - i/n`,
    /// and multiplied by [`MemoryPolicy::rank_factor`].
    pub(super) async fn by_strength(
        &self,
        hits: Vec<Memory>,
        policy: &MemoryPolicy,
    ) -> (Vec<Memory>, HashMap<String, f32>) {
        if hits.is_empty() {
            return (hits, HashMap::new());
        }
        let ids: Vec<MemoryId> = hits.iter().map(|m| m.id.clone()).collect();
        let usage = memory_use::for_memories(&self.store, &self.tenant, &ids)
            .await
            .unwrap_or_else(|e| {
                eprintln!("antumbra-mcp: memory use unreadable, ranking without it: {e}");
                Vec::new()
            });
        let seen: HashMap<&str, _> = usage.iter().map(|u| (u.memory.as_str(), u)).collect();
        let now = Utc::now();
        let strengths: HashMap<String, f32> = hits
            .iter()
            .map(|m| {
                let s = strength(m, seen.get(m.id.as_str()).copied(), policy, now);
                (m.id.as_str().to_string(), s)
            })
            .collect();
        let n = hits.len() as f32;
        let mut scored: Vec<(f32, Memory)> = hits
            .into_iter()
            .enumerate()
            .map(|(i, m)| {
                let relevance = 1.0 - i as f32 / n;
                (relevance * policy.rank_factor(strengths[m.id.as_str()]), m)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        (scored.into_iter().map(|(_, m)| m).collect(), strengths)
    }

    /// Who a recall's memories were shown to: the session it named, else this
    /// user's agent on the calling machine.
    fn shown_key(&self, session: Option<&str>) -> (String, Duration) {
        match session.map(str::trim).filter(|s| !s.is_empty()) {
            Some(s) => (format!("{}|session:{s}", self.tenant), SESSION_SHOWN_TTL),
            None => (self.agent_key(), AGENT_SHOWN_TTL),
        }
    }

    fn agent_key(&self) -> String {
        format!("{}|agent:{}@{}", self.tenant, self.user, self.device())
    }

    /// Count what a recall returned, and remember who it was shown to. In the
    /// call, not after it: on an embedded server the connection is signed in
    /// as this caller only while the call runs.
    pub(super) async fn record_shown(
        &self,
        ids: &[MemoryId],
        session: Option<&str>,
        policy: &MemoryPolicy,
    ) {
        if !policy.auto || ids.is_empty() {
            return;
        }
        if let Err(e) =
            memory_use::record(&self.store, &self.tenant, ids, Seen::RECALLED, Utc::now()).await
        {
            eprintln!("antumbra-mcp: could not count a recall: {e}");
        }
        let (key, ttl) = self.shown_key(session);
        self.steward.remember_shown(key, ids, ttl);
    }

    /// Count an agent's own judgment of a memory beside the system's.
    pub(super) async fn record_judgment(&self, id: &MemoryId, good: bool) {
        let seen = if good { Seen::GOOD } else { Seen::BAD };
        if let Err(e) = memory_use::record(
            &self.store,
            &self.tenant,
            std::slice::from_ref(id),
            seen,
            Utc::now(),
        )
        .await
        {
            eprintln!("antumbra-mcp: could not count a judgment: {e}");
        }
    }

    /// Which of `memories` `text` drew on: the relevance head's probability, or
    /// the cosine where no head is configured or it fails, against
    /// `threshold`.
    async fn drawn_on(&self, text: &str, memories: &[Memory], threshold: f32) -> Vec<bool> {
        use antumbra_core::ports::{Answer, Question};
        if let Some(decider) = &self.decider {
            let state = antumbra_core::ports::RelevanceState {
                query: prefix(text, ANSWER_CHARS),
                memories: memories
                    .iter()
                    .map(|m| prefix(&m.content, MEMORY_CHARS))
                    .collect(),
            }
            .encode();
            let questions = vec![Question::Noul; memories.len()];
            match decider.decide(&state, &questions).await {
                Ok(answers) if answers.len() == memories.len() => {
                    return answers
                        .iter()
                        .map(|a| matches!(a, Answer::Noul { probability } if *probability >= threshold))
                        .collect();
                }
                Ok(_) => eprintln!(
                    "antumbra-mcp: the relevance head misanswered a turn; judging by cosine"
                ),
                Err(e) => eprintln!(
                    "antumbra-mcp: the relevance head failed on a turn, judging by cosine: {e}"
                ),
            }
        }
        match self.embedder.embed(&prefix(text, ANSWER_CHARS)).await {
            Ok(answer) => memories
                .iter()
                .map(|m| {
                    m.embedding
                        .as_deref()
                        .is_some_and(|e| antumbra_core::cosine_similarity(&answer, e) >= threshold)
                })
                .collect(),
            Err(e) => {
                eprintln!("antumbra-mcp: could not embed a turn to judge it: {e}");
                Vec::new()
            }
        }
    }

    /// Judge the end of a turn: which of the memories shown for it the answer
    /// drew on. Reached by the client's Stop hook through `/mcp/call`; no
    /// agent is offered it.
    pub(super) async fn observe_turn(
        &self,
        Parameters(p): Parameters<ObserveParams>,
    ) -> Result<Json<ObservedOut>, ErrorData> {
        let policy = self.policy().await;
        let nothing = ObservedOut {
            judged: 0,
            used: 0,
            repeated: false,
        };
        if !policy.auto {
            return Ok(Json(nothing));
        }
        if let Some(turn) = p.turn.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
            if !self.steward.first_time(format!("{}|{turn}", self.tenant)) {
                return Ok(Json(ObservedOut {
                    repeated: true,
                    ..nothing
                }));
            }
        }
        let judged: Vec<MemoryId> = match p.session.as_deref() {
            Some(s) if !s.trim().is_empty() => self.steward.take_shown(&self.shown_key(Some(s)).0),
            _ => Vec::new(),
        };
        let mut own = self.steward.take_shown(&self.agent_key());
        own.retain(|id| !judged.contains(id));
        let text = p.text.trim();
        if text.is_empty() || (judged.is_empty() && own.is_empty()) {
            return Ok(Json(nothing));
        }
        let ids: Vec<String> = judged
            .iter()
            .chain(&own)
            .map(|m| m.as_str().to_string())
            .collect();
        let memories = memory::get_many(&self.store, &self.tenant, &ids)
            .await
            .map_err(err)?;
        let drawn = self.drawn_on(text, &memories, policy.use_threshold).await;
        if drawn.len() != memories.len() {
            return Ok(Json(nothing));
        }
        let (mut used, mut passed, mut used_own) = (Vec::new(), Vec::new(), Vec::new());
        for (m, drew) in memories.iter().zip(drawn) {
            let was_shown = judged.contains(&m.id);
            match (was_shown, drew) {
                (true, true) => used.push(m.id.clone()),
                (true, false) => passed.push(m.id.clone()),
                (false, true) => used_own.push(m.id.clone()),
                (false, false) => {}
            }
        }
        let now = Utc::now();
        // What the agent recalled for itself is recorded only when drawn on,
        // and then as any use is: a judgment that came out used.
        for (ids, seen) in [
            (&used, Seen::USED),
            (&passed, Seen::PASSED_OVER),
            (&used_own, Seen::USED),
        ] {
            memory_use::record(&self.store, &self.tenant, ids, seen, now)
                .await
                .map_err(err)?;
        }
        Ok(Json(ObservedOut {
            judged: (used.len() + passed.len()) as u32,
            used: (used.len() + used_own.len()) as u32,
            repeated: false,
        }))
    }

    /// Link a memory just stored to the existing ones it comes close to, and
    /// count one close enough to be the same knowledge as recurring. Returns
    /// the ids linked. Best-effort: a failure is said on stderr and leaves the
    /// memory stored.
    pub(super) async fn link_new(&self, m: &Memory, policy: &MemoryPolicy) -> Vec<String> {
        let Some(embedding) = m.embedding.as_deref() else {
            return Vec::new();
        };
        if !policy.auto || policy.max_links == 0 {
            return Vec::new();
        }
        let k = policy.max_links as usize + 1;
        let near = match memory::recall(&self.store, &self.tenant, embedding, k, None).await {
            Ok(near) => near,
            Err(e) => {
                eprintln!("antumbra-mcp: could not look for a new memory's neighbors: {e}");
                return Vec::new();
            }
        };
        let mut close: Vec<(Memory, f32)> = near
            .into_iter()
            .filter(|n| n.id != m.id)
            .filter_map(|n| {
                let c = antumbra_core::cosine_similarity(embedding, n.embedding.as_deref()?);
                (c >= policy.link_threshold).then_some((n, c))
            })
            .collect();
        close.sort_by(|a, b| b.1.total_cmp(&a.1));
        close.truncate(policy.max_links as usize);
        if close.is_empty() {
            return Vec::new();
        }
        // Cosine alone favors short rows; where a relevance head is configured
        // it confirms each link, as it judges a turn.
        if self.decider.is_some() {
            let candidates: Vec<Memory> = close.iter().map(|(n, _)| n.clone()).collect();
            let confirmed = self
                .drawn_on(&m.content, &candidates, policy.use_threshold)
                .await;
            if confirmed.len() == close.len() {
                close = close
                    .into_iter()
                    .zip(confirmed)
                    .filter_map(|(pair, ok)| ok.then_some(pair))
                    .collect();
            }
        }
        let now = Utc::now();
        let mut linked = Vec::new();
        let mut recurred = Vec::new();
        for (n, c) in &close {
            let e = MemoryEdge::new(
                self.tenant.clone(),
                m.id.clone(),
                n.id.clone(),
                EdgeType::References,
                *c,
                now,
            );
            match edge::relate(&self.store, &e).await {
                Ok(()) => linked.push(n.id.as_str().to_string()),
                Err(e) => eprintln!("antumbra-mcp: could not link a new memory: {e}"),
            }
            if *c >= policy.recurrence_threshold {
                recurred.push(n.id.clone());
            }
        }
        if let Err(e) =
            memory_use::record(&self.store, &self.tenant, &recurred, Seen::RECURRED, now).await
        {
            eprintln!("antumbra-mcp: could not count a restatement: {e}");
        }
        linked
    }
}

#[cfg(test)]
mod tests;
