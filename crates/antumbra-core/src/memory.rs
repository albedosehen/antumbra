//! Penumbra — the soft, editable memory store (the partial shadow).
//!
//! A memory is a trace that has not yet hardened into the umbra (a frozen
//! expert). It is fast to write, editable, reinforced over use, and tenant-
//! scoped to a [`WorkspaceId`]. Memories are the *consolidation source*: a
//! reinforced, stable, verifiable trace graduates store→weights (EXP-021); a
//! contradicted one retires the expert it produced. This is the hippocampus to
//! the population's neocortex.
//!
//! The three networks mirror the rule/fact/judgment split a memory store keeps:
//! `World` (facts), `Bank` (experiences), `Opinion` (judgments/preferences).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{CompartmentId, ExpertId, MemoryId, TenantId, UserId};

/// Which network a memory belongs to — the coarse skill/kind it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryNetwork {
    /// Durable facts about the world.
    World,
    /// Lived experiences / incidents.
    Bank,
    /// Judgments, preferences, feedback.
    Opinion,
}

impl MemoryNetwork {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryNetwork::World => "world",
            MemoryNetwork::Bank => "bank",
            MemoryNetwork::Opinion => "opinion",
        }
    }
}

/// Whether a memory is a committed trace or a *planned* one — an announced
/// intent that other agents can see before it is acted on (the "planned
/// changes" awareness signal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryStatus {
    #[default]
    Committed,
    Planned,
}

impl MemoryStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryStatus::Committed => "committed",
            MemoryStatus::Planned => "planned",
        }
    }
}

/// A typed relationship between two memories (the Penumbra graph). `Supersedes`
/// and `Contradicts` are the native signal for the consolidation→retire loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeType {
    /// Cites or relates to another memory.
    References,
    /// Replaces an older memory.
    Supersedes,
    /// Conflicts with another memory.
    Contradicts,
    /// Comes after another in a sequence.
    Follows,
    /// Was caused by another.
    Caused,
}

impl EdgeType {
    pub fn as_str(self) -> &'static str {
        match self {
            EdgeType::References => "references",
            EdgeType::Supersedes => "supersedes",
            EdgeType::Contradicts => "contradicts",
            EdgeType::Follows => "follows",
            EdgeType::Caused => "caused",
        }
    }
}

/// A directed, tenant-scoped edge `from -> to` in the Penumbra graph.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEdge {
    pub tenant: TenantId,
    pub from_id: MemoryId,
    pub to_id: MemoryId,
    pub edge_type: EdgeType,
    pub weight: f32,
    pub created_at: DateTime<Utc>,
}

impl MemoryEdge {
    pub fn new(
        tenant: impl Into<TenantId>,
        from_id: impl Into<MemoryId>,
        to_id: impl Into<MemoryId>,
        edge_type: EdgeType,
        weight: f32,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            tenant: tenant.into(),
            from_id: from_id.into(),
            to_id: to_id.into(),
            edge_type,
            weight,
            created_at: now,
        }
    }
}

/// A tenant-scoped memory trace in the Penumbra.
#[derive(Debug, Clone, PartialEq)]
pub struct Memory {
    /// Globally-unique id (the SurrealDB record key).
    pub id: MemoryId,
    /// The owning tenant — the isolation key (engine-enforced via `tenant_id =
    /// $auth.tenant`; the repo also filters on it as the second layer).
    pub tenant: TenantId,
    pub network: MemoryNetwork,
    pub content: String,
    /// Semantic embedding for recall (KNN). `None` until embedded.
    pub embedding: Option<Vec<f32>>,
    /// Confidence / strength in `[0, 1]`; rises as the trace is reinforced.
    pub confidence: f32,
    /// How many times the trace has been reinforced/accessed — the recurrence
    /// signal the consolidation gate scores.
    pub reinforcement: u32,
    /// Provenance — the sources/evidence that justify the trace.
    pub evidence: Vec<String>,
    /// `true` if the fact changes over time; volatile traces never graduate.
    pub volatile: bool,
    /// Set once the trace has graduated into the umbra: the expert it produced.
    /// A contradiction against a consolidated memory retires this expert.
    pub consolidated_expert: Option<ExpertId>,
    /// The compartment this memory lives in. `None` = the author's default
    /// compartment (resolved at the boundary). The unit of sharing/reference.
    pub compartment: Option<CompartmentId>,
    /// Provenance: the user who authored the memory, and the host/device it was
    /// written from (so a recall can say who learned this, and where).
    pub author: Option<UserId>,
    pub author_host: Option<String>,
    /// Committed vs a planned/announced intent (cross-agent awareness).
    pub status: MemoryStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// When set, this trace is a **tombstone**: forgotten, but retained so the
    /// deletion propagates (last-write-wins sync, R-1) and is routed to grantees
    /// (R-2) instead of silently resurfacing from another replica. Read paths
    /// hide tombstones; a grace-windowed purge removes them for good. `None` for a
    /// live trace.
    pub deleted_at: Option<DateTime<Utc>>,
}

impl Memory {
    /// A fresh, un-reinforced trace at the given confidence.
    pub fn new(
        id: impl Into<MemoryId>,
        tenant: impl Into<TenantId>,
        network: MemoryNetwork,
        content: impl Into<String>,
        confidence: f32,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            id: id.into(),
            tenant: tenant.into(),
            network,
            content: content.into(),
            embedding: None,
            confidence: confidence.clamp(0.0, 1.0),
            reinforcement: 0,
            evidence: Vec::new(),
            volatile: false,
            consolidated_expert: None,
            compartment: None,
            author: None,
            author_host: None,
            status: MemoryStatus::Committed,
            created_at: now,
            updated_at: now,
            deleted_at: None,
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Place the memory in a compartment (the unit of sharing).
    pub fn in_compartment(mut self, compartment: impl Into<CompartmentId>) -> Self {
        self.compartment = Some(compartment.into());
        self
    }

    /// Stamp provenance: who authored it and from which host/device.
    pub fn by(mut self, author: impl Into<UserId>, host: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self.author_host = Some(host.into());
        self
    }

    /// Mark the memory as a planned/announced intent rather than committed.
    pub fn planned(mut self) -> Self {
        self.status = MemoryStatus::Planned;
        self
    }

    pub fn with_evidence(mut self, evidence: Vec<String>) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn volatile(mut self, volatile: bool) -> Self {
        self.volatile = volatile;
        self
    }

    /// Reinforce the trace: bump the recurrence count and nudge confidence up
    /// toward 1.0 (diminishing returns), stamping the update time.
    pub fn reinforce(&mut self, now: DateTime<Utc>) {
        self.reinforcement = self.reinforcement.saturating_add(1);
        self.confidence = (self.confidence + (1.0 - self.confidence) * 0.25).clamp(0.0, 1.0);
        self.updated_at = now;
    }

    /// Record that this trace graduated into the umbra as `expert`.
    pub fn mark_consolidated(&mut self, expert: ExpertId, now: DateTime<Utc>) {
        self.consolidated_expert = Some(expert);
        self.updated_at = now;
    }

    /// Forget this trace as a **tombstone**: mark it deleted at `now` (also
    /// stamping `updated_at`, so the deletion is the trace's newest version and
    /// wins under last-write-wins). The row is kept, not dropped, so the deletion
    /// propagates and routes rather than resurfacing from another replica.
    pub fn soft_delete(&mut self, now: DateTime<Utc>) {
        self.deleted_at = Some(now);
        self.updated_at = now;
    }

    /// `true` if this trace is a tombstone (forgotten).
    pub fn is_deleted(&self) -> bool {
        self.deleted_at.is_some()
    }

    /// Whether the trace has already hardened into the umbra.
    pub fn is_consolidated(&self) -> bool {
        self.consolidated_expert.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reinforce_raises_confidence_and_count() {
        let now = Utc::now();
        let mut m = Memory::new(
            "memory:a",
            "ws:1",
            MemoryNetwork::World,
            "use deno",
            0.4,
            now,
        );
        assert_eq!(m.reinforcement, 0);
        m.reinforce(now);
        assert_eq!(m.reinforcement, 1);
        assert!(m.confidence > 0.4 && m.confidence < 1.0);
    }

    #[test]
    fn network_and_status_as_str() {
        assert_eq!(MemoryNetwork::World.as_str(), "world");
        assert_eq!(MemoryNetwork::Bank.as_str(), "bank");
        assert_eq!(MemoryNetwork::Opinion.as_str(), "opinion");
        assert_eq!(MemoryStatus::Committed.as_str(), "committed");
        assert_eq!(MemoryStatus::Planned.as_str(), "planned");
    }

    #[test]
    fn builder_chain_sets_every_field() {
        let now = Utc::now();
        let m = Memory::new("memory:b", "ws:1", MemoryNetwork::Bank, "x", 1.5, now)
            .with_embedding(vec![0.1, 0.2])
            .in_compartment("comp:1")
            .by("user:a", "host-1")
            .with_evidence(vec!["src".into()])
            .volatile(true)
            .planned();
        assert_eq!(m.confidence, 1.0, "confidence clamps to [0,1]");
        assert_eq!(m.embedding.as_deref(), Some(&[0.1f32, 0.2][..]));
        assert_eq!(m.compartment.as_ref().unwrap().as_str(), "comp:1");
        assert_eq!(m.author.as_ref().unwrap().as_str(), "user:a");
        assert_eq!(m.author_host.as_deref(), Some("host-1"));
        assert_eq!(m.evidence, vec!["src".to_string()]);
        assert!(m.volatile);
        assert_eq!(m.status, MemoryStatus::Planned);
    }

    #[test]
    fn soft_delete_sets_the_tombstone() {
        let now = Utc::now();
        let mut m = Memory::new("memory:c", "ws:1", MemoryNetwork::World, "x", 0.5, now);
        assert!(!m.is_deleted());
        m.soft_delete(now);
        assert!(m.is_deleted());
        assert_eq!(m.deleted_at, Some(now));
        assert_eq!(m.updated_at, now);
    }

    #[test]
    fn memory_edge_new_carries_its_fields() {
        let now = Utc::now();
        let e = MemoryEdge::new(
            "ws:1",
            "memory:a",
            "memory:b",
            EdgeType::Supersedes,
            0.9,
            now,
        );
        assert_eq!(e.from_id.as_str(), "memory:a");
        assert_eq!(e.to_id.as_str(), "memory:b");
        assert_eq!(e.edge_type, EdgeType::Supersedes);
        assert_eq!(e.edge_type.as_str(), "supersedes");
    }

    #[test]
    fn consolidation_link_round_trips() {
        let now = Utc::now();
        let mut m = Memory::new("memory:a", "ws:1", MemoryNetwork::World, "x", 1.0, now);
        assert!(!m.is_consolidated());
        m.mark_consolidated(ExpertId::new("expert:consolidated-pm"), now);
        assert!(m.is_consolidated());
        assert_eq!(
            m.consolidated_expert,
            Some(ExpertId::new("expert:consolidated-pm"))
        );
    }

    #[test]
    fn network_serializes_lowercase() {
        let json = serde_json::to_string(&MemoryNetwork::Opinion).unwrap();
        assert_eq!(json, "\"opinion\"");
    }
}
