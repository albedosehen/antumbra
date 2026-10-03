//! The evidence graph (ADR-0019 section 2): which services depend on which,
//! held as claims with evidence rather than as facts a parser extracted.
//!
//! An edge is an ordinary memory. Its evidence says what depends on what and
//! by which source: **declared** (a manifest names the other service's
//! package), **observed** (telemetry saw the traffic), **learned** (the two
//! change together), or **claimed** (a model read the code and said so). The
//! source sets how much the edge is believed to start with and how fast that
//! belief fades when nobody sees the edge again; seeing it again reinforces
//! the same memory, because the memory's id is derived from the edge and its
//! source. The graph so stays current without a job that re-extracts it.
//!
//! Blast radius is a weighted walk over the edges ([`blast_radius`]) that
//! returns each path with the evidence of every edge on it, so an answer can be
//! checked rather than taken on faith. Several sources for one edge combine as
//! independent evidence, so a model's claim alone stays under the default
//! floor and rises above it only when a declaration or an observation
//! corroborates it, as the ADR asks.

use std::collections::{BTreeMap, HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::TenantId;
use crate::memory::Memory;
use crate::provenance::{normalize_repo, GitProvenance};

const EDGE: &str = "dep:";
const SOURCE: &str = "dep-source:";
const DETAIL: &str = "dep-detail:";
const ARROW: &str = " -> ";

/// Where a dependency claim comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// A manifest, lockfile, client configuration or infrastructure definition
    /// names the other service.
    Declared,
    /// Telemetry over a window saw one call the other.
    Observed,
    /// The two change together, deploy together, or share incidents.
    Learned,
    /// A model read a service and said so. Believed little until corroborated.
    Claimed,
}

impl Source {
    pub const ALL: [Source; 4] = [
        Source::Declared,
        Source::Observed,
        Source::Learned,
        Source::Claimed,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Source::Declared => "declared",
            Source::Observed => "observed",
            Source::Learned => "learned",
            Source::Claimed => "claimed",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Source::ALL.into_iter().find(|v| v.as_str() == s.trim())
    }

    /// The evidence entry that marks an edge as this source's, for a store
    /// asked for one source's edges.
    pub fn evidence_entry(self) -> String {
        format!("{SOURCE}{}", self.as_str())
    }

    /// How much an edge from this source is believed when first recorded.
    pub fn confidence(self) -> f32 {
        match self {
            Source::Declared => 0.9,
            Source::Observed => 0.85,
            Source::Learned => 0.6,
            Source::Claimed => 0.3,
        }
    }

    /// Days for an edge nobody has seen again to lose half its weight. An
    /// observation is about a window and goes stale fastest; a declaration
    /// holds while the file says so, and is re-declared whenever it is read.
    pub fn half_life_days(self) -> f64 {
        match self {
            Source::Declared => 180.0,
            Source::Observed => 30.0,
            Source::Learned => 90.0,
            Source::Claimed => 60.0,
        }
    }
}

/// The evidence entries that mark a memory as an edge, one per source. Every
/// edge carries exactly one, so a store asked for any of these returns the
/// edges and nothing else.
pub fn source_entries() -> Vec<String> {
    Source::ALL.iter().map(|s| s.evidence_entry()).collect()
}

/// One claim that `from` depends on `to`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    pub from: String,
    pub to: String,
    pub source: Source,
    /// What the evidence is, in a line: "package.json names @acme/orders".
    pub detail: String,
}

impl Claim {
    /// A claim between two services, their names normalized the way repository
    /// slugs are, so two spellings of one service never split it in two.
    pub fn new(from: &str, to: &str, source: Source, detail: &str) -> Self {
        Self {
            from: normalize_repo(from),
            to: normalize_repo(to),
            source,
            detail: detail.trim().to_string(),
        }
    }

    /// The memory this claim is held in, in `tenant`. Derived from the
    /// workspace, the edge and the source, so recording the same claim again
    /// lands on the same memory. The workspace is part of it because a memory
    /// id is one key across every tenant: two workspaces recording the same
    /// edge must not land on one row.
    pub fn memory_id(&self, tenant: &TenantId) -> String {
        let slug = |s: &str| -> String {
            s.chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
                .collect()
        };
        format!(
            "memory:dep-{}-{}--{}-{}",
            slug(tenant.as_str()),
            slug(&self.from),
            slug(&self.to),
            self.source.as_str()
        )
    }

    /// The memory's text, for recall to find and a person to read.
    pub fn content(&self) -> String {
        let mut text = format!(
            "{} depends on {} ({} dependency)",
            self.from,
            self.to,
            self.source.as_str()
        );
        if !self.detail.is_empty() {
            text.push_str(": ");
            text.push_str(&self.detail);
        }
        text.push('.');
        text
    }

    /// The memory's evidence: the edge, its source, the detail, and the git
    /// anchor of the file that declares it when there is one.
    pub fn evidence(&self, anchor: Option<&GitProvenance>) -> Vec<String> {
        let mut out = vec![
            format!("{EDGE}{}{ARROW}{}", self.from, self.to),
            format!("{SOURCE}{}", self.source.as_str()),
        ];
        if !self.detail.is_empty() {
            out.push(format!("{DETAIL}{}", self.detail));
        }
        if let Some(a) = anchor {
            out.push(a.to_evidence());
        }
        out
    }
}

/// An edge as the graph reads it back from a memory.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub source: Source,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// The memory's confidence: the source's start, moved by reinforcement.
    pub confidence: f32,
    pub reinforcement: u32,
    /// When it was last recorded or reinforced.
    pub last_seen: DateTime<Utc>,
    pub memory_id: String,
    /// The file that declares it, when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<GitProvenance>,
}

impl Edge {
    /// Read a memory. `None` when it holds no dependency claim.
    pub fn of(m: &Memory) -> Option<Self> {
        let (from, to) = m
            .evidence
            .iter()
            .find_map(|e| e.strip_prefix(EDGE)?.split_once(ARROW))?;
        let source = m
            .evidence
            .iter()
            .find_map(|e| Source::parse(e.strip_prefix(SOURCE)?))?;
        let detail = m
            .evidence
            .iter()
            .find_map(|e| e.strip_prefix(DETAIL))
            .map(str::to_string);
        Some(Self {
            from: normalize_repo(from),
            to: normalize_repo(to),
            source,
            detail,
            confidence: m.confidence,
            reinforcement: m.reinforcement,
            last_seen: m.updated_at,
            memory_id: m.id.as_str().to_string(),
            anchor: GitProvenance::from_evidence(&m.evidence),
        })
    }

    /// How much it is believed now: its confidence, halved for every half-life
    /// of its source since it was last seen.
    pub fn weight(&self, now: DateTime<Utc>) -> f32 {
        let days = (now - self.last_seen).num_seconds().max(0) as f64 / 86_400.0;
        let fade = 0.5f64.powf(days / self.source.half_life_days());
        (f64::from(self.confidence.clamp(0.0, 1.0)) * fade) as f32
    }
}

/// Which way a walk goes from the service it starts at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    /// Who depends on it: what breaks if it does. Blast radius proper.
    Dependents,
    /// What it depends on: what it needs to keep working.
    Dependencies,
}

/// One hop of a path: a pair of services and every edge between them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Hop {
    pub from: String,
    pub to: String,
    /// The pair's combined weight: its edges as independent evidence,
    /// `1 - product(1 - weight)`.
    pub weight: f32,
    pub edges: Vec<Edge>,
}

/// A service a walk reached, by the strongest path to it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Reached {
    pub service: String,
    /// Hops from the start.
    pub depth: u32,
    /// The path's weight: the product of its hops' weights.
    pub weight: f32,
    pub path: Vec<Hop>,
}

/// Each pair's edges and combined weight, keyed by the pair.
fn pairs(edges: &[Edge], now: DateTime<Utc>) -> BTreeMap<(String, String), Hop> {
    let mut pairs: BTreeMap<(String, String), Hop> = BTreeMap::new();
    for e in edges {
        let hop = pairs
            .entry((e.from.clone(), e.to.clone()))
            .or_insert_with(|| Hop {
                from: e.from.clone(),
                to: e.to.clone(),
                weight: 0.0,
                edges: Vec::new(),
            });
        hop.edges.push(e.clone());
    }
    for hop in pairs.values_mut() {
        let doubt: f32 = hop.edges.iter().map(|e| 1.0 - e.weight(now)).product();
        hop.weight = 1.0 - doubt;
        hop.edges
            .sort_by(|a, b| b.weight(now).total_cmp(&a.weight(now)));
    }
    pairs
}

/// Walk the graph from `start`, up to `max_depth` hops, through pairs whose
/// combined weight is at least `min_weight`, keeping for each service the
/// strongest path to it. Strongest first, then nearest, then by name.
pub fn blast_radius(
    edges: &[Edge],
    start: &str,
    direction: Direction,
    max_depth: u32,
    min_weight: f32,
    now: DateTime<Utc>,
) -> Vec<Reached> {
    let start = normalize_repo(start);
    let mut next: HashMap<String, Vec<Hop>> = HashMap::new();
    for ((from, to), hop) in pairs(edges, now) {
        if hop.weight < min_weight {
            continue;
        }
        let key = match direction {
            Direction::Dependents => to,
            Direction::Dependencies => from,
        };
        next.entry(key).or_default().push(hop);
    }

    let mut best: HashMap<String, Reached> = HashMap::new();
    let mut queue: VecDeque<(String, u32, f32, Vec<Hop>)> = VecDeque::new();
    queue.push_back((start.clone(), 0, 1.0, Vec::new()));
    while let Some((at, depth, weight, path)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        for hop in next.get(&at).into_iter().flatten() {
            let service = match direction {
                Direction::Dependents => &hop.from,
                Direction::Dependencies => &hop.to,
            };
            if *service == start || path.iter().any(|h| h.from == *service || h.to == *service) {
                continue; // no cycles back through the start or the path
            }
            let w = weight * hop.weight;
            if best.get(service).is_some_and(|r| r.weight >= w) {
                continue;
            }
            let mut through = path.clone();
            through.push(hop.clone());
            best.insert(
                service.clone(),
                Reached {
                    service: service.clone(),
                    depth: depth + 1,
                    weight: w,
                    path: through.clone(),
                },
            );
            queue.push_back((service.clone(), depth + 1, w, through));
        }
    }
    let mut reached: Vec<Reached> = best.into_values().collect();
    reached.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then(a.depth.cmp(&b.depth))
            .then(a.service.cmp(&b.service))
    });
    reached
}

#[cfg(test)]
mod tests;
