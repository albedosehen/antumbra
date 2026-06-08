//! Penumbra clustering: the antumbra *proposes* compartments by grouping a
//! user's uncompartmented memories into competence-coherent regions (the
//! compartments, latent-spaces of memory).
//!
//! This is the "second part of the antumbra": beyond drawing competence
//! boundaries, it surfaces structure in the penumbra so the user can
//! organize, share, and ultimately consolidate a region into a private expert.
//! It is pure analysis over the embeddings already stored on each [`Memory`]
//! (no model, no IO), so the whole proposal pass is deterministic and testable on
//! CPU. The antumbra only *suggests*; the user curates (keep / name / share),
//! exactly as `Origin::Proposed` records.

use crate::expert::cosine_similarity;
use crate::ids::MemoryId;
use crate::memory::Memory;

/// Knobs for [`propose_compartments`].
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Cosine at/above which a memory joins an existing cluster (measured
    /// against the cluster's running centroid).
    pub similarity_threshold: f32,
    /// Smallest cluster worth proposing; singletons and pairs are noise.
    pub min_size: usize,
    /// How many leading words of the medoid's content to use as a label.
    pub label_words: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.6,
            min_size: 3,
            label_words: 5,
        }
    }
}

/// A proposed compartment: a coherent cluster of the user's memories the
/// antumbra suggests grouping. `members` are the memory ids; `label` is a
/// heuristic name derived from the cluster's most central memory (the user
/// renames it); `cohesion` is the mean cosine of members to the centroid, a
/// confidence the user can rank proposals by.
#[derive(Debug, Clone, PartialEq)]
pub struct ProposedCompartment {
    pub label: String,
    pub members: Vec<MemoryId>,
    pub centroid: Vec<f32>,
    pub cohesion: f32,
}

/// Internal accumulator: a centroid maintained as a running mean (sum / count)
/// so adding a member is O(dim), not O(members * dim).
struct Cluster {
    sum: Vec<f32>,
    centroid: Vec<f32>,
    members: Vec<usize>,
}

impl Cluster {
    fn new(emb: &[f32], idx: usize) -> Self {
        Self {
            sum: emb.to_vec(),
            centroid: emb.to_vec(),
            members: vec![idx],
        }
    }

    fn add(&mut self, emb: &[f32], idx: usize) {
        for (s, x) in self.sum.iter_mut().zip(emb) {
            *s += x;
        }
        let n = self.members.len() as f32 + 1.0;
        for (c, s) in self.centroid.iter_mut().zip(&self.sum) {
            *c = s / n;
        }
        self.members.push(idx);
    }
}

/// Cluster a pool of memories into proposed compartments.
///
/// This is policy-free: it clusters every memory in the slice that carries an
/// embedding. **Choosing the pool is the caller's job**: the caller passes the
/// *unorganized* memories (those in the inbox / default compartment, or with no
/// compartment) and leaves deliberately-filed memory out, since the core cannot
/// know which compartment is the inbox. Clustering is a single deterministic
/// pass: candidates are taken in `id` order, and each joins the most-similar
/// existing cluster whose centroid is within `similarity_threshold`, else seeds
/// a new cluster. Clusters below `min_size` are dropped as noise. Proposals are
/// returned best-cohesion first.
pub fn propose_compartments(memories: &[Memory], cfg: &ClusterConfig) -> Vec<ProposedCompartment> {
    // Embedded candidates in a stable id order so the greedy pass (and thus the
    // proposals) is reproducible run to run.
    let mut candidates: Vec<&Memory> = memories.iter().filter(|m| m.embedding.is_some()).collect();
    candidates.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));

    let mut clusters: Vec<Cluster> = Vec::new();
    for (idx, m) in candidates.iter().enumerate() {
        let emb = m.embedding.as_deref().expect("filtered to Some above");
        // Best existing cluster by centroid cosine.
        let best = clusters
            .iter()
            .enumerate()
            .map(|(ci, c)| (ci, cosine_similarity(&c.centroid, emb)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((ci, sim)) if sim >= cfg.similarity_threshold => clusters[ci].add(emb, idx),
            _ => clusters.push(Cluster::new(emb, idx)),
        }
    }

    let mut proposals: Vec<ProposedCompartment> = clusters
        .into_iter()
        // `min_size` is at least 1: a 0 from a request is meaningless (a cluster
        // always has its seed member) and would only emit singleton noise.
        .filter(|c| c.members.len() >= cfg.min_size.max(1))
        .map(|c| {
            // Medoid: the member nearest the centroid. Its content names the
            // region, and its cosine anchors the cohesion average.
            let sims: Vec<f32> = c
                .members
                .iter()
                .map(|&i| {
                    cosine_similarity(
                        &c.centroid,
                        candidates[i].embedding.as_deref().expect("embedded"),
                    )
                })
                .collect();
            let medoid_pos = sims
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(p, _)| p)
                .unwrap_or(0);
            let cohesion = sims.iter().sum::<f32>() / sims.len() as f32;
            let label = label_from(&candidates[c.members[medoid_pos]].content, cfg.label_words);
            let members: Vec<MemoryId> = c
                .members
                .iter()
                .map(|&i| candidates[i].id.clone())
                .collect();
            ProposedCompartment {
                label,
                members,
                centroid: c.centroid,
                cohesion,
            }
        })
        .collect();

    // Best (most cohesive) proposals first; stable label tie-break for determinism.
    proposals.sort_by(|a, b| {
        b.cohesion
            .total_cmp(&a.cohesion)
            .then_with(|| a.label.cmp(&b.label))
    });
    proposals
}

/// Derive a short, slug-like label from a memory's content: the first
/// `max_words` alphanumeric words, lowercased and hyphen-joined. Falls back to a
/// fixed stem when the content has no usable words.
fn label_from(content: &str, max_words: usize) -> String {
    let words: Vec<String> = content
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .take(max_words)
        .collect();
    if words.is_empty() {
        "proposed-region".to_string()
    } else {
        words.join("-")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{CompartmentId, MemoryId, TenantId};
    use crate::memory::{Memory, MemoryNetwork};
    use chrono::Utc;

    fn mem(id: &str, content: &str, emb: Vec<f32>, compartment: Option<&str>) -> Memory {
        let mut m = Memory::new(
            MemoryId::new(id),
            TenantId::new("ws:1"),
            MemoryNetwork::World,
            content,
            0.8,
            Utc::now(),
        );
        m.embedding = Some(emb);
        m.compartment = compartment.map(CompartmentId::new);
        m
    }

    #[test]
    fn groups_two_coherent_regions() {
        // Two tight clusters along orthogonal axes, plus one outlier singleton.
        let memories = vec![
            mem("m:1", "deno run typescript", vec![1.0, 0.0, 0.0], None),
            mem("m:2", "deno test typescript", vec![0.98, 0.05, 0.0], None),
            mem("m:3", "deno bundle module", vec![0.97, 0.0, 0.05], None),
            mem("m:4", "rust cargo build", vec![0.0, 1.0, 0.0], None),
            mem("m:5", "rust cargo clippy", vec![0.0, 0.98, 0.04], None),
            mem("m:6", "rust cargo test", vec![0.05, 0.97, 0.0], None),
            mem("m:7", "unrelated note", vec![0.0, 0.0, 1.0], None),
        ];
        let out = propose_compartments(&memories, &ClusterConfig::default());
        assert_eq!(out.len(), 2, "two regions of >=3, the singleton dropped");
        for p in &out {
            assert_eq!(p.members.len(), 3);
            assert!(p.cohesion > 0.9);
        }
    }

    #[test]
    fn clustering_is_policy_free_over_the_given_pool() {
        // The core clusters whatever it is handed; pool selection (skipping
        // already-filed memory) is the caller's job, not this function's.
        let memories = vec![
            mem("m:1", "a one", vec![1.0, 0.0], Some("comp:x")),
            mem("m:2", "a two", vec![1.0, 0.0], Some("comp:x")),
            mem("m:3", "a three", vec![1.0, 0.0], Some("comp:x")),
        ];
        let out = propose_compartments(&memories, &ClusterConfig::default());
        assert_eq!(out.len(), 1, "core clusters the pool it is given");
    }

    #[test]
    fn drops_clusters_below_min_size() {
        let memories = vec![
            mem("m:1", "lonely pair one", vec![1.0, 0.0], None),
            mem("m:2", "lonely pair two", vec![0.99, 0.0], None),
        ];
        let out = propose_compartments(&memories, &ClusterConfig::default());
        assert!(out.is_empty(), "a pair is below the default min_size of 3");
    }

    #[test]
    fn ignores_memories_without_embeddings() {
        let mut m = Memory::new(
            MemoryId::new("m:1"),
            TenantId::new("ws:1"),
            MemoryNetwork::World,
            "no vector",
            0.8,
            Utc::now(),
        );
        m.embedding = None;
        let out = propose_compartments(&[m], &ClusterConfig::default());
        assert!(out.is_empty());
    }

    #[test]
    fn label_is_a_slug_of_the_medoid() {
        let memories = vec![
            mem("m:1", "Deno Run TypeScript!", vec![1.0, 0.0, 0.0], None),
            mem("m:2", "deno test", vec![0.99, 0.02, 0.0], None),
            mem("m:3", "deno bundle", vec![0.98, 0.0, 0.03], None),
        ];
        let out = propose_compartments(&memories, &ClusterConfig::default());
        assert_eq!(out.len(), 1);
        assert!(
            out[0].label.starts_with("deno"),
            "label slugged from content: {}",
            out[0].label
        );
        assert!(!out[0].label.contains('!'), "punctuation stripped");
    }

    #[test]
    fn proposals_sorted_by_cohesion() {
        let memories = vec![
            // Tight cluster (cohesion ~1.0).
            mem("m:1", "tight a", vec![1.0, 0.0, 0.0], None),
            mem("m:2", "tight b", vec![1.0, 0.0, 0.0], None),
            mem("m:3", "tight c", vec![1.0, 0.0, 0.0], None),
            // Looser cluster along another axis.
            mem("m:4", "loose a", vec![0.0, 1.0, 0.0], None),
            mem("m:5", "loose b", vec![0.0, 0.85, 0.2], None),
            mem("m:6", "loose c", vec![0.0, 0.8, 0.3], None),
        ];
        let out = propose_compartments(&memories, &ClusterConfig::default());
        assert_eq!(out.len(), 2);
        assert!(
            out[0].cohesion >= out[1].cohesion,
            "most cohesive first: {} then {}",
            out[0].cohesion,
            out[1].cohesion
        );
    }
}
