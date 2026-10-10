//! The MCP tool parameter and view types (the wire shapes of every tool), plus
//! the constants the tool bodies read. Kept beside `server.rs` so the tool impl
//! reads as behavior, not as a wall of DTOs; `pub(super)` because nothing outside
//! the server module speaks these shapes.

use super::*;

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct StoreParams {
    /// The text to remember.
    pub(super) content: String,
    /// Network: `world` (facts), `bank` (experiences), `opinion` (judgments).
    #[serde(default = "default_network")]
    pub(super) network: String,
    /// Initial confidence in `[0,1]` (default 0.6).
    pub(super) confidence: Option<f32>,
    /// Provenance sources for the memory.
    pub(super) evidence: Option<Vec<String>>,
    /// `true` if the fact changes over time (kept in store, never consolidated).
    pub(super) volatile: Option<bool>,
    /// The compartment to store into. Omit to use this session's default space.
    pub(super) compartment: Option<String>,
    /// Where this memory was learned, for memories about code: the repository,
    /// commit, and branch (and the file, when it is about one file). Stored as a
    /// `git:` evidence entry so a later recall can judge whether it still applies
    /// (the session-start hook checks the commit against HEAD and whether the
    /// branch still exists). Omit for memories that are not about code.
    pub(super) provenance: Option<ProvenanceParams>,
}

/// A git anchor as the caller states it; becomes one `git:` evidence entry.
#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ProvenanceParams {
    /// Repository slug, `host/org/name` (for example `github.com/oneiriq/antumbra`).
    pub(super) repo: String,
    /// The commit (7 to 40 hex digits) the memory was learned at.
    pub(super) commit: String,
    /// The branch checked out at the time.
    pub(super) branch: Option<String>,
    /// The file the memory is about, when it is about one file.
    pub(super) path: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct StoredOut {
    pub(super) id: String,
    /// Compartment ids the antumbra auto-created from the inbox on this write
    /// (only when the autonomous propose trigger is enabled and fired).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) auto_proposed: Vec<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RecallParams {
    /// What to recall, embedded and matched by semantic similarity.
    pub(super) query: String,
    /// How many to return (default 5).
    pub(super) top_k: Option<u32>,
    /// Optional network filter (`world`/`bank`/`opinion`).
    pub(super) network: Option<String>,
    /// The caller's current repository slug (`host/org/name`). With it, every
    /// result carries a `scope` against this context, and results from another
    /// branch or repository are demoted below in-scope ones (never hidden).
    pub(super) repo: Option<String>,
    /// The caller's checked-out branch, to scope results by branch as well.
    pub(super) branch: Option<String>,
    /// Lower (or raise) the relevance floor for this call, as a probability in
    /// `[0, 1]`. Only meaningful where a typed decider is configured; without
    /// one no floor runs and every recalled row is returned.
    ///
    /// The floor is a default rather than a rule, because "the best of a bad
    /// lot" is occasionally what a caller wants. What it may not be is the only
    /// option.
    pub(super) floor: Option<f32>,
    /// Return each memory's whole `content` instead of the bounded prefix.
    /// Default `false`: a recall is a survey, and a survey that spends the
    /// context window cannot be followed by the work it was for. Ask for `true`
    /// once a row is known to be the one that matters.
    pub(super) full: Option<bool>,
    /// Only memories written from this machine, by host name (as `devices`
    /// lists them; either case). Drawn from a wider pool than `top_k`, but
    /// still a filter on what recall found: a machine whose memories are few
    /// can return fewer than `top_k`, and `list_memories` with `host` reads
    /// them all.
    pub(super) host: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ListParams {
    /// Optional network filter (`world`/`bank`/`opinion`).
    pub(super) network: Option<String>,
    /// Only memories written from this machine, by host name (as `devices`
    /// lists them; either case): `{host: "windows", limit: 5}` is what that
    /// machine wrote last. A memory written through a hosted server before it
    /// read its client's name carries the server's name instead.
    pub(super) host: Option<String>,
    /// Return one page of at most this many (up to 200), the most recently
    /// updated first. Omitted: every memory, in no set order.
    pub(super) limit: Option<u32>,
    /// With `limit`, how many of the most recently updated to skip: the
    /// previous page's offset plus its length. Default 0.
    pub(super) offset: Option<u32>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct IdParams {
    /// The memory's `id`, as store_memory, recall_memories and list_memories
    /// return it (`memory:...`). Passed as `id` it is accepted too: agents copy
    /// the field name from the results, and refusing it cost a retry.
    #[serde(alias = "id")]
    pub(super) memory_id: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct MemoryView {
    pub(super) id: String,
    /// The memory's text, cut to a 900-character prefix unless the call asked
    /// for `full`. `content_chars` is always the STORED length and `truncated`
    /// says whether this is a prefix, so a reader never has to infer the cut.
    ///
    /// Deliberately carries no `maxLength`: the bound is a default a caller may
    /// lift, and a schema asserting 900 would be false on every `full: true`
    /// response. The bound is documented here instead, which is what a consumer
    /// reading the schema actually needs.
    pub(super) content: String,
    pub(super) network: String,
    pub(super) confidence: f32,
    pub(super) reinforcement: u32,
    /// The machine it was written from, by host name: the one its client
    /// named, else the server's own. A memory written through a hosted server
    /// before the server read its client's name carries the server's name,
    /// which says which hub took it and not which device sent it. Absent on a
    /// memory written before writes were stamped at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) author_host: Option<String>,
    /// The git anchor parsed from the memory's evidence, when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) provenance: Option<ProvenanceView>,
    /// How the anchor relates to the caller's `repo`/`branch` context
    /// (`in_scope`, `other_branch`, `other_repo`, `orphaned`, `unknown`); only
    /// when recall was given a context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) scope: Option<String>,
    /// When the branch this memory was learned on was deleted (recorded by the
    /// GitHub integration on the delete event), if it was and nothing has
    /// re-anchored the memory since.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) orphaned_at: Option<String>,
    /// The dense cosine between the query and this memory, in `[-1, 1]`, on a
    /// recall that had a query. Absent when the memory carries no embedding, and
    /// on the paths that never had a query.
    ///
    /// DO NOT READ THIS AS RELEVANCE, and do not threshold on it. It is not the
    /// ordering key: results are ranked by fusion over a dense and a lexical leg
    /// (and a cross-encoder where one is configured), so the first row routinely
    /// carries the lowest number here. Worse, it tracks a memory's LENGTH more
    /// than its topic -- a 66-character row scores 0.774 against a query about
    /// banana bread, while a 1058-character row scores 0.224 against a query
    /// about its own contents. Measured on a 5,538-memory store, which is why
    /// recall's relevance floor is judged by a trained head and not built on
    /// this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) similarity: Option<f32>,
    /// How long `content` is in the store, in characters. Always present, so a
    /// caller can tell a short memory from a long one it is seeing the front of.
    pub(super) content_chars: u32,
    /// Present, and `true`, exactly when `content` above is a prefix of the
    /// stored text. Absent when the row is whole. Re-request with `full: true`
    /// to get the rest.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) truncated: bool,
    /// When the memory was last written: stored, reinforced, or penalized. With
    /// `reinforcement`, this is what lets a memory serve as a counter that also
    /// says when it last counted, which is how skill usage is kept.
    pub(super) updated_at: String,
}

/// A memory's git anchor as returned to a caller.
#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ProvenanceView {
    pub(super) repo: String,
    pub(super) commit: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) branch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) path: Option<String>,
}

impl From<&GitProvenance> for ProvenanceView {
    fn from(p: &GitProvenance) -> Self {
        Self {
            repo: p.repo.clone(),
            commit: p.commit.clone(),
            branch: p.branch.clone(),
            path: p.path.clone(),
        }
    }
}

impl From<&Memory> for MemoryView {
    fn from(m: &Memory) -> Self {
        Self {
            id: m.id.as_str().to_string(),
            content: m.content.clone(),
            network: m.network.as_str().to_string(),
            confidence: m.confidence,
            reinforcement: m.reinforcement,
            author_host: m.author_host.clone(),
            provenance: GitProvenance::from_evidence(&m.evidence)
                .as_ref()
                .map(ProvenanceView::from),
            scope: None,
            orphaned_at: orphan_of(&m.evidence).map(|o| o.at.to_rfc3339()),
            updated_at: m.updated_at.to_rfc3339(),
            similarity: None,
            content_chars: m.content.chars().count() as u32,
            truncated: false,
        }
    }
}

/// How much of a memory's `content` a recall returns before it is cut.
///
/// Recall returns `top_k` rows and a memory runs to thousands of characters, so
/// the default answer was unbounded in the one place an agent cannot afford it:
/// five rows of a thousand characters is most of a session-start budget spent
/// before the session has begun. 900 is what the shipped hooks already cut at,
/// so this makes the surface agree with its own clients rather than inventing a
/// second number, and five of them sit comfortably inside the 10,000-character
/// limit a hook's context has.
pub(super) const RECALL_CONTENT_CHARS: usize = 900;

impl MemoryView {
    /// The view of `m` judged against the caller's git context.
    pub(super) fn scoped(m: &Memory, ctx: &GitContext) -> Self {
        let mut view = Self::from(m);
        view.scope = Some(scope_of_evidence(&m.evidence, ctx).as_str().to_string());
        view
    }

    /// Cut `content` to a prefix of `max` characters unless `full`.
    ///
    /// A prefix, never a summary: no model belongs in the recall path, and two
    /// identical recalls must return identical text. The cut counts CHARACTERS
    /// rather than bytes, so it can never land inside a multi-byte codepoint.
    pub(super) fn bounded(mut self, full: bool, max: usize) -> Self {
        if full {
            return self;
        }
        // `content_chars` is the stored length and is set before any cut, so it
        // stays true whichever branch runs.
        if self.content.chars().count() > max {
            let cut: String = self.content.chars().take(max).collect();
            self.content = cut;
            self.truncated = true;
        }
        self
    }
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct MemoriesOut {
    /// At most `top_k` for a recall, at most `limit` (up to 200) for a paged
    /// list, and every memory for a list that was not paged.
    pub(super) memories: Vec<MemoryView>,
    /// Present, and `true`, when a relevance floor ran and NOTHING cleared it.
    /// Absent otherwise, including when no floor ran at all.
    ///
    /// This is the difference between "nothing here answers you" and "here are
    /// five weak rows, you decide", which an empty list alone cannot express and
    /// a caller would otherwise have to infer from scores it should not be
    /// reading. An agent that cannot tell those apart re-runs the query with
    /// different flags to find out, which costs a turn and more context than the
    /// answer would have.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) nothing_cleared_the_floor: bool,
    /// Present, and `true`, when this is a page of `list_memories` and another
    /// follows it. Absent otherwise: the last page, or a call that was not
    /// paged.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) more: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct IngestDocumentParams {
    /// The document's title (used to group and name its chunks).
    pub(super) title: String,
    /// The full document text to ingest.
    pub(super) content: String,
    /// Where the document came from (path / url / note).
    #[serde(default)]
    pub(super) source: Option<String>,
    /// The git anchor the document describes, folded into every chunk's
    /// source so a recalled chunk names the commit (see store_memory).
    #[serde(default)]
    pub(super) provenance: Option<ProvenanceParams>,
    /// The compartment to keep the document in. Then only you and the people you
    /// share that compartment with can recall it. Omit it for the workspace's
    /// shared pool, which every member of the workspace can recall: that is the
    /// right place for reference material, and the wrong place for anything
    /// private.
    #[serde(default)]
    pub(super) compartment: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct IngestedOut {
    pub(super) title: String,
    pub(super) chunks: u32,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RecallDocumentsParams {
    /// What to recall from the ingested documents.
    pub(super) query: String,
    /// How many chunks to return (default 5).
    pub(super) top_k: Option<u32>,
}

/// A document chunk as returned to a caller (without its embedding).
#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct DocumentChunkView {
    pub(super) title: String,
    pub(super) source: Option<String>,
    pub(super) ordinal: u32,
    pub(super) content: String,
    /// The copal file holding the document's original content (the document of
    /// record); omitted when it was ingested without an archive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) copal_file: Option<String>,
    /// The content digest copal reported for that archived original.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) copal_digest: Option<String>,
    /// The compartment the document is kept in; omitted for the shared pool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) compartment: Option<String>,
}

impl From<&DocumentChunk> for DocumentChunkView {
    fn from(c: &DocumentChunk) -> Self {
        DocumentChunkView {
            title: c.title.clone(),
            source: c.source.clone(),
            ordinal: c.ordinal,
            content: c.content.clone(),
            copal_file: c.copal_file.clone(),
            copal_digest: c.copal_digest.clone(),
            compartment: c.compartment.as_ref().map(|c| c.as_str().to_string()),
        }
    }
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct DocumentChunksOut {
    /// At most `top_k` chunks (default 5), best match first. Empty means
    /// nothing matched: there is no relevance floor on this path, so an empty
    /// answer is not one whose matches were rejected.
    pub(super) chunks: Vec<DocumentChunkView>,
}

/// One expert in the visible population (read-only view, no embedding/adapter).
#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ExpertView {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) generation: u32,
    pub(super) fitness: f32,
    /// True if this is the caller's own private expert (else a shared one).
    pub(super) private: bool,
    /// Where it stands: `active` (routed to), `dormant` (served
    /// only when named), `archived` (kept, not served) or `deleted`.
    pub(super) status: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct PopulationOut {
    /// One row per expert visible to you: the shared population plus your own
    /// private ones. Empty means the population has never been seeded, not that
    /// a filter hid it -- the ACL scopes this list and does not empty it.
    pub(super) experts: Vec<ExpertView>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct StatsOut {
    pub(super) memories: u32,
    pub(super) documents: u32,
    pub(super) experts: u32,
    pub(super) boundaries: u32,
    pub(super) compartments: u32,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ReinforceOut {
    pub(super) found: bool,
    pub(super) reinforcement: u32,
    pub(super) confidence: f32,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ForgetOut {
    pub(super) forgotten: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RelateParams {
    pub(super) from_id: String,
    pub(super) to_id: String,
    /// `references` / `supersedes` / `contradicts` / `follows` / `caused`.
    #[serde(default = "default_edge_type")]
    pub(super) edge_type: String,
    /// Edge strength (default 1.0).
    pub(super) weight: Option<f32>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RelateOut {
    pub(super) related: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct NeighborsParams {
    /// The memory's `id`, as recall_memories returns it (`id` is accepted too).
    #[serde(alias = "id")]
    pub(super) memory_id: String,
    /// Optional edge-type filter.
    pub(super) edge_type: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct NeighborView {
    pub(super) edge_type: String,
    pub(super) weight: f32,
    pub(super) memory: MemoryView,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct NeighborsOut {
    /// Every relation on the memory, or every one of the requested `edge_type`.
    /// Empty means the memory has no relations at all when `edge_type` was
    /// omitted, and none of that type when it was given -- so an empty answer
    /// to a filtered call is worth retrying unfiltered before concluding the
    /// memory is isolated.
    pub(super) neighbors: Vec<NeighborView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RouteParams {
    /// The task to route across the shared expert population.
    pub(super) task: String,
    /// How many candidate experts to return (default 3).
    pub(super) top_k: Option<u32>,
}

/// Minimum cosine similarity for one of the user's *private* experts to be
/// offered as a route candidate (a heuristic floor; private experts are not in
/// the shared learned router, so they are matched directly by centroid; a
/// per-private-expert learned boundary is the eventual refinement).
pub(super) const PRIVATE_ROUTE_FLOOR: f32 = 0.3;

/// Inhibition radius for the legacy absolute-scope boundary path; mirrors
/// `antumbra_gate::GateConfig::default().inhibition_radius`. Correction-derived
/// boundaries use the relative C/C' margin, which ignores the radius, so this
/// only bites a hypothetical absolute boundary.
pub(super) const INHIBITION_RADIUS: f32 = 0.5;

/// Escalate (route to no one) when a boundary inhibits the task above this,
/// mirroring the CLI learned-route gate. A task inside a known failure scope is
/// handed up rather than served by an expert that provably fails there.
pub(super) const BOUNDARY_ESCALATE_THRESHOLD: f32 = 0.5;

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RouteHit {
    pub(super) expert_id: String,
    pub(super) probability: f32,
    /// `true` if this is one of *your* private experts (consolidated from your
    /// compartment), matched by centroid; `false` for a shared expert.
    pub(super) private: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RouteOut {
    /// Whether the population covers this task (vs out-of-distribution).
    pub(super) covered: bool,
    /// `true` when no expert covers it: defer to the generalist.
    pub(super) escalate: bool,
    /// Why it escalated, when it did: a failure boundary, no router yet, or
    /// outside what the population covers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reason: Option<String>,
    pub(super) routes: Vec<RouteHit>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct AnswerParams {
    /// The task to route and answer through the covering expert.
    pub(super) task: String,
    /// The repository the task is in (`host/org/name`), so your standing
    /// behaviors for it are composed into the answer. Those that apply
    /// everywhere always are.
    #[serde(default)]
    pub(super) repo: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct AnswerOut {
    /// The generated answer (empty when escalating).
    pub(super) answer: String,
    /// The expert that served it, when one covered the task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) expert_id: Option<String>,
    /// `true` when no expert covered it, or serving is not configured.
    pub(super) escalate: bool,
    /// Why it escalated, when it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) note: Option<String>,
    /// Your standing experts composed into the answer: the behaviors you
    /// accepted, for everywhere and for the repository.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) standing: Vec<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct DocumentView {
    /// The title it was ingested under, which is what identifies it.
    pub(super) title: String,
    /// How many chunks it was cut into.
    pub(super) chunks: u32,
    /// Whether its file of record is archived (copal).
    pub(super) archived: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct DocumentsOut {
    /// One row per document you can see, sorted by title. Empty means none
    /// has been ingested rather than none matched, since this lists rather
    /// than searches.
    pub(super) documents: Vec<DocumentView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct CreateCompartmentParams {
    /// A display name for the new compartment.
    pub(super) name: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct CompartmentView {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) origin: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct CompartmentsOut {
    /// One row per compartment you can see: your own, plus any shared with you.
    /// Empty means none exist rather than none matched, since this lists rather
    /// than searches.
    pub(super) compartments: Vec<CompartmentView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ShareParams {
    pub(super) compartment_id: String,
    /// The user to share with (a user id in this tenant).
    pub(super) grantee: String,
    /// `reference` (recall) or `link` (also connect). Defaults to reference.
    #[serde(default = "default_capability")]
    pub(super) capability: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ShareOut {
    pub(super) shared: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RevokeParams {
    pub(super) compartment_id: String,
    pub(super) grantee: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RevokeOut {
    pub(super) revoked: bool,
}

pub(super) fn default_threshold() -> f32 {
    0.6
}

pub(super) fn default_min_size() -> usize {
    3
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ProposeCompartmentsParams {
    /// Cosine at/above which two memories cluster together (default 0.6).
    #[serde(default = "default_threshold")]
    pub(super) similarity_threshold: f32,
    /// Smallest cluster worth proposing; singletons and pairs are noise (default 3).
    #[serde(default = "default_min_size")]
    pub(super) min_size: usize,
    /// Persist each proposal as an `Origin::Proposed` compartment you own and
    /// move its members into it. Default false (suggest only; reversible by
    /// deleting the compartment).
    #[serde(default)]
    pub(super) apply: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ProposalView {
    /// Heuristic label from the cluster's most central memory; rename on accept.
    pub(super) label: String,
    /// The memory ids grouped into this proposed region.
    pub(super) members: Vec<String>,
    /// Mean cosine of members to the centroid; rank proposals by this.
    pub(super) cohesion: f32,
    /// Set when `apply` was true: the id of the created proposed compartment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) compartment_id: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ProposalsOut {
    /// One row per cluster found among your uncompartmented memories. Empty is
    /// ambiguous on purpose and the two readings want different moves: either
    /// there is nothing uncompartmented to cluster, or nothing clustered at this
    /// `similarity_threshold` and `min_size`, which a lower threshold may fix.
    pub(super) proposals: Vec<ProposalView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct LeaveHandoffParams {
    /// What to leave: the first line is the title it is announced under.
    pub(super) content: String,
    /// The machine it is for, by host name, or `any` (the default) for
    /// whichever of your machines starts a session next.
    pub(super) for_host: Option<String>,
    /// The machine leaving it, by host name. Defaults to the machine this call
    /// came from, as its client names it (`X-Antumbra-Host`), else the
    /// server's own host; a client that sends no name should pass its own (the
    /// session-start block names it).
    pub(super) from_host: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct LeftHandoffOut {
    pub(super) id: String,
    /// The machine it is for, normalized (trimmed, lowercased), or `any`.
    pub(super) for_host: String,
    /// Whether `for_host` is one of your registered devices (or `any`). False
    /// is not an error: a machine that has not registered still receives it
    /// when a session there reports that host. It is how a typo shows.
    pub(super) registered_device: bool,
    /// Your registered devices' host names, at most a few dozen.
    pub(super) devices: Vec<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct HandoffsParams {
    /// The machine asking. Defaults to the machine this call came from, as
    /// its client names it, else the server's own host; a client that sends
    /// no name should pass its own.
    pub(super) host: Option<String>,
    /// Also list handoffs already marked done. Default false.
    pub(super) include_done: Option<bool>,
    /// Return each handoff's whole text instead of a 900-character prefix.
    pub(super) full: Option<bool>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct HandoffView {
    pub(super) id: String,
    /// Its first line, as it is announced, at most 80 characters.
    pub(super) title: String,
    /// The text, a 900-character prefix unless `full` was asked for.
    pub(super) content: String,
    pub(super) content_chars: u32,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) truncated: bool,
    /// The machine that left it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) from_host: Option<String>,
    /// The machine it is for, or `any`.
    pub(super) for_host: String,
    pub(super) left_at: String,
    /// When it was marked done and by which machine; absent while it waits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) done_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) done_by: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct HandoffsOut {
    /// The machine the list is for, normalized.
    pub(super) host: String,
    /// Newest first, at most 50. Without `include_done`, only the ones still
    /// waiting; empty when nothing waits.
    pub(super) handoffs: Vec<HandoffView>,
    /// The lines a session-start block shows for what waits: a count and one
    /// line per handoff. Absent when nothing waits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) announcement: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct CompleteHandoffParams {
    /// The handoff's `id`, as handoffs lists it (`id` is accepted too).
    #[serde(alias = "id")]
    pub(super) handoff_id: String,
    /// The machine that dealt with it. Defaults to the machine this call came
    /// from, as its client names it, else the server's own host.
    pub(super) host: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct CompletedHandoffOut {
    /// False when no handoff of yours has that id.
    pub(super) found: bool,
    /// True when it had already been marked done; it is left as it was.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) already_done: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RecordDependencyParams {
    /// The service that depends, as a repository slug (`host/org/name`).
    pub(super) from: String,
    /// The service depended on, as a repository slug.
    pub(super) to: String,
    /// `declared`, `observed`, `learned` or `claimed`.
    pub(super) source: String,
    /// What the evidence is, in a line: "package.json names @acme/orders".
    pub(super) detail: Option<String>,
    /// The file that declares it, at the commit it was read.
    pub(super) provenance: Option<ProvenanceParams>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RecordedDependencyOut {
    pub(super) id: String,
    /// False when the edge was already recorded from this source and was
    /// reinforced instead.
    pub(super) created: bool,
    pub(super) confidence: f32,
    pub(super) reinforcement: u32,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct BlastRadiusParams {
    /// The service to start from, as a repository slug.
    pub(super) service: String,
    /// `dependents` (the default) or `dependencies`.
    pub(super) direction: Option<String>,
    /// Hops to walk, default 3, at most 6.
    pub(super) depth: Option<u32>,
    /// The weakest pair walked, in `[0, 1]`, default 0.4.
    pub(super) min_weight: Option<f32>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct EdgeView {
    /// `declared`, `observed`, `learned` or `claimed`.
    pub(super) source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) detail: Option<String>,
    pub(super) confidence: f32,
    /// The confidence faded by the time since it was last seen.
    pub(super) weight: f32,
    pub(super) reinforcement: u32,
    pub(super) last_seen: String,
    pub(super) memory_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) anchor: Option<ProvenanceView>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct HopView {
    pub(super) from: String,
    pub(super) to: String,
    /// The pair's sources combined as independent evidence.
    pub(super) weight: f32,
    /// Every recorded edge between the two, strongest first, at most 4 (one
    /// per source).
    pub(super) evidence: Vec<EdgeView>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ReachedView {
    pub(super) service: String,
    pub(super) depth: u32,
    /// The path's weight: the product of its hops' weights.
    pub(super) weight: f32,
    /// The strongest path to it, from the start outward, at most 6 hops.
    pub(super) path: Vec<HopView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ListDependenciesParams {
    /// Only the pairs this service (a repository slug) is on either end of.
    pub(super) service: Option<String>,
    /// Pairs to return, default 100, at most 200.
    pub(super) limit: Option<u32>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct DependenciesOut {
    /// Strongest first, at most 200.
    pub(super) pairs: Vec<HopView>,
    /// How many pairs there are in all, listed or not.
    pub(super) total: u32,
    /// Every service on either end of a pair, by name, at most 400.
    pub(super) services: Vec<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BlastRadiusOut {
    /// The start, normalized.
    pub(super) service: String,
    pub(super) direction: String,
    /// How many dependency edges the workspace holds, walked or not.
    pub(super) edges: u32,
    /// Strongest first, at most 100. Empty when nothing reaches the start
    /// above `min_weight`, including when the service has no edges at all.
    pub(super) reached: Vec<ReachedView>,
}
