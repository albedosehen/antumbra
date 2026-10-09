//! What a caller can tell the server about the history memories are anchored
//! to. `record_merges` is the GitHub webhook's merge handler for a
//! deployment GitHub cannot reach: the caller, who can ask GitHub, says which
//! branches merged where, and the memories anchored to each move to its merge
//! commit on the base branch. Without it, everything learned on a feature
//! branch reads `other_branch` from the base branch for good once the branch
//! merges, and recall ranks it below memories that were never anchored at all.
//!
//! It runs as the caller, where the webhook runs in owner mode. A webhook
//! delivery is signed by GitHub; this report is signed only by the caller, so
//! it moves only memories the caller could already rewrite.
//!
//! One call carries every merge the caller knows of, because finding the
//! memories a merge moves means reading every memory's anchor: on kuskokwim
//! that read is the cost of the call, and one call a merge read it thirty
//! times for a thirty-merge report (111 seconds).
//!
//! Its own router, joined to the memory tools' in `engine.rs`, because
//! `server.rs` is past the size rule.

use chrono::DateTime;

use antumbra_core::normalize_repo;
use antumbra_github::{reanchor_merged, Merge};

use super::*;

/// One merge, as the caller reports it.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct MergeReport {
    /// The branch that merged.
    pub head_branch: String,
    /// The branch it merged into.
    pub base_branch: String,
    /// The commit the merge produced on the base branch (7 to 40 hex digits).
    pub merge_commit: String,
    /// When it merged, RFC 3339. A memory created after this stays where it is
    /// unless it is anchored at one of `commits`: a branch name reused for new
    /// work after the merge names a different branch. Omit to move every memory
    /// on the head branch.
    #[serde(default)]
    pub merged_at: Option<String>,
    /// The commits the branch carried when it merged. A memory anchored at one
    /// of them moves even when it was stored after `merged_at`, as happens when
    /// a session is still on the branch after it merges. Matched by prefix, so
    /// a short sha on either side is fine.
    #[serde(default)]
    pub commits: Vec<String>,
}

/// Merges in one repository, reported together.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecordMergesParams {
    /// Repository slug, `host/org/name` (for example `github.com/oneiriq/antumbra`).
    pub repo: String,
    /// The merges, oldest first. In that order a branch merged into another
    /// branch is carried on when that branch merges in turn.
    pub merges: Vec<MergeReport>,
}

/// What one reported merge moved.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MergeOutcome {
    pub head_branch: String,
    /// Memories moved to the merge commit on the base branch.
    pub moved: usize,
    /// Memories on the merged branch that you can read but not write (a
    /// compartment shared with you read-only), so they stay where they are.
    pub not_writable: usize,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MergesRecorded {
    /// Memories moved, over every merge.
    pub moved: usize,
    /// Memories left in place because they are not yours to write, over every merge.
    pub not_writable: usize,
    /// Each merge's own counts, in the order reported.
    pub merges: Vec<MergeOutcome>,
}

/// Whether `anchor` names one of `commits`, either written short.
fn carried(commits: &[String], anchor: &str) -> bool {
    let anchor = anchor.to_ascii_lowercase();
    commits.iter().any(|c| {
        let c = c.to_ascii_lowercase();
        c.starts_with(&anchor) || anchor.starts_with(&c)
    })
}

/// A reported merge, checked.
struct Reported {
    merge: Merge,
    merged_at: Option<DateTime<Utc>>,
    commits: Vec<String>,
}

impl Reported {
    /// Whether this merge moves the memory anchored as `a`: its anchor is on the
    /// merged branch, and it was stored before the merge or at one of the
    /// branch's own commits.
    fn moves(&self, a: &memory::Anchored) -> bool {
        self.merge.moves(&a.evidence)
            && (self.merged_at.is_none_or(|at| a.created_at <= at)
                || GitProvenance::from_evidence(&a.evidence)
                    .is_some_and(|anchor| carried(&self.commits, &anchor.commit)))
    }
}

/// The merge `m` describes in `repo`, or why it cannot be one.
fn reported(repo: &str, m: &MergeReport) -> Result<Reported, ErrorData> {
    let repo = normalize_repo(repo);
    let anchor_on = |branch: &str| {
        GitProvenance::new(repo.clone(), m.merge_commit.clone())
            .on_branch(branch)
            .is_valid()
    };
    // A slug is at least `host/name`; the anchor's own check allows a bare word.
    if !repo.contains('/') || !anchor_on(&m.head_branch) || !anchor_on(&m.base_branch) {
        return Err(ErrorData::invalid_params(
            format!(
                "the merge of '{}' needs a repo slug without '@' (host/org/name), a 7-40 hex \
                 digit merge commit, and branches without ':'",
                m.head_branch
            ),
            None,
        ));
    }
    if m.head_branch == m.base_branch {
        return Err(ErrorData::invalid_params(
            format!("'{}' cannot merge into itself", m.head_branch),
            None,
        ));
    }
    let merged_at = m
        .merged_at
        .as_deref()
        .map(|at| {
            DateTime::parse_from_rfc3339(at)
                .map(|t| t.with_timezone(&Utc))
                .map_err(|e| {
                    ErrorData::invalid_params(
                        format!("merged_at '{at}' is not RFC 3339: {e}"),
                        None,
                    )
                })
        })
        .transpose()?;
    Ok(Reported {
        merge: Merge {
            repo,
            head_branch: m.head_branch.clone(),
            base_branch: m.base_branch.clone(),
            merge_commit: m.merge_commit.clone(),
        },
        merged_at,
        commits: m.commits.clone(),
    })
}

#[tool_router(router = provenance_router, vis = "pub(super)")]
impl McpServer {
    /// Move the memories of merged branches onto the branches they merged into.
    #[tool(
        description = "Record that branches merged: memories anchored to each merged branch move to its merge commit on the base branch, so recall from the base branch treats them as its own. What the GitHub webhook does, for a server GitHub cannot reach. Report every merge in one call, oldest first. Safe to repeat."
    )]
    pub(super) async fn record_merges(
        &self,
        Parameters(p): Parameters<RecordMergesParams>,
    ) -> Result<Json<MergesRecorded>, ErrorData> {
        let reports = p
            .merges
            .iter()
            .map(|m| reported(&p.repo, m))
            .collect::<Result<Vec<_>, _>>()?;
        let mut anchors = if reports.is_empty() {
            Vec::new()
        } else {
            memory::list_anchored(&self.store, &self.tenant)
                .await
                .map_err(err)?
        };
        let mut merges = Vec::with_capacity(reports.len());
        for r in &reports {
            let (moved, not_writable) = self.apply_merge(r, &mut anchors).await?;
            merges.push(MergeOutcome {
                head_branch: r.merge.head_branch.clone(),
                moved,
                not_writable,
            });
        }
        Ok(Json(MergesRecorded {
            moved: merges.iter().map(|m| m.moved).sum(),
            not_writable: merges.iter().map(|m| m.not_writable).sum(),
            merges,
        }))
    }
}

impl McpServer {
    /// Move what `r` moves. Which memories that is comes from their anchors
    /// alone, and only those are read in full: the write replaces the row, and
    /// a row written back without its embedding would lose it. `anchors` is kept
    /// current, so a later merge sees where this one put a memory.
    async fn apply_merge(
        &self,
        r: &Reported,
        anchors: &mut [memory::Anchored],
    ) -> Result<(usize, usize), ErrorData> {
        let (mut moved, mut not_writable) = (0, 0);
        for a in anchors.iter_mut().filter(|a| r.moves(a)) {
            let Some(m) = memory::get(&self.store, &self.tenant, &a.id)
                .await
                .map_err(err)?
            else {
                continue;
            };
            for changed in reanchor_merged(vec![m], &r.merge, Utc::now()) {
                memory::upsert(&self.store, &changed).await.map_err(err)?;
                // A write into a compartment the caller may only read succeeds
                // and persists nothing, so count what actually landed.
                let landed = memory::get(&self.store, &self.tenant, &changed.id)
                    .await
                    .map_err(err)?
                    .is_some_and(|stored| stored.evidence == changed.evidence);
                if landed {
                    moved += 1;
                    a.evidence = changed.evidence;
                } else {
                    not_writable += 1;
                }
            }
        }
        Ok((moved, not_writable))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::testing::FixedEmbedder;
    use antumbra_store::EMBED_DIM;

    const REPO: &str = "github.com/acme/orders";

    async fn server_as(store: &Store, tenant: &str, user: &str) -> McpServer {
        let (tenant, user) = (TenantId::new(tenant), UserId::new(user));
        let compartment = crate::provision_identity(store, &tenant, &user)
            .await
            .unwrap();
        McpServer::new(
            store.clone(),
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            tenant,
            user,
            "h".into(),
            compartment,
            None,
        )
    }

    async fn remember(s: &McpServer, content: &str, repo: &str, branch: &str) -> String {
        remember_in(s, content, repo, branch, None).await
    }

    async fn remember_in(
        s: &McpServer,
        content: &str,
        repo: &str,
        branch: &str,
        compartment: Option<String>,
    ) -> String {
        s.store_memory(Parameters(StoreParams {
            provenance: Some(ProvenanceParams {
                repo: repo.into(),
                commit: "b697da7".into(),
                branch: Some(branch.into()),
                path: Some("src/orders.rs".into()),
            }),
            content: content.into(),
            network: "world".into(),
            confidence: None,
            evidence: None,
            volatile: None,
            compartment,
        }))
        .await
        .unwrap()
        .0
        .id
    }

    fn merge(merged_at: Option<&str>) -> MergeReport {
        MergeReport {
            head_branch: "feat/outbox".into(),
            base_branch: "main".into(),
            merge_commit: "4e1318f".into(),
            merged_at: merged_at.map(str::to_string),
            commits: Vec::new(),
        }
    }

    async fn report(s: &McpServer, merges: Vec<MergeReport>) -> MergesRecorded {
        s.record_merges(Parameters(RecordMergesParams {
            repo: REPO.into(),
            merges,
        }))
        .await
        .unwrap()
        .0
    }

    /// The scope the one memory recalled for "outbox" has, seen from `main`.
    async fn scope_from_main(s: &McpServer) -> Option<String> {
        s.recall_memories(Parameters(RecallParams {
            host: None,
            repo: Some(REPO.into()),
            branch: Some("main".into()),
            query: "outbox".into(),
            top_k: Some(1),
            network: None,
            full: None,
            floor: None,
        }))
        .await
        .unwrap()
        .0
        .memories
        .first()
        .and_then(|m| m.scope.clone())
    }

    async fn anchor(s: &McpServer, id: &str) -> Option<GitProvenance> {
        let m = memory::get(&s.store, &s.tenant, &MemoryId::new(id))
            .await
            .unwrap()
            .unwrap();
        GitProvenance::from_evidence(&m.evidence)
    }

    #[tokio::test]
    async fn a_merge_moves_the_branchs_memories_and_nothing_else() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        let on_branch = remember(&s, "the outbox drains every second", REPO, "feat/outbox").await;
        let elsewhere = remember(&s, "retries back off", REPO, "feat/retries").await;
        let other_repo =
            remember(&s, "same branch name", "github.com/acme/web", "feat/outbox").await;

        let recorded = report(&s, vec![merge(None)]).await;
        assert_eq!((recorded.moved, recorded.not_writable), (1, 0));

        let kept = memory::get(&s.store, &s.tenant, &MemoryId::new(&on_branch))
            .await
            .unwrap()
            .unwrap();
        assert!(
            kept.embedding.is_some(),
            "a moved memory keeps its embedding"
        );
        let moved = anchor(&s, &on_branch).await.unwrap();
        assert_eq!(moved.branch.as_deref(), Some("main"));
        assert_eq!(moved.commit, "4e1318f");
        assert_eq!(
            moved.path.as_deref(),
            Some("src/orders.rs"),
            "the path is kept"
        );
        assert_eq!(
            anchor(&s, &elsewhere).await.unwrap().branch.as_deref(),
            Some("feat/retries")
        );
        assert_eq!(
            anchor(&s, &other_repo).await.unwrap().branch.as_deref(),
            Some("feat/outbox")
        );

        // Reported again, the merge finds nothing left on the branch.
        assert_eq!(report(&s, vec![merge(None)]).await.moved, 0);
    }

    /// The point of the tool: from the base branch, a merged branch's memory
    /// reads as in scope instead of `other_branch`.
    #[tokio::test]
    async fn after_the_merge_recall_from_the_base_branch_counts_it_in_scope() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        remember(&s, "the outbox drains every second", REPO, "feat/outbox").await;
        assert_eq!(scope_from_main(&s).await.as_deref(), Some("other_branch"));
        report(&s, vec![merge(None)]).await;
        assert_eq!(scope_from_main(&s).await.as_deref(), Some("in_scope"));
    }

    /// A branch merged into another branch that then merged: reported in one
    /// call, oldest first, the memory is carried on to the second base, which
    /// only works because the anchors read once are kept current.
    #[tokio::test]
    async fn a_stacked_branch_is_carried_on_within_one_report() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        let id = remember(&s, "the retry budget is three", REPO, "feat/retries").await;
        let child = MergeReport {
            head_branch: "feat/retries".into(),
            base_branch: "feat/outbox".into(),
            merge_commit: "c0ffee1".into(),
            ..merge(None)
        };
        let recorded = report(&s, vec![child, merge(None)]).await;
        let each: Vec<usize> = recorded.merges.iter().map(|m| m.moved).collect();
        assert_eq!(each, vec![1, 1]);
        let now = anchor(&s, &id).await.unwrap();
        assert_eq!(
            (now.branch.as_deref(), now.commit.as_str()),
            (Some("main"), "4e1318f")
        );
    }

    #[tokio::test]
    async fn an_empty_report_moves_nothing() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        remember(&s, "the outbox drains every second", REPO, "feat/outbox").await;
        let recorded = report(&s, Vec::new()).await;
        assert_eq!((recorded.moved, recorded.merges.len()), (0, 0));
    }

    #[tokio::test]
    async fn a_memory_created_after_the_merge_belongs_to_a_new_branch_of_that_name() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        let id = remember(&s, "new work on a reused name", REPO, "feat/outbox").await;
        let recorded = report(&s, vec![merge(Some("2020-01-01T00:00:00Z"))]).await;
        assert_eq!(recorded.moved, 0);
        assert_eq!(
            anchor(&s, &id).await.unwrap().branch.as_deref(),
            Some("feat/outbox")
        );
    }

    /// A session still on the branch after it merged stores memories anchored
    /// at the branch's last commit. Those are the merged work, whenever stored.
    #[tokio::test]
    async fn a_memory_anchored_at_a_merged_commit_moves_whenever_it_was_stored() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        let id = remember(&s, "stored after the merge", REPO, "feat/outbox").await;
        let late = MergeReport {
            commits: vec!["B697DA70000000000000000000000000000000ff".into()],
            ..merge(Some("2020-01-01T00:00:00Z"))
        };
        assert_eq!(report(&s, vec![late]).await.moved, 1);
        assert_eq!(
            anchor(&s, &id).await.unwrap().branch.as_deref(),
            Some("main")
        );
    }

    #[test]
    fn a_commit_matches_written_short_on_either_side() {
        let merged = vec!["b697da7e1f".to_string(), "c0ffee1".to_string()];
        assert!(carried(&merged, "b697da7"));
        assert!(carried(&merged, "C0FFEE1234"));
        assert!(!carried(&merged, "b697da8"));
        assert!(!carried(&[], "b697da7"));
    }

    /// Two people in one workspace, signed in in turn over one connection as the
    /// HTTP server does. B's report moves nothing of A's: A's private memory is
    /// invisible to B, and the one A shared read-only is visible but not B's to
    /// rewrite, and is counted as such. A's own report moves both.
    #[tokio::test]
    async fn a_merge_moves_only_what_the_caller_may_write() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let a = server_as(&store, "ws:t", "user:a").await;
        let b = server_as(&store, "ws:t", "user:b").await;

        store.signin(&a.tenant, &a.user).await.unwrap();
        let private = remember(&a, "a's own outbox note", REPO, "feat/outbox").await;
        let shared_space = a
            .create_compartment(Parameters(CreateCompartmentParams {
                name: "outbox".into(),
            }))
            .await
            .unwrap()
            .0
            .id;
        let shared = remember_in(
            &a,
            "the outbox drains every second",
            REPO,
            "feat/outbox",
            Some(shared_space.clone()),
        )
        .await;
        a.share_compartment(Parameters(ShareParams {
            compartment_id: shared_space,
            grantee: "user:b".into(),
            capability: "reference".into(),
        }))
        .await
        .unwrap();

        store.signin(&b.tenant, &b.user).await.unwrap();
        let by_b = report(&b, vec![merge(None)]).await;
        assert_eq!((by_b.moved, by_b.not_writable), (0, 1));

        store.signin(&a.tenant, &a.user).await.unwrap();
        for id in [&private, &shared] {
            assert_eq!(
                anchor(&a, id).await.unwrap().branch.as_deref(),
                Some("feat/outbox")
            );
        }
        let by_a = report(&a, vec![merge(None)]).await;
        assert_eq!((by_a.moved, by_a.not_writable), (2, 0));
    }

    #[test]
    fn a_report_that_is_not_a_merge_is_refused() {
        let bad = |edit: fn(&mut MergeReport)| {
            let mut m = merge(None);
            edit(&mut m);
            reported(REPO, &m).is_err()
        };
        assert!(bad(|m| m.merge_commit = "not-a-sha".into()));
        assert!(bad(|m| m.head_branch = "main".into()));
        assert!(bad(|m| m.base_branch = "a:b".into()));
        assert!(bad(|m| m.merged_at = Some("yesterday".into())));
        assert!(!bad(|m| m.merged_at = Some("2026-09-23T17:47:07Z".into())));
        assert!(
            reported("orders", &merge(None)).is_err(),
            "a bare repo name"
        );
    }

    /// Every tool the server advertises is one the REST shim can dispatch. The
    /// shim matches names by hand, so a tool added to a router and not to the
    /// shim is one a hook cannot call; bad arguments are fine here, an unknown
    /// name is not.
    #[tokio::test]
    async fn every_advertised_tool_is_reachable_over_rest() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        for name in McpServer::all_tool_names() {
            if let Err(e) = s.call_tool(&name, serde_json::json!({})).await {
                assert!(!e.message.contains("unknown tool"), "{name}: {}", e.message);
            }
        }
    }

    /// Both reachable over the REST shim, which dispatches by name: a tool the
    /// shim forgets is one a hook cannot call. `penalize_memory` was missing,
    /// so the session bootstrap's orphan penalty and `antumbra claude remember`
    /// both got "unknown tool".
    #[tokio::test]
    async fn the_rest_shim_reaches_record_merges_and_penalize_memory() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        let id = remember(&s, "the outbox drains every second", REPO, "feat/outbox").await;
        let recorded = s
            .call_tool(
                "record_merges",
                serde_json::json!({
                    "repo": REPO,
                    "merges": [{
                        "head_branch": "feat/outbox",
                        "base_branch": "main",
                        "merge_commit": "4e1318f",
                    }],
                }),
            )
            .await
            .unwrap();
        assert_eq!(recorded["moved"], 1);
        assert_eq!(recorded["merges"][0]["head_branch"], "feat/outbox");
        let penalized = s
            .call_tool("penalize_memory", serde_json::json!({ "memory_id": id }))
            .await
            .unwrap();
        assert_eq!(penalized["found"], true);
    }
}
