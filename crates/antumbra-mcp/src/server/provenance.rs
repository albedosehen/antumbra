//! What a caller can tell the server about the history memories are anchored
//! to. `record_merge` is the GitHub webhook's merge handler (ADR-0019) for a
//! deployment GitHub cannot reach: the caller, who can ask GitHub, says which
//! branch merged where, and the memories anchored to that branch move to the
//! merge commit on the base branch. Without it, everything learned on a feature
//! branch reads `other_branch` from the base branch for good once the branch
//! merges, and recall ranks it below memories that were never anchored at all.
//!
//! It runs as the caller, where the webhook runs in owner mode. A webhook
//! delivery is signed by GitHub; this report is signed only by the caller, so
//! it moves only memories the caller could already rewrite.
//!
//! Its own router, joined to the memory tools' in `engine.rs`, because
//! `server.rs` is past the size rule.

use chrono::DateTime;

use antumbra_core::normalize_repo;
use antumbra_github::{reanchor_merged, Merge};

use super::*;

/// A merge, as the caller reports it.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecordMergeParams {
    /// Repository slug, `host/org/name` (for example `github.com/oneiriq/antumbra`).
    pub repo: String,
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

/// Whether `anchor` names one of `commits`, either written short.
fn carried(commits: &[String], anchor: &str) -> bool {
    let anchor = anchor.to_ascii_lowercase();
    commits.iter().any(|c| {
        let c = c.to_ascii_lowercase();
        c.starts_with(&anchor) || anchor.starts_with(&c)
    })
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MergeRecorded {
    /// Memories moved to the merge commit on the base branch.
    pub moved: usize,
    /// Memories on the merged branch that you can read but not write (a
    /// compartment shared with you read-only), so they stay where they are.
    pub not_writable: usize,
}

/// The merge `p` describes, or why it cannot be one.
fn merge_of(p: &RecordMergeParams) -> Result<(Merge, Option<DateTime<Utc>>), ErrorData> {
    let repo = normalize_repo(&p.repo);
    let anchor_on = |branch: &str| {
        GitProvenance::new(repo.clone(), p.merge_commit.clone())
            .on_branch(branch)
            .is_valid()
    };
    // A slug is at least `host/name`; the anchor's own check allows a bare word.
    if !repo.contains('/') || !anchor_on(&p.head_branch) || !anchor_on(&p.base_branch) {
        return Err(ErrorData::invalid_params(
            "a merge needs a repo slug without '@' (host/org/name), a 7-40 hex digit merge \
             commit, and branches without ':'"
                .to_string(),
            None,
        ));
    }
    if p.head_branch == p.base_branch {
        return Err(ErrorData::invalid_params(
            format!("'{}' cannot merge into itself", p.head_branch),
            None,
        ));
    }
    let merged_at = p
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
    let merge = Merge {
        repo,
        head_branch: p.head_branch.clone(),
        base_branch: p.base_branch.clone(),
        merge_commit: p.merge_commit.clone(),
    };
    Ok((merge, merged_at))
}

#[tool_router(router = provenance_router, vis = "pub(super)")]
impl McpServer {
    /// Move the memories of a merged branch onto the branch it merged into.
    #[tool(
        description = "Record that a branch merged: memories anchored to the merged branch move to the merge commit on the base branch, so recall from the base branch treats them as its own. What the GitHub webhook does, for a server GitHub cannot reach. Safe to repeat."
    )]
    pub(super) async fn record_merge(
        &self,
        Parameters(p): Parameters<RecordMergeParams>,
    ) -> Result<Json<MergeRecorded>, ErrorData> {
        let (merge, merged_at) = merge_of(&p)?;
        let belongs = |m: &Memory| {
            merged_at.is_none_or(|at| m.created_at <= at)
                || GitProvenance::from_evidence(&m.evidence)
                    .is_some_and(|anchor| carried(&p.commits, &anchor.commit))
        };
        let candidates: Vec<Memory> = memory::list(&self.store, &self.tenant)
            .await
            .map_err(err)?
            .into_iter()
            .filter(belongs)
            .collect();
        let changed = reanchor_merged(candidates, &merge, Utc::now());
        let (mut moved, mut not_writable) = (0, 0);
        for m in &changed {
            memory::upsert(&self.store, m).await.map_err(err)?;
            // A write into a compartment the caller may only read succeeds and
            // persists nothing, so count what actually landed.
            let landed = memory::get(&self.store, &self.tenant, &m.id)
                .await
                .map_err(err)?
                .is_some_and(|stored| stored.evidence == m.evidence);
            if landed {
                moved += 1;
            } else {
                not_writable += 1;
            }
        }
        Ok(Json(MergeRecorded {
            moved,
            not_writable,
        }))
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

    fn merge(merged_at: Option<&str>) -> RecordMergeParams {
        RecordMergeParams {
            repo: REPO.into(),
            head_branch: "feat/outbox".into(),
            base_branch: "main".into(),
            merge_commit: "4e1318f".into(),
            merged_at: merged_at.map(str::to_string),
            commits: Vec::new(),
        }
    }

    /// The scope the one memory recalled for "outbox" has, seen from `main`.
    async fn scope_from_main(s: &McpServer) -> Option<String> {
        s.recall_memories(Parameters(RecallParams {
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

        let recorded = s.record_merge(Parameters(merge(None))).await.unwrap().0;
        assert_eq!((recorded.moved, recorded.not_writable), (1, 0));

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
        let again = s.record_merge(Parameters(merge(None))).await.unwrap().0;
        assert_eq!(again.moved, 0);
    }

    /// The point of the tool: from the base branch, a merged branch's memory
    /// reads as in scope instead of `other_branch`.
    #[tokio::test]
    async fn after_the_merge_recall_from_the_base_branch_counts_it_in_scope() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        remember(&s, "the outbox drains every second", REPO, "feat/outbox").await;
        assert_eq!(scope_from_main(&s).await.as_deref(), Some("other_branch"));
        s.record_merge(Parameters(merge(None))).await.unwrap();
        assert_eq!(scope_from_main(&s).await.as_deref(), Some("in_scope"));
    }

    #[tokio::test]
    async fn a_memory_created_after_the_merge_belongs_to_a_new_branch_of_that_name() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        let id = remember(&s, "new work on a reused name", REPO, "feat/outbox").await;
        let long_ago = "2020-01-01T00:00:00Z";
        let recorded = s
            .record_merge(Parameters(merge(Some(long_ago))))
            .await
            .unwrap()
            .0;
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
        let report = RecordMergeParams {
            commits: vec!["B697DA70000000000000000000000000000000ff".into()],
            ..merge(Some("2020-01-01T00:00:00Z"))
        };
        assert_eq!(s.record_merge(Parameters(report)).await.unwrap().0.moved, 1);
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
        let by_b = b.record_merge(Parameters(merge(None))).await.unwrap().0;
        assert_eq!((by_b.moved, by_b.not_writable), (0, 1));

        store.signin(&a.tenant, &a.user).await.unwrap();
        for id in [&private, &shared] {
            assert_eq!(
                anchor(&a, id).await.unwrap().branch.as_deref(),
                Some("feat/outbox")
            );
        }
        let by_a = a.record_merge(Parameters(merge(None))).await.unwrap().0;
        assert_eq!((by_a.moved, by_a.not_writable), (2, 0));
    }

    #[test]
    fn a_report_that_is_not_a_merge_is_refused() {
        let bad = |edit: fn(&mut RecordMergeParams)| {
            let mut p = merge(None);
            edit(&mut p);
            merge_of(&p).is_err()
        };
        assert!(bad(|p| p.merge_commit = "not-a-sha".into()));
        assert!(bad(|p| p.repo = "orders".into()));
        assert!(bad(|p| p.head_branch = "main".into()));
        assert!(bad(|p| p.base_branch = "a:b".into()));
        assert!(bad(|p| p.merged_at = Some("yesterday".into())));
        assert!(!bad(|p| p.merged_at = Some("2026-09-23T17:47:07Z".into())));
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
    async fn the_rest_shim_reaches_record_merge_and_penalize_memory() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = server_as(&store, "ws:a", "user:a").await;
        let id = remember(&s, "the outbox drains every second", REPO, "feat/outbox").await;
        let recorded = s
            .call_tool(
                "record_merge",
                serde_json::json!({
                    "repo": REPO, "head_branch": "feat/outbox",
                    "base_branch": "main", "merge_commit": "4e1318f",
                }),
            )
            .await
            .unwrap();
        assert_eq!(recorded["moved"], 1);
        let penalized = s
            .call_tool("penalize_memory", serde_json::json!({ "memory_id": id }))
            .await
            .unwrap();
        assert_eq!(penalized["found"], true);
    }
}
