//! Document ingest driven by GitHub events: what a merge changed, or
//! everything a newly installed repository holds, read through the App and
//! ingested by the one shared path (`antumbra-ingest`), each document anchored
//! to the commit it was read at.
//!
//! Planning (which paths, at which commit) happens inside the delivery so the
//! response can say how many documents were queued; the reads and the ingest
//! run afterwards, in a detached task, because GitHub gives a receiver ten
//! seconds and an ingest through a remote embedder can take longer. Every
//! step is idempotent (a title's chunks are replaced in place), so a
//! redelivery after a failure repeats nothing harmful.

use std::sync::Arc;

use anyhow::{Context, Result};

use antumbra_core::manifest;
use antumbra_core::{GitProvenance, TenantId};
use antumbra_github::{is_knowledge_document, Merge, PullRequestEvent, MAX_DOCUMENT_BYTES};
use antumbra_ingest::Document;

use super::{dependencies, GithubConfig, HttpState};

/// The documents to read and ingest, and the anchor they get.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct IngestPlan {
    pub tenant: TenantId,
    /// `owner/name`, for the API.
    pub full_name: String,
    /// The repository slug memories and chunks anchor to.
    pub slug: String,
    /// The commit every document is read at and anchored to.
    pub commit: String,
    /// The branch that commit sits on.
    pub branch: String,
    pub paths: Vec<String>,
    /// Knowledge documents the change took out of the tree (removed, or renamed
    /// away from), whose chunks have to go with them.
    pub vacated: Vec<String>,
    /// Every manifest in the tree at the commit, to read for the declared
    /// edges; `None` when nothing changed them, or the commit is not on the
    /// default branch, whose reading is the repository's.
    pub manifests: Option<Vec<String>>,
    /// The listing GitHub returned was incomplete (a very large tree).
    pub truncated: bool,
    /// The installation token the reads authenticate with.
    pub token: String,
}

/// What a plan's run did, per document.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct IngestReport {
    pub ingested: Vec<String>,
    /// Documents dropped because their file is gone from the tree.
    pub dropped: Vec<String>,
    /// Absent at the commit, binary, or over the size cap.
    pub skipped: Vec<String>,
    pub failed: Vec<(String, String)>,
}

/// The documents a merged pull request changed (still present after it),
/// read at the merge commit. `None` when the receiver cannot read contents
/// (no App credentials, or a delivery that did not come through an
/// installation).
pub(super) async fn plan_for_merge(
    cfg: &GithubConfig,
    event: &PullRequestEvent,
    merge: &Merge,
    tenant: &TenantId,
) -> Result<Option<IngestPlan>> {
    let Some(installation) = event.installation else {
        return Ok(None);
    };
    let Some(token) = cfg.installation_token(installation.id).await? else {
        return Ok(None);
    };
    let full_name = event.repository.full_name.clone();
    let files = cfg
        .api()
        .pull_request_files(&token, &full_name, event.pull_request.number)
        .await
        .with_context(|| {
            format!(
                "listing the files of pull request #{}",
                event.pull_request.number
            )
        })?;
    // A removed or renamed document leaves chunks behind that describe a file
    // which no longer exists, anchored to a commit that is an ancestor of HEAD,
    // so recall would call them live. They go when the file goes.
    let vacated = files
        .iter()
        .filter_map(|f| f.vacated_path())
        .filter(|path| is_knowledge_document(path))
        .map(str::to_string)
        .collect();
    let paths = files
        .iter()
        .filter(|f| f.is_present() && is_knowledge_document(&f.path))
        .map(|f| f.path.clone())
        .collect();
    let manifests = if touches_a_manifest(&files)
        && on_default_branch(cfg, event, merge, &token).await?
    {
        let tree = cfg
            .api()
            .tree_paths(&token, &full_name, &merge.merge_commit)
            .await
            .with_context(|| format!("listing the tree of {full_name}@{}", merge.merge_commit))?;
        // A listing GitHub cut short would read as manifests removed.
        (!tree.truncated).then(|| manifest_paths(tree.paths))
    } else {
        None
    };
    Ok(Some(IngestPlan {
        tenant: tenant.clone(),
        full_name,
        slug: merge.repo.clone(),
        commit: merge.merge_commit.clone(),
        branch: merge.base_branch.clone(),
        paths,
        vacated,
        manifests,
        truncated: false,
        token,
    }))
}

/// Whether a change added, changed or took away a manifest.
fn touches_a_manifest(files: &[antumbra_github::ChangedFile]) -> bool {
    files.iter().any(|f| {
        manifest::is_manifest(&f.path) || f.vacated_path().is_some_and(manifest::is_manifest)
    })
}

/// Whether the merge landed on the repository's default branch, from the
/// delivery when GitHub sent it, else asked.
async fn on_default_branch(
    cfg: &GithubConfig,
    event: &PullRequestEvent,
    merge: &Merge,
    token: &str,
) -> Result<bool> {
    let default = match event.repository.default_branch.clone() {
        Some(branch) => branch,
        None => cfg
            .api()
            .default_branch(token, &event.repository.full_name)
            .await
            .context("reading the default branch")?,
    };
    Ok(merge.base_branch == default)
}

fn manifest_paths(paths: Vec<String>) -> Vec<String> {
    paths
        .into_iter()
        .filter(|p| manifest::is_manifest(p))
        .collect()
}

/// Every knowledge document in a repository at the head of its default
/// branch: the cold start for a repository the App was just installed on.
pub(super) async fn plan_for_repository(
    cfg: &GithubConfig,
    full_name: &str,
    installation_id: u64,
    tenant: &TenantId,
) -> Result<IngestPlan> {
    let token = cfg
        .installation_token(installation_id)
        .await?
        .context("no App credentials configured, so repository contents cannot be read")?;
    let api = cfg.api();
    let branch = api
        .default_branch(&token, full_name)
        .await
        .with_context(|| format!("reading the default branch of {full_name}"))?;
    let commit = api
        .branch_head(&token, full_name, &branch)
        .await
        .with_context(|| format!("reading the head of {full_name}@{branch}"))?;
    let tree = api
        .tree_paths(&token, full_name, &commit)
        .await
        .with_context(|| format!("listing the tree of {full_name}@{commit}"))?;
    Ok(IngestPlan {
        tenant: tenant.clone(),
        full_name: full_name.to_string(),
        slug: api.repo_slug(full_name),
        commit,
        branch,
        paths: tree
            .paths
            .iter()
            .filter(|p| is_knowledge_document(p))
            .cloned()
            .collect(),
        // A cold start reads the tree as it is; nothing has been taken out of it.
        vacated: Vec::new(),
        // A listing GitHub cut short would read as manifests removed.
        manifests: (!tree.truncated).then(|| manifest_paths(tree.paths)),
        truncated: tree.truncated,
        token,
    })
}

/// Run `plan` after the response, logging what it did.
pub(super) fn spawn(state: Arc<HttpState>, plan: IngestPlan) {
    let what = format!(
        "{} @ {} ({} document(s){})",
        plan.slug,
        &plan.commit[..plan.commit.len().min(12)],
        plan.paths.len(),
        if plan.truncated {
            ", listing truncated"
        } else {
            ""
        }
    );
    tokio::spawn(async move {
        let reading = plan.clone();
        let report = run(&state, plan).await;
        eprintln!(
            "antumbra-mcp: github ingest {what}: {} ingested, {} dropped, {} skipped, {} failed",
            report.ingested.len(),
            report.dropped.len(),
            report.skipped.len(),
            report.failed.len()
        );
        for (path, why) in &report.failed {
            eprintln!("antumbra-mcp: github ingest failed for {path}: {why}");
        }
        if let Some(paths) = &reading.manifests {
            let declared = dependencies::run(&state, &reading, paths).await;
            eprintln!(
                "antumbra-mcp: github manifests {what}: {} read, {} edge(s) recorded, {} retracted, {} failed",
                declared.read,
                declared.recorded,
                declared.retracted,
                declared.failed.len()
            );
            for (path, why) in &declared.failed {
                eprintln!("antumbra-mcp: github manifest reading failed for {path}: {why}");
            }
        }
    });
}

/// Read and ingest every path in `plan`; one document's failure never stops
/// the rest. Each ingest holds the auth lock in owner mode, the way every
/// owner-side write does.
pub(super) async fn run(state: &HttpState, plan: IngestPlan) -> IngestReport {
    let mut report = IngestReport::default();
    let Some(cfg) = state.github.as_ref() else {
        return report;
    };
    let embedder = {
        let _guard = state.auth.lock().await;
        if let Err(e) = state.store.signin_root().await {
            report.failed = plan
                .paths
                .into_iter()
                .map(|p| (p, format!("owner signin failed: {e}")))
                .collect();
            return report;
        }
        state.embedder_for(&plan.tenant).await
    };
    for path in plan.vacated {
        let title = document_title(&plan.slug, &path);
        let result = {
            let _guard = state.auth.lock().await;
            match state.store.signin_root().await {
                Ok(()) => antumbra_store::repo::document::delete_title(
                    &state.store,
                    &plan.tenant,
                    &title,
                    None,
                )
                .await
                .map_err(anyhow::Error::from),
                Err(e) => Err(anyhow::Error::from(e)),
            }
        };
        match result {
            Ok(()) => report.dropped.push(path),
            Err(e) => report.failed.push((path, format!("{e:#}"))),
        }
    }
    for path in plan.paths {
        let text = match cfg
            .api()
            .file_text(&plan.token, &plan.full_name, &path, &plan.commit)
            .await
        {
            Ok(Some(text)) if text.len() <= MAX_DOCUMENT_BYTES => text,
            Ok(_) => {
                report.skipped.push(path);
                continue;
            }
            Err(e) => {
                report.failed.push((path, e.to_string()));
                continue;
            }
        };
        let doc = Document {
            title: document_title(&plan.slug, &path),
            source: Some(cfg.api().blob_url(&plan.full_name, &plan.commit, &path)),
            content: text,
            provenance: Some(
                GitProvenance::new(plan.slug.clone(), plan.commit.clone())
                    .on_branch(plan.branch.clone())
                    .at_path(path.clone()),
            ),
            // Repository documents are the organization's reference material:
            // the shared pool, readable by every member of the workspace.
            compartment: None,
        };
        let result = {
            let _guard = state.auth.lock().await;
            match state.store.signin_root().await {
                Ok(()) => {
                    antumbra_ingest::ingest_text(
                        &state.store,
                        embedder.as_ref(),
                        &plan.tenant,
                        state.copal.as_deref(),
                        &doc,
                    )
                    .await
                }
                Err(e) => Err(anyhow::Error::from(e)),
            }
        };
        match result {
            Ok(_) => report.ingested.push(path),
            Err(e) => report.failed.push((path, format!("{e:#}"))),
        }
    }
    report
}

/// The title a repository's document is ingested under: the repository slug and
/// the path. A document's identity is (workspace, title), and one workspace
/// usually holds many repositories, so a bare path would make every repository's
/// `README.md` the same document: the second ingest would delete the first one's
/// chunks and revision its archived original, and at a cold start the last
/// repository read would win every path the repositories share.
pub(super) fn document_title(slug: &str, path: &str) -> String {
    format!("{slug}:{path}")
}
