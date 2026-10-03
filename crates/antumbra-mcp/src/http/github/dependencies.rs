//! The declared source through the App (ADR-0019): a repository's manifests
//! read at a commit, kept as its latest reading, and the declared edges into
//! and out of it brought in line with every repository's latest reading.
//!
//! A repository is read when the App is installed on it (every manifest at the
//! head of its default branch) and when a merge into its default branch
//! changes a manifest (every manifest at the merge commit: one file's change
//! can only be judged against the repository's whole set). It runs in the same
//! detached task as the document ingest, and like it is idempotent: reading
//! the same commit again reinforces the same edges.

use chrono::Utc;

use antumbra_core::depgraph::{Edge, Source};
use antumbra_core::manifest::{self, ManifestSet};
use antumbra_core::UserId;
use antumbra_github::{MAX_DOCUMENT_BYTES, SYSTEM_HOST, SYSTEM_USER};
use antumbra_store::repo::{manifest_set, memory};

use super::ingest::IngestPlan;
use super::HttpState;

/// What one reading did.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct DeclaredReport {
    /// Manifests read and understood.
    pub read: usize,
    /// Declared edges recorded or reinforced.
    pub recorded: usize,
    /// Declared edges no manifest makes any more, retracted.
    pub retracted: usize,
    pub failed: Vec<(String, String)>,
}

/// Read `paths` (the repository's manifests) at the plan's commit, keep them
/// as the repository's latest reading, and record and retract the declared
/// edges into and out of it. A manifest that cannot be read is reported and
/// left out; one failure in the owner section stops the reading, since a
/// partial set would retract edges that still stand.
pub(super) async fn run(state: &HttpState, plan: &IngestPlan, paths: &[String]) -> DeclaredReport {
    let mut report = DeclaredReport::default();
    let Some(cfg) = state.github.as_ref() else {
        return report;
    };
    let mut manifests = Vec::new();
    for path in paths {
        match cfg
            .api()
            .file_text(&plan.token, &plan.full_name, path, &plan.commit)
            .await
        {
            Ok(Some(text)) if text.len() <= MAX_DOCUMENT_BYTES => {
                if let Some(m) = manifest::read(path, &text) {
                    manifests.push(m);
                }
            }
            Ok(_) => {}
            Err(e) => report.failed.push((path.clone(), e.to_string())),
        }
    }
    if !report.failed.is_empty() {
        // A set missing a manifest that could not be read would retract every
        // edge that manifest declares, so nothing is kept from this reading.
        return report;
    }
    report.read = manifests.len();
    let set = ManifestSet {
        repo: plan.slug.clone(),
        commit: plan.commit.clone(),
        branch: Some(plan.branch.clone()),
        manifests,
    };
    if let Err(e) = keep(state, plan, &set, &mut report).await {
        report.failed.push((plan.slug.clone(), format!("{e:#}")));
    }
    report
}

/// The owner section: the reading kept, every repository's latest reading
/// listed, and the edges into and out of this one recorded and retracted.
async fn keep(
    state: &HttpState,
    plan: &IngestPlan,
    set: &ManifestSet,
    report: &mut DeclaredReport,
) -> anyhow::Result<()> {
    let _guard = state.auth.lock().await;
    state.store.signin_root().await?;
    let embedder = state.embedder_for(&plan.tenant).await;
    let now = Utc::now();
    manifest_set::upsert(&state.store, &plan.tenant, set, now).await?;
    let sets = manifest_set::list(&state.store, &plan.tenant).await?;
    let existing: Vec<(String, String)> = memory::with_any_evidence(
        &state.store,
        &plan.tenant,
        &[Source::Declared.evidence_entry()],
    )
    .await?
    .iter()
    .filter_map(Edge::of)
    .filter(|e| e.source == Source::Declared)
    .map(|e| (e.from, e.to))
    .collect();
    let sync = manifest::sync(&sets, &plan.slug, &existing);
    let author = UserId::new(SYSTEM_USER);
    for (claim, anchor) in &sync.record {
        crate::dependencies::record(
            &state.store,
            embedder.as_ref(),
            &plan.tenant,
            (&author, SYSTEM_HOST),
            claim,
            Some(anchor),
            now,
        )
        .await?;
        report.recorded += 1;
    }
    for claim in &sync.retract {
        if crate::dependencies::retract(&state.store, &plan.tenant, claim, now).await? {
            report.retracted += 1;
        }
    }
    Ok(())
}
