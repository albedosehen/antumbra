//! The knowledge diff on a pull request (ADR-0019): when one is opened, pushed
//! to, reopened or marked ready, the memories and documents anchored to the
//! paths it changes and to its branch are posted as a neutral check run. It
//! runs after the response, like ingest, since it reads the pull request's
//! files and the workspace's memories. Off unless the server was started with
//! `--github-knowledge-diff`, and the App needs the Checks permission (write).

use std::sync::Arc;

use anyhow::{Context, Result};

use antumbra_core::TenantId;
use antumbra_github::{KnowledgeDiff, PullRequestEvent};
use antumbra_store::repo::{document, memory};

use super::{config, HttpState, Outcome};

/// The actions that put new commits, or a new reader, in front of the diff.
const ACTIONS: [&str; 4] = ["opened", "synchronize", "reopened", "ready_for_review"];

/// What the check run is called on the pull request.
pub(super) const CHECK_NAME: &str = "Antumbra knowledge diff";

/// Whether `event` is one the knowledge diff answers.
pub(super) fn answers(event: &PullRequestEvent) -> bool {
    ACTIONS.contains(&event.action.as_str())
}

/// Queue the knowledge diff for `event`, or say why it is not posted.
pub(super) fn queue(state: &Arc<HttpState>, event: &PullRequestEvent) -> Outcome {
    const KIND: &str = "pull_request";
    let cfg = config(state);
    let repo = event.repository.slug();
    if !cfg.knowledge_diff {
        return Outcome::ignored(
            KIND,
            repo,
            format!("action '{}': the knowledge diff is off", event.action),
        );
    }
    let Some(slug) = repo else {
        return Outcome::ignored(KIND, None, "the repository has no slug");
    };
    let Some(tenant) = cfg.repos.tenant_for(&slug).cloned() else {
        return Outcome::ignored(KIND, Some(slug), "repository is not mapped to a workspace");
    };
    let Some(installation) = event.installation else {
        return Outcome::ignored(
            KIND,
            Some(slug),
            "delivered outside an App installation: no token to post a check run with",
        );
    };
    let (state, event, posting) = (state.clone(), event.clone(), slug.clone());
    let task_tenant = tenant.clone();
    tokio::spawn(async move {
        let number = event.pull_request.number;
        match post(&state, &event, &posting, &task_tenant, installation.id).await {
            Ok(title) => eprintln!("antumbra-mcp: knowledge diff on {posting}#{number}: {title}"),
            Err(e) => eprintln!("antumbra-mcp: knowledge diff on {posting}#{number} failed: {e:#}"),
        }
    });
    Outcome {
        event: KIND.into(),
        repo: Some(slug),
        tenant: Some(tenant.as_str().to_string()),
        knowledge_diff_queued: true,
        ..Outcome::default()
    }
}

/// Read the change and the workspace, and post the check run. Returns its
/// title.
async fn post(
    state: &HttpState,
    event: &PullRequestEvent,
    slug: &str,
    tenant: &TenantId,
    installation_id: u64,
) -> Result<String> {
    let cfg = config(state);
    let token = cfg
        .installation_token(installation_id)
        .await?
        .context("no App credentials to post a check run with")?;
    let pr = &event.pull_request;
    let full_name = &event.repository.full_name;
    let files = cfg
        .api()
        .pull_request_files(&token, full_name, pr.number)
        .await
        .with_context(|| format!("listing the files of pull request #{}", pr.number))?;
    // Owner mode across the workspace, serialized with every other owner-side
    // call; two reads.
    let (memories, titles) = {
        let _guard = state.auth.lock().await;
        state.store.signin_root().await?;
        (
            memory::list(&state.store, tenant).await?,
            document::list_titles(&state.store, tenant).await?,
        )
    };
    let diff = KnowledgeDiff::of(slug, &pr.head.name, &files, &memories, &titles);
    let output = diff.render(&pr.head.name, &pr.base.name);
    cfg.api()
        .create_check_run(&token, full_name, &pr.head.sha, CHECK_NAME, &output)
        .await
        .context("posting the check run")?;
    Ok(output.title)
}
