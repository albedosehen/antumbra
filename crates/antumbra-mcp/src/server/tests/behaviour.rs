//! Behaviours (ADR-0027): recorded only with a check shown to discriminate,
//! listed by status, accepted and retired by their owner, superseded by a
//! later one, and private to the user who recorded them.

use super::*;
use crate::server::behaviour::{
    BehaviourIdParams, ExampleParams, ListBehavioursParams, RecordBehaviourParams,
};

fn server_for(store: &Store, user: &str) -> McpServer {
    McpServer::new(
        store.clone(),
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new(user),
        "windows".into(),
        CompartmentId::new(format!("comp:test:{user}:default")),
        None,
    )
}

fn example(task: &str, answer: &str) -> ExampleParams {
    ExampleParams {
        task: task.into(),
        answer: answer.into(),
    }
}

fn branch_rule(supersedes: Option<String>, accepted: bool) -> RecordBehaviourParams {
    RecordBehaviourParams {
        rule: "Name a branch for an issue feat/{issue}-{slug}.".into(),
        must: vec![r"feat/\d+-[a-z0-9-]+".into()],
        must_not: vec![r"feature/".into()],
        examples: vec![
            example(
                "Start issue #42, Add login page.",
                "git checkout -b feat/42-add-login-page",
            ),
            example(
                "Branch for issue 7: fix crash.",
                "git switch -c feat/7-fix-crash",
            ),
            example(
                "New branch for ticket #128 dark mode.",
                "git checkout -b feat/128-dark-mode",
            ),
        ],
        violations: vec![
            "git checkout -b feature/add-login-page".into(),
            "git checkout -b issue-42".into(),
        ],
        scope: None,
        supersedes,
        accepted: Some(accepted),
    }
}

async fn list(s: &McpServer, status: Option<&str>) -> Vec<(String, String, String)> {
    s.list_behaviours(Parameters(ListBehavioursParams {
        status: status.map(str::to_string),
        scope: None,
    }))
    .await
    .unwrap()
    .0
    .behaviours
    .into_iter()
    .map(|b| (b.id, b.status, b.rule))
    .collect()
}

#[tokio::test]
async fn a_behaviour_is_recorded_only_with_a_check_that_discriminates() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");

    let mut weak = branch_rule(None, false);
    weak.violations
        .push("git checkout -b feat/9-anything".into());
    let refused = s.record_behaviour(Parameters(weak)).await.unwrap().0;
    assert!(!refused.recorded);
    assert!(
        refused
            .problems
            .iter()
            .any(|p| p.contains("passes the violating answer")),
        "{:?}",
        refused.problems
    );
    assert!(list(&s, None).await.is_empty(), "nothing written");

    let kept = s
        .record_behaviour(Parameters(branch_rule(None, false)))
        .await
        .unwrap()
        .0;
    assert!(kept.recorded, "{:?}", kept.problems);
    assert_eq!(kept.status.as_deref(), Some("proposed"));
    let listed = list(&s, None).await;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].1, "proposed");
    assert!(listed[0].2.starts_with("Name a branch"));

    // A behaviour is a memory in the user's behaviour compartment, kept off
    // the write-time consolidation that would teach it to echo itself.
    let m = memory::get(
        &store,
        &TenantId::new("ws:test"),
        &MemoryId::new(kept.id.unwrap()),
    )
    .await?
    .expect("stored");
    assert!(m.volatile);
    assert_eq!(
        m.compartment.as_ref().map(|c| c.as_str().to_string()),
        Some("comp:ws:test:user:lily:behaviour".to_string())
    );
    Ok(())
}

#[tokio::test]
async fn behaviours_are_accepted_retired_and_superseded() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");
    let first = s
        .record_behaviour(Parameters(branch_rule(None, false)))
        .await
        .unwrap()
        .0
        .id
        .unwrap();

    let accepted = s
        .accept_behaviour(Parameters(BehaviourIdParams {
            behaviour_id: first.clone(),
        }))
        .await
        .unwrap()
        .0;
    assert!(accepted.found);
    assert_eq!(list(&s, Some("accepted")).await.len(), 1);
    assert!(list(&s, Some("proposed")).await.is_empty());

    // The user restates it: the new one is accepted, the old one retired.
    let second = s
        .record_behaviour(Parameters(branch_rule(Some(first.clone()), true)))
        .await
        .unwrap()
        .0;
    assert_eq!(second.superseded, Some(true));
    assert_eq!(second.status.as_deref(), Some("accepted"));
    let retired = list(&s, Some("retired")).await;
    assert_eq!(retired.len(), 1);
    assert_eq!(retired[0].0, first);
    assert!(s
        .list_behaviours(Parameters(ListBehavioursParams {
            status: Some("bogus".into()),
            scope: None,
        }))
        .await
        .is_err());
    Ok(())
}

#[tokio::test]
async fn behaviours_are_private_to_the_user_who_recorded_them() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let lily = server_for(&store, "user:lily");
    let oslo = server_for(&store, "user:oslo");
    let id = lily
        .record_behaviour(Parameters(branch_rule(None, false)))
        .await
        .unwrap()
        .0
        .id
        .unwrap();
    assert!(list(&oslo, None).await.is_empty());
    let attempt = oslo
        .retire_behaviour(Parameters(BehaviourIdParams { behaviour_id: id }))
        .await
        .unwrap()
        .0;
    assert!(!attempt.found, "another user cannot retire it");
    assert_eq!(list(&lily, Some("proposed")).await.len(), 1);
    Ok(())
}
