//! Behaviors: recorded only with a check shown to discriminate,
//! listed by status, accepted and retired by their owner, superseded by a
//! later one, and private to the user who recorded them.

use super::*;
use crate::server::behavior::{
    BehaviorIdParams, ExampleParams, ListBehaviorsParams, RecordBehaviorParams,
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

fn branch_rule(supersedes: Option<String>, accepted: bool) -> RecordBehaviorParams {
    RecordBehaviorParams {
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
            example(
                "Make a branch for issue 3, rename the CLI.",
                "git switch -c feat/3-rename-the-cli",
            ),
        ],
        violations: vec![
            "git checkout -b feature/add-login-page".into(),
            "git checkout -b issue-42".into(),
        ],
        scope: None,
        supersedes,
        accepted: Some(accepted),
        sources: Vec::new(),
    }
}

async fn list(s: &McpServer, status: Option<&str>) -> Vec<(String, String, String)> {
    s.list_behaviors(Parameters(ListBehaviorsParams {
        status: status.map(str::to_string),
        scope: None,
    }))
    .await
    .unwrap()
    .0
    .behaviors
    .into_iter()
    .map(|b| (b.id, b.status, b.rule))
    .collect()
}

#[tokio::test]
async fn a_behavior_is_recorded_only_with_a_check_that_discriminates() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");

    let mut weak = branch_rule(None, false);
    weak.violations
        .push("git checkout -b feat/9-anything".into());
    let refused = s.record_behavior(Parameters(weak)).await.unwrap().0;
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
        .record_behavior(Parameters(branch_rule(None, false)))
        .await
        .unwrap()
        .0;
    assert!(kept.recorded, "{:?}", kept.problems);
    assert_eq!(kept.status.as_deref(), Some("proposed"));
    let listed = list(&s, None).await;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].1, "proposed");
    assert!(listed[0].2.starts_with("Name a branch"));

    // A behavior is a memory in the user's behavior compartment, kept off
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
        Some("comp:ws:test:user:lily:behaviour".to_string()),
        "the compartment keeps the name it is already stored under"
    );
    Ok(())
}

#[tokio::test]
async fn behaviors_are_accepted_retired_and_superseded() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");
    let first = s
        .record_behavior(Parameters(branch_rule(None, false)))
        .await
        .unwrap()
        .0
        .id
        .unwrap();

    let accepted = s
        .accept_behavior(Parameters(BehaviorIdParams {
            behavior_id: first.clone(),
        }))
        .await
        .unwrap()
        .0;
    assert!(accepted.found);
    assert_eq!(list(&s, Some("accepted")).await.len(), 1);
    assert!(list(&s, Some("proposed")).await.is_empty());

    // The user restates it: the new one is accepted, the old one retired.
    let second = s
        .record_behavior(Parameters(branch_rule(Some(first.clone()), true)))
        .await
        .unwrap()
        .0;
    assert_eq!(second.superseded, Some(true));
    assert_eq!(second.status.as_deref(), Some("accepted"));
    let retired = list(&s, Some("retired")).await;
    assert_eq!(retired.len(), 1);
    assert_eq!(retired[0].0, first);
    assert!(s
        .list_behaviors(Parameters(ListBehaviorsParams {
            status: Some("bogus".into()),
            scope: None,
        }))
        .await
        .is_err());
    Ok(())
}

#[tokio::test]
async fn three_examples_are_refused_and_one_stored_with_three_says_why_it_is_not_taught(
) -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");
    let mut three = branch_rule(None, true);
    three.examples.truncate(3);
    let refused = s.record_behavior(Parameters(three)).await.unwrap().0;
    assert!(!refused.recorded);
    assert!(
        refused.problems.iter().any(|p| p.contains("at least 4")),
        "{:?}",
        refused.problems
    );

    // One stored with three, before four were required, is listed with the
    // reason it never trains.
    let legacy = antumbra_core::behavior::Spec {
        rule: "Name a branch for an issue feat/{issue}-{slug}.".into(),
        must: vec![r"feat/\d+-[a-z0-9-]+".into()],
        must_not: vec![],
        examples: (1..=3)
            .map(|i| antumbra_core::behavior::Example {
                task: format!("Issue {i}."),
                answer: format!("git switch -c feat/{i}-x"),
            })
            .collect(),
        violations: vec!["git switch -c issue-1".into()],
    };
    antumbra_store::repo::behavior::record(
        &store,
        &TenantId::new("ws:test"),
        &UserId::new("user:lily"),
        "windows",
        MemoryId::new("memory:legacy"),
        &legacy,
        antumbra_core::behavior::Status::Accepted,
        "everywhere",
        None,
        &[],
        vec![0.1; EMBED_DIM],
    )
    .await?;
    let listed = s
        .list_behaviors(Parameters(ListBehaviorsParams {
            status: Some("accepted".into()),
            scope: None,
        }))
        .await
        .unwrap()
        .0
        .behaviors;
    assert_eq!(listed.len(), 1);
    assert!(
        listed[0].problems.iter().any(|p| p.contains("one in four")),
        "{:?}",
        listed[0].problems
    );
    Ok(())
}

#[tokio::test]
async fn a_behavior_lists_how_its_last_training_went() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");
    let id = s
        .record_behavior(Parameters(branch_rule(None, true)))
        .await
        .unwrap()
        .0
        .id
        .unwrap();
    let tenant = TenantId::new("ws:test");
    let mut m = memory::get(&store, &tenant, &MemoryId::new(id.clone()))
        .await?
        .expect("stored");
    antumbra_core::behavior::mark_training(
        &mut m.evidence,
        &antumbra_core::behavior::Training {
            set: "00112233aabbccdd".into(),
            base: 0.0,
            expert: 0.5,
            learned: false,
            admitted: true,
        },
    );
    memory::upsert(&store, &m).await?;
    let listed = s
        .list_behaviors(Parameters(ListBehaviorsParams {
            status: None,
            scope: None,
        }))
        .await
        .unwrap()
        .0
        .behaviors;
    let said = listed[0].last_training.as_deref().unwrap_or_default();
    assert!(
        said.starts_with("not learned (held-out 0.50 against the base model's 0.00); its scope's expert serves the others without it"),
        "{said}"
    );
    Ok(())
}

#[tokio::test]
async fn the_tools_answer_to_the_names_they_had_before_the_us_spelling() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");
    let p = branch_rule(None, false);
    let args = serde_json::json!({
        "rule": p.rule, "must": p.must, "must_not": p.must_not,
        "examples": p.examples.iter().map(|e| serde_json::json!({"task": e.task, "answer": e.answer})).collect::<Vec<_>>(),
        "violations": p.violations,
    });
    let recorded = s.call_tool("record_behaviour", args).await.unwrap();
    let id = recorded["id"].as_str().unwrap().to_string();
    let accepted = s
        .call_tool(
            "accept_behaviour",
            serde_json::json!({ "behaviour_id": id }),
        )
        .await
        .unwrap();
    assert_eq!(accepted["status"], "accepted");
    let listed = s
        .call_tool("list_behaviours", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(listed["behaviors"].as_array().map(Vec::len), Some(1));
    assert!(s
        .call_tool("retire_behavior", serde_json::json!({ "behavior_id": id }))
        .await
        .is_ok());
    Ok(())
}

#[tokio::test]
async fn behaviors_are_private_to_the_user_who_recorded_them() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let lily = server_for(&store, "user:lily");
    let oslo = server_for(&store, "user:oslo");
    let id = lily
        .record_behavior(Parameters(branch_rule(None, false)))
        .await
        .unwrap()
        .0
        .id
        .unwrap();
    assert!(list(&oslo, None).await.is_empty());
    let attempt = oslo
        .retire_behavior(Parameters(BehaviorIdParams { behavior_id: id }))
        .await
        .unwrap()
        .0;
    assert!(!attempt.found, "another user cannot retire it");
    assert_eq!(list(&lily, Some("proposed")).await.len(), 1);
    Ok(())
}

/// A listed behavior with nothing wrong and no training yet is what most
/// of a store holds, and a client that checks results against the tool's
/// `outputSchema` refuses the whole list if the schema requires a field the
/// server left out. Every key the schema requires has to be in the result.
#[test]
fn a_behavior_with_nothing_wrong_and_no_training_matches_the_output_schema() {
    let view = crate::server::behavior::BehaviorView {
        id: "memory:b1".into(),
        rule: "Name a branch for an issue feat/{issue}-{slug}.".into(),
        scope: "everywhere".into(),
        status: "proposed".into(),
        must: vec![r"feat/\d+".into()],
        must_not: Vec::new(),
        examples: 4,
        violations: 2,
        supersedes: None,
        updated_at: "2026-10-09T00:00:00+00:00".into(),
        sources: Vec::new(),
        problems: Vec::new(),
        last_training: None,
    };
    assert_required_keys_present(&view);
}

#[tokio::test]
async fn candidates_are_what_may_state_a_rule_oldest_first_and_page_by_creation(
) -> anyhow::Result<()> {
    use crate::server::candidates::CandidatesParams;
    use antumbra_core::MemoryNetwork::{Bank, Opinion, World};

    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");
    let tenant = TenantId::new("ws:test");
    let t0 = Utc::now() - chrono::Duration::minutes(10);
    let at = |i: i64| t0 + chrono::Duration::seconds(i);
    let mem = |id: &str, net: antumbra_core::MemoryNetwork, content: &str, i: i64| {
        antumbra_core::Memory::new(id, tenant.clone(), net, content, 0.8, at(i))
            .with_embedding(vec![0.1; EMBED_DIM])
    };
    memory::upsert(
        &store,
        &mem("memory:fact", World, "surql-rs 0.28 shipped on Tuesday.", 1),
    )
    .await?;
    memory::upsert(
        &store,
        &mem(
            "memory:rule",
            World,
            "Never stack a pull request on an unmerged branch.",
            2,
        ),
    )
    .await?;
    memory::upsert(
        &store,
        &mem(
            "memory:taste",
            Opinion,
            "Short pull requests read better.",
            3,
        ),
    )
    .await?;
    memory::upsert(
        &store,
        &mem("memory:counter", Bank, "[skill-use:x] always 3", 4).volatile(true),
    )
    .await?;
    memory::upsert(
        &store,
        &mem(
            "memory:note",
            Bank,
            "Fixed the build; it should pass now.",
            5,
        ),
    )
    .await?;

    let page = s
        .behavior_candidates(Parameters(CandidatesParams {
            after: None,
            scan: Some(2),
        }))
        .await
        .unwrap()
        .0;
    assert_eq!(page.scanned, 2);
    assert!(page.more);
    let ids: Vec<&str> = page.candidates.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids,
        ["memory:rule"],
        "a fact without a rule's wording is passed over"
    );
    assert_eq!(page.candidates[0].reasons, ["never"]);
    assert_eq!(page.candidates[0].network, "world");
    let cursor = page.scanned_through.clone().expect("a cursor");
    assert_eq!(cursor, at(2).to_rfc3339());

    let rest = s
        .behavior_candidates(Parameters(CandidatesParams {
            after: Some(cursor),
            scan: None,
        }))
        .await
        .unwrap()
        .0;
    assert_eq!(rest.scanned, 3);
    assert!(!rest.more);
    let ids: Vec<&str> = rest.candidates.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids,
        ["memory:taste", "memory:note"],
        "an opinion is picked whatever it says, a volatile counter never is"
    );
    assert_eq!(rest.candidates[0].reasons, ["opinion"]);
    assert_eq!(rest.candidates[1].reasons, ["should"]);

    let end = s
        .behavior_candidates(Parameters(CandidatesParams {
            after: rest.scanned_through.clone(),
            scan: None,
        }))
        .await
        .unwrap()
        .0;
    assert_eq!(end.scanned, 0);
    assert!(
        end.scanned_through.is_none(),
        "nothing read: the end of the store"
    );
    assert!(!end.more);

    assert!(
        s.behavior_candidates(Parameters(CandidatesParams {
            after: Some("yesterday".into()),
            scan: None,
        }))
        .await
        .is_err(),
        "a cursor that is not a time is refused"
    );
    Ok(())
}

#[tokio::test]
async fn a_behavior_cites_the_memories_it_came_from_and_they_are_no_longer_candidates(
) -> anyhow::Result<()> {
    use crate::server::candidates::{CandidatesOut, CandidatesParams};

    async fn fresh(s: &McpServer) -> CandidatesOut {
        s.behavior_candidates(Parameters(CandidatesParams {
            after: None,
            scan: None,
        }))
        .await
        .unwrap()
        .0
    }

    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");
    let said = antumbra_core::Memory::new(
        "memory:said",
        TenantId::new("ws:test"),
        antumbra_core::MemoryNetwork::Opinion,
        "Name branches feat/{issue}-{slug}, never feature/.",
        0.9,
        Utc::now(),
    )
    .with_embedding(vec![0.1; EMBED_DIM]);
    memory::upsert(&store, &said).await?;
    assert_eq!(fresh(&s).await.candidates.len(), 1);

    let mut p = branch_rule(None, false);
    p.sources = vec!["memory:said".into(), " ".into(), "memory:said".into()];
    let recorded = s.record_behavior(Parameters(p)).await.unwrap().0;
    assert!(recorded.recorded, "{:?}", recorded.problems);
    let listed = s
        .list_behaviors(Parameters(ListBehaviorsParams {
            status: None,
            scope: None,
        }))
        .await
        .unwrap()
        .0
        .behaviors;
    assert_eq!(
        listed[0].sources,
        ["memory:said"],
        "cited once, blanks dropped"
    );

    let after = fresh(&s).await;
    assert!(
        after.candidates.is_empty(),
        "{:?}",
        after.candidates.iter().map(|c| &c.id).collect::<Vec<_>>()
    );
    assert_eq!(
        after.scanned, 2,
        "the memory and the behavior itself were read, and neither is offered"
    );
    Ok(())
}
