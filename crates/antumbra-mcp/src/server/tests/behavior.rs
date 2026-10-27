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
        listed[0]
            .problems
            .iter()
            .any(|p| p.contains("one in three")),
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
            held: 2,
            worse: 0.2,
        },
    );
    antumbra_core::behavior::mark_audit(
        &mut m.evidence,
        &antumbra_core::behavior::Audit {
            set: "00112233aabbccdd".into(),
            fresh: 2,
            base: 0,
            expert: 1,
            worse: Some(0.74),
        },
    );
    antumbra_core::behavior::missed::mark(
        &mut m.evidence,
        Some(&antumbra_core::behavior::missed::Missed::new(
            "Start issue #7.",
            "git switch -c issue-7",
        )),
    );
    memory::upsert(&store, &m).await?;
    let out = s
        .list_behaviors(Parameters(ListBehaviorsParams {
            status: None,
            scope: None,
        }))
        .await
        .unwrap()
        .0;
    assert!(
        out.tuned_confidence.is_none() && out.tuning.is_empty(),
        "no reading yet: the recipe's confidence stands"
    );
    let listed = out.behaviors;
    let said = listed[0].last_training.as_deref().unwrap_or_default();
    assert!(
        said.starts_with("not learned (held-out 0.50 against the base model's 0.00); its scope's expert serves the others without it"),
        "{said}"
    );
    assert_eq!(
        listed[0].last_audit.as_deref(),
        Some("audited on 2 later examples: the expert passed 1, the base model 0")
    );
    let missed = listed[0].missed.as_ref().expect("what it was missed on");
    assert_eq!(
        (missed.task.as_str(), missed.answer.as_str()),
        ("Start issue #7.", "git switch -c issue-7")
    );
    Ok(())
}

/// [`branch_rule`] as the JSON arguments an agent sends, its `violations`
/// written as given.
fn branch_rule_args(violations: serde_json::Value) -> serde_json::Value {
    let p = branch_rule(None, false);
    serde_json::json!({
        "rule": p.rule, "must": p.must, "must_not": p.must_not,
        "examples": p.examples.iter().map(|e| serde_json::json!({"task": e.task, "answer": e.answer})).collect::<Vec<_>>(),
        "violations": violations,
    })
}

fn problems_of(out: &serde_json::Value) -> Vec<&str> {
    out["problems"]
        .as_array()
        .map(|ps| ps.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default()
}

/// The first call an agent made from the capture hook's old wording, which
/// asked for "the rule, a check and examples": a `check` the tool has no field
/// for, one example, no violations. It is told everything it lacks in one
/// answer, where deserialization named one missing field per call.
#[tokio::test]
async fn a_call_that_leaves_parts_out_is_told_everything_it_lacks_at_once() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");
    let out = s
        .call_tool(
            "record_behavior",
            serde_json::json!({
                "rule": "Name a branch for an issue feat/{issue}-{slug}.",
                "check": { "must": [r"feat/\d+-[a-z0-9-]+"] },
                "examples": [{ "task": "Start issue #42.", "answer": "git checkout -b feat/42-add-login-page" }],
            }),
        )
        .await
        .unwrap();
    assert_eq!(out["recorded"], false);
    let problems = problems_of(&out);
    for lacking in ["`must` pattern", "1 example(s)", "`violations`"] {
        assert!(
            problems.iter().any(|p| p.contains(lacking)),
            "{lacking}: {problems:?}"
        );
    }
    assert!(list(&s, None).await.is_empty(), "nothing written");

    let empty = s
        .call_tool("record_behavior", serde_json::json!({}))
        .await
        .unwrap();
    assert!(
        problems_of(&empty).contains(&"the rule is empty"),
        "{empty}"
    );
    Ok(())
}

/// An agent that writes a violation the way it wrote the examples, as a
/// `{task, answer}` pair, is read: the answer is the violation, and the check
/// is held to it.
#[tokio::test]
async fn a_violation_written_as_an_example_is_read_as_its_answer() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");

    let passed = s
        .call_tool(
            "record_behavior",
            branch_rule_args(serde_json::json!([
                { "task": "Start issue #9.", "answer": "git checkout -b feat/9-anything" },
            ])),
        )
        .await
        .unwrap();
    assert_eq!(passed["recorded"], false);
    assert_eq!(
        problems_of(&passed),
        ["the check passes the violating answer `git checkout -b feat/9-anything`"]
    );

    let recorded = s
        .call_tool(
            "record_behavior",
            branch_rule_args(serde_json::json!([
                { "task": "Start issue #42, Add login page.", "answer": "git checkout -b feature/add-login-page" },
                "git checkout -b issue-42",
            ])),
        )
        .await
        .unwrap();
    assert_eq!(recorded["recorded"], true, "{recorded}");
    let listed = s
        .call_tool("list_behaviors", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(listed["behaviors"][0]["violations"], 2);
    Ok(())
}

/// Read leniently, the spec's fields are still what the schema asks for: an
/// agent is told to send all four, none offers a default to lean on, and a
/// violation is a string.
#[test]
fn the_schema_still_asks_for_the_rule_the_check_the_examples_and_the_violations() {
    let tools = McpServer::tool_router().list_all();
    let tool = tools
        .iter()
        .find(|t| t.name == "record_behavior")
        .expect("listed");
    let schema = serde_json::to_value(&tool.input_schema).unwrap();
    let spec = ["rule", "must", "examples", "violations"];
    assert_eq!(schema["required"], serde_json::json!(spec));
    for field in spec {
        let prop = &schema["properties"][field];
        assert!(prop.get("default").is_none(), "{field}: {prop}");
    }
    assert_eq!(
        schema["properties"]["violations"]["items"]["type"],
        "string"
    );
    assert_eq!(
        schema["properties"]["must_not"]["default"],
        serde_json::json!([]),
        "a field that may be left out keeps its default"
    );
}

#[tokio::test]
async fn the_tools_answer_to_the_names_they_had_before_the_us_spelling() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let s = server_for(&store, "user:lily");
    let args = branch_rule_args(serde_json::json!(branch_rule(None, false).violations));
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
        last_audit: None,
        missed: None,
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
            "Session note: the fix landed. Deploy it yourself; do not leave it to the user.",
            5,
        ),
    )
    .await?;
    memory::upsert(
        &store,
        &mem(
            "memory:state",
            Bank,
            "The gate must never fire twice, and the dispatcher never reads the claim.",
            6,
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
        "a fact without a rule's shape is passed over"
    );
    assert_eq!(page.candidates[0].reasons, ["opens:never"]);
    assert_eq!(
        page.candidates[0].hits,
        ["Never stack a pull request on an unmerged branch"]
    );
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
    assert_eq!(rest.scanned, 4);
    assert!(!rest.more);
    let ids: Vec<&str> = rest.candidates.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids,
        ["memory:taste", "memory:note"],
        "an opinion is picked whatever it says; a volatile counter and a note \
         whose negations sit mid-sentence are not"
    );
    assert_eq!(rest.candidates[0].reasons, ["opinion"]);
    assert_eq!(
        rest.candidates[1].reasons,
        ["opens:deploy", "opens:do not"],
        "`yourself` is the condition that makes `Deploy it yourself` a rule"
    );
    assert_eq!(
        rest.candidates[1].hits,
        ["Deploy it yourself", "do not leave it to the user"]
    );

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
