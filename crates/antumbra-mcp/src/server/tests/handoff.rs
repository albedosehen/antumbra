//! Handoffs: left for a machine, announced there until done, readable
//! after, and private to the user who left them.

use super::*;

fn server_for(store: &Store, user: &str, host: &str) -> McpServer {
    McpServer::new(
        store.clone(),
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new(user),
        host.into(),
        CompartmentId::new(format!("comp:test:{user}:default")),
        None,
    )
}

async fn leave(s: &McpServer, content: &str, for_host: Option<&str>) -> LeftHandoffOut {
    s.leave_handoff(Parameters(LeaveHandoffParams {
        content: content.into(),
        for_host: for_host.map(str::to_string),
        from_host: Some("windows".into()),
    }))
    .await
    .unwrap()
    .0
}

async fn waiting(s: &McpServer, host: &str, include_done: bool) -> HandoffsOut {
    s.handoffs(Parameters(HandoffsParams {
        host: Some(host.into()),
        include_done: Some(include_done),
        full: None,
    }))
    .await
    .unwrap()
    .0
}

/// A handoff for one machine is announced there and nowhere else, one for any
/// machine everywhere, and once marked done it stops being announced but stays
/// readable with who dealt with it.
#[tokio::test]
async fn a_handoff_waits_where_it_is_addressed_until_it_is_done() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = server_for(&store, "user:a", "kuskokwim");
    let for_gpu = leave(
        &s,
        "Rerun the relevance probe\nwith 2048 tokens",
        Some(" Kuskokwim "),
    )
    .await;
    assert_eq!(for_gpu.for_host, "kuskokwim");
    let for_any = leave(&s, "Check the dashboard", None).await;
    assert_eq!(for_any.for_host, "any");
    assert!(for_any.registered_device, "any is always deliverable");

    let here = waiting(&s, "kuskokwim", false).await;
    let ids: Vec<&str> = here.handoffs.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        ids,
        [for_any.id.as_str(), for_gpu.id.as_str()],
        "newest first"
    );
    assert_eq!(here.handoffs[1].title, "Rerun the relevance probe");
    assert_eq!(here.handoffs[1].from_host.as_deref(), Some("windows"));
    let said = here.announcement.expect("announced");
    assert!(
        said.starts_with("2 handoffs waiting for this machine (kuskokwim):"),
        "{said}"
    );

    let elsewhere = waiting(&s, "shaman", false).await;
    assert_eq!(elsewhere.handoffs.len(), 1, "only the one for any machine");

    let done = s
        .complete_handoff(Parameters(CompleteHandoffParams {
            handoff_id: for_gpu.id.clone(),
            host: Some("kuskokwim".into()),
        }))
        .await
        .unwrap()
        .0;
    assert!(done.found && !done.already_done);
    let after = waiting(&s, "kuskokwim", false).await;
    assert_eq!(after.handoffs.len(), 1);
    assert!(after.announcement.unwrap().starts_with("1 handoff waiting"));

    let history = waiting(&s, "kuskokwim", true).await;
    let kept = history
        .handoffs
        .iter()
        .find(|h| h.id == for_gpu.id)
        .expect("still readable");
    assert_eq!(kept.done_by.as_deref(), Some("kuskokwim"));
    assert!(kept.done_at.is_some());

    let again = s
        .complete_handoff(Parameters(CompleteHandoffParams {
            handoff_id: for_gpu.id,
            host: None,
        }))
        .await
        .unwrap()
        .0;
    assert!(again.found && again.already_done);
}

/// Nothing waiting announces nothing, an unknown id is not found, and a memory
/// that is not a handoff cannot be completed as one.
#[tokio::test]
async fn nothing_waiting_says_nothing_and_only_handoffs_complete() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = server_for(&store, "user:a", "windows");
    let empty = waiting(&s, "windows", false).await;
    assert!(empty.handoffs.is_empty() && empty.announcement.is_none());

    let missing = s
        .complete_handoff(Parameters(CompleteHandoffParams {
            handoff_id: "memory:nope".into(),
            host: None,
        }))
        .await
        .unwrap()
        .0;
    assert!(!missing.found);

    let plain = s
        .call_tool(
            "store_memory",
            serde_json::json!({ "content": "an ordinary note" }),
        )
        .await
        .unwrap();
    let not_one = s
        .complete_handoff(Parameters(CompleteHandoffParams {
            handoff_id: plain["id"].as_str().unwrap().into(),
            host: None,
        }))
        .await
        .unwrap()
        .0;
    assert!(!not_one.found, "only a handoff can be completed as one");

    let Err(refused) = s
        .leave_handoff(Parameters(LeaveHandoffParams {
            content: "   ".into(),
            for_host: None,
            from_host: None,
        }))
        .await
    else {
        panic!("an empty handoff was accepted");
    };
    assert!(
        refused.message.contains("needs content"),
        "{}",
        refused.message
    );
}

/// A host nobody registered is still deliverable, and the answer says so, with
/// the registered devices beside it, so a typo shows.
#[tokio::test]
async fn an_unregistered_host_is_accepted_and_flagged() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = server_for(&store, "user:a", "windows");
    let left = leave(&s, "For a box that never registered", Some("kuskokwm")).await;
    assert_eq!(left.for_host, "kuskokwm");
    assert!(!left.registered_device);
}

/// One user's handoffs never reach another's sessions, even on the same host.
#[tokio::test]
async fn a_handoff_is_private_to_the_user_who_left_it() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let a = server_for(&store, "user:a", "kuskokwim");
    let b = server_for(&store, "user:b", "kuskokwim");
    leave(&a, "For my own GPU box", Some("kuskokwim")).await;
    assert_eq!(waiting(&a, "kuskokwim", false).await.handoffs.len(), 1);
    assert!(waiting(&b, "kuskokwim", true).await.handoffs.is_empty());
}

/// All three are reachable over the REST dispatcher, which is how the
/// session-start hook asks.
#[tokio::test]
async fn the_rest_dispatcher_reaches_the_handoff_tools() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = server_for(&store, "user:a", "windows");
    let left = s
        .call_tool(
            "leave_handoff",
            serde_json::json!({ "content": "From the dispatcher", "for_host": "windows" }),
        )
        .await
        .unwrap();
    let listed = s
        .call_tool("handoffs", serde_json::json!({ "host": "windows" }))
        .await
        .unwrap();
    assert!(listed["announcement"]
        .as_str()
        .unwrap()
        .contains("From the dispatcher"));
    let done = s
        .call_tool(
            "complete_handoff",
            serde_json::json!({ "handoff_id": left["id"] }),
        )
        .await
        .unwrap();
    assert_eq!(done["found"], true);
}
