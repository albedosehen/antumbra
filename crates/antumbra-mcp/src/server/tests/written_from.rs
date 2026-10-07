//! Which machine wrote a memory (#193): a write is stamped with the machine
//! its call names, two machines sharing one identity's server stamp their own
//! names, and a read can say and filter by it.

use super::*;
use crate::server::device::DevicesOut;

/// A hosted server, named as a hub is: what a call that names no machine is
/// stamped with.
fn hub(store: &Store) -> McpServer {
    McpServer::new(
        store.clone(),
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:a"),
        "Kuskokwim".into(),
        CompartmentId::new("comp:test:user:a:default"),
        None,
    )
}

async fn store(s: &McpServer, content: &str) -> String {
    s.store_memory(Parameters(StoreParams {
        content: content.into(),
        network: "world".into(),
        confidence: None,
        evidence: None,
        volatile: None,
        compartment: None,
        provenance: None,
    }))
    .await
    .unwrap()
    .0
    .id
}

async fn listed(s: &McpServer, host: Option<&str>, limit: Option<u32>) -> Vec<MemoryView> {
    s.list_memories(Parameters(ListParams {
        host: host.map(str::to_string),
        network: None,
        limit,
        offset: None,
    }))
    .await
    .unwrap()
    .0
    .memories
}

fn host_of(views: &[MemoryView], id: &str) -> Option<String> {
    views
        .iter()
        .find(|v| v.id == id)
        .and_then(|v| v.author_host.clone())
}

/// The acceptance case: a memory stored from a session that names `mac` reads
/// back as written from `mac`, not from the hub that took it. A call that names
/// no machine is stamped with the server's own name, normalized as host names
/// are everywhere.
#[tokio::test]
async fn a_memory_is_stamped_with_the_machine_its_call_names() {
    let store_ = Store::connect_memory(EMBED_DIM).await.unwrap();
    let hub = hub(&store_);
    let from_mac = store(&hub.called_from("mac".into()), "mac wrote this").await;
    let unnamed = store(&hub, "nobody said where this came from").await;

    let all = listed(&hub, None, None).await;
    assert_eq!(host_of(&all, &from_mac).as_deref(), Some("mac"));
    assert_eq!(host_of(&all, &unnamed).as_deref(), Some("kuskokwim"));
}

/// Two machines sharing one identity share the server cached for it; each
/// call's copy stamps its own machine, and the shared server keeps none.
#[tokio::test]
async fn two_machines_sharing_one_server_stamp_their_own_names() {
    let store_ = Store::connect_memory(EMBED_DIM).await.unwrap();
    let shared = hub(&store_);
    let mac = store(&shared.called_from("mac".into()), "from the laptop").await;
    let windows = store(&shared.called_from("windows".into()), "from the desk").await;

    let all = listed(&shared, None, None).await;
    assert_eq!(host_of(&all, &mac).as_deref(), Some("mac"));
    assert_eq!(host_of(&all, &windows).as_deref(), Some("windows"));
    let said = shared.devices().await.unwrap().0;
    assert_eq!(said.this_device, "kuskokwim");
    assert!(!said.named_by_client, "the cached server names no machine");
}

/// `list_memories {host}` is one machine's, in either case, paged or whole;
/// what the hub stamped is the hub's, and a memory with no stamp is nobody's.
#[tokio::test]
async fn list_memories_returns_only_the_named_machines() {
    let store_ = Store::connect_memory(EMBED_DIM).await.unwrap();
    let hub = hub(&store_);
    let win_1 = store(&hub.called_from("windows".into()), "windows first").await;
    store(&hub.called_from("mac".into()), "mac between").await;
    let win_2 = store(&hub.called_from("windows".into()), "windows last").await;
    store(&hub, "the hub's own").await;
    let old = Memory::new(
        "memory:before-stamps",
        TenantId::new("ws:test"),
        MemoryNetwork::World,
        "written before writes were stamped",
        0.6,
        Utc::now(),
    );
    memory::upsert(&store_, &old).await.unwrap();

    let ids = |views: Vec<MemoryView>| views.into_iter().map(|v| v.id).collect::<Vec<_>>();
    // Paged: newest first, cut by the engine.
    assert_eq!(
        ids(listed(&hub, Some("Windows"), Some(5)).await),
        [win_2.clone(), win_1.clone()]
    );
    // Whole: every one of them, in no set order.
    let mut whole = ids(listed(&hub, Some(" windows "), None).await);
    whole.sort();
    let mut expected = vec![win_1, win_2];
    expected.sort();
    assert_eq!(whole, expected);
    assert_eq!(listed(&hub, Some("kuskokwim"), None).await.len(), 1);
    assert!(listed(&hub, Some("linux"), Some(5)).await.is_empty());
    // A blank filter is no filter.
    assert_eq!(listed(&hub, Some("  "), None).await.len(), 5);
    let all = listed(&hub, None, None).await;
    assert_eq!(host_of(&all, "memory:before-stamps"), None);
}

/// `recall_memories {host}` keeps only what that machine wrote.
#[tokio::test]
async fn recall_can_be_one_machines() {
    let store_ = Store::connect_memory(EMBED_DIM).await.unwrap();
    let hub = hub(&store_);
    let mac = store(
        &hub.called_from("mac".into()),
        "the deno runtime on the laptop",
    )
    .await;
    store(
        &hub.called_from("windows".into()),
        "the deno runtime on the desk",
    )
    .await;

    let recalled = |host: Option<&str>| {
        hub.recall_memories(Parameters(RecallParams {
            query: "deno runtime".into(),
            top_k: Some(5),
            network: None,
            repo: None,
            branch: None,
            floor: None,
            full: None,
            host: host.map(str::to_string),
        }))
    };
    let mine = recalled(Some("MAC")).await.unwrap().0.memories;
    assert_eq!(mine.len(), 1);
    assert_eq!(mine[0].id, mac);
    assert_eq!(mine[0].author_host.as_deref(), Some("mac"));
    assert_eq!(recalled(None).await.unwrap().0.memories.len(), 2);
}

/// A handoff left without `from_host`, and the handoffs asked for without
/// `host`, default to the machine the call names rather than the hub.
#[tokio::test]
async fn handoffs_default_to_the_calling_machine() {
    let store_ = Store::connect_memory(EMBED_DIM).await.unwrap();
    let hub = hub(&store_);
    let mac = hub.called_from("mac".into());
    let left = mac
        .leave_handoff(Parameters(LeaveHandoffParams {
            content: "For the desk".into(),
            for_host: Some("windows".into()),
            from_host: None,
        }))
        .await
        .unwrap()
        .0;
    let waiting = hub
        .called_from("windows".into())
        .handoffs(Parameters(HandoffsParams {
            host: None,
            include_done: None,
            full: None,
        }))
        .await
        .unwrap()
        .0;
    assert_eq!(waiting.host, "windows");
    let [only] = waiting.handoffs.as_slice() else {
        panic!("one handoff waits for windows: {}", waiting.handoffs.len());
    };
    assert_eq!(only.id, left.id);
    assert_eq!(only.from_host.as_deref(), Some("mac"));
}

/// `devices` says which machine this session's writes are stamped with, and
/// whether its client named it: what `antumbra setup check` reports.
#[tokio::test]
async fn devices_says_which_machine_writes_are_stamped_with() {
    let store_ = Store::connect_memory(EMBED_DIM).await.unwrap();
    let said: DevicesOut = hub(&store_)
        .called_from("mac".into())
        .devices()
        .await
        .unwrap()
        .0;
    assert_eq!(said.this_device, "mac");
    assert!(said.named_by_client);
}

/// A name is one machine's, normalized as handoffs compare names; what names
/// no machine in particular, or is not a name, stamps nothing.
#[test]
fn only_a_machines_name_is_stamped() {
    assert_eq!(machine_name(" Mac ").as_deref(), Some("mac"));
    assert_eq!(
        machine_name("shons-macbook.local").as_deref(),
        Some("shons-macbook.local")
    );
    for none in ["", "   ", "any", "ANY", "local", " Local "] {
        assert_eq!(machine_name(none), None, "{none:?}");
    }
    assert_eq!(machine_name(&"a".repeat(128)).map(|n| n.len()), Some(128));
    assert_eq!(machine_name(&"a".repeat(129)), None);
    assert_eq!(machine_name("mac\u{7}"), None);
}
