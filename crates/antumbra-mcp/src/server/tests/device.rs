//! Devices: a session names the machine it runs on, the fabric
//! lists it, and a row a server wrote about its own machine is never
//! overwritten by a session that cannot see that machine's hardware.

use super::*;
use crate::server::device::{DevicesOut, RegisterDeviceParams, RegisteredDeviceOut};
use antumbra_core::DeviceProfile;
use antumbra_store::repo::device as store_device;

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

async fn register(s: &McpServer, host: &str) -> RegisteredDeviceOut {
    s.register_device(Parameters(RegisterDeviceParams { host: host.into() }))
        .await
        .unwrap()
        .0
}

async fn devices(s: &McpServer) -> DevicesOut {
    s.devices().await.unwrap().0
}

/// A laptop talking to a hosted hub is listed once its session names it, as a
/// memory node, under the name its handoffs are addressed to.
#[tokio::test]
async fn a_session_names_its_machine_and_the_fabric_lists_it() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let hub = server_for(&store, "user:a", "kuskokwim");

    let mac = register(&hub, " Mac ").await;
    assert_eq!(mac.host, "mac", "the same normalization handoffs use");
    assert!(mac.registered && mac.note.is_none());
    assert_eq!(mac.role, "memory");

    let listed = devices(&hub).await;
    let [only] = listed.devices.as_slice() else {
        panic!("one machine named, one listed: {}", listed.devices.len());
    };
    assert_eq!(only.host, "mac");
    assert_eq!(only.role, "memory");
    assert_eq!(only.backend, antumbra_core::CLIENT_BACKEND);
    assert!(only.vram_mib.is_none());
    assert!(listed.trainer.is_none(), "a laptop is nobody's trainer");

    // A handoff addressed to it now says it reached a registered machine.
    let left = hub
        .leave_handoff(Parameters(LeaveHandoffParams {
            content: "For the laptop".into(),
            for_host: Some("mac".into()),
            from_host: Some("windows".into()),
        }))
        .await
        .unwrap()
        .0;
    assert!(left.registered_device, "{:?}", left.devices);
}

/// Every session start names the machine again: that refreshes when it was
/// last seen and never adds a second row.
#[tokio::test]
async fn naming_a_machine_again_refreshes_it_and_adds_no_row() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let hub = server_for(&store, "user:a", "kuskokwim");
    register(&hub, "mac").await;
    let first = devices(&hub).await.devices[0].last_seen.clone();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    register(&hub, "MAC").await;

    let listed = devices(&hub).await;
    assert_eq!(listed.devices.len(), 1, "one machine, one row");
    assert!(listed.devices[0].last_seen > first, "seen again");
}

/// A row a server wrote about its own machine stays as it is: a session
/// naming that machine cannot demote a trainer to a memory node.
#[tokio::test]
async fn a_server_registered_machine_is_left_as_its_server_wrote_it() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let (tenant, user) = (TenantId::new("ws:test"), UserId::new("user:a"));
    let rig = DeviceProfile::detected(
        tenant.clone(),
        user.clone(),
        "kuskokwim",
        "cuda",
        Some(24_576),
        Utc::now(),
    );
    store_device::upsert(&store, &rig).await.unwrap();
    let hub = server_for(&store, "user:a", "shaman");

    let named = register(&hub, "Kuskokwim").await;
    assert!(!named.registered);
    assert_eq!(named.role, "genesis");
    assert!(named.note.unwrap().contains("registers it itself"));

    let listed = devices(&hub).await;
    assert_eq!(listed.devices.len(), 1);
    assert_eq!(listed.devices[0].backend, "cuda");
    assert_eq!(listed.devices[0].vram_mib, Some(24_576));
    assert_eq!(listed.trainer.as_deref(), Some("kuskokwim"));
    let trainer = store_device::genesis_for_user(&store, &tenant, &user)
        .await
        .unwrap();
    assert_eq!(trainer.map(|d| d.host), Some("kuskokwim".to_string()));
}

/// A session on the server's own machine does not write a client row over the
/// server's: the server registers itself.
#[tokio::test]
async fn the_servers_own_machine_is_not_written_as_a_client() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let hub = server_for(&store, "user:a", "kuskokwim");
    let named = register(&hub, "kuskokwim").await;
    assert!(!named.registered);
    assert!(named.note.unwrap().contains("server's own machine"));
    assert!(devices(&hub).await.devices.is_empty());
}

/// A name that stands for no machine in particular is refused: `local` is what
/// every machine without ANTUMBRA_HOST_ID says, and `any` is a handoff address.
#[tokio::test]
async fn a_name_that_is_no_machine_is_refused() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let hub = server_for(&store, "user:a", "kuskokwim");
    for unnamed in ["local", " LOCAL ", "any", "", "   "] {
        let Err(refused) = hub
            .register_device(Parameters(RegisterDeviceParams {
                host: unnamed.into(),
            }))
            .await
        else {
            panic!("{unnamed:?} was registered");
        };
        assert!(
            refused.message.contains("ANTUMBRA_HOST_ID"),
            "{}",
            refused.message
        );
    }
    assert!(devices(&hub).await.devices.is_empty());
}

/// Under the record session a hosted server runs each identity on, a user
/// writes and lists only their own machines; two people naming the same
/// machine get a row each.
#[tokio::test]
async fn each_user_names_and_sees_only_their_own_machines_under_signin() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let tenant = TenantId::new("ws:test");
    let (ua, ub) = (UserId::new("user:a"), UserId::new("user:b"));
    crate::provision_identity(&store, &tenant, &ua)
        .await
        .unwrap();
    crate::provision_identity(&store, &tenant, &ub)
        .await
        .unwrap();
    let a = server_for(&store, "user:a", "kuskokwim");
    let b = server_for(&store, "user:b", "kuskokwim");

    store.signin(&tenant, &ua).await.unwrap();
    assert!(register(&a, "mac").await.registered);
    store.signin(&tenant, &ub).await.unwrap();
    assert!(register(&b, "mac").await.registered);
    assert!(register(&b, "chromebook").await.registered);

    let theirs: Vec<String> = devices(&b)
        .await
        .devices
        .into_iter()
        .map(|d| d.host)
        .collect();
    assert_eq!(theirs.len(), 2);
    assert!(theirs.contains(&"chromebook".to_string()));
    store.signin(&tenant, &ua).await.unwrap();
    let mine: Vec<String> = devices(&a)
        .await
        .devices
        .into_iter()
        .map(|d| d.host)
        .collect();
    assert_eq!(mine, ["mac"]);
}

/// Both are reachable over the REST dispatcher, which is how the session-start
/// hook and `antumbra setup` call them.
#[tokio::test]
async fn the_rest_dispatcher_reaches_the_device_tools() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let hub = server_for(&store, "user:a", "kuskokwim");
    let named = hub
        .call_tool("register_device", serde_json::json!({ "host": "mac" }))
        .await
        .unwrap();
    assert_eq!(named["registered"], true);
    assert_eq!(named["role"], "memory");
    let listed = hub
        .call_tool("devices", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(listed["devices"][0]["host"], "mac");
    assert!(listed.get("trainer").is_none());
}
