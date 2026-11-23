//! Devices: a session names the machine it runs on, the fabric
//! lists it, and a row a server wrote about its own machine is never
//! overwritten by a session that cannot see that machine's hardware.

use super::*;
use crate::server::device::{DevicesOut, RegisterDeviceParams, RegisteredDeviceOut};
use antumbra_core::DeviceProfile;
use antumbra_store::repo::device as store_device;
use antumbra_store::testing::mem_store;

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

async fn register(s: &McpServer, host: &str) -> anyhow::Result<RegisteredDeviceOut> {
    Ok(s.register_device(Parameters(RegisterDeviceParams {
        host: host.into(),
        os: None,
    }))
    .await?
    .0)
}

async fn devices(s: &McpServer) -> anyhow::Result<DevicesOut> {
    Ok(s.devices().await?.0)
}

/// A laptop talking to a hosted hub is listed once its session names it, as a
/// memory node, under the name its handoffs are addressed to.
#[tokio::test]
async fn a_session_names_its_machine_and_the_fabric_lists_it() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let hub = server_for(&store, "user:a", "kestrel");

    let mac = register(&hub, " Mac ").await?;
    assert_eq!(mac.host, "mac", "the same normalization handoffs use");
    assert!(mac.registered && mac.note.is_none());
    assert_eq!(mac.role, "memory");

    let listed = devices(&hub).await?;
    let [only] = listed.devices.as_slice() else {
        anyhow::bail!("one machine named, one listed: {}", listed.devices.len());
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
        .await?
        .0;
    assert!(left.registered_device, "{:?}", left.devices);
    Ok(())
}

/// Every session start names the machine again: that refreshes when it was
/// last seen and never adds a second row.
#[tokio::test]
async fn naming_a_machine_again_refreshes_it_and_adds_no_row() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let hub = server_for(&store, "user:a", "kestrel");
    register(&hub, "mac").await?;
    let first = devices(&hub).await?.devices[0].last_seen.clone();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    register(&hub, "MAC").await?;

    let listed = devices(&hub).await?;
    assert_eq!(listed.devices.len(), 1, "one machine, one row");
    assert!(listed.devices[0].last_seen > first, "seen again");
    Ok(())
}

/// A row a server wrote about its own machine stays as it is: a session
/// naming that machine cannot demote a trainer to a memory node.
#[tokio::test]
async fn a_server_registered_machine_is_left_as_its_server_wrote_it() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let (tenant, user) = (TenantId::new("ws:test"), UserId::new("user:a"));
    let rig = DeviceProfile::detected(
        tenant.clone(),
        user.clone(),
        "kestrel",
        "cuda",
        Some(24_576),
        Utc::now(),
    );
    store_device::upsert(&store, &rig).await?;
    let hub = server_for(&store, "user:a", "sparrow");

    let named = register(&hub, "Kestrel").await?;
    assert!(!named.registered);
    assert_eq!(named.role, "genesis");
    let Some(note) = named.note else {
        anyhow::bail!("naming a server-registered machine came with no note");
    };
    assert!(note.contains("registers it itself"));

    let listed = devices(&hub).await?;
    assert_eq!(listed.devices.len(), 1);
    assert_eq!(listed.devices[0].backend, "cuda");
    assert_eq!(listed.devices[0].vram_mib, Some(24_576));
    assert_eq!(listed.trainer.as_deref(), Some("kestrel"));
    let trainer = store_device::genesis_for_user(&store, &tenant, &user).await?;
    assert_eq!(trainer.map(|d| d.host), Some("kestrel".to_string()));
    Ok(())
}

/// A session on the server's own machine does not write a client row over the
/// server's: the server registers itself.
#[tokio::test]
async fn the_servers_own_machine_is_not_written_as_a_client() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let hub = server_for(&store, "user:a", "kestrel");
    let named = register(&hub, "kestrel").await?;
    assert!(!named.registered);
    let Some(note) = named.note else {
        anyhow::bail!("naming the server's own machine came with no note");
    };
    assert!(note.contains("server's own machine"));
    assert!(devices(&hub).await?.devices.is_empty());
    Ok(())
}

/// A name that stands for no machine in particular is refused: `local` is what
/// every machine without ANTUMBRA_HOST_ID says, and `any` is a handoff address.
#[tokio::test]
async fn a_name_that_is_no_machine_is_refused() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let hub = server_for(&store, "user:a", "kestrel");
    for unnamed in ["local", " LOCAL ", "any", "", "   "] {
        let Err(refused) = hub
            .register_device(Parameters(RegisterDeviceParams {
                host: unnamed.into(),
                os: None,
            }))
            .await
        else {
            anyhow::bail!("{unnamed:?} was registered");
        };
        assert!(
            refused.message.contains("ANTUMBRA_HOST_ID"),
            "{}",
            refused.message
        );
    }
    assert!(devices(&hub).await?.devices.is_empty());
    Ok(())
}

/// Under the record session a hosted server runs each identity on, a user
/// writes and lists only their own machines; two people naming the same
/// machine get a row each.
#[tokio::test]
async fn each_user_names_and_sees_only_their_own_machines_under_signin() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let tenant = TenantId::new("ws:test");
    let (ua, ub) = (UserId::new("user:a"), UserId::new("user:b"));
    crate::provision_identity(&store, &tenant, &ua).await?;
    crate::provision_identity(&store, &tenant, &ub).await?;
    let a = server_for(&store, "user:a", "kestrel");
    let b = server_for(&store, "user:b", "kestrel");

    store.signin(&tenant, &ua).await?;
    assert!(register(&a, "mac").await?.registered);
    store.signin(&tenant, &ub).await?;
    assert!(register(&b, "mac").await?.registered);
    assert!(register(&b, "chromebook").await?.registered);

    let theirs: Vec<String> = devices(&b)
        .await?
        .devices
        .into_iter()
        .map(|d| d.host)
        .collect();
    assert_eq!(theirs.len(), 2);
    assert!(theirs.contains(&"chromebook".to_string()));
    store.signin(&tenant, &ua).await?;
    let mine: Vec<String> = devices(&a)
        .await?
        .devices
        .into_iter()
        .map(|d| d.host)
        .collect();
    assert_eq!(mine, ["mac"]);
    Ok(())
}

/// Both are reachable over the REST dispatcher, which is how the session-start
/// hook and `antumbra setup` call them.
#[tokio::test]
async fn the_rest_dispatcher_reaches_the_device_tools() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let hub = server_for(&store, "user:a", "kestrel");
    let named = hub
        .call_tool("register_device", serde_json::json!({ "host": "mac" }))
        .await?;
    assert_eq!(named["registered"], true);
    assert_eq!(named["role"], "memory");
    let listed = hub.call_tool("devices", serde_json::json!({})).await?;
    assert_eq!(listed["devices"][0]["host"], "mac");
    assert!(listed.get("trainer").is_none());
    Ok(())
}

/// #597: a call moves the last-seen of the machine it came from and nothing
/// else of its row; a second call inside the throttle writes nothing; a
/// call from a machine with no row makes none.
#[tokio::test]
async fn a_call_moves_its_machines_last_seen_and_is_throttled() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let hub = server_for(&store, "user:a", "hub");
    let (tenant, user) = (TenantId::new("ws:test"), UserId::new("user:a"));
    let then = Utc::now() - chrono::Duration::hours(11);
    store_device::upsert(
        &store,
        &DeviceProfile::client(tenant.clone(), user.clone(), "laptop", then),
    )
    .await?;
    let laptop = || async {
        anyhow::Ok(
            store_device::list_for_user(&store, &tenant, &user)
                .await?
                .into_iter()
                .find(|d| d.host == "laptop"),
        )
    };

    // A call that names no machine is the server's own; the laptop stays.
    hub.call_tool("devices", serde_json::json!({})).await?;
    assert_eq!(laptop().await?.map(|d| d.last_seen()), Some(then));
    // A machine with no row is given none by calling.
    hub.called_from("desktop".into())
        .call_tool("devices", serde_json::json!({}))
        .await?;
    assert_eq!(
        store_device::list_for_user(&store, &tenant, &user)
            .await?
            .len(),
        1
    );

    let from_laptop = hub.called_from("laptop".into());
    let said = from_laptop
        .call_tool("devices", serde_json::json!({}))
        .await?;
    let Some(seen) = laptop().await? else {
        anyhow::bail!("the laptop's row is gone");
    };
    assert!(
        seen.last_seen() > then + chrono::Duration::hours(10),
        "{seen:?}"
    );
    assert_eq!((seen.updated_at, seen.backend.as_str()), (then, "client"));
    // The answer says both: seen now, registered then.
    let row = &said["devices"][0];
    assert_eq!(row["registered"], then.to_rfc3339());
    assert_ne!(row["last_seen"], row["registered"]);

    // Inside the throttle: nothing is written, though the row was set back.
    store_device::saw(&store, &tenant, &user, "laptop", then).await?;
    from_laptop
        .call_tool("devices", serde_json::json!({}))
        .await?;
    assert_eq!(laptop().await?.and_then(|d| d.seen_at), Some(then));
    Ok(())
}

/// A session names its machine's operating system, and what it said stays
/// when a later registration names none.
#[tokio::test]
async fn a_machine_reports_its_operating_system() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let hub = server_for(&store, "user:a", "hub");
    hub.call_tool(
        "register_device",
        serde_json::json!({ "host": "laptop", "os": " macOS 15.6\n" }),
    )
    .await?;
    hub.call_tool("register_device", serde_json::json!({ "host": "laptop" }))
        .await?;
    let said = hub.call_tool("devices", serde_json::json!({})).await?;
    assert_eq!(said["devices"][0]["os"], "macOS 15.6");
    assert!(said["devices"][0].get("name").is_none());
    Ok(())
}
