//! Relics end to end through the tools, over the REST dispatcher the dashboard
//! uses: publish, view, republish, make public and private again, list and
//! search, and the refusals. Copal is a fake that answers like the real one and
//! records every call, so a test can show which file went public and that a
//! refused publish never reached it.

use super::*;

use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::Mutex;

use serde_json::{json, Value};

const ORIGIN: &str = "http://10.0.0.130:8083";

/// A copal that mints a fresh file id per create, answers render links on the
/// render origin, accepts access changes, and records every call by URL.
#[derive(Default)]
struct RelicCopal {
    calls: Mutex<Vec<(String, Value)>>,
    files: AtomicU32,
}

impl RelicCopal {
    fn calls(&self) -> Vec<(String, Value)> {
        self.calls.lock().unwrap().clone()
    }

    /// The access each file was last set to, by file id.
    fn access_of(&self, file: &str) -> Option<String> {
        self.calls()
            .into_iter()
            .rev()
            .find(|(url, _)| url.ends_with(&format!("/v1/files/{file}/access")))
            .map(|(_, body)| body["access"].as_str().unwrap_or_default().to_string())
    }
}

impl antumbra_copal::CopalTransport for RelicCopal {
    fn post_json(
        &self,
        url: &str,
        _credential: &antumbra_copal::CopalCredential,
        body: &Value,
    ) -> antumbra_core::Result<Value> {
        self.calls
            .lock()
            .unwrap()
            .push((url.to_string(), body.clone()));
        if url.ends_with("/v1/files") {
            let n = self.files.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            Ok(json!({ "id": format!("01relicfile{n:015}"), "state": "draft" }))
        } else if url.ends_with("/render-url") {
            Ok(json!({
                "url": format!("{ORIGIN}/v1/render/cg1.grant.secret"),
                "expires_at": "2026-10-24T20:00:00Z",
            }))
        } else {
            Ok(json!({ "access": body["access"] }))
        }
    }

    fn put_bytes(
        &self,
        url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _content_type: &str,
        digest: &str,
        _body: &[u8],
    ) -> antumbra_core::Result<Value> {
        self.calls
            .lock()
            .unwrap()
            .push((url.to_string(), json!({ "uploaded": true })));
        Ok(json!({ "digest": digest, "state": "ready" }))
    }
}

/// A provisioned identity's server, with relics in `copal` when given.
async fn as_user(
    store: &Store,
    user: &str,
    copal: Option<(Arc<RelicCopal>, Option<&str>)>,
) -> McpServer {
    let (tenant, user) = (TenantId::new("ws:org"), UserId::new(user));
    let compartment = crate::provision_identity(store, &tenant, &user)
        .await
        .unwrap();
    let s = McpServer::new(
        store.clone(),
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        tenant,
        user,
        "h".into(),
        compartment,
        None,
    );
    match copal {
        Some((transport, origin)) => s.with_copal_archive(Arc::new(
            antumbra_copal::CopalArchive::with_transport(
                "127.0.0.1:9010",
                antumbra_copal::CopalTenancy::Shared("antumbra".into()),
                transport,
            )
            .with_render_origin(origin.map(str::to_string)),
        )),
        None => s,
    }
}

async fn lily() -> (Store, McpServer, Arc<RelicCopal>) {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let copal = Arc::new(RelicCopal::default());
    let s = as_user(&store, "user:lily", Some((copal.clone(), Some(ORIGIN)))).await;
    (store, s, copal)
}

async fn publish(s: &McpServer, args: Value) -> Value {
    s.call_tool("publish_relic", args).await.unwrap()
}

#[tokio::test]
async fn a_relic_is_published_viewed_republished_and_made_public() {
    let (_store, s, copal) = lily().await;

    let first = publish(
        &s,
        json!({
            "title": "The Ceiling Pays for the Floor",
            "html": "<h1>The ceiling pays</h1><script>var DATA = [1947, 65.2];</script>",
        }),
    )
    .await;
    assert_eq!(first["slug"], "the-ceiling-pays-for-the-floor");
    assert_eq!(first["version"], 1);
    assert_eq!(first["visibility"], "private");
    assert_eq!(first["path"], "/relic/the-ceiling-pays-for-the-floor");
    assert!(first.get("public_url").is_none(), "{first}");
    assert!(first.get("note").is_none(), "{first}");

    // Private: shown through a render link made for this view.
    let viewed = s
        .call_tool(
            "get_relic",
            json!({ "slug": "the-ceiling-pays-for-the-floor" }),
        )
        .await
        .unwrap();
    assert!(
        viewed["view_url"]
            .as_str()
            .unwrap()
            .starts_with(&format!("{ORIGIN}/v1/render/")),
        "{viewed}"
    );
    assert!(viewed["expires_at"].is_string());
    assert_eq!(viewed["yours"], true);
    let first_file = "01relicfile000000000000001";

    // Same slug, new page: version 2, at the same address, now public.
    let second = publish(
        &s,
        json!({
            "title": "The Ceiling Pays for the Floor",
            "slug": "the-ceiling-pays-for-the-floor",
            "html": "<h1>Version two</h1>",
            "visibility": "public",
        }),
    )
    .await;
    assert_eq!(second["version"], 2);
    assert_eq!(second["visibility"], "public");
    let second_file = "01relicfile000000000000002";
    assert_eq!(
        second["public_url"],
        format!("{ORIGIN}/v1/public/{second_file}")
    );
    // Only the current version is ever public: the one before it closes.
    assert_eq!(copal.access_of(first_file).as_deref(), Some("private"));
    let create = copal
        .calls()
        .into_iter()
        .filter(|(url, _)| url.ends_with("/v1/files"))
        .nth(1)
        .unwrap();
    assert_eq!(create.1["access"], "public");
    assert_eq!(
        create.1["path"],
        "relics/the-ceiling-pays-for-the-floor/v2.html"
    );

    let viewed = s
        .call_tool(
            "get_relic",
            json!({ "slug": "the-ceiling-pays-for-the-floor" }),
        )
        .await
        .unwrap();
    assert_eq!(
        viewed["view_url"],
        format!("{ORIGIN}/v1/public/{second_file}")
    );
    assert!(viewed.get("expires_at").is_none());
    assert_eq!(viewed["versions"].as_array().unwrap().len(), 2);

    // And private again, which copal is told first.
    let changed = s
        .call_tool(
            "set_relic_visibility",
            json!({ "slug": "the-ceiling-pays-for-the-floor", "visibility": "private" }),
        )
        .await
        .unwrap();
    assert_eq!(changed["visibility"], "private");
    assert_eq!(copal.access_of(second_file).as_deref(), Some("private"));
    let listed = s.call_tool("list_relics", json!({})).await.unwrap();
    let cards = listed["relics"].as_array().unwrap();
    assert_eq!(cards.len(), 1, "one relic, however many versions");
    assert_eq!(cards[0]["visibility"], "private");
    assert!(cards[0].get("preview_url").is_none());
}

#[tokio::test]
async fn relics_are_found_by_what_they_say_and_by_whose_they_are() {
    let (store, s, copal) = lily().await;
    publish(
        &s,
        json!({ "title": "Child poverty", "html": "<p>The expanded credit cut it in half.</p>" }),
    )
    .await;
    publish(
        &s,
        json!({
            "title": "Wealth ceiling",
            "html": "<p>A zymurgy of brackets, two percent above fifty million.</p>",
            "visibility": "public",
        }),
    )
    .await;

    // A word only the page's body holds finds it.
    let found = s
        .call_tool("list_relics", json!({ "query": "zymurgy" }))
        .await
        .unwrap();
    assert_eq!(found["relics"][0]["slug"], "wealth-ceiling", "{found}");
    assert_eq!(
        found["relics"][0]["preview_url"]
            .as_str()
            .map(|u| u.starts_with(ORIGIN)),
        Some(true)
    );
    // A title word finds it too.
    let by_title = s
        .call_tool("list_relics", json!({ "query": "Child" }))
        .await
        .unwrap();
    assert_eq!(by_title["relics"][0]["slug"], "child-poverty", "{by_title}");

    let mine = s
        .call_tool("list_relics", json!({ "scope": "mine" }))
        .await
        .unwrap();
    assert_eq!(mine["relics"].as_array().unwrap().len(), 2);
    let shared = s
        .call_tool("list_relics", json!({ "scope": "shared" }))
        .await
        .unwrap();
    assert!(shared["relics"].as_array().unwrap().is_empty());
    assert!(shared["note"].is_string(), "an empty list says why");

    // Another member's view of the same store: lily's relics are not theirs.
    let oslo = as_user(&store, "user:oslo", Some((copal, Some(ORIGIN)))).await;
    let theirs = oslo
        .call_tool("list_relics", json!({ "scope": "mine" }))
        .await
        .unwrap();
    assert!(theirs["relics"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn only_the_owner_republishes_or_changes_who_sees_a_relic() {
    let (store, s, copal) = lily().await;
    publish(
        &s,
        json!({ "title": "Ceiling", "slug": "ceiling", "html": "<p>lily's</p>" }),
    )
    .await;
    let before = copal.calls().len();

    let oslo = as_user(&store, "user:oslo", Some((copal.clone(), Some(ORIGIN)))).await;
    let taken = oslo
        .call_tool(
            "publish_relic",
            json!({ "title": "Mine now", "slug": "ceiling", "html": "<p>oslo's</p>" }),
        )
        .await
        .unwrap_err();
    assert!(
        taken.message.contains("someone else's"),
        "{}",
        taken.message
    );
    let flipped = oslo
        .call_tool(
            "set_relic_visibility",
            json!({ "slug": "ceiling", "visibility": "public" }),
        )
        .await
        .unwrap_err();
    assert!(
        flipped.message.contains("only its owner"),
        "{}",
        flipped.message
    );
    assert_eq!(copal.calls().len(), before, "neither refusal reached copal");

    // A title that draws an existing slug of someone else's steps aside.
    let drawn = publish(
        &oslo,
        json!({ "title": "Ceiling", "html": "<p>oslo's own</p>" }),
    )
    .await;
    assert_ne!(drawn["slug"], "ceiling");
    assert!(drawn["slug"].as_str().unwrap().starts_with("ceiling-"));
}

#[tokio::test]
async fn a_publish_that_cannot_land_is_refused_before_copal() {
    let (store, s, copal) = lily().await;
    for (args, says) in [
        (json!({ "title": "", "html": "<p>x</p>" }), "title"),
        (json!({ "title": "T", "html": "  " }), "html is empty"),
        (
            json!({ "title": "T", "html": "<p>x</p>", "slug": "Not A Slug" }),
            "cannot be a relic's address",
        ),
        (
            json!({ "title": "T", "html": "<p>x</p>", "visibility": "unlisted" }),
            "unknown visibility",
        ),
        (
            json!({ "title": "T", "html": "<p>x</p>", "compartment": "comp:nobodys" }),
            "not one you can write to",
        ),
    ] {
        let refused = s
            .call_tool("publish_relic", args.clone())
            .await
            .unwrap_err();
        assert!(
            refused.message.contains(says),
            "{args}: {}",
            refused.message
        );
    }
    assert!(copal.calls().is_empty(), "nothing reached copal");

    // Without a render origin, or without copal, relics are refused outright.
    let blind = as_user(
        &store,
        "user:blind",
        Some((Arc::new(RelicCopal::default()), None)),
    )
    .await;
    let none = as_user(&store, "user:none", None).await;
    for server in [blind, none] {
        let refused = server
            .call_tool("publish_relic", json!({ "title": "T", "html": "<p>x</p>" }))
            .await
            .unwrap_err();
        assert!(
            refused.message.contains("--copal-render-origin"),
            "{}",
            refused.message
        );
    }
    let missing = s
        .call_tool("get_relic", json!({ "slug": "nothing-here" }))
        .await
        .unwrap_err();
    assert!(missing.message.contains("no relic"), "{}", missing.message);
}
