//! The console's routes over the real router, on the embedded deployment,
//! where the owner's connection and every caller's are one connection: a
//! token reads its own workspace, an operator's also reads the substrate and
//! may act on it, and an operator's read leaves no later caller reading as
//! the owner.

use super::*;
use crate::http::tests::state_embedding;
use antumbra_core::testing::FixedEmbedder;
use antumbra_core::{
    Expert, ExpertId, FailureBoundary, Generation, Memory, MemoryNetwork, Shadow, ShadowId, UserId,
};
use antumbra_store::repo::{boundary, expert, memory, shadow};
use antumbra_store::{Store, EMBED_DIM};
use chrono::Utc;
use tower::ServiceExt;

const SECRET: &[u8] = b"test-secret";

fn plain(tenant: &str, user: &str) -> String {
    crate::auth::mint_hs256(SECRET, tenant, user, std::time::Duration::from_secs(3600)).unwrap()
}

fn operator(tenant: &str, user: &str) -> String {
    crate::auth::mint_hs256_operator(SECRET, tenant, user, std::time::Duration::from_secs(3600))
        .unwrap()
}

fn expert_owned_by(id: &str, owner: Option<&str>) -> Expert {
    Expert {
        id: ExpertId::new(id),
        name: id.to_string(),
        base_model: "base".into(),
        artifact_uri: "adapters/x.safetensors".into(),
        capability_card: serde_json::json!({}),
        capability_vec: None,
        fitness: 0.5,
        frozen_at: None,
        generation: Generation::ZERO,
        owner: owner.map(UserId::new),
        compartment: None,
        placed_on: None,
        created_at: Utc::now(),
    }
}

/// Two workspaces' memories, a shared expert and another user's private one,
/// and a shadow and a boundary, which belong to no workspace.
async fn seeded() -> Arc<HttpState> {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    for (id, tenant) in [("memory:a", "ws:a"), ("memory:b", "ws:b")] {
        memory::upsert(
            &store,
            &Memory::new(id, tenant, MemoryNetwork::World, "a fact", 0.6, Utc::now()),
        )
        .await
        .unwrap();
    }
    expert::insert(&store, &expert_owned_by("expert:shared", None))
        .await
        .unwrap();
    expert::insert(&store, &expert_owned_by("expert:theirs", Some("user:b")))
        .await
        .unwrap();
    shadow::upsert(
        &store,
        &Shadow::spawn(
            ShadowId::new("shadow:a"),
            Generation::ZERO,
            None,
            Utc::now(),
        ),
    )
    .await
    .unwrap();
    boundary::upsert(
        &store,
        &FailureBoundary {
            id: antumbra_core::BoundaryId::new("boundary:a"),
            behavior: "run `npm install`".into(),
            fail_context: serde_json::json!({ "client": "someone else's" }),
            near_ok_context: None,
            governing_features: Vec::new(),
            grain: None,
            context_vec: None,
            ok_context_vec: None,
            confidence: 0.8,
            generation: Generation::ZERO,
            owner: None,
            created_at: Utc::now(),
        },
    )
    .await
    .unwrap();
    state_embedding(store, Arc::new(FixedEmbedder::new(EMBED_DIM))).await
}

async fn send(state: &Arc<HttpState>, req: Request<Body>) -> (StatusCode, serde_json::Value) {
    let res = crate::http::router(state.clone())
        .oneshot(req)
        .await
        .unwrap();
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

async fn read(state: &Arc<HttpState>, token: Option<&str>) -> (StatusCode, serde_json::Value) {
    let mut req = Request::builder().uri("/console/snapshot");
    if let Some(t) = token {
        req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    send(state, req.body(Body::empty()).unwrap()).await
}

async fn act(
    state: &Arc<HttpState>,
    token: Option<&str>,
    body: &str,
) -> (StatusCode, serde_json::Value) {
    let mut req = Request::builder()
        .method("POST")
        .uri("/console/action")
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(t) = token {
        req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    send(state, req.body(Body::from(body.to_string())).unwrap()).await
}

fn ids(snapshot: &serde_json::Value, list: &str) -> Vec<String> {
    snapshot[list]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x["id"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn neither_route_answers_without_a_token() {
    let state = seeded().await;
    assert_eq!(read(&state, None).await.0, StatusCode::UNAUTHORIZED);
    assert_eq!(
        act(&state, None, r#"{"action":"halt_loop","run":"run:a"}"#)
            .await
            .0,
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn a_token_reads_its_workspace_and_may_not_act() {
    let state = seeded().await;
    let token = plain("ws:a", "user:a");
    let (status, snapshot) = read(&state, Some(&token)).await;
    assert_eq!(status, StatusCode::OK, "{snapshot}");
    assert_eq!(snapshot["operator"], false);
    assert_eq!(ids(&snapshot, "memories"), ["memory:a"]);
    assert_eq!(
        ids(&snapshot, "experts"),
        ["expert:shared"],
        "another user's private expert stays theirs"
    );
    assert!(ids(&snapshot, "shadows").is_empty());
    assert!(
        ids(&snapshot, "boundaries").is_empty(),
        "a boundary belongs to no workspace"
    );

    let (status, body) = act(
        &state,
        Some(&token),
        r#"{"action":"freeze_expert","expert":"expert:shared"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
    assert!(expert::get(&state.store, &ExpertId::new("expert:shared"))
        .await
        .unwrap()
        .expect("kept")
        .frozen_at
        .is_none());
}

#[tokio::test]
async fn an_operator_reads_the_substrate_and_acts_on_it() {
    let state = seeded().await;
    let token = operator("ws:a", "user:a");
    let (status, snapshot) = read(&state, Some(&token)).await;
    assert_eq!(status, StatusCode::OK, "{snapshot}");
    assert_eq!(snapshot["operator"], true);
    assert_eq!(
        ids(&snapshot, "memories"),
        ["memory:a"],
        "an operator's memories are still its own workspace's"
    );
    assert_eq!(ids(&snapshot, "shadows"), ["shadow:a"]);
    assert_eq!(ids(&snapshot, "boundaries"), ["boundary:a"]);

    let (status, body) = act(
        &state,
        Some(&token),
        r#"{"action":"freeze_expert","expert":"expert:shared"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["applied"], true);
    assert!(expert::get(&state.store, &ExpertId::new("expert:shared"))
        .await
        .unwrap()
        .expect("kept")
        .frozen_at
        .is_some());

    let (status, _) = act(
        &state,
        Some(&token),
        r#"{"action":"prune_shadow","shadow":"shadow:none"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = act(&state, Some(&token), r#"{"action":"explode"}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// The console asks every few seconds: an unchanged snapshot is a 304 with no
/// body, and an action changes the tag so the next read is whole.
#[tokio::test]
async fn an_unchanged_snapshot_is_not_sent_again() {
    let state = seeded().await;
    let token = operator("ws:a", "user:a");
    let get = |tag: Option<&str>| {
        let mut req = Request::builder()
            .uri("/console/snapshot")
            .header(header::AUTHORIZATION, format!("Bearer {token}"));
        if let Some(tag) = tag {
            req = req.header(header::IF_NONE_MATCH, tag);
        }
        req.body(Body::empty()).unwrap()
    };
    let first = crate::http::router(state.clone())
        .oneshot(get(None))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let tag = first.headers()[header::ETAG].to_str().unwrap().to_string();

    let again = crate::http::router(state.clone())
        .oneshot(get(Some(&tag)))
        .await
        .unwrap();
    assert_eq!(again.status(), StatusCode::NOT_MODIFIED);
    let body = axum::body::to_bytes(again.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(body.is_empty());

    act(
        &state,
        Some(&token),
        r#"{"action":"prune_shadow","shadow":"shadow:a"}"#,
    )
    .await;
    let changed = crate::http::router(state.clone())
        .oneshot(get(Some(&tag)))
        .await
        .unwrap();
    assert_eq!(changed.status(), StatusCode::OK);
}

/// Embedded, an operator's read signs the one connection in as the owner.
/// The next caller's read must still be its own workspace's, not everyone's.
#[tokio::test]
async fn a_read_after_an_operators_is_still_scoped() {
    let state = seeded().await;
    let (status, _) = read(&state, Some(&operator("ws:a", "user:a"))).await;
    assert_eq!(status, StatusCode::OK);
    let (status, snapshot) = read(&state, Some(&plain("ws:b", "user:b"))).await;
    assert_eq!(status, StatusCode::OK, "{snapshot}");
    assert_eq!(ids(&snapshot, "memories"), ["memory:b"]);
    assert_eq!(snapshot["operator"], false);
}
