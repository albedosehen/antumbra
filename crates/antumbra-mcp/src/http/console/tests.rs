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
use antumbra_store::testing::mem_store;
use antumbra_store::EMBED_DIM;
use chrono::Utc;
use tower::ServiceExt;

const SECRET: &[u8] = b"test-secret";

fn plain(tenant: &str, user: &str) -> anyhow::Result<String> {
    Ok(crate::auth::mint_hs256(
        SECRET,
        tenant,
        user,
        std::time::Duration::from_secs(3600),
    )?)
}

fn operator(tenant: &str, user: &str) -> anyhow::Result<String> {
    Ok(crate::auth::mint_hs256_operator(
        SECRET,
        tenant,
        user,
        std::time::Duration::from_secs(3600),
    )?)
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
async fn seeded() -> anyhow::Result<Arc<HttpState>> {
    let store = mem_store().await?;
    for (id, tenant) in [("memory:a", "ws:a"), ("memory:b", "ws:b")] {
        memory::upsert(
            &store,
            &Memory::new(id, tenant, MemoryNetwork::World, "a fact", 0.6, Utc::now()),
        )
        .await?;
    }
    expert::insert(&store, &expert_owned_by("expert:shared", None)).await?;
    expert::insert(&store, &expert_owned_by("expert:theirs", Some("user:b"))).await?;
    shadow::upsert(
        &store,
        &Shadow::spawn(
            ShadowId::new("shadow:a"),
            Generation::ZERO,
            None,
            Utc::now(),
        ),
    )
    .await?;
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
    .await?;
    Ok(state_embedding(store, Arc::new(FixedEmbedder::new(EMBED_DIM))).await)
}

async fn send(
    state: &Arc<HttpState>,
    req: Request<Body>,
) -> anyhow::Result<(StatusCode, serde_json::Value)> {
    let res = crate::http::router(state.clone()).oneshot(req).await?;
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await?;
    Ok((
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    ))
}

async fn read(
    state: &Arc<HttpState>,
    token: Option<&str>,
) -> anyhow::Result<(StatusCode, serde_json::Value)> {
    let mut req = Request::builder().uri("/console/snapshot");
    if let Some(t) = token {
        req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    send(state, req.body(Body::empty())?).await
}

async fn act(
    state: &Arc<HttpState>,
    token: Option<&str>,
    body: &str,
) -> anyhow::Result<(StatusCode, serde_json::Value)> {
    let mut req = Request::builder()
        .method("POST")
        .uri("/console/action")
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(t) = token {
        req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    send(state, req.body(Body::from(body.to_string()))?).await
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
async fn neither_route_answers_without_a_token() -> anyhow::Result<()> {
    let state = seeded().await?;
    assert_eq!(read(&state, None).await?.0, StatusCode::UNAUTHORIZED);
    assert_eq!(
        act(&state, None, r#"{"action":"halt_loop","run":"run:a"}"#)
            .await?
            .0,
        StatusCode::UNAUTHORIZED
    );
    Ok(())
}

#[tokio::test]
async fn a_token_reads_its_workspace_and_may_not_act() -> anyhow::Result<()> {
    let state = seeded().await?;
    let token = plain("ws:a", "user:a")?;
    let (status, snapshot) = read(&state, Some(&token)).await?;
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
    .await?;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
    let Some(shared) = expert::get(&state.store, &ExpertId::new("expert:shared")).await? else {
        anyhow::bail!("the shared expert was not kept");
    };
    assert!(shared.frozen_at.is_none());
    Ok(())
}

#[tokio::test]
async fn an_operator_reads_the_substrate_and_acts_on_it() -> anyhow::Result<()> {
    let state = seeded().await?;
    let token = operator("ws:a", "user:a")?;
    let (status, snapshot) = read(&state, Some(&token)).await?;
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
    .await?;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["applied"], true);
    let Some(shared) = expert::get(&state.store, &ExpertId::new("expert:shared")).await? else {
        anyhow::bail!("the shared expert was not kept");
    };
    assert!(shared.frozen_at.is_some());

    let (status, _) = act(
        &state,
        Some(&token),
        r#"{"action":"prune_shadow","shadow":"shadow:none"}"#,
    )
    .await?;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = act(&state, Some(&token), r#"{"action":"explode"}"#).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    Ok(())
}

/// The console asks every few seconds: an unchanged snapshot is a 304 with no
/// body, and an action changes the tag so the next read is whole.
#[tokio::test]
async fn an_unchanged_snapshot_is_not_sent_again() -> anyhow::Result<()> {
    let state = seeded().await?;
    let token = operator("ws:a", "user:a")?;
    let get = |tag: Option<&str>| -> anyhow::Result<Request<Body>> {
        let mut req = Request::builder()
            .uri("/console/snapshot")
            .header(header::AUTHORIZATION, format!("Bearer {token}"));
        if let Some(tag) = tag {
            req = req.header(header::IF_NONE_MATCH, tag);
        }
        Ok(req.body(Body::empty())?)
    };
    let first = crate::http::router(state.clone())
        .oneshot(get(None)?)
        .await?;
    assert_eq!(first.status(), StatusCode::OK);
    let Some(tag) = first.headers().get(header::ETAG) else {
        anyhow::bail!("the first snapshot carries no ETag");
    };
    let tag = tag.to_str()?.to_string();

    let again = crate::http::router(state.clone())
        .oneshot(get(Some(&tag))?)
        .await?;
    assert_eq!(again.status(), StatusCode::NOT_MODIFIED);
    let body = axum::body::to_bytes(again.into_body(), usize::MAX).await?;
    assert!(body.is_empty());

    act(
        &state,
        Some(&token),
        r#"{"action":"prune_shadow","shadow":"shadow:a"}"#,
    )
    .await?;
    let changed = crate::http::router(state.clone())
        .oneshot(get(Some(&tag))?)
        .await?;
    assert_eq!(changed.status(), StatusCode::OK);
    Ok(())
}

/// Embedded, an operator's read signs the one connection in as the owner.
/// The next caller's read must still be its own workspace's, not everyone's.
#[tokio::test]
async fn a_read_after_an_operators_is_still_scoped() -> anyhow::Result<()> {
    let state = seeded().await?;
    let (status, _) = read(&state, Some(&operator("ws:a", "user:a")?)).await?;
    assert_eq!(status, StatusCode::OK);
    let (status, snapshot) = read(&state, Some(&plain("ws:b", "user:b")?)).await?;
    assert_eq!(status, StatusCode::OK, "{snapshot}");
    assert_eq!(ids(&snapshot, "memories"), ["memory:b"]);
    assert_eq!(snapshot["operator"], false);
    Ok(())
}
