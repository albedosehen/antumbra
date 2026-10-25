//! The relics pages over the real router: the gallery and viewer are served to
//! anyone with a policy that frames copal's render origin only, and a public
//! relic is described without a token while a private one and a missing one
//! are the same 404, saying nothing about who owns either.

use super::*;

use antumbra_core::{CompartmentId, Relic, RelicVersion, Visibility};
use antumbra_store::repo::relic;

const ORIGIN: &str = "http://10.0.0.130:8083";

fn page(slug: &str, visibility: Visibility) -> Relic {
    let now = chrono::Utc::now();
    Relic {
        slug: slug.into(),
        tenant: TenantId::new("ws:org"),
        owner: UserId::new("user:lily"),
        compartment: CompartmentId::new("comp:lily"),
        title: format!("The {slug} page"),
        visibility,
        versions: vec![RelicVersion {
            number: 1,
            copal_file: format!("01file{slug}"),
            copal_digest: "d".into(),
            created_at: now,
            author_host: None,
        }],
        source: None,
        created_at: now,
        updated_at: now,
    }
}

async fn app() -> Router {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    relic::put(&store, &page("open", Visibility::Public))
        .await
        .unwrap();
    relic::put(&store, &page("closed", Visibility::Private))
        .await
        .unwrap();
    let copal = antumbra_copal::CopalArchive::new(
        "127.0.0.1:9",
        antumbra_copal::CopalTenancy::Shared("antumbra".into()),
    )
    .with_render_origin(Some(ORIGIN.into()));
    router(
        state_archived(
            store,
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            Some(Arc::new(copal)),
        )
        .await,
    )
}

async fn get(app: &Router, uri: &str) -> Response {
    let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
    app.clone().oneshot(req).await.unwrap()
}

async fn json_of(res: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn a_public_relic_is_described_to_anyone_and_a_private_one_to_no_one() {
    let app = app().await;

    let open = get(&app, "/relic/open/public").await;
    assert_eq!(open.status(), StatusCode::OK);
    assert_eq!(open.headers()[header::CACHE_CONTROL], "no-store");
    let body = json_of(open).await;
    assert_eq!(body["iframe_src"], format!("{ORIGIN}/v1/public/01fileopen"));
    assert_eq!(body["title"], "The open page");
    // Nothing about who owns it or where it lives.
    assert!(
        body.get("owner").is_none() && body.get("tenant").is_none(),
        "{body}"
    );

    for uri in [
        "/relic/closed/public",
        "/relic/missing/public",
        "/relic/Not%20A%20Slug/public",
    ] {
        let res = get(&app, uri).await;
        assert_eq!(res.status(), StatusCode::NOT_FOUND, "{uri}");
    }
    // A private relic and a missing one answer exactly alike.
    let closed = json_of(get(&app, "/relic/closed/public").await).await;
    let missing = json_of(get(&app, "/relic/missing/public").await).await;
    assert_eq!(closed, missing);
}

#[tokio::test]
async fn the_pages_are_served_to_anyone_and_frame_only_the_render_origin() {
    let app = app().await;
    for uri in ["/relics", "/relic/closed", "/relic/missing"] {
        let res = get(&app, uri).await;
        assert_eq!(res.status(), StatusCode::OK, "{uri}");
        let csp = res.headers()[header::CONTENT_SECURITY_POLICY]
            .to_str()
            .unwrap();
        assert!(
            csp.contains(&format!("frame-src {ORIGIN};")),
            "{uri}: {csp}"
        );
        assert!(csp.contains("frame-ancestors 'none'"), "{uri}: {csp}");
    }
    assert_eq!(
        get(&app, "/relic/Not_A_Slug").await.status(),
        StatusCode::NOT_FOUND
    );
    let slash = get(&app, "/relics/").await;
    assert!(slash.status().is_redirection());
    assert_eq!(slash.headers()[header::LOCATION], "../relics");
    // The relics script and style are dashboard assets.
    for uri in ["/dashboard/relics.js", "/dashboard/relics.css"] {
        assert_eq!(get(&app, uri).await.status(), StatusCode::OK, "{uri}");
    }
}
