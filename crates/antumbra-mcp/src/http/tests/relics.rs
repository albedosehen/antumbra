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

const PAGE: &str = "<!doctype html><title>Plan</title><h1>The plan</h1>\
    <p>Cut <strong>65</strong> to 22.</p><script>steal()</script>";

/// A copal that serves [`PAGE`] for every file and records which it read.
#[derive(Default)]
struct Pages {
    reads: std::sync::Mutex<Vec<String>>,
}

impl antumbra_copal::CopalTransport for Pages {
    fn post_json(
        &self,
        _url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _body: &serde_json::Value,
    ) -> antumbra_core::Result<serde_json::Value> {
        Err(antumbra_core::AntumbraError::other(
            "an export never writes",
        ))
    }

    fn put_bytes(
        &self,
        _url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _content_type: &str,
        _digest: &str,
        _body: &[u8],
    ) -> antumbra_core::Result<serde_json::Value> {
        Err(antumbra_core::AntumbraError::other(
            "an export never writes",
        ))
    }

    fn get_bytes(
        &self,
        url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _limit: u64,
    ) -> antumbra_core::Result<Vec<u8>> {
        self.reads.lock().unwrap().push(url.to_string());
        Ok(PAGE.as_bytes().to_vec())
    }
}

/// A public relic `open`, and a private relic `closed` in lily's own
/// compartment, over a copal that serves [`PAGE`].
async fn exporting() -> (Router, Arc<Pages>) {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let compartment =
        crate::provision_identity(&store, &TenantId::new("ws:org"), &UserId::new("user:lily"))
            .await
            .unwrap();
    relic::put(&store, &page("open", Visibility::Public))
        .await
        .unwrap();
    relic::put(
        &store,
        &Relic {
            compartment,
            ..page("closed", Visibility::Private)
        },
    )
    .await
    .unwrap();
    let pages = Arc::new(Pages::default());
    let copal = antumbra_copal::CopalArchive::with_transport(
        "127.0.0.1:9",
        antumbra_copal::CopalTenancy::Shared("antumbra".into()),
        pages.clone(),
    )
    .with_render_origin(Some(ORIGIN.into()));
    let app = router(
        state_archived(
            store,
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            Some(Arc::new(copal)),
        )
        .await,
    );
    (app, pages)
}

async fn get_as(app: &Router, uri: &str, auth: Option<&str>) -> Response {
    let mut req = Request::builder().uri(uri);
    if let Some(auth) = auth {
        req = req.header(header::AUTHORIZATION, auth);
    }
    app.clone()
        .oneshot(req.body(Body::empty()).unwrap())
        .await
        .unwrap()
}

async fn text_of(res: Response) -> String {
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn a_public_relic_exports_to_anyone_as_a_file_this_origin_never_runs() {
    let (app, pages) = exporting().await;

    let html = get_as(&app, "/relic/open/export?format=html", None).await;
    assert_eq!(html.status(), StatusCode::OK);
    let h = html.headers();
    assert_eq!(h[header::CONTENT_TYPE], "text/html; charset=utf-8");
    assert_eq!(
        h[header::CONTENT_DISPOSITION],
        "attachment; filename=\"open-v1.html\""
    );
    assert!(h[header::CONTENT_SECURITY_POLICY]
        .to_str()
        .unwrap()
        .starts_with("sandbox"));
    assert_eq!(h[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
    assert_eq!(h[header::CACHE_CONTROL], "no-store");
    assert_eq!(text_of(html).await, PAGE);
    assert_eq!(
        *pages.reads.lock().unwrap(),
        vec!["http://127.0.0.1:9/v1/files/01fileopen/content".to_string()]
    );

    let md = get_as(&app, "/relic/open/export?format=md", None).await;
    assert_eq!(md.status(), StatusCode::OK);
    assert_eq!(
        md.headers()[header::CONTENT_TYPE],
        "text/markdown; charset=utf-8"
    );
    assert_eq!(
        md.headers()[header::CONTENT_DISPOSITION],
        "attachment; filename=\"open-v1.md\""
    );
    assert_eq!(text_of(md).await, "# The plan\n\nCut **65** to 22.\n");

    // Without a format, the page as it was published.
    let plain = get_as(&app, "/relic/open/export", None).await;
    assert_eq!(
        plain.headers()[header::CONTENT_DISPOSITION],
        "attachment; filename=\"open-v1.html\""
    );
}

#[tokio::test]
async fn a_private_relic_exports_only_to_who_may_see_it() {
    let (app, pages) = exporting().await;

    // No token: a private relic, a missing one and a malformed name alike.
    let unseen = text_of(get_as(&app, "/relic/closed/export", None).await).await;
    for uri in [
        "/relic/closed/export",
        "/relic/missing/export",
        "/relic/Not_A_Slug/export",
    ] {
        let res = get_as(&app, uri, None).await;
        assert_eq!(res.status(), StatusCode::NOT_FOUND, "{uri}");
        assert_eq!(text_of(res).await, unseen, "{uri}");
    }
    assert_eq!(
        get_as(&app, "/relic/closed/export", Some("Bearer not-a-jwt"))
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert!(pages.reads.lock().unwrap().is_empty());

    let lily = format!("Bearer {}", token("ws:org", "user:lily"));
    let mine = get_as(&app, "/relic/closed/export?format=markdown", Some(&lily)).await;
    assert_eq!(mine.status(), StatusCode::OK);
    assert_eq!(
        mine.headers()[header::CONTENT_DISPOSITION],
        "attachment; filename=\"closed-v1.md\""
    );

    // Someone else in the workspace: nothing of lily's, but a public relic.
    let rosa = format!("Bearer {}", token("ws:org", "user:rosa"));
    assert_eq!(
        get_as(&app, "/relic/closed/export", Some(&rosa))
            .await
            .status(),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        get_as(&app, "/relic/open/export", Some(&rosa))
            .await
            .status(),
        StatusCode::OK
    );
}

#[tokio::test]
async fn an_export_says_what_it_cannot_make() {
    let (app, pages) = exporting().await;
    assert_eq!(
        get_as(&app, "/relic/open/export?format=docx", None)
            .await
            .status(),
        StatusCode::BAD_REQUEST
    );
    let pdf = get_as(&app, "/relic/open/export?format=pdf", None).await;
    assert_eq!(pdf.status(), StatusCode::NOT_IMPLEMENTED);
    let said = json_of(pdf).await;
    assert!(
        said["error"].as_str().unwrap().contains("--pdf-renderer"),
        "{said}"
    );
    assert!(pages.reads.lock().unwrap().is_empty());
}
