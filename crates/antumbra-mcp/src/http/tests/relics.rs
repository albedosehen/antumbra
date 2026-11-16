//! The relics pages over the real router: the gallery and viewer are served to
//! anyone with a policy that frames copal's render origin only, and a public
//! relic is described without a token while a private one and a missing one
//! are the same 404, saying nothing about who owns either.

use super::*;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use antumbra_core::{CompartmentId, Relic, RelicVersion, Visibility};
use antumbra_store::repo::relic;
use antumbra_store::testing::mem_store;

use crate::http::relics::{public_row, EXPORTS_PER_CLIENT, PUBLIC_READS_PER_CLIENT};

const ORIGIN: &str = "http://render.example:8083";

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

async fn app() -> anyhow::Result<Router> {
    let store = mem_store().await?;
    relic::put(&store, &page("open", Visibility::Public)).await?;
    relic::put(&store, &page("closed", Visibility::Private)).await?;
    let copal = antumbra_copal::CopalArchive::new(
        "127.0.0.1:9",
        antumbra_copal::CopalTenancy::Shared("antumbra".into()),
    )
    .with_render_origin(Some(ORIGIN.into()));
    Ok(router(
        state_archived(
            store,
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            Some(Arc::new(copal)),
        )
        .await,
    ))
}

async fn get(app: &Router, uri: &str) -> anyhow::Result<Response> {
    let req = Request::builder().uri(uri).body(Body::empty())?;
    Ok(app.clone().oneshot(req).await?)
}

async fn json_of(res: Response) -> anyhow::Result<serde_json::Value> {
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[tokio::test]
async fn a_public_relic_is_described_to_anyone_and_a_private_one_to_no_one() -> anyhow::Result<()> {
    let app = app().await?;

    let open = get(&app, "/relic/open/public").await?;
    assert_eq!(open.status(), StatusCode::OK);
    assert_eq!(open.headers()[header::CACHE_CONTROL], "no-store");
    let body = json_of(open).await?;
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
        let res = get(&app, uri).await?;
        assert_eq!(res.status(), StatusCode::NOT_FOUND, "{uri}");
    }
    // A private relic and a missing one answer exactly alike.
    let closed = json_of(get(&app, "/relic/closed/public").await?).await?;
    let missing = json_of(get(&app, "/relic/missing/public").await?).await?;
    assert_eq!(closed, missing);
    Ok(())
}

#[tokio::test]
async fn the_pages_are_served_to_anyone_and_frame_only_the_render_origin() -> anyhow::Result<()> {
    let app = app().await?;
    for uri in ["/relics", "/relic/closed", "/relic/missing"] {
        let res = get(&app, uri).await?;
        assert_eq!(res.status(), StatusCode::OK, "{uri}");
        let csp = res.headers()[header::CONTENT_SECURITY_POLICY].to_str()?;
        assert!(
            csp.contains(&format!("frame-src {ORIGIN};")),
            "{uri}: {csp}"
        );
        assert!(csp.contains("frame-ancestors 'none'"), "{uri}: {csp}");
    }
    assert_eq!(
        get(&app, "/relic/Not_A_Slug").await?.status(),
        StatusCode::NOT_FOUND
    );
    let slash = get(&app, "/relics/").await?;
    assert!(slash.status().is_redirection());
    assert_eq!(slash.headers()[header::LOCATION], "../relics");
    // The relics script and style are dashboard assets.
    for uri in ["/dashboard/relics.js", "/dashboard/relics.css"] {
        assert_eq!(get(&app, uri).await?.status(), StatusCode::OK, "{uri}");
    }
    Ok(())
}

// On a per-identity state the owner connection is no one else's, so the public
// lookup reads it without the auth lock: it answers while another task holds
// the lock. On the shared-connection state it must wait, since that connection
// is every identity's.
#[tokio::test]
async fn a_public_lookup_waits_on_the_lock_only_where_the_connection_is_shared(
) -> anyhow::Result<()> {
    let store = mem_store().await?;
    relic::put(&store, &page("open", Visibility::Public)).await?;
    let embedder: Arc<dyn Embedder> = Arc::new(FixedEmbedder::new(EMBED_DIM));
    let wait = Duration::from_millis(300);

    let st = state_per_identity(store.clone(), embedder.clone(), None).await;
    let _held = st.auth.lock().await;
    let found = tokio::time::timeout(wait, public_row(&st, "open")).await;
    anyhow::ensure!(
        matches!(found, Ok(Some(_))),
        "a per-identity lookup answered neither in time nor with the relic"
    );

    let st = state_archived(store, embedder, None).await;
    let _held = st.auth.lock().await;
    let found = tokio::time::timeout(wait, public_row(&st, "open")).await;
    anyhow::ensure!(
        found.is_err(),
        "a shared-connection lookup must wait for the lock"
    );
    Ok(())
}

// One client may read the public lookup its burst's worth at once; the next
// request is refused with a retry hint, before the lookup does any work.
#[tokio::test]
async fn public_lookups_past_a_clients_burst_are_refused() -> anyhow::Result<()> {
    let app = app().await?;
    for n in 0..PUBLIC_READS_PER_CLIENT.burst {
        let res = get(&app, "/relic/open/public").await?;
        anyhow::ensure!(res.status() == StatusCode::OK, "request {n}");
    }
    // The bucket refills while the burst is sent (two tokens a second at this
    // rate), so a slow runner may find one back by now: a few more requests
    // spend what came back, and the refusal is waited for, bounded. Twenty
    // find none only where each request takes half a second.
    let mut refused = None;
    for _ in 0..20 {
        let res = get(&app, "/relic/open/public").await?;
        if res.status() == StatusCode::TOO_MANY_REQUESTS {
            refused = Some(res);
            break;
        }
    }
    let Some(refused) = refused else {
        anyhow::bail!("twenty requests past the burst and none refused");
    };
    anyhow::ensure!(refused.headers().get(header::RETRY_AFTER).is_some());
    anyhow::ensure!(refused.headers()[header::CACHE_CONTROL] == "no-store");
    Ok(())
}

/// A renderer that prints every page as the same few bytes and counts the
/// pages it was sent: a socket answering the one POST the renderer makes.
async fn fake_renderer() -> anyhow::Result<(String, Arc<AtomicUsize>)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let base = format!("http://{}", listener.local_addr()?);
    let prints = Arc::new(AtomicUsize::new(0));
    let counted = prints.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let counted = counted.clone();
            tokio::spawn(async move {
                // Read the head, then as much body as it declares, so the
                // client is not cut off mid-upload.
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let (head_end, declared) = loop {
                    let Ok(n) = socket.read(&mut chunk).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..at]).to_ascii_lowercase();
                        let declared = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        break (at + 4, declared);
                    }
                };
                while buf.len() < head_end + declared {
                    match socket.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => buf.extend_from_slice(&chunk[..n]),
                    }
                }
                counted.fetch_add(1, Ordering::SeqCst);
                let body = b"%PDF-1.4 printed";
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                if socket.write_all(head.as_bytes()).await.is_err()
                    || socket.write_all(body).await.is_err()
                {
                    return;
                }
                // The client may have gone already; there is nothing left to
                // do about it either way.
                socket.shutdown().await.ok();
            });
        }
    });
    Ok((base, prints))
}

// A version is printed once. The second request for it is answered from the
// copy kept, with no second print and no second read from copal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pdf_is_printed_once_per_version() -> anyhow::Result<()> {
    let (base, prints) = fake_renderer().await?;
    let renderer = Arc::new(crate::pdf::PdfRenderer::new(&base)?);
    let (app, pages) = exporting_with(Some(renderer)).await?;
    for n in 0..2 {
        let res = get_as(&app, "/relic/open/export?format=pdf", None).await?;
        anyhow::ensure!(res.status() == StatusCode::OK, "export {n}");
        anyhow::ensure!(
            res.headers()[header::CONTENT_TYPE] == "application/pdf",
            "export {n}"
        );
        anyhow::ensure!(
            res.headers()[header::CONTENT_DISPOSITION]
                .to_str()?
                .contains("open-v1.pdf"),
            "export {n}"
        );
        let body = text_of(res).await?;
        anyhow::ensure!(body.starts_with("%PDF"), "export {n}: {body}");
    }
    anyhow::ensure!(prints.load(Ordering::SeqCst) == 1, "printed once");
    let reads = pages.read().len();
    anyhow::ensure!(reads == 1, "copal read once, got {reads}");
    Ok(())
}

// A print is refused past a client's burst before any work is done for it:
// here, before the renderer is even looked for.
#[tokio::test]
async fn pdf_exports_past_a_clients_burst_are_refused_before_any_work() -> anyhow::Result<()> {
    let (app, pages) = exporting().await?;
    for n in 0..EXPORTS_PER_CLIENT.burst {
        let res = get_as(&app, "/relic/open/export?format=pdf", None).await?;
        anyhow::ensure!(res.status() == StatusCode::NOT_IMPLEMENTED, "export {n}");
    }
    let refused = get_as(&app, "/relic/open/export?format=pdf", None).await?;
    anyhow::ensure!(refused.status() == StatusCode::TOO_MANY_REQUESTS);
    anyhow::ensure!(pages.read().is_empty(), "nothing was read from copal");
    // The other formats are not budgeted.
    let html = get_as(&app, "/relic/open/export?format=html", None).await?;
    anyhow::ensure!(html.status() == StatusCode::OK);
    Ok(())
}

const PAGE: &str = "<!doctype html><title>Plan</title><h1>The plan</h1>\
    <p>Cut <strong>65</strong> to 22.</p><script>steal()</script>";

/// A copal that serves [`PAGE`] for every file and records which it read.
#[derive(Default)]
struct Pages {
    reads: std::sync::Mutex<Vec<String>>,
}

impl Pages {
    /// The files read so far. The list is only ever pushed to, so a lock
    /// poisoned mid-push still holds a whole list.
    fn read(&self) -> Vec<String> {
        self.reads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
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
        // Only ever pushed to, so a poisoned lock still holds a whole list.
        self.reads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(url.to_string());
        Ok(PAGE.as_bytes().to_vec())
    }
}

/// A public relic `open`, and a private relic `closed` in lily's own
/// compartment, over a copal that serves [`PAGE`].
async fn exporting() -> anyhow::Result<(Router, Arc<Pages>)> {
    exporting_with(None).await
}

/// As [`exporting`], printing through `pdf` when one is given.
async fn exporting_with(
    pdf: Option<Arc<crate::pdf::PdfRenderer>>,
) -> anyhow::Result<(Router, Arc<Pages>)> {
    let store = mem_store().await?;
    let compartment =
        crate::provision_identity(&store, &TenantId::new("ws:org"), &UserId::new("user:lily"))
            .await?;
    relic::put(&store, &page("open", Visibility::Public)).await?;
    relic::put(
        &store,
        &Relic {
            compartment,
            ..page("closed", Visibility::Private)
        },
    )
    .await?;
    let pages = Arc::new(Pages::default());
    let copal = Arc::new(
        antumbra_copal::CopalArchive::with_transport(
            "127.0.0.1:9",
            antumbra_copal::CopalTenancy::Shared("antumbra".into()),
            pages.clone(),
        )
        .with_render_origin(Some(ORIGIN.into())),
    );
    let embedder: Arc<dyn Embedder> = Arc::new(FixedEmbedder::new(EMBED_DIM));
    let state = match pdf {
        Some(pdf) => state_printing(store, embedder, Some(copal), pdf).await,
        None => state_archived(store, embedder, Some(copal)).await,
    };
    Ok((router(state), pages))
}

async fn get_as(app: &Router, uri: &str, auth: Option<&str>) -> anyhow::Result<Response> {
    let mut req = Request::builder().uri(uri);
    if let Some(auth) = auth {
        req = req.header(header::AUTHORIZATION, auth);
    }
    Ok(app.clone().oneshot(req.body(Body::empty())?).await?)
}

async fn text_of(res: Response) -> anyhow::Result<String> {
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await?;
    Ok(String::from_utf8(bytes.to_vec())?)
}

#[tokio::test]
async fn a_public_relic_exports_to_anyone_as_a_file_this_origin_never_runs() -> anyhow::Result<()> {
    let (app, pages) = exporting().await?;

    let html = get_as(&app, "/relic/open/export?format=html", None).await?;
    assert_eq!(html.status(), StatusCode::OK);
    let h = html.headers();
    assert_eq!(h[header::CONTENT_TYPE], "text/html; charset=utf-8");
    assert_eq!(
        h[header::CONTENT_DISPOSITION],
        "attachment; filename=\"open-v1.html\""
    );
    assert!(h[header::CONTENT_SECURITY_POLICY]
        .to_str()?
        .starts_with("sandbox"));
    assert_eq!(h[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
    assert_eq!(h[header::CACHE_CONTROL], "no-store");
    assert_eq!(text_of(html).await?, PAGE);
    assert_eq!(
        pages.read(),
        vec!["http://127.0.0.1:9/v1/files/01fileopen/content".to_string()]
    );

    let md = get_as(&app, "/relic/open/export?format=md", None).await?;
    assert_eq!(md.status(), StatusCode::OK);
    assert_eq!(
        md.headers()[header::CONTENT_TYPE],
        "text/markdown; charset=utf-8"
    );
    assert_eq!(
        md.headers()[header::CONTENT_DISPOSITION],
        "attachment; filename=\"open-v1.md\""
    );
    assert_eq!(text_of(md).await?, "# The plan\n\nCut **65** to 22.\n");

    // Without a format, the page as it was published.
    let plain = get_as(&app, "/relic/open/export", None).await?;
    assert_eq!(
        plain.headers()[header::CONTENT_DISPOSITION],
        "attachment; filename=\"open-v1.html\""
    );
    Ok(())
}

#[tokio::test]
async fn a_private_relic_exports_only_to_who_may_see_it() -> anyhow::Result<()> {
    let (app, pages) = exporting().await?;

    // No token: a private relic, a missing one and a malformed name alike.
    let unseen = text_of(get_as(&app, "/relic/closed/export", None).await?).await?;
    for uri in [
        "/relic/closed/export",
        "/relic/missing/export",
        "/relic/Not_A_Slug/export",
    ] {
        let res = get_as(&app, uri, None).await?;
        assert_eq!(res.status(), StatusCode::NOT_FOUND, "{uri}");
        assert_eq!(text_of(res).await?, unseen, "{uri}");
    }
    assert_eq!(
        get_as(&app, "/relic/closed/export", Some("Bearer not-a-jwt"))
            .await?
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert!(pages.read().is_empty());

    let lily = format!("Bearer {}", token("ws:org", "user:lily")?);
    let mine = get_as(&app, "/relic/closed/export?format=markdown", Some(&lily)).await?;
    assert_eq!(mine.status(), StatusCode::OK);
    assert_eq!(
        mine.headers()[header::CONTENT_DISPOSITION],
        "attachment; filename=\"closed-v1.md\""
    );

    // Someone else in the workspace: nothing of lily's, but a public relic.
    let rosa = format!("Bearer {}", token("ws:org", "user:rosa")?);
    assert_eq!(
        get_as(&app, "/relic/closed/export", Some(&rosa))
            .await?
            .status(),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        get_as(&app, "/relic/open/export", Some(&rosa))
            .await?
            .status(),
        StatusCode::OK
    );
    Ok(())
}

#[tokio::test]
async fn an_export_says_what_it_cannot_make() -> anyhow::Result<()> {
    let (app, pages) = exporting().await?;
    assert_eq!(
        get_as(&app, "/relic/open/export?format=docx", None)
            .await?
            .status(),
        StatusCode::BAD_REQUEST
    );
    let pdf = get_as(&app, "/relic/open/export?format=pdf", None).await?;
    assert_eq!(pdf.status(), StatusCode::NOT_IMPLEMENTED);
    let said = json_of(pdf).await?;
    let Some(error) = said["error"].as_str() else {
        anyhow::bail!("the refusal carries no error text: {said}");
    };
    assert!(error.contains("--pdf-renderer"), "{said}");
    assert!(pages.read().is_empty());
    Ok(())
}
