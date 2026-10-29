//! PDF copies of relics, for the viewer's download. A faithful PDF of a page
//! takes a browser engine, so the server hands the page to a Gotenberg service
//! (gotenberg.dev: headless Chromium behind an HTTP API) that the operator runs
//! beside it, and returns what it prints. Optional: without one, a relic still
//! downloads as HTML and Markdown.
//!
//! The page goes to Chromium as an uploaded file and prints with its
//! backgrounds and its own `@page` size. What it may fetch while it renders
//! (web fonts, a charting library) is the renderer's to restrict; the Compose
//! service refuses private addresses, so a page cannot reach into the network
//! the server sits on.

use std::time::Duration;

use antumbra_core::{AntumbraError, Result};

/// How long one page may take to print, fonts and scripts included.
const TIMEOUT_SECS: u64 = 60;
/// The most PDF read back from the renderer.
const MAX_PDF_BYTES: u64 = 64 * 1024 * 1024;

/// The settings every print takes: the page's colors and bars kept, its own
/// `@page` size honored when it sets one, and a second for web fonts to land.
const FIELDS: &[(&str, &str)] = &[
    ("printBackground", "true"),
    ("preferCssPageSize", "true"),
    ("waitDelay", "1s"),
];

/// A Gotenberg service that prints HTML pages as PDF.
pub struct PdfRenderer {
    url: String,
    agent: ureq::Agent,
}

impl PdfRenderer {
    /// Print through the Gotenberg service at `base`, e.g. `http://pdf:3000`.
    pub fn new(base: &str) -> anyhow::Result<Self> {
        let base = base.trim().trim_end_matches('/');
        if !(base.starts_with("http://") || base.starts_with("https://"))
            || base.contains(char::is_whitespace)
        {
            anyhow::bail!(
                "--pdf-renderer must be an http or https URL such as http://pdf:3000, not {base:?}"
            );
        }
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(TIMEOUT_SECS)))
            .http_status_as_error(false)
            .build()
            .into();
        Ok(Self {
            url: format!("{base}/forms/chromium/convert/html"),
            agent,
        })
    }

    /// The PDF of one HTML page.
    pub async fn render(&self, html: Vec<u8>) -> Result<Vec<u8>> {
        let agent = self.agent.clone();
        let url = self.url.clone();
        tokio::task::spawn_blocking(move || {
            let boundary = boundary_for(&html);
            let body = form(&boundary, &html);
            let mut resp = agent
                .post(&url)
                .header(
                    "content-type",
                    &format!("multipart/form-data; boundary={boundary}"),
                )
                .send(&body[..])
                .map_err(|e| {
                    AntumbraError::other(format!("the PDF renderer is unreachable: {e}"))
                })?;
            let status = resp.status();
            if !status.is_success() {
                let said = resp.body_mut().read_to_string().unwrap_or_default();
                let said: String = said.chars().take(200).collect();
                return Err(AntumbraError::other(format!(
                    "the PDF renderer answered {}: {}",
                    status.as_u16(),
                    said.trim()
                )));
            }
            resp.body_mut()
                .with_config()
                .limit(MAX_PDF_BYTES)
                .read_to_vec()
                .map_err(|e| AntumbraError::other(format!("the PDF renderer's answer: {e}")))
        })
        .await
        .map_err(|e| AntumbraError::other(format!("the PDF task panicked: {e}")))?
    }
}

/// The `multipart/form-data` body Gotenberg's HTML route reads: the page as
/// the file `index.html`, then [`FIELDS`].
fn form(boundary: &str, html: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(html.len() + 512);
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"index.html\"\r\n\
             Content-Type: text/html; charset=utf-8\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(html);
    body.extend_from_slice(b"\r\n");
    for (name, value) in FIELDS {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

/// A boundary the page does not contain, so the form cannot be cut short by
/// what the page says.
fn boundary_for(html: &[u8]) -> String {
    (0u32..)
        .map(|n| format!("antumbra-relic-{n:08x}"))
        .find(|b| !html.windows(b.len()).any(|w| w == b.as_bytes()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_goes_as_index_html_with_the_print_settings() {
        let body = String::from_utf8(form("b0", b"<h1>Plan</h1>")).unwrap();
        assert!(body.starts_with(
            "--b0\r\nContent-Disposition: form-data; name=\"files\"; filename=\"index.html\"\r\n\
             Content-Type: text/html; charset=utf-8\r\n\r\n<h1>Plan</h1>\r\n"
        ));
        for (name, value) in FIELDS {
            assert!(
                body.contains(&format!("name=\"{name}\"\r\n\r\n{value}\r\n")),
                "{name}"
            );
        }
        assert!(body.ends_with("--b0--\r\n"));
    }

    #[test]
    fn the_boundary_is_never_in_the_page() {
        assert_eq!(boundary_for(b"<p>plain</p>"), "antumbra-relic-00000000");
        assert_eq!(
            boundary_for(b"antumbra-relic-00000000 antumbra-relic-00000001"),
            "antumbra-relic-00000002"
        );
    }

    /// Prints through a real renderer: `ANTUMBRA_TEST_PDF_RENDERER` names it
    /// (e.g. `docker run -p 127.0.0.1:3000:3000 gotenberg/gotenberg:8.37.0`),
    /// `ANTUMBRA_TEST_PDF_PAGE` optionally names a page to print and
    /// `ANTUMBRA_TEST_PDF_OUT` where to keep the result.
    #[tokio::test]
    #[ignore = "needs a running Gotenberg"]
    async fn prints_through_a_real_renderer() {
        let url = std::env::var("ANTUMBRA_TEST_PDF_RENDERER").expect("ANTUMBRA_TEST_PDF_RENDERER");
        let html = match std::env::var("ANTUMBRA_TEST_PDF_PAGE") {
            Ok(path) => std::fs::read(path).unwrap(),
            Err(_) => b"<!doctype html><title>t</title><h1>Printed</h1>".to_vec(),
        };
        let pdf = PdfRenderer::new(&url).unwrap().render(html).await.unwrap();
        assert!(pdf.starts_with(b"%PDF-"), "{:?}", &pdf[..pdf.len().min(16)]);
        if let Ok(out) = std::env::var("ANTUMBRA_TEST_PDF_OUT") {
            std::fs::write(out, &pdf).unwrap();
        }
    }

    #[test]
    fn the_renderer_is_an_http_url() {
        let r = PdfRenderer::new("http://pdf:3000/").unwrap();
        assert_eq!(r.url, "http://pdf:3000/forms/chromium/convert/html");
        for bad in ["pdf:3000", "ftp://pdf", "http://pdf 3000", ""] {
            assert!(PdfRenderer::new(bad).is_err(), "{bad:?}");
        }
    }
}
