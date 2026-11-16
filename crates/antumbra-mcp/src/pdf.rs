//! PDF copies of relics, for the viewer's download. A faithful PDF of a page
//! takes a browser engine, so the server hands the page to a Gotenberg service
//! (gotenberg.dev: headless Chromium behind an HTTP API) that the operator runs
//! beside it, and returns what it prints. Optional: without one, a relic still
//! downloads as HTML and Markdown.
//!
//! The page goes to Chromium as an uploaded file and prints with its
//! backgrounds and its own `@page` size. What it may fetch while it renders
//! (web fonts, a charting library) is restricted twice: the renderer refuses
//! private addresses when started as the compose profile starts it
//! (`--chromium-deny-private-ips=true`), and before every print this module
//! replaces a reference to a private, loopback or link-local address, literal
//! or resolved, with `about:blank` ([`without_private_hosts`]), so a page
//! cannot reach into the network the server sits on where the renderer was
//! started otherwise.

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::Duration;

use antumbra_core::{AntumbraError, Result};

/// How long one page may take to print, fonts and scripts included.
const TIMEOUT_SECS: u64 = 60;
/// How long one host name may take to resolve before a print; one that does
/// not resolve in time is left as it is, since the renderer cannot reach it
/// either.
const RESOLVE_TIMEOUT: Duration = Duration::from_secs(2);
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

    /// The PDF of one HTML page, with its references to private addresses made
    /// inert first.
    pub async fn render(&self, html: Vec<u8>) -> Result<Vec<u8>> {
        let html = without_private_hosts(html).await;
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

/// `html` with every URL whose host is a private, loopback, link-local or
/// unspecified address, or a name that resolves to one, replaced by
/// `about:blank`. Each name is resolved once, under [`RESOLVE_TIMEOUT`]; one
/// that does not resolve is left as it is.
pub async fn without_private_hosts(html: Vec<u8>) -> Vec<u8> {
    let text = String::from_utf8_lossy(&html);
    let mut verdicts: HashMap<String, bool> = HashMap::new();
    for host in hosts_in(&text) {
        let private = match host.parse::<IpAddr>() {
            Ok(ip) => is_private_ip(ip),
            Err(_) => resolves_private(&host).await,
        };
        verdicts.insert(host, private);
    }
    if !verdicts.values().any(|&private| private) {
        return html;
    }
    without_references_to(&text, &|host| verdicts.get(host).copied().unwrap_or(false)).into_bytes()
}

/// Whether `host` resolves to a private address, taking any such answer as yes.
async fn resolves_private(host: &str) -> bool {
    match tokio::time::timeout(RESOLVE_TIMEOUT, tokio::net::lookup_host((host, 0u16))).await {
        Ok(Ok(addrs)) => addrs.into_iter().any(|addr| is_private_ip(addr.ip())),
        _ => false,
    }
}

/// An address a page has no business fetching from while it prints: loopback,
/// RFC 1918 and the shared range, link-local (the cloud metadata address among
/// them), unspecified and broadcast, and their IPv6 counterparts.
fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, _, _] = v4.octets();
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || (a == 100 && (64..=127).contains(&b))
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| is_private_ip(IpAddr::V4(v4)))
        }
    }
}

/// The URL schemes a page fetches over.
const SCHEMES: [&str; 4] = ["http", "https", "ws", "wss"];

/// What ends a URL written into HTML, CSS or script.
fn ends_a_url(c: char) -> bool {
    c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | ')' | '`' | '\\')
}

/// Every URL in `text` as `(start, end, host)`, the host lowercased, without
/// userinfo, brackets or port.
fn url_spans(text: &str) -> Vec<(usize, usize, String)> {
    let mut spans = Vec::new();
    for (sep, _) in text.match_indices("://") {
        // The scheme is the letters just before the separator.
        let scheme_start = text[..sep]
            .char_indices()
            .rev()
            .take_while(|(_, c)| c.is_ascii_alphabetic())
            .last()
            .map_or(sep, |(i, _)| i);
        if !SCHEMES.contains(&text[scheme_start..sep].to_ascii_lowercase().as_str()) {
            continue;
        }
        let rest = &text[sep + 3..];
        let end = sep + 3 + rest.find(ends_a_url).unwrap_or(rest.len());
        let authority = &text[sep + 3..end];
        let authority = authority.split(['/', '?', '#']).next().unwrap_or(authority);
        let authority = authority.rsplit('@').next().unwrap_or(authority);
        let host = match authority.strip_prefix('[') {
            Some(bracketed) => bracketed.split(']').next().unwrap_or(bracketed),
            None => match authority.rsplit_once(':') {
                Some((host, port))
                    if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) =>
                {
                    host
                }
                _ => authority,
            },
        };
        if !host.is_empty() {
            spans.push((scheme_start, end, host.to_ascii_lowercase()));
        }
    }
    spans
}

/// The hosts `text` references, once each.
fn hosts_in(text: &str) -> Vec<String> {
    let mut hosts: Vec<String> = url_spans(text).into_iter().map(|(_, _, h)| h).collect();
    hosts.sort();
    hosts.dedup();
    hosts
}

/// `text` with every URL whose host `is_private` replaced by `about:blank`.
fn without_references_to(text: &str, is_private: &dyn Fn(&str) -> bool) -> String {
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for (start, end, host) in url_spans(text) {
        if start < at || !is_private(&host) {
            continue;
        }
        out.push_str(&text[at..start]);
        out.push_str("about:blank");
        at = end;
    }
    out.push_str(&text[at..]);
    out
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, Ipv6Addr};

    use super::*;

    #[test]
    fn references_to_private_addresses_become_blank_and_public_ones_stay() {
        let loopback = Ipv4Addr::LOCALHOST;
        let lan = Ipv4Addr::new(10, 1, 2, 3);
        let metadata = Ipv4Addr::new(169, 254, 169, 254);
        let page = format!(
            r#"<img src="http://{loopback}/a"> <script src="HTTPS://{lan}:8443/b.js"></script>
               <link href='http://[::1]/c.css'> <style>@import url(http://{metadata}/latest/meta-data)</style>
               <a href="https://user:pw@{lan}/d">x</a> <img src="https://cdn.example/lib.js">
               <img src="http://internal.example/e"> <img src="data:image/png;base64,AAAA">
               <a href="ws://{loopback}:9000/socket">s</a>"#
        );
        let verdict = |host: &str| {
            host.parse::<IpAddr>()
                .map(is_private_ip)
                .unwrap_or(host == "internal.example")
        };
        let out = without_references_to(&page, &verdict);
        for gone in [
            loopback.to_string(),
            lan.to_string(),
            "[::1]".to_string(),
            metadata.to_string(),
            "internal.example".to_string(),
            "user:pw@".to_string(),
        ] {
            assert!(!out.contains(&gone), "{gone} is still in: {out}");
        }
        assert!(out.contains("https://cdn.example/lib.js"), "{out}");
        assert!(out.contains("data:image/png;base64,AAAA"), "{out}");
        assert_eq!(out.matches("about:blank").count(), 7, "{out}");
    }

    #[test]
    fn hosts_are_read_once_each_without_port_userinfo_or_brackets() {
        let text = "see http://A.example:8080/x and https://a.example/y, ws://[::1]:1/z, \
                    https://u:p@b.example/w, mailto:someone@c.example, and plain text";
        assert_eq!(hosts_in(text), vec!["::1", "a.example", "b.example"]);
    }

    #[test]
    fn the_private_ranges_are_the_ones_a_page_may_not_reach() {
        for private in [
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254)),
            IpAddr::V4(Ipv4Addr::new(100, 64, 0, 1)),
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            IpAddr::V6(Ipv6Addr::LOCALHOST),
            IpAddr::V6(Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 1)),
            IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1)),
            IpAddr::V6(Ipv4Addr::new(10, 0, 0, 1).to_ipv6_mapped()),
        ] {
            assert!(is_private_ip(private), "{private}");
        }
        for public in [
            IpAddr::V4(Ipv4Addr::new(203, 0, 113, 5)),
            IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7)),
            IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1)),
        ] {
            assert!(!is_private_ip(public), "{public}");
        }
    }

    // The whole path a print takes: literal private addresses go without any
    // resolving, and a name that cannot resolve stays, since the renderer
    // cannot reach it either.
    #[tokio::test]
    async fn a_page_is_scrubbed_before_it_prints() {
        let lan = Ipv4Addr::new(192, 168, 7, 7);
        let page = format!(
            r#"<img src="http://{lan}/pixel"> <img src="https://nowhere.invalid/x"> <p>text</p>"#
        );
        let out =
            String::from_utf8_lossy(&without_private_hosts(page.into_bytes()).await).into_owned();
        assert!(!out.contains(&lan.to_string()), "{out}");
        assert!(out.contains("https://nowhere.invalid/x"), "{out}");
        assert!(out.contains("<p>text</p>"), "{out}");
        let plain = b"<p>no references</p>".to_vec();
        assert_eq!(without_private_hosts(plain.clone()).await, plain);
    }

    #[test]
    fn the_compose_renderer_is_started_denying_private_addresses() {
        let compose = include_str!("../../../docker/docker-compose.yml");
        assert!(
            compose.contains("--chromium-deny-private-ips=true"),
            "the pdf service must deny private addresses"
        );
    }

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
