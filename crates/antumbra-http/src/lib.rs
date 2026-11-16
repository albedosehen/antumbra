//! HTTP as Antumbra's own binaries speak it to the server.
//!
//! Three doors called `POST /mcp/call` with their own client, and each refused
//! its own way: one read `401` as a refused token before reading the server's
//! words, one never did, and one dropped the body, so a console user saw
//! "answered 400" where the CLI showed what the server said. The client here
//! is the one they share; [`SurfaceError`] is the one set of refusals.
//!
//! Security: the surface address and the token come from the operator's own
//! settings or flags, never from the server or from stored data, so this is
//! not a request-forgery sink. The token rides the `Authorization` header and
//! appears in no error; [`SurfaceClient`] has no `Debug` impl, so it cannot
//! leak through `{:?}`.

use std::time::Duration;

use serde_json::{json, Value};

/// An agent that waits `patience` for a whole call and hands back a refusal's
/// body rather than turning its status into an error, so the server's own
/// words reach whoever asked.
pub fn agent(patience: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(patience))
        .http_status_as_error(false)
        .build()
        .into()
}

/// The header a client names its machine in, which the server stamps on what
/// that machine writes. The hooks send it from `ANTUMBRA_HOST_ID`.
pub const DEVICE_HEADER: &str = "X-Antumbra-Host";

/// The name this process's machine goes by, as the hooks name it: the
/// `ANTUMBRA_HOST_ID` the operator set, lower-cased; `None` when nothing
/// names it, and the server stamps its writes as before.
pub fn host_id() -> Option<String> {
    std::env::var("ANTUMBRA_HOST_ID")
        .ok()
        .map(|h| h.trim().to_lowercase())
        .filter(|h| !h.is_empty())
}

/// One tool call's failure, in words the one who asked can act on. Every door
/// shows the same ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceError {
    /// No answer at all: the address is wrong, the server is down, or the call
    /// outlasted the client's patience.
    Unreachable {
        tool: String,
        url: String,
        reason: String,
    },
    /// `401` or `403`: the token was not accepted, with the server's words
    /// when it gave any.
    TokenRefused {
        tool: String,
        url: String,
        status: u16,
        said: Option<String>,
    },
    /// The server answered with an `error`: its own words, whatever the status.
    Refused {
        tool: String,
        status: u16,
        said: String,
    },
    /// A failing status with no words to go with it.
    Answered {
        tool: String,
        url: String,
        status: u16,
    },
    /// A success whose body is not JSON.
    Unreadable { tool: String, url: String },
}

impl std::fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable { tool, url, reason } => {
                write!(f, "{tool}: could not reach {url}: {reason}")
            }
            Self::TokenRefused {
                tool, url, said, ..
            } => {
                write!(
                    f,
                    "{tool}: the server at {url} refused the token; it may have expired, or be \
                     for another server"
                )?;
                match said {
                    Some(said) => write!(f, " ({said})"),
                    None => Ok(()),
                }
            }
            Self::Refused { tool, said, .. } => write!(f, "{tool}: {said}"),
            Self::Answered { tool, url, status } => {
                write!(f, "{tool}: the server at {url} answered {status}")
            }
            Self::Unreadable { tool, url } => {
                write!(
                    f,
                    "{tool}: the server at {url} sent an answer this client cannot read"
                )
            }
        }
    }
}

impl std::error::Error for SurfaceError {}

/// The client for the server's tool surface, `POST {surface}/mcp/call`.
pub struct SurfaceClient {
    agent: ureq::Agent,
    url: String,
    token: Option<String>,
    device: Option<String>,
}

impl SurfaceClient {
    /// A client for the server at `surface` (a trailing slash is fine),
    /// presenting `token` when there is one, waiting `patience` for a call.
    pub fn new(surface: &str, token: Option<&str>, patience: Duration) -> Self {
        Self::with_agent(surface, token, agent(patience))
    }

    /// The same, on an agent the caller already has (a console that also
    /// reads the server's other endpoints with it).
    pub fn with_agent(surface: &str, token: Option<&str>, agent: ureq::Agent) -> Self {
        Self {
            agent,
            url: surface.trim_end_matches('/').to_string(),
            token: token.map(str::to_owned),
            device: None,
        }
    }

    /// Name the machine the calls come from, so the server stamps what they
    /// write with it; `None` leaves the header off.
    pub fn from_machine(mut self, device: Option<String>) -> Self {
        self.device = device.filter(|d| !d.trim().is_empty());
        self
    }

    /// The server's address, without a trailing slash.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Call `tool` with `arguments`; its answer on success, and on anything
    /// else one of the [`SurfaceError`]s, in this order: no answer; `401` or
    /// `403`; an `error` in the body, whatever the status; a failing status
    /// with no words; a success that is not JSON.
    pub fn call(&self, tool: &str, arguments: Value) -> Result<Value, SurfaceError> {
        let endpoint = format!("{}/mcp/call", self.url);
        let mut request = self
            .agent
            .post(&endpoint)
            .header("content-type", "application/json");
        if let Some(token) = &self.token {
            request = request.header("authorization", &format!("Bearer {token}"));
        }
        if let Some(device) = &self.device {
            request = request.header(DEVICE_HEADER, device);
        }
        let mut response = request
            .send_json(json!({ "tool": tool, "arguments": arguments }))
            .map_err(|e| SurfaceError::Unreachable {
                tool: tool.into(),
                url: self.url.clone(),
                reason: e.to_string(),
            })?;
        let status = response.status();
        let body: Option<Value> = response.body_mut().read_json().ok();
        let said = body
            .as_ref()
            .and_then(|b| b.get("error"))
            .map(|e| match e.as_str() {
                Some(text) => text.to_string(),
                None => e.to_string(),
            });
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(SurfaceError::TokenRefused {
                tool: tool.into(),
                url: self.url.clone(),
                status: status.as_u16(),
                said,
            });
        }
        if let Some(said) = said {
            return Err(SurfaceError::Refused {
                tool: tool.into(),
                status: status.as_u16(),
                said,
            });
        }
        if !status.is_success() {
            return Err(SurfaceError::Answered {
                tool: tool.into(),
                url: self.url.clone(),
                status: status.as_u16(),
            });
        }
        body.ok_or_else(|| SurfaceError::Unreadable {
            tool: tool.into(),
            url: self.url.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread::JoinHandle;

    use anyhow::{anyhow, bail, ensure, Context};

    use super::*;

    const PATIENCE: Duration = Duration::from_secs(5);

    /// What a case asserts of the client's answer.
    type Holds = fn(&Result<Value, SurfaceError>) -> bool;

    /// A server that answers one request with `status` and `body`, and hands
    /// back the request it read.
    fn serve_once(status: u16, body: &'static str) -> anyhow::Result<(String, Served)> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}/", listener.local_addr()?);
        let thread = std::thread::spawn(move || -> anyhow::Result<String> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(PATIENCE))?;
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = stream.read(&mut chunk)?;
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&chunk[..n]);
                let Some(head_end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
                let length = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if request.len() >= head_end + 4 + length {
                    break;
                }
            }
            let reason = match status {
                200 => "OK",
                400 => "Bad Request",
                401 => "Unauthorized",
                403 => "Forbidden",
                404 => "Not Found",
                _ => "Error",
            };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes())?;
            stream.flush()?;
            Ok(String::from_utf8_lossy(&request).into_owned())
        });
        Ok((url, Served(thread)))
    }

    struct Served(JoinHandle<anyhow::Result<String>>);

    impl Served {
        fn request(self) -> anyhow::Result<String> {
            self.0
                .join()
                .map_err(|_| anyhow!("the server thread panicked"))?
        }
    }

    #[test]
    fn every_answer_has_its_one_refusal() -> anyhow::Result<()> {
        let cases: [(u16, &str, Holds, &str); 8] = [
            (
                200,
                r#"{"routes":[]}"#,
                |r| matches!(r, Ok(v) if v["routes"].is_array()),
                "a success is its body",
            ),
            (
                400,
                r#"{"error":"task is required"}"#,
                |r| {
                    matches!(r, Err(SurfaceError::Refused { status: 400, said, .. })
                        if said == "task is required")
                },
                "a 400 with an error is the server's words",
            ),
            (
                200,
                r#"{"error":{"code":7}}"#,
                |r| {
                    matches!(r, Err(SurfaceError::Refused { status: 200, said, .. })
                        if said == r#"{"code":7}"#)
                },
                "an error in a success is still a refusal, rendered as JSON",
            ),
            (
                401,
                r#"{"error":"token expired"}"#,
                |r| {
                    matches!(r, Err(SurfaceError::TokenRefused { status: 401, said: Some(s), .. })
                        if s == "token expired")
                },
                "a 401 is the token, with the server's words",
            ),
            (
                403,
                "",
                |r| {
                    matches!(
                        r,
                        Err(SurfaceError::TokenRefused {
                            status: 403,
                            said: None,
                            ..
                        })
                    )
                },
                "a 403 is the token, without words",
            ),
            (
                404,
                "",
                |r| matches!(r, Err(SurfaceError::Answered { status: 404, .. })),
                "a 404 without words is the status",
            ),
            (
                502,
                "<html>bad gateway</html>",
                |r| matches!(r, Err(SurfaceError::Answered { status: 502, .. })),
                "a failing status with a non-JSON body is the status",
            ),
            (
                200,
                "not json",
                |r| matches!(r, Err(SurfaceError::Unreadable { .. })),
                "a success that is not JSON is unreadable",
            ),
        ];
        for (status, body, holds, what) in cases {
            let (url, served) = serve_once(status, body)?;
            let client = SurfaceClient::new(&url, Some("t0ken"), PATIENCE);
            let answer = client.call("route", json!({ "task": "add" }));
            ensure!(holds(&answer), "{what}: {answer:?}");
            if let Err(e) = &answer {
                ensure!(e.to_string().starts_with("route: "), "{what}: {e}");
            }
            served.request()?;
        }
        Ok(())
    }

    #[test]
    fn the_request_carries_the_token_the_machine_and_the_call() -> anyhow::Result<()> {
        let (url, served) = serve_once(200, "{}")?;
        let client = SurfaceClient::new(&url, Some("t0ken"), PATIENCE)
            .from_machine(Some("one-device".into()));
        ensure!(
            !client.url().ends_with('/'),
            "the trailing slash is trimmed"
        );
        client
            .call("devices", json!({}))
            .map_err(|e| anyhow!("{e}"))?;
        let request = served.request()?;
        let (head, body) = request
            .split_once("\r\n\r\n")
            .context("a head and a body")?;
        let head = head.to_lowercase();
        ensure!(head.starts_with("post /mcp/call http/1.1"), "{head}");
        ensure!(head.contains("authorization: bearer t0ken"), "{head}");
        ensure!(head.contains("x-antumbra-host: one-device"), "{head}");
        ensure!(head.contains("content-type: application/json"), "{head}");
        let sent: Value = serde_json::from_str(body)?;
        ensure!(
            sent == json!({ "tool": "devices", "arguments": {} }),
            "{sent}"
        );
        Ok(())
    }

    #[test]
    fn no_token_and_no_machine_send_no_header() -> anyhow::Result<()> {
        let (url, served) = serve_once(200, "{}")?;
        SurfaceClient::new(&url, None, PATIENCE)
            .from_machine(Some("  ".into()))
            .call("devices", json!({}))
            .map_err(|e| anyhow!("{e}"))?;
        let head = served.request()?.to_lowercase();
        ensure!(!head.contains("authorization:"), "{head}");
        ensure!(!head.contains("x-antumbra-host:"), "{head}");
        Ok(())
    }

    #[test]
    fn a_server_that_is_not_there_is_unreachable() -> anyhow::Result<()> {
        let url = {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            format!("http://{}", listener.local_addr()?)
        };
        let Err(e) = SurfaceClient::new(&url, None, PATIENCE).call("route", json!({})) else {
            bail!("a closed port answered");
        };
        ensure!(matches!(e, SurfaceError::Unreachable { .. }), "{e}");
        ensure!(e.to_string().starts_with("route: could not reach "), "{e}");
        Ok(())
    }

    #[test]
    fn the_refusals_read_as_one_set() {
        let tool = || "route".to_string();
        let url = || "http://workspace.example".to_string();
        assert_eq!(
            SurfaceError::TokenRefused {
                tool: tool(),
                url: url(),
                status: 401,
                said: None
            }
            .to_string(),
            "route: the server at http://workspace.example refused the token; it may have \
             expired, or be for another server"
        );
        assert_eq!(
            SurfaceError::TokenRefused {
                tool: tool(),
                url: url(),
                status: 401,
                said: Some("token expired".into())
            }
            .to_string(),
            "route: the server at http://workspace.example refused the token; it may have \
             expired, or be for another server (token expired)"
        );
        assert_eq!(
            SurfaceError::Refused {
                tool: tool(),
                status: 400,
                said: "task is required".into()
            }
            .to_string(),
            "route: task is required"
        );
        assert_eq!(
            SurfaceError::Answered {
                tool: tool(),
                url: url(),
                status: 404
            }
            .to_string(),
            "route: the server at http://workspace.example answered 404"
        );
    }
}
