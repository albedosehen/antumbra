//! HTTP as Antumbra's own binaries speak it, to the server and to the
//! endpoints beside it.
//!
//! Three doors called `POST /mcp/call` with their own client, and each refused
//! its own way: one read `401` as a refused token before reading the server's
//! words, one never did, and one dropped the body, so a console user saw
//! "answered 400" where the CLI showed what the server said. The client here
//! is the one they share; [`SurfaceError`] is the one set of refusals.
//!
//! Four crates built a `ureq` agent with a patience from the environment and
//! posted JSON, and two of them turned a refusal into a bare status, so an
//! embedder's or a reranker's `4xx` lost the reason. [`agent`], [`patience`],
//! [`JsonPost`] with its [`UreqJson`], and [`read_json_or_refusal`] are the one
//! transport, and every refusal carries the endpoint's own words.
//!
//! Security: every address and token here comes from the operator's own
//! settings or flags, never from the server or from stored data, so none of
//! this is a request-forgery sink. A token rides the `Authorization` header
//! and appears in no error; [`SurfaceClient`] and [`UreqJson`] have no `Debug`
//! impl, so a key cannot leak through `{:?}`.

use std::str::FromStr;
use std::time::Duration;

use serde_json::{json, Value};

use antumbra_core::{AntumbraError, Result};

/// An agent that waits `patience` for a whole call and hands back a refusal's
/// body rather than turning its status into an error, so the endpoint's own
/// words reach whoever asked.
pub fn agent(patience: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(patience))
        .http_status_as_error(false)
        .build()
        .into()
}

/// A positive setting from the environment, else `default`: an unset `name`,
/// an empty value, one that is not a number and a zero all mean the default,
/// so an operator cannot set a patience of nothing by mistake.
pub fn env_positive<T>(name: &str, default: T) -> T
where
    T: FromStr + PartialOrd + Default,
{
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<T>().ok())
        .filter(|v| *v > T::default())
        .unwrap_or(default)
}

/// How long one call to an endpoint may take: the seconds in `name` when the
/// operator set them, else `default_secs`. Without a bound a hung endpoint
/// pins a worker forever, and the server runs a tool's embed under its
/// per-request lock, so an unbounded call would stall every tenant.
pub fn patience(name: &str, default_secs: u64) -> Duration {
    Duration::from_secs(env_positive(name, default_secs))
}

/// What an endpoint said in a refusal, in one line. Antumbra's own surface and
/// text-embeddings-inference say `{"error": "text"}`, copal says
/// `{"error": {"kind", "message"}}`, GitHub says `{"message": "text"}`;
/// anything else is shown as it came, cut short, and no body is said to be
/// none.
pub fn refusal(body: &str) -> String {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let said = parsed
        .as_ref()
        .and_then(|v| v.get("error").or_else(|| v.get("message")));
    match said {
        Some(Value::Object(error)) => format!(
            "{} ({})",
            error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("no message"),
            error
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("no kind")
        ),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None if body.trim().is_empty() => "no body".to_string(),
        None => body.chars().take(200).collect(),
    }
}

/// A response's JSON, or an error carrying the endpoint's own account of why
/// it refused: `{what} answered {status}: {refusal}`.
pub fn read_json_or_refusal(
    what: &str,
    mut resp: ureq::http::Response<ureq::Body>,
) -> Result<Value> {
    let status = resp.status();
    if status.is_success() {
        return resp
            .body_mut()
            .read_json::<Value>()
            .map_err(|e| AntumbraError::other(format!("{what}: the answer was not JSON: {e}")));
    }
    let text = resp.body_mut().read_to_string().unwrap_or_default();
    Err(AntumbraError::other(format!(
        "{what} answered {}: {}",
        status.as_u16(),
        refusal(&text)
    )))
}

/// The one network operation an embedder or a reranker performs: POST a JSON
/// body, with a bearer when the endpoint wants one, and read the JSON answer.
/// Behind a trait so their request shaping and parsing are tested without a
/// socket.
pub trait JsonPost: Send + Sync {
    fn post(&self, url: &str, bearer: Option<&str>, body: &Value) -> Result<Value>;
}

/// The production [`JsonPost`]: a blocking `ureq` POST over an agent with a
/// bounded patience, so a dead endpoint fails fast instead of hanging the
/// worker, and a refusal comes back with the endpoint's words. `what` names
/// the caller in its errors (`embed`, `rerank`).
pub struct UreqJson {
    what: &'static str,
    agent: ureq::Agent,
}

impl UreqJson {
    pub fn new(what: &'static str, patience: Duration) -> Self {
        Self {
            what,
            agent: agent(patience),
        }
    }
}

impl JsonPost for UreqJson {
    fn post(&self, url: &str, bearer: Option<&str>, body: &Value) -> Result<Value> {
        let mut req = self
            .agent
            .post(url)
            .header("content-type", "application/json");
        if let Some(key) = bearer {
            req = req.header("authorization", &format!("Bearer {key}"));
        }
        let resp = req
            .send_json(body)
            .map_err(|e| AntumbraError::other(format!("{} POST {url} failed: {e}", self.what)))?;
        read_json_or_refusal(&format!("{} POST {url}", self.what), resp)
    }
}

/// The header a client names its machine in, which the server stamps on what
/// that machine writes. The hooks send it from `ANTUMBRA_HOST_ID`.
pub const DEVICE_HEADER: &str = "X-Antumbra-Host";

/// This machine's operating system as a person would name it: `macOS 15.6`,
/// `Windows 11 Pro`, `Ubuntu 24.04.1 LTS`. `None` when it could not be told.
/// Found once a process.
///
/// `ANTUMBRA_HOST_OS` says it instead when set: a server in a container
/// would otherwise name its image's system, not the machine's, and setup
/// passes the machine's in.
pub fn os_label() -> Option<String> {
    static FOUND: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    FOUND
        .get_or_init(|| told_os(std::env::var("ANTUMBRA_HOST_OS").ok().as_deref()).or_else(find_os))
        .clone()
}

/// The operating system a deployment named, when it named one.
fn told_os(told: Option<&str>) -> Option<String> {
    told.map(str::trim)
        .filter(|os| !os.is_empty())
        .map(str::to_string)
}

fn said(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn find_os() -> Option<String> {
    match std::env::consts::OS {
        "macos" => Some(match said("sw_vers", &["-productVersion"]) {
            Some(version) => format!("macOS {version}"),
            None => "macOS".to_string(),
        }),
        "windows" => {
            let key = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion";
            let value = |name: &str| {
                said("reg", &["query", key, "/v", name]).and_then(|out| reg_value(&out, name))
            };
            Some(windows_name(
                value("ProductName").as_deref(),
                value("CurrentBuildNumber").as_deref(),
            ))
        }
        "linux" => Some(
            std::fs::read_to_string("/etc/os-release")
                .ok()
                .and_then(|text| os_release_name(&text))
                .unwrap_or_else(|| "Linux".to_string()),
        ),
        other => Some(other.to_string()),
    }
}

/// The value `reg query ... /v name` printed: the line's text after its type.
fn reg_value(out: &str, name: &str) -> Option<String> {
    out.lines().find_map(|line| {
        let rest = line.trim().strip_prefix(name)?.trim_start();
        let value = rest.strip_prefix("REG_SZ")?.trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

/// Windows as it names itself. The registry's product name still says
/// `Windows 10` on Windows 11, which is told by its build (22000 and later).
fn windows_name(product: Option<&str>, build: Option<&str>) -> String {
    let product = product.unwrap_or("Windows");
    let eleven = build.and_then(|b| b.trim().parse::<u32>().ok()) >= Some(22000);
    if eleven {
        product.replacen("Windows 10", "Windows 11", 1)
    } else {
        product.to_string()
    }
}

/// A distribution's name and version from `/etc/os-release`: its
/// `PRETTY_NAME`, else its `NAME` and `VERSION_ID`.
fn os_release_name(text: &str) -> Option<String> {
    let field = |key: &str| {
        text.lines().find_map(|line| {
            let value = line.trim().strip_prefix(key)?.strip_prefix('=')?;
            let value = value.trim().trim_matches(|c| c == '"' || c == '\'').trim();
            (!value.is_empty()).then(|| value.to_string())
        })
    };
    field("PRETTY_NAME").or_else(|| {
        let name = field("NAME")?;
        Some(match field("VERSION_ID") {
            Some(version) => format!("{name} {version}"),
            None => name,
        })
    })
}

/// The header an agent's connection asks for fewer tools in. A coding
/// session needs the `agent` profile's tools and pays, on every turn, for the
/// description of every tool it is advertised.
pub const TOOLS_HEADER: &str = "X-Antumbra-Tools";

/// What an agent's connection asks to be given: `agent` unless
/// `ANTUMBRA_AGENT_TOOLS` says otherwise (`all`, or tool names as the
/// server's `--tools` reads them); `None` for `all`, which asks nothing.
pub fn agent_tools() -> Option<String> {
    let asked = std::env::var("ANTUMBRA_AGENT_TOOLS").unwrap_or_default();
    match asked.trim() {
        "" => Some("agent".to_string()),
        all if all.eq_ignore_ascii_case("all") => None,
        other => Some(other.to_string()),
    }
}

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
    /// when it gave any. `presented` says whether there was a token to
    /// refuse: a call that carried none was turned away for want of one, and
    /// saying its token "may have expired" sends the reader looking for a
    /// token that was never sent.
    TokenRefused {
        tool: String,
        url: String,
        status: u16,
        said: Option<String>,
        presented: bool,
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
                tool,
                url,
                said,
                presented,
                ..
            } => {
                if *presented {
                    write!(
                        f,
                        "{tool}: the server at {url} refused the token; it may have expired, \
                         or be for another server"
                    )?;
                } else {
                    write!(
                        f,
                        "{tool}: the server at {url} wants a token and none was given; set \
                         ANTUMBRA_TOKEN, or run `antumbra setup`"
                    )?;
                }
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
    pub fn call(&self, tool: &str, arguments: Value) -> std::result::Result<Value, SurfaceError> {
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
                presented: self.token.is_some(),
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

/// What a crate's tests need to drive a transport without a real endpoint: a
/// canned [`JsonPost`], and a server on a local port that answers a few
/// requests with one status and body.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread::JoinHandle;
    use std::time::Duration;

    use serde_json::Value;

    use antumbra_core::{AntumbraError, Result};

    use super::JsonPost;

    /// A [`JsonPost`] that answers every call with the same canned result, so
    /// request shaping and parsing are tested without a socket.
    pub struct Canned(pub Result<Value>);

    impl JsonPost for Canned {
        fn post(&self, _url: &str, _bearer: Option<&str>, _body: &Value) -> Result<Value> {
            self.0
                .as_ref()
                .map(Clone::clone)
                .map_err(|e| AntumbraError::other(e.to_string()))
        }
    }

    /// How long the server waits for each request before giving up on it.
    const PATIENCE: Duration = Duration::from_secs(5);

    /// A server on a local port that answers `times` requests with `status`
    /// and `body` (one connection each), and hands back the requests it read.
    /// The URL has a trailing slash, so a client's trimming is exercised.
    pub fn serve(
        times: usize,
        status: u16,
        body: &'static str,
    ) -> std::io::Result<(String, Served)> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}/", listener.local_addr()?);
        let thread = std::thread::spawn(move || -> std::io::Result<Vec<String>> {
            let mut seen = Vec::with_capacity(times);
            for _ in 0..times {
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
                    409 => "Conflict",
                    422 => "Unprocessable Entity",
                    _ => "Error",
                };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes())?;
                stream.flush()?;
                seen.push(String::from_utf8_lossy(&request).into_owned());
            }
            Ok(seen)
        });
        Ok((url, Served(thread)))
    }

    /// The server's side of a [`serve`], joined for the requests it read.
    pub struct Served(JoinHandle<std::io::Result<Vec<String>>>);

    impl Served {
        /// Every request the server read, in order.
        pub fn requests(self) -> std::io::Result<Vec<String>> {
            self.0
                .join()
                .map_err(|_| std::io::Error::other("the server thread panicked"))?
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use anyhow::{anyhow, bail, ensure, Context};

    use super::testing::serve;
    use super::*;

    const PATIENCE: Duration = Duration::from_secs(5);

    /// What a case asserts of the client's answer.
    type Holds = fn(&std::result::Result<Value, SurfaceError>) -> bool;

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
            let (url, served) = serve(1, status, body)?;
            let client = SurfaceClient::new(&url, Some("t0ken"), PATIENCE);
            let answer = client.call("route", json!({ "task": "add" }));
            ensure!(holds(&answer), "{what}: {answer:?}");
            if let Err(e) = &answer {
                ensure!(e.to_string().starts_with("route: "), "{what}: {e}");
            }
            served.requests()?;
        }
        Ok(())
    }

    /// A `401` to a call that carried no token is for want of one, and one to
    /// a call that carried a token is that token's: the client knows which it
    /// sent, and the two read differently to whoever has to act on them.
    #[test]
    fn a_call_that_carried_no_token_is_told_none_was_given() -> anyhow::Result<()> {
        let (url, served) = serve(1, 401, r#"{"error":"missing bearer token"}"#)?;
        let Err(e) = SurfaceClient::new(&url, None, PATIENCE).call("route", json!({})) else {
            bail!("a 401 was taken for an answer");
        };
        ensure!(
            matches!(
                e,
                SurfaceError::TokenRefused {
                    presented: false,
                    ..
                }
            ),
            "{e:?}"
        );
        ensure!(
            e.to_string().contains("wants a token and none was given"),
            "{e}"
        );
        let request = served
            .requests()?
            .into_iter()
            .next()
            .context("one request")?;
        ensure!(
            !request.to_lowercase().contains("authorization:"),
            "no token, no header: {request}"
        );

        let (url, served) = serve(1, 401, r#"{"error":"token expired"}"#)?;
        let Err(e) = SurfaceClient::new(&url, Some("t0ken"), PATIENCE).call("route", json!({}))
        else {
            bail!("a 401 was taken for an answer");
        };
        ensure!(
            matches!(
                e,
                SurfaceError::TokenRefused {
                    presented: true,
                    ..
                }
            ),
            "{e:?}"
        );
        ensure!(e.to_string().contains("refused the token"), "{e}");
        served.requests()?;
        Ok(())
    }

    #[test]
    fn the_request_carries_the_token_the_machine_and_the_call() -> anyhow::Result<()> {
        let (url, served) = serve(1, 200, "{}")?;
        let client = SurfaceClient::new(&url, Some("t0ken"), PATIENCE)
            .from_machine(Some("one-device".into()));
        ensure!(
            !client.url().ends_with('/'),
            "the trailing slash is trimmed"
        );
        client
            .call("devices", json!({}))
            .map_err(|e| anyhow!("{e}"))?;
        let request = served
            .requests()?
            .into_iter()
            .next()
            .context("one request")?;
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
        let (url, served) = serve(1, 200, "{}")?;
        SurfaceClient::new(&url, None, PATIENCE)
            .from_machine(Some("  ".into()))
            .call("devices", json!({}))
            .map_err(|e| anyhow!("{e}"))?;
        let head = served
            .requests()?
            .into_iter()
            .next()
            .context("one request")?
            .to_lowercase();
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
    fn the_refusals_read_as_one_set() -> anyhow::Result<()> {
        let tool = || "route".to_string();
        let url = || "http://workspace.example".to_string();
        assert_eq!(
            SurfaceError::TokenRefused {
                tool: tool(),
                url: url(),
                status: 401,
                said: None,
                presented: true
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
                said: Some("token expired".into()),
                presented: true
            }
            .to_string(),
            "route: the server at http://workspace.example refused the token; it may have \
             expired, or be for another server (token expired)"
        );
        // A call that carried no token is told so, not that its token expired.
        assert_eq!(
            SurfaceError::TokenRefused {
                tool: tool(),
                url: url(),
                status: 401,
                said: Some("missing bearer token".into()),
                presented: false
            }
            .to_string(),
            "route: the server at http://workspace.example wants a token and none was given; \
             set ANTUMBRA_TOKEN, or run `antumbra setup` (missing bearer token)"
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
        Ok(())
    }

    #[test]
    fn a_refusal_reads_every_endpoints_shape() -> anyhow::Result<()> {
        // copal
        assert_eq!(
            refusal(
                r#"{"error":{"kind":"conflict","message":"idempotency key consumed by a deleted file"}}"#
            ),
            "idempotency key consumed by a deleted file (conflict)"
        );
        // text-embeddings-inference, and Antumbra's own surface
        assert_eq!(
            refusal(
                r#"{"error":"batch size 100 > maximum allowed batch size 32","error_type":"Validation"}"#
            ),
            "batch size 100 > maximum allowed batch size 32"
        );
        // GitHub
        assert_eq!(refusal(r#"{"message":"Not Found"}"#), "Not Found");
        // an error that is neither text nor an object is shown as JSON
        assert_eq!(refusal(r#"{"error":[1,2]}"#), "[1,2]");
        // anything else, as it came and cut short
        assert_eq!(refusal(""), "no body");
        assert_eq!(refusal("   "), "no body");
        assert_eq!(refusal("upstream timed out"), "upstream timed out");
        assert_eq!(refusal(&"x".repeat(500)).chars().count(), 200);
        Ok(())
    }

    #[test]
    fn a_json_post_surfaces_a_refusals_body_and_a_successs_json() -> anyhow::Result<()> {
        let (url, served) = serve(
            1,
            422,
            r#"{"error":"batch size 100 > maximum allowed batch size 32","error_type":"Validation"}"#,
        )?;
        let post = UreqJson::new("rerank", PATIENCE);
        let Err(e) = post.post(&url, Some("k3y"), &json!({ "query": "q" })) else {
            bail!("a 422 was accepted");
        };
        let shown = e.to_string();
        ensure!(
            shown.starts_with("rerank POST http://")
                && shown.contains("answered 422: batch size 100"),
            "{shown}"
        );
        let head = served
            .requests()?
            .into_iter()
            .next()
            .context("one request")?
            .to_lowercase();
        ensure!(head.contains("authorization: bearer k3y"), "{head}");

        let (url, served) = serve(1, 200, r#"{"data":[]}"#)?;
        let answer = post
            .post(&url, None, &json!({}))
            .map_err(|e| anyhow!("{e}"))?;
        ensure!(answer == json!({ "data": [] }), "{answer}");
        let head = served
            .requests()?
            .into_iter()
            .next()
            .context("one request")?
            .to_lowercase();
        ensure!(!head.contains("authorization:"), "{head}");

        let (url, _served) = serve(1, 200, "not json")?;
        let Err(e) = post.post(&url, None, &json!({})) else {
            bail!("a non-JSON success was accepted");
        };
        ensure!(e.to_string().contains("the answer was not JSON"), "{e}");
        Ok(())
    }

    #[test]
    fn a_setting_from_the_environment_must_be_positive() -> anyhow::Result<()> {
        // Read through a name no test sets, so the default is what comes back;
        // the parsing rules are exercised on values directly.
        assert_eq!(env_positive("ANTUMBRA_HTTP_TEST_UNSET_SETTING", 30u64), 30);
        let parse = |s: &str| {
            s.trim()
                .parse::<u64>()
                .ok()
                .filter(|v| *v > 0)
                .unwrap_or(30)
        };
        assert_eq!(parse("45"), 45);
        assert_eq!(parse(" 7 "), 7);
        assert_eq!(parse("0"), 30);
        assert_eq!(parse(""), 30);
        assert_eq!(parse("soon"), 30);
        assert_eq!(
            patience("ANTUMBRA_HTTP_TEST_UNSET_SETTING", 20),
            Duration::from_secs(20)
        );
        Ok(())
    }

    /// A machine's operating system is named from what the system says of
    /// itself: Windows 11 by its build, a distribution by its release file.
    #[test]
    fn an_operating_system_is_named_as_a_person_would() {
        let reg = "\r\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\r\n    ProductName    REG_SZ    Windows 10 Pro\r\n\r\n";
        assert_eq!(
            reg_value(reg, "ProductName").as_deref(),
            Some("Windows 10 Pro")
        );
        assert_eq!(reg_value(reg, "CurrentBuildNumber"), None);
        assert_eq!(
            windows_name(Some("Windows 10 Pro"), Some("26100")),
            "Windows 11 Pro"
        );
        assert_eq!(
            windows_name(Some("Windows 10 Pro"), Some("19045")),
            "Windows 10 Pro"
        );
        assert_eq!(
            windows_name(Some("Windows Server 2022 Standard"), Some("20348")),
            "Windows Server 2022 Standard"
        );
        assert_eq!(windows_name(None, None), "Windows");

        let ubuntu = "NAME=\"Ubuntu\"\nVERSION_ID=\"24.04\"\nPRETTY_NAME=\"Ubuntu 24.04.1 LTS\"\n";
        assert_eq!(
            os_release_name(ubuntu).as_deref(),
            Some("Ubuntu 24.04.1 LTS")
        );
        assert_eq!(
            os_release_name("NAME=Alpine\nVERSION_ID=3.20\n").as_deref(),
            Some("Alpine 3.20")
        );
        assert_eq!(
            os_release_name("NAME='Arch Linux'\n").as_deref(),
            Some("Arch Linux")
        );
        assert_eq!(os_release_name("ID=x\n"), None);
        assert_eq!(told_os(Some(" macOS 15.6 ")).as_deref(), Some("macOS 15.6"));
        assert_eq!(told_os(Some("  ")), None);
        assert_eq!(told_os(None), None);
        // Whatever this machine is, it has a name.
        assert!(os_label().is_some_and(|os| !os.is_empty()));
    }
}
