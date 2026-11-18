//! `antumbra mcp bridge`: the server's MCP endpoint, on standard input and
//! output.
//!
//! An agent connects to an MCP server one of two ways: it runs a command and
//! talks over the command's pipes, or it is given an address and a credential.
//! Claude Code is given the address, with a helper that reads the token from
//! its file for each connection. No other agent has such a helper: each wants
//! the token written into its own config file, or set in the environment of
//! every shell it is started from. Some refuse a plain `http://` address,
//! which is what a lab's server has. And each treats a server that no longer
//! knows its session (it restarted) its own way, most by failing every call
//! until the agent is started again.
//!
//! So for those agents the address and the credential stay with Antumbra. The
//! agent runs this command, which carries each message to `POST {url}/mcp`
//! with the token read from its file for that request, and each answer back.
//! What the server says is passed on as it says it, with two exceptions:
//!
//! - **Instructions.** A server that gives none in its handshake has
//!   [`INSTRUCTIONS`] added, which an agent places in its system prompt. They
//!   are what keeps an agent recalling and storing where its hooks do not run.
//! - **A session the server forgot.** The bridge opens another with the
//!   handshake the agent first sent, and asks again. The agent sees one
//!   answer, a little late.
//!
//! Two ways of speaking MCP are carried. An agent that opens a session
//! (`initialize`, then requests that ride it) is given one, kept, and reopened
//! as above. An agent that speaks the 2026-07-28 revision opens none: it asks
//! `server/discover`, and every request after says for itself which revision
//! it speaks. Over HTTP that revision wants the same said in headers, so a
//! proxy can route a request without reading it; [`described`] writes them
//! from the request. The server answers both ways, an agent picks one.
//!
//! Not carried: what the server sends unasked on its `GET` stream, which today
//! is a log line when another session changes a memory; and `Mcp-Param-*`
//! headers, which a tool asks for in its schema and none of the server's do.
//!
//! The exchange ([`Bridge`]) is written against "send this message to the
//! endpoint" ([`Endpoint`]), so it is tested with no socket. [`HttpEndpoint`]
//! is the edge, tested against a server on a local port.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use base64::Engine;
use clap::Args;
use serde_json::{json, Value};

use antumbra_http::DEVICE_HEADER;

use super::Reach;

/// What an agent is told when the server's handshake says nothing: when to
/// recall, what to store, and how to mark a memory right or wrong.
pub const INSTRUCTIONS: &str =
    "Antumbra is the user's memory across sessions, agents and machines. \
Before you start a task, call recall_memories with what the task is about, and inside a git \
repository pass repo (host/org/name) and branch. Before you finish, store what this session \
learned that the code does not say with store_memory: decisions and the reasons for them, the \
user's corrections and preferences, conventions, bugs and incidents, and results you verified. \
Pick the network: world (facts), bank (experiences), opinion (judgments and preferences). For \
anything about code, pass provenance {repo, commit, branch}. When the user states or corrects how \
an agent should act, also call record_behavior. When your work proves a recalled memory right or \
wrong, call reinforce_memory or penalize_memory with its id.";

/// The most of a refusal's body that is read: a server's words, not a page.
const MOST_SAID: u64 = 64 * 1024;

/// `antumbra mcp bridge`.
#[derive(Args, Debug, Clone)]
pub struct BridgeArgs {
    /// The agent this bridge is run by, named in the requests it makes.
    #[arg(long)]
    pub agent: Option<String>,
    /// The server's address, when it is not the one `antumbra setup` recorded.
    /// Named as the other commands name it: `--url` is the store's.
    #[arg(long = "surface", env = "ANTUMBRA_URL")]
    pub surface: Option<String>,
    /// The file holding the token, when it is not the one setup wrote.
    #[arg(long, env = "ANTUMBRA_TOKEN_FILE")]
    pub token_file: Option<PathBuf>,
}

/// What became of one message sent to the endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sent {
    /// Taken. Any answers were handed over as they arrived; `session` is the
    /// one the server named in its reply, when it named one.
    Taken { session: Option<String> },
    /// The server does not know the session the message named.
    SessionGone,
    /// Refused, with the status and the server's words.
    Refused { status: u16, said: String },
}

/// Where the bridge carries an agent's messages.
pub trait Endpoint: Send + Sync {
    /// Send `message` in `session`, speaking `protocol`, and hand each answer
    /// to `each` as it arrives. An error is no answer at all.
    fn send(
        &self,
        message: &Value,
        session: Option<&str>,
        protocol: Option<&str>,
        each: &mut dyn FnMut(Value),
    ) -> anyhow::Result<Sent>;

    /// The endpoint, in the words an error names it with.
    fn name(&self) -> String;
}

/// The connection as it stands: the session and protocol the server gave, and
/// the agent's own handshake, kept to open another session with.
#[derive(Default)]
struct Link {
    session: Option<String>,
    protocol: Option<String>,
    opening: Option<Value>,
    opened: Option<Value>,
}

/// A lock's contents, whatever became of a thread that held it: one answer
/// lost to a panic must not end every call after it.
fn held<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One agent's connection, carried to an [`Endpoint`].
pub struct Bridge<E, W> {
    endpoint: E,
    out: Mutex<W>,
    link: Mutex<Link>,
    /// Held while a session is opened again, so messages that all found the
    /// session gone open one between them.
    reopening: Mutex<()>,
}

fn method_of(message: &Value) -> Option<&str> {
    message.get("method").and_then(Value::as_str)
}

/// Whether `message` is a request: it names a method and wants an answer.
fn asks(message: &Value) -> bool {
    method_of(message).is_some() && message.get("id").is_some_and(|id| !id.is_null())
}

/// Whether `message` opens the connection: a session's handshake, or the
/// question an agent that keeps no session asks first. Its answer is the one
/// that carries the server's instructions.
fn opens(message: &Value) -> bool {
    matches!(method_of(message), Some("initialize" | "server/discover"))
}

/// Whether `message` is handled before the next one is read. What opens the
/// connection and everything that is not a request are: a notification is
/// taken at once, and the order they reach the server in is the order the
/// agent sent them. Every other request runs beside the rest, so a slow tool
/// call does not hold up a quick one.
pub fn takes_its_turn(message: &Value) -> bool {
    !asks(message) || opens(message)
}

/// The revision from which a request over HTTP is described in its headers.
const DESCRIBED_FROM: &str = "2026-07-28";

/// Where a request names the revision it speaks.
const SPOKEN: &str = "/params/_meta/io.modelcontextprotocol~1protocolVersion";

/// A value as an `Mcp-*` header carries it: as it is, or wrapped in Base64
/// when it would not survive as a header (a leading or trailing blank, a
/// character outside printable ASCII, or text that already looks wrapped).
fn header_value(value: &str) -> String {
    let wrapped = value.starts_with("=?base64?") && value.ends_with("?=");
    let edges = value.starts_with([' ', '\t']) || value.ends_with([' ', '\t']);
    let plain = value.chars().all(|c| (' '..='~').contains(&c));
    if value.is_empty() || (plain && !edges && !wrapped) {
        value.to_string()
    } else {
        format!(
            "=?base64?{}?=",
            base64::engine::general_purpose::STANDARD.encode(value)
        )
    }
}

/// The headers that describe `message` to the server and to anything on the
/// way, when the message says which revision it speaks: that revision, and
/// from 2026-07-28 its method and the name it is about (a tool's, a prompt's,
/// a resource's address, a task's id). A message that names no revision rides
/// a session, and has none of these.
pub fn described(message: &Value) -> Vec<(&'static str, String)> {
    let Some(spoken) = message.pointer(SPOKEN).and_then(Value::as_str) else {
        return Vec::new();
    };
    let mut headers = vec![("mcp-protocol-version", spoken.to_string())];
    let Some(method) = method_of(message) else {
        return headers;
    };
    if spoken < DESCRIBED_FROM {
        return headers;
    }
    headers.push(("mcp-method", method.to_string()));
    let named_by = match method {
        "tools/call" | "prompts/get" => Some("name"),
        "resources/read" | "resources/subscribe" | "resources/unsubscribe" => Some("uri"),
        "tasks/get" | "tasks/update" | "tasks/cancel" => Some("taskId"),
        _ => None,
    };
    let name = named_by
        .and_then(|key| message.pointer(&format!("/params/{key}")))
        .and_then(Value::as_str);
    if let Some(name) = name {
        headers.push(("mcp-name", header_value(name)));
    }
    headers
}

impl<E: Endpoint, W: Write> Bridge<E, W> {
    pub fn new(endpoint: E, out: W) -> Self {
        Bridge {
            endpoint,
            out: Mutex::new(out),
            link: Mutex::new(Link::default()),
            reopening: Mutex::new(()),
        }
    }

    /// One line to the agent. A pipe the agent closed is not worth a word:
    /// the next read of standard input ends the bridge.
    fn write(&self, message: &Value) {
        let mut out = held(&self.out);
        let _ = writeln!(out, "{message}");
        let _ = out.flush();
    }

    /// Take one message from the agent, and see it answered: by the server,
    /// or, for a request the server never answered, with the reason.
    pub fn take(&self, message: Value) {
        match method_of(&message) {
            Some("initialize") => {
                let mut link = held(&self.link);
                link.opening = Some(message.clone());
                link.session = None;
                link.protocol = None;
            }
            Some("notifications/initialized") => {
                held(&self.link).opened = Some(message.clone());
            }
            _ => {}
        }
        let mut answered = false;
        if let Err(problem) = self.carry(&message, &mut answered) {
            eprintln!("antumbra mcp bridge: {problem}");
            // An answer already passed on stands; a second one for the same
            // request would be a protocol error of the bridge's own.
            if asks(&message) && !answered {
                self.write(&json!({
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": { "code": -32000, "message": problem.to_string() },
                }));
            }
        }
    }

    fn carry(&self, message: &Value, answered: &mut bool) -> anyhow::Result<()> {
        let opening = method_of(message) == Some("initialize");
        let mut tried_again = false;
        loop {
            let (session, protocol) = {
                let link = held(&self.link);
                (link.session.clone(), link.protocol.clone())
            };
            let mut each = |answer: Value| {
                let answer = self.passed_on(message, answer);
                if answer.get("id") == message.get("id") && method_of(&answer).is_none() {
                    *answered = true;
                }
                self.write(&answer);
            };
            let sent = self.endpoint.send(
                message,
                if opening { None } else { session.as_deref() },
                if opening { None } else { protocol.as_deref() },
                &mut each,
            )?;
            match sent {
                Sent::Taken { session: named } => {
                    if opening {
                        held(&self.link).session = named;
                    }
                    return Ok(());
                }
                Sent::SessionGone if !opening && !tried_again => {
                    self.reopen(session.as_deref())?;
                    tried_again = true;
                }
                Sent::SessionGone => anyhow::bail!(
                    "{} does not know this session, and a new one was no better",
                    self.endpoint.name()
                ),
                Sent::Refused { status, said } => anyhow::bail!("{}", self.refusal(status, &said)),
            }
        }
    }

    /// A refusal in words the person reading the agent's error can act on.
    fn refusal(&self, status: u16, said: &str) -> String {
        let said = antumbra_http::refusal(said);
        let said = said.trim();
        let tail = if said.is_empty() {
            String::new()
        } else {
            format!(": {said}")
        };
        if status == 401 || status == 403 {
            format!(
                "{} refused the token ({status}){tail}. Check it with: antumbra setup check",
                self.endpoint.name()
            )
        } else {
            format!("{} answered {status}{tail}", self.endpoint.name())
        }
    }

    /// An answer as the agent gets it: the server's, with the protocol noted
    /// and [`INSTRUCTIONS`] added when it answers what opened the connection
    /// and gives none.
    fn passed_on(&self, asked: &Value, mut answer: Value) -> Value {
        if !opens(asked) || answer.get("id") != asked.get("id") {
            return answer;
        }
        let Some(result) = answer.get_mut("result").and_then(Value::as_object_mut) else {
            return answer;
        };
        // A session's handshake settles the revision its requests are sent
        // under. An agent that keeps no session says it in each request.
        if let Some(protocol) = result.get("protocolVersion").and_then(Value::as_str) {
            held(&self.link).protocol = Some(protocol.to_string());
        }
        let silent = result
            .get("instructions")
            .and_then(Value::as_str)
            .is_none_or(|text| text.trim().is_empty());
        if silent {
            result.insert("instructions".into(), Value::String(INSTRUCTIONS.into()));
        }
        answer
    }

    /// Open another session with the agent's own handshake, unless a message
    /// that found `was` gone at the same time already has.
    fn reopen(&self, was: Option<&str>) -> anyhow::Result<()> {
        let _one_at_a_time = held(&self.reopening);
        let (opening, opened) = {
            let link = held(&self.link);
            if link.session.as_deref() != was {
                return Ok(());
            }
            (link.opening.clone(), link.opened.clone())
        };
        let opening = opening.ok_or_else(|| {
            anyhow::anyhow!(
                "{} does not know this session, and the agent sent no handshake to open another with",
                self.endpoint.name()
            )
        })?;
        let mut protocol = None;
        let sent = self.endpoint.send(&opening, None, None, &mut |answer| {
            if let Some(said) = answer
                .pointer("/result/protocolVersion")
                .and_then(Value::as_str)
            {
                protocol = Some(said.to_string());
            }
        })?;
        let session = match sent {
            Sent::Taken { session } => session,
            Sent::SessionGone => {
                anyhow::bail!("{} would not open a new session", self.endpoint.name())
            }
            Sent::Refused { status, said } => anyhow::bail!("{}", self.refusal(status, &said)),
        };
        {
            let mut link = held(&self.link);
            link.session = session.clone();
            if protocol.is_some() {
                link.protocol = protocol.clone();
            }
        }
        if let Some(opened) = opened {
            self.endpoint.send(
                &opened,
                session.as_deref(),
                protocol.as_deref(),
                &mut |_| {},
            )?;
        }
        eprintln!(
            "antumbra mcp bridge: {} no longer knew the session; opened another",
            self.endpoint.name()
        );
        Ok(())
    }
}

/// Every message in a `text/event-stream` body, in order: each event's `data`
/// lines joined and read as JSON. Comments, event names, ids and events with
/// no data (a server's keep-alive, its first event) are passed over, and so is
/// data that is not JSON, with a line on standard error.
pub fn read_events(reader: impl BufRead, each: &mut dyn FnMut(Value)) -> std::io::Result<()> {
    let mut data = String::new();
    let flush = |data: &mut String, each: &mut dyn FnMut(Value)| {
        if data.trim().is_empty() {
            data.clear();
            return;
        }
        match serde_json::from_str::<Value>(data) {
            Ok(message) => each(message),
            Err(e) => {
                eprintln!("antumbra mcp bridge: an event that is not JSON was passed over ({e})")
            }
        }
        data.clear();
    };
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            flush(&mut data, each);
        } else if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    flush(&mut data, each);
    Ok(())
}

/// Whether `body` is one JSON-RPC message, or a batch of them.
fn is_message(body: &Value) -> bool {
    match body {
        Value::Array(all) => !all.is_empty() && all.iter().all(|m| m.get("jsonrpc").is_some()),
        other => other.get("jsonrpc").is_some(),
    }
}

/// The server's `/mcp`, over HTTP.
pub struct HttpEndpoint {
    agent: ureq::Agent,
    reach: Reach,
    /// What the requests say they come from.
    client: String,
}

impl HttpEndpoint {
    /// The endpoint `reach` names, for the agent `by` when one is named.
    pub fn new(reach: Reach, by: Option<&str>) -> Self {
        // A tool call that asks an expert can run for minutes, and its answer
        // is one stream held open for that long.
        let patience = antumbra_http::patience("ANTUMBRA_BRIDGE_PATIENCE_SECS", 600);
        let client = match by.map(str::trim).filter(|b| !b.is_empty()) {
            Some(by) => format!("antumbra-bridge/{} ({by})", env!("CARGO_PKG_VERSION")),
            None => format!("antumbra-bridge/{}", env!("CARGO_PKG_VERSION")),
        };
        HttpEndpoint {
            agent: antumbra_http::agent(patience),
            reach,
            client,
        }
    }
}

impl Endpoint for HttpEndpoint {
    fn send(
        &self,
        message: &Value,
        session: Option<&str>,
        protocol: Option<&str>,
        each: &mut dyn FnMut(Value),
    ) -> anyhow::Result<Sent> {
        let token = self.reach.token()?;
        let mut request = self
            .agent
            .post(format!("{}/mcp", self.reach.url))
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("user-agent", &self.client)
            .header("authorization", format!("Bearer {token}"));
        if let Some(session) = session {
            request = request.header("mcp-session-id", session);
        }
        // What the message says of itself wins over what a session settled.
        let described = described(message);
        if described.is_empty() {
            if let Some(protocol) = protocol {
                request = request.header("mcp-protocol-version", protocol);
            }
        }
        for (name, value) in &described {
            request = request.header(*name, value);
        }
        if let Some(host) = &self.reach.host {
            request = request.header(DEVICE_HEADER, host);
        }
        let response = request
            .send_json(message)
            .map_err(|e| anyhow::anyhow!("{} did not answer: {e}", self.name()))?;
        let status = response.status().as_u16();
        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        let named = header("mcp-session-id");
        let streamed = header("content-type")
            .is_some_and(|kind| kind.to_lowercase().starts_with("text/event-stream"));
        let ok = (200..300).contains(&status);
        let body = response.into_body().into_reader();
        if ok && streamed {
            read_events(BufReader::new(body), each)
                .map_err(|e| anyhow::anyhow!("{} stopped answering: {e}", self.name()))?;
            return Ok(Sent::Taken { session: named });
        }
        let mut bytes = Vec::new();
        // A body cut short still says what it says; what was read is kept.
        let _ = body.take(MOST_SAID).read_to_end(&mut bytes);
        let text = String::from_utf8_lossy(&bytes).into_owned();
        // A JSON-RPC error rides some failing statuses, and is an answer.
        if let Some(body) = serde_json::from_str::<Value>(&text).ok().filter(is_message) {
            each(body);
            return Ok(Sent::Taken { session: named });
        }
        if ok {
            return Ok(Sent::Taken { session: named });
        }
        if status == 404 && session.is_some() {
            return Ok(Sent::SessionGone);
        }
        Ok(Sent::Refused { status, said: text })
    }

    fn name(&self) -> String {
        format!("Antumbra at {}", self.reach.url)
    }
}

/// Run the bridge until the agent closes its side.
pub fn run(args: &BridgeArgs) -> anyhow::Result<()> {
    let home = crate::machine::home().ok_or_else(|| {
        anyhow::anyhow!("no home directory (neither USERPROFILE nor HOME is set)")
    })?;
    let mut reach = Reach::read(&home, |name| match name {
        // The flag, which clap also fills from the environment.
        "ANTUMBRA_URL" => args.surface.clone(),
        other => std::env::var(other).ok(),
    })?;
    if let Some(file) = &args.token_file {
        reach.token_file.clone_from(file);
    }
    let bridge = Arc::new(Bridge::new(
        HttpEndpoint::new(reach, args.agent.as_deref()),
        std::io::stdout(),
    ));
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(text) {
            Ok(message) => message,
            Err(e) => {
                eprintln!("antumbra mcp bridge: a line that is not JSON was passed over ({e})");
                bridge.write(&json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": -32700, "message": format!("not JSON: {e}") },
                }));
                continue;
            }
        };
        if takes_its_turn(&message) {
            bridge.take(message);
        } else {
            let bridge = Arc::clone(&bridge);
            std::thread::spawn(move || bridge.take(message));
        }
    }
    // The agent closed its side. Whatever was still being asked has nobody
    // left to answer, so the bridge ends without waiting for it.
    Ok(())
}

#[cfg(test)]
mod tests;
