use std::collections::VecDeque;
use std::io::{Cursor, Read as _, Write as _};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use serde_json::{json, Value};

use super::*;
use crate::agents::scratch::Scratch;

/// What a scripted endpoint was sent: the message, and the session and
/// protocol it rode.
type Seen = (Value, Option<String>, Option<String>);

/// One step of a script: the answers to hand over, and what became of the
/// message.
type Step = (Vec<Value>, anyhow::Result<Sent>);

/// An endpoint that answers from a script, in order, and keeps what it was
/// sent.
#[derive(Clone, Default)]
struct Scripted {
    steps: Arc<Mutex<VecDeque<Step>>>,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Scripted {
    fn then(self, answers: Vec<Value>, sent: anyhow::Result<Sent>) -> Self {
        held(&self.steps).push_back((answers, sent));
        self
    }

    fn seen(&self) -> Vec<Seen> {
        held(&self.seen).clone()
    }
}

impl Endpoint for Scripted {
    fn send(
        &self,
        message: &Value,
        session: Option<&str>,
        protocol: Option<&str>,
        each: &mut dyn FnMut(Value),
    ) -> anyhow::Result<Sent> {
        held(&self.seen).push((
            message.clone(),
            session.map(str::to_owned),
            protocol.map(str::to_owned),
        ));
        let (answers, sent) = held(&self.steps)
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("the script has no step for {message}"))?;
        for answer in answers {
            each(answer);
        }
        sent
    }

    fn name(&self) -> String {
        "Antumbra at https://memory.example".into()
    }
}

/// Where a bridge under test writes, and the test reads.
#[derive(Clone, Default)]
struct Written(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Written {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        held(&self.0).extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Written {
    /// Every message written so far, one a line.
    fn messages(&self) -> anyhow::Result<Vec<Value>> {
        let text = String::from_utf8(held(&self.0).clone())?;
        text.lines()
            .map(|line| Ok(serde_json::from_str(line)?))
            .collect()
    }
}

fn taken(session: Option<&str>) -> anyhow::Result<Sent> {
    Ok(Sent::Taken {
        session: session.map(str::to_owned),
    })
}

fn initialize() -> Value {
    json!({ "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": { "protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": { "name": "an-agent", "version": "1" } } })
}

fn initialized() -> Value {
    json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })
}

fn welcome(instructions: Option<&str>) -> Value {
    let mut result = json!({ "protocolVersion": "2025-06-18", "capabilities": { "tools": {} },
                             "serverInfo": { "name": "antumbra", "version": "0" } });
    if let Some(text) = instructions {
        result["instructions"] = json!(text);
    }
    json!({ "jsonrpc": "2.0", "id": 0, "result": result })
}

fn list(id: u64) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "method": "tools/list" })
}

fn tools(id: u64) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": { "tools": [] } })
}

#[test]
fn a_session_is_opened_once_and_everything_after_rides_it() -> anyhow::Result<()> {
    let endpoint = Scripted::default()
        .then(vec![welcome(Some("as the server says"))], taken(Some("s1")))
        .then(vec![], taken(None))
        .then(vec![tools(1)], taken(None));
    let out = Written::default();
    let bridge = Bridge::new(endpoint.clone(), out.clone());
    bridge.take(initialize());
    bridge.take(initialized());
    bridge.take(list(1));

    let seen = endpoint.seen();
    assert_eq!(seen.len(), 3);
    assert_eq!((seen[0].1.as_deref(), seen[0].2.as_deref()), (None, None));
    for later in &seen[1..] {
        assert_eq!(later.1.as_deref(), Some("s1"));
        assert_eq!(later.2.as_deref(), Some("2025-06-18"));
    }
    let written = out.messages()?;
    assert_eq!(written, vec![welcome(Some("as the server says")), tools(1)]);
    Ok(())
}

#[test]
fn a_handshake_that_says_nothing_is_given_the_instructions() -> anyhow::Result<()> {
    for silent in [None, Some("  ")] {
        let endpoint = Scripted::default().then(vec![welcome(silent)], taken(Some("s1")));
        let out = Written::default();
        Bridge::new(endpoint, out.clone()).take(initialize());
        let written = out.messages()?;
        assert_eq!(written.len(), 1);
        assert_eq!(written[0]["result"]["instructions"], json!(INSTRUCTIONS));
        // The rest of the handshake is the server's, untouched.
        assert_eq!(
            written[0]["result"]["serverInfo"],
            json!({ "name": "antumbra", "version": "0" })
        );
    }
    assert!(INSTRUCTIONS.contains("recall_memories") && INSTRUCTIONS.contains("store_memory"));
    Ok(())
}

#[test]
fn a_session_the_server_forgot_is_opened_again_and_the_call_answered_once() -> anyhow::Result<()> {
    let endpoint = Scripted::default()
        .then(vec![welcome(None)], taken(Some("s1")))
        .then(vec![], taken(None))
        // The server restarted: it knows nothing of s1.
        .then(vec![], Ok(Sent::SessionGone))
        .then(vec![welcome(None)], taken(Some("s2")))
        .then(vec![], taken(None))
        .then(vec![tools(7)], taken(None))
        .then(vec![tools(8)], taken(None));
    let out = Written::default();
    let bridge = Bridge::new(endpoint.clone(), out.clone());
    bridge.take(initialize());
    bridge.take(initialized());
    bridge.take(list(7));
    bridge.take(list(8));

    let seen = endpoint.seen();
    let sessions: Vec<Option<&str>> = seen.iter().map(|s| s.1.as_deref()).collect();
    assert_eq!(
        sessions,
        vec![
            None,
            Some("s1"),
            Some("s1"),
            None,
            Some("s2"),
            Some("s2"),
            Some("s2")
        ]
    );
    // The handshake sent again is the agent's own, both halves of it.
    assert_eq!(seen[3].0, initialize());
    assert_eq!(seen[4].0, initialized());
    // The agent saw its first handshake and one answer per call: the second
    // handshake's answer is the bridge's business.
    let ids: Vec<Value> = out.messages()?.iter().map(|m| m["id"].clone()).collect();
    assert_eq!(ids, vec![json!(0), json!(7), json!(8)]);
    Ok(())
}

#[test]
fn a_session_gone_twice_is_an_error_the_agent_is_told() -> anyhow::Result<()> {
    let endpoint = Scripted::default()
        .then(vec![welcome(None)], taken(Some("s1")))
        .then(vec![], Ok(Sent::SessionGone))
        .then(vec![welcome(None)], taken(Some("s2")))
        .then(vec![], Ok(Sent::SessionGone));
    let out = Written::default();
    let bridge = Bridge::new(endpoint, out.clone());
    bridge.take(initialize());
    bridge.take(list(3));
    let written = out.messages()?;
    assert_eq!(written.len(), 2);
    assert_eq!(written[1]["id"], json!(3));
    let said = written[1]["error"]["message"].as_str().unwrap_or_default();
    assert!(said.contains("does not know this session"), "{said}");
    Ok(())
}

#[test]
fn a_refused_token_is_said_to_a_request_and_not_to_a_notification() -> anyhow::Result<()> {
    let refused = || {
        Ok(Sent::Refused {
            status: 401,
            said: "invalid or missing bearer token".into(),
        })
    };
    let endpoint = Scripted::default()
        .then(vec![], refused())
        .then(vec![], refused());
    let out = Written::default();
    let bridge = Bridge::new(endpoint, out.clone());
    bridge.take(initialize());
    bridge.take(initialized());
    let written = out.messages()?;
    assert_eq!(written.len(), 1, "{written:?}");
    assert_eq!(written[0]["id"], json!(0));
    assert_eq!(written[0]["error"]["code"], json!(-32000));
    let said = written[0]["error"]["message"].as_str().unwrap_or_default();
    assert!(said.contains("refused the token (401)"), "{said}");
    assert!(said.contains("invalid or missing bearer token"), "{said}");
    assert!(said.contains("antumbra setup check"), "{said}");
    Ok(())
}

#[test]
fn a_server_that_does_not_answer_is_said_once() -> anyhow::Result<()> {
    let endpoint = Scripted::default()
        .then(vec![welcome(None)], taken(Some("s1")))
        .then(
            vec![],
            Err(anyhow::anyhow!(
                "Antumbra at https://memory.example did not answer: timed out"
            )),
        )
        // An answer that arrived before the stream broke stands by itself.
        .then(
            vec![tools(5)],
            Err(anyhow::anyhow!(
                "Antumbra at https://memory.example stopped answering"
            )),
        );
    let out = Written::default();
    let bridge = Bridge::new(endpoint, out.clone());
    bridge.take(initialize());
    bridge.take(list(4));
    bridge.take(list(5));
    let written = out.messages()?;
    assert_eq!(written.len(), 3, "{written:?}");
    assert!(written[1]["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("did not answer"));
    assert_eq!(written[2], tools(5));
    Ok(())
}

#[test]
fn the_handshake_and_notifications_take_their_turn_and_calls_run_beside() {
    assert!(takes_its_turn(&initialize()));
    assert!(takes_its_turn(&initialized()));
    // An answer of the agent's own, to something the server asked.
    assert!(takes_its_turn(
        &json!({ "jsonrpc": "2.0", "id": 9, "result": {} })
    ));
    assert!(!takes_its_turn(&list(1)));
    assert!(!takes_its_turn(
        &json!({ "jsonrpc": "2.0", "id": "a", "method": "tools/call", "params": {} })
    ));
}

#[test]
fn an_event_stream_is_read_message_by_message() -> anyhow::Result<()> {
    let body = concat!(
        ": keep-alive\r\n",
        "id: 0\r\n",
        "data:\r\n",
        "\r\n",
        "event: message\r\n",
        "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\r\n",
        "\r\n",
        "data: {\"jsonrpc\":\"2.0\",\r\n",
        "data:  \"id\":1,\"result\":{}}\r\n",
        "\r\n",
        "data: not json\r\n",
        "\r\n",
        "data: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{}}",
    );
    let mut read = Vec::new();
    read_events(Cursor::new(body), &mut |message| read.push(message))?;
    assert_eq!(
        read,
        vec![
            json!({ "jsonrpc": "2.0", "method": "notifications/progress", "params": {} }),
            json!({ "jsonrpc": "2.0", "id": 1, "result": {} }),
            json!({ "jsonrpc": "2.0", "id": 2, "result": {} }),
        ]
    );
    Ok(())
}

/// A server on a local port that answers each request, in order, with the
/// next canned response (written as given), and hands back what it read.
fn serve(
    responses: Vec<String>,
) -> anyhow::Result<(String, JoinHandle<std::io::Result<Vec<String>>>)> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let url = format!("http://{}", listener.local_addr()?);
    let thread = std::thread::spawn(move || -> std::io::Result<Vec<String>> {
        let mut seen = Vec::new();
        for response in responses {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
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
            stream.write_all(response.as_bytes())?;
            stream.flush()?;
            seen.push(String::from_utf8_lossy(&request).into_owned());
        }
        Ok(seen)
    });
    Ok((url, thread))
}

fn response(status: &str, kind: &str, extra: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\ncontent-type: {kind}\r\n{extra}content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

#[test]
fn the_endpoint_speaks_to_a_server_as_an_mcp_client_does() -> anyhow::Result<()> {
    let stream =
        "data: {\"jsonrpc\":\"2.0\",\"id\":0,\"result\":{\"protocolVersion\":\"2025-06-18\"}}\n\n";
    let (url, served) = serve(vec![
        response("200 OK", "text/event-stream", "mcp-session-id: s-9\r\n", stream),
        response("202 Accepted", "text/plain", "", ""),
        response("404 Not Found", "text/plain", "", "Not Found: Session not found"),
        response(
            "404 Not Found",
            "application/json",
            "",
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"error\":{\"code\":-32601,\"message\":\"no such method\"}}",
        ),
        response("401 Unauthorized", "text/plain", "", "invalid or missing bearer token"),
    ])?;
    let home = Scratch::new("bridge-http")?;
    let token_file = home.path().join("token.txt");
    std::fs::write(&token_file, "tok-123\n")?;
    let endpoint = HttpEndpoint::new(
        Reach {
            url: url.clone(),
            workspace: None,
            token_given: None,
            token_file,
            host: Some("bench".into()),
        },
        Some("codex"),
    );

    let mut answers = Vec::new();
    let opened = endpoint.send(&initialize(), None, None, &mut |m| answers.push(m))?;
    assert_eq!(
        opened,
        Sent::Taken {
            session: Some("s-9".into())
        }
    );
    assert_eq!(answers.len(), 1);
    assert_eq!(answers[0]["result"]["protocolVersion"], json!("2025-06-18"));

    let noted = endpoint.send(&initialized(), Some("s-9"), Some("2025-06-18"), &mut |_| {})?;
    assert_eq!(noted, Sent::Taken { session: None });

    let gone = endpoint.send(&list(2), Some("s-9"), Some("2025-06-18"), &mut |_| {})?;
    assert_eq!(gone, Sent::SessionGone);

    // A JSON-RPC error on a failing status is an answer, not a lost session.
    let mut errors = Vec::new();
    let unknown = endpoint.send(&list(3), Some("s-9"), None, &mut |m| errors.push(m))?;
    assert_eq!(unknown, Sent::Taken { session: None });
    assert_eq!(errors[0]["error"]["code"], json!(-32601));

    let refused = endpoint.send(&list(4), Some("s-9"), None, &mut |_| {})?;
    assert_eq!(
        refused,
        Sent::Refused {
            status: 401,
            said: "invalid or missing bearer token".into()
        }
    );

    let requests = served
        .join()
        .map_err(|_| anyhow::anyhow!("the server thread panicked"))??;
    let first = requests[0].to_lowercase();
    assert!(first.starts_with("post /mcp http/1.1"), "{first}");
    assert!(first.contains("authorization: bearer tok-123"), "{first}");
    assert!(
        first.contains("accept: application/json, text/event-stream"),
        "{first}"
    );
    assert!(first.contains("x-antumbra-host: bench"), "{first}");
    assert!(first.contains("user-agent: antumbra-bridge/"), "{first}");
    assert!(first.contains("(codex)"), "{first}");
    assert!(!first.contains("mcp-session-id:"), "{first}");
    let second = requests[1].to_lowercase();
    assert!(second.contains("mcp-session-id: s-9"), "{second}");
    assert!(
        second.contains("mcp-protocol-version: 2025-06-18"),
        "{second}"
    );
    Ok(())
}

#[test]
fn a_token_replaced_while_the_agent_runs_is_the_one_the_next_call_presents() -> anyhow::Result<()> {
    let ok = || response("202 Accepted", "text/plain", "", "");
    let (url, served) = serve(vec![ok(), ok()])?;
    let home = Scratch::new("bridge-token")?;
    let token_file = home.path().join("token.txt");
    std::fs::write(&token_file, "old-token")?;
    let endpoint = HttpEndpoint::new(
        Reach {
            url,
            workspace: None,
            token_given: None,
            token_file: token_file.clone(),
            host: None,
        },
        None,
    );
    endpoint.send(&initialized(), None, None, &mut |_| {})?;
    std::fs::write(&token_file, "new-token")?;
    endpoint.send(&initialized(), None, None, &mut |_| {})?;
    let requests = served
        .join()
        .map_err(|_| anyhow::anyhow!("the server thread panicked"))??;
    assert!(requests[0].contains("Bearer old-token"));
    assert!(requests[1].contains("Bearer new-token"));
    // No name, no header: the server stamps its own.
    assert!(!requests[0].to_lowercase().contains("x-antumbra-host"));
    Ok(())
}

#[test]
fn no_token_is_an_error_before_anything_is_sent() -> anyhow::Result<()> {
    let home = Scratch::new("bridge-no-token")?;
    let endpoint = HttpEndpoint::new(
        Reach {
            // Nothing listens here, and nothing is asked to.
            url: "http://127.0.0.1:9".into(),
            workspace: None,
            token_given: None,
            token_file: home.path().join("token.txt"),
            host: None,
        },
        None,
    );
    let Err(problem) = endpoint.send(&initialize(), None, None, &mut |_| {}) else {
        anyhow::bail!("a request was sent with no token");
    };
    assert!(problem.to_string().contains("no token in"), "{problem}");
    Ok(())
}

/// A request as an agent that keeps no session sends it: it says for itself
/// which revision it speaks. Recorded from goose 1.60.
fn modern(id: u64, method: &str, mut params: Value) -> Value {
    if let Some(params) = params.as_object_mut() {
        params.insert(
            "_meta".into(),
            json!({
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientInfo": { "name": "goose-cli", "version": "1.60.1" },
                "io.modelcontextprotocol/clientCapabilities": {},
            }),
        );
    }
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}

#[test]
fn a_request_that_says_its_revision_is_described_in_its_headers() {
    let discover = described(&modern(0, "server/discover", json!({})));
    assert_eq!(
        discover,
        vec![
            ("mcp-protocol-version", "2026-07-28".to_string()),
            ("mcp-method", "server/discover".to_string()),
        ]
    );
    let call = described(&modern(
        2,
        "tools/call",
        json!({ "name": "recall_memories", "arguments": { "query": "x" } }),
    ));
    assert_eq!(
        call,
        vec![
            ("mcp-protocol-version", "2026-07-28".to_string()),
            ("mcp-method", "tools/call".to_string()),
            ("mcp-name", "recall_memories".to_string()),
        ]
    );
    let read = described(&modern(
        3,
        "resources/read",
        json!({ "uri": "relic://a b" }),
    ));
    assert_eq!(read[2], ("mcp-name", "relic://a b".to_string()));
    // A request that rides a session names no revision, and has none.
    assert!(described(&list(1)).is_empty());
    // An earlier revision named in a request is said, and no more.
    let earlier = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        "params": { "_meta": { "io.modelcontextprotocol/protocolVersion": "2025-11-25" } } });
    assert_eq!(
        described(&earlier),
        vec![("mcp-protocol-version", "2025-11-25".to_string())]
    );
}

#[test]
fn a_name_that_would_not_survive_as_a_header_is_wrapped() {
    assert_eq!(header_value("recall_memories"), "recall_memories");
    assert_eq!(header_value(""), "");
    // "caf\u{e9}" in Base64, as the server unwraps it.
    assert_eq!(header_value("caf\u{e9}"), "=?base64?Y2Fmw6k=?=");
    assert_eq!(header_value(" padded"), "=?base64?IHBhZGRlZA==?=");
    assert_eq!(header_value("line\nbreak"), "=?base64?bGluZQpicmVhaw==?=");
    // Text that already looks wrapped is wrapped, so it is read back as itself.
    assert!(header_value("=?base64?eA==?=").starts_with("=?base64?PT9iYXNlNjQ"));
}

#[test]
fn an_agent_that_keeps_no_session_is_carried_and_given_the_instructions() -> anyhow::Result<()> {
    let discovered = json!({ "jsonrpc": "2.0", "id": 0, "result": {
        "resultType": "complete", "supportedVersions": ["2025-11-25", "2026-07-28"],
        "capabilities": { "tools": {} }, "ttlMs": 0, "cacheScope": "private" } });
    let endpoint = Scripted::default()
        .then(vec![discovered], taken(None))
        .then(vec![tools(1)], taken(None));
    let out = Written::default();
    let bridge = Bridge::new(endpoint.clone(), out.clone());
    let discover = modern(0, "server/discover", json!({}));
    assert!(takes_its_turn(&discover));
    bridge.take(discover);
    bridge.take(modern(1, "tools/list", json!({})));

    // No session is opened, kept or sent.
    for seen in endpoint.seen() {
        assert_eq!((seen.1, seen.2), (None, None));
    }
    let written = out.messages()?;
    assert_eq!(written[0]["result"]["instructions"], json!(INSTRUCTIONS));
    assert_eq!(
        written[0]["result"]["supportedVersions"],
        json!(["2025-11-25", "2026-07-28"])
    );
    assert_eq!(written[1], tools(1));
    Ok(())
}

#[test]
fn the_endpoint_sends_what_a_request_says_of_itself() -> anyhow::Result<()> {
    let answer =
        "data: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"resultType\":\"complete\"}}\n\n";
    let (url, served) = serve(vec![
        response("200 OK", "text/event-stream", "", answer),
        response("200 OK", "text/event-stream", "", answer),
    ])?;
    let home = Scratch::new("bridge-modern")?;
    let token_file = home.path().join("token.txt");
    std::fs::write(&token_file, "tok-123")?;
    let endpoint = HttpEndpoint::new(
        Reach {
            url,
            workspace: None,
            token_given: None,
            token_file,
            host: None,
        },
        Some("goose"),
    );
    let call = modern(
        2,
        "tools/call",
        json!({ "name": "recall_memories", "arguments": {} }),
    );
    endpoint.send(&call, None, None, &mut |_| {})?;
    // A session's revision is not sent over what the request says of itself.
    endpoint.send(&call, None, Some("2025-06-18"), &mut |_| {})?;
    let requests = served
        .join()
        .map_err(|_| anyhow::anyhow!("the server thread panicked"))??;
    for request in &requests {
        let head = request.to_lowercase();
        assert!(head.contains("mcp-protocol-version: 2026-07-28"), "{head}");
        assert!(head.contains("mcp-method: tools/call"), "{head}");
        assert!(head.contains("mcp-name: recall_memories"), "{head}");
        assert!(!head.contains("mcp-session-id:"), "{head}");
        assert_eq!(head.matches("mcp-protocol-version:").count(), 1, "{head}");
    }
    Ok(())
}
