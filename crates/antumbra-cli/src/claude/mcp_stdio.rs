//! Just enough of an MCP client to ask a stdio server what tools it has: the
//! handshake, then `tools/list` until the server stops handing back a cursor.
//!
//! The exchange ([`list_tools`]) is a function over "send a message" and
//! "receive the next one", so it is tested with no process. [`ask`] is the edge:
//! it starts the server, wires those two to its pipes, and ends it.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{json, Value};

/// The revision this client speaks. A server that speaks another answers with
/// its own, and `tools/list` reads the same in every revision so far.
const PROTOCOL: &str = "2025-06-18";

/// A server that pages forever is a broken server, not a large one.
const MOST_PAGES: usize = 200;

type Send<'a> = &'a mut dyn FnMut(&Value) -> anyhow::Result<()>;
type Receive<'a> = &'a mut dyn FnMut() -> anyhow::Result<Value>;

/// Wait for the answer to request `id`, letting anything else go by: a server
/// may log, notify, or ask something of its own in between.
fn answer_to(id: u64, receive: Receive<'_>) -> anyhow::Result<Value> {
    loop {
        let message = receive()?;
        if message.get("id").and_then(Value::as_u64) != Some(id) || message.get("method").is_some()
        {
            continue;
        }
        if let Some(error) = message.get("error") {
            anyhow::bail!("the server refused request {id}: {error}");
        }
        return message
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("the server's answer to request {id} has no result"));
    }
}

/// Shake hands and collect every tool, as the server listed them.
pub fn list_tools(send: Send<'_>, receive: Receive<'_>) -> anyhow::Result<Vec<Value>> {
    send(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL,
            "capabilities": {},
            "clientInfo": { "name": "antumbra-mcp-lint", "version": env!("CARGO_PKG_VERSION") }
        }
    }))?;
    answer_to(1, receive)?;
    send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))?;

    let mut tools = Vec::new();
    let mut cursor: Option<String> = None;
    for page in 0..MOST_PAGES {
        let id = 2 + page as u64;
        let params = match &cursor {
            Some(cursor) => json!({ "cursor": cursor }),
            None => json!({}),
        };
        send(&json!({ "jsonrpc": "2.0", "id": id, "method": "tools/list", "params": params }))?;
        let result = answer_to(id, receive)?;
        tools.extend(
            result
                .get("tools")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .cloned(),
        );
        cursor = result
            .get("nextCursor")
            .and_then(Value::as_str)
            .map(str::to_string);
        if cursor.is_none() {
            return Ok(tools);
        }
    }
    anyhow::bail!("the server was still paging after {MOST_PAGES} pages of tools")
}

/// Start `command`, ask it for its tools, and end it. The server's stderr is
/// dropped: it is the server's log, and some servers are loud.
pub fn ask(command: &[String], patience: Duration) -> anyhow::Result<Vec<Value>> {
    let Some((program, arguments)) = command.split_first() else {
        anyhow::bail!("no server command given: put it after `--`");
    };
    // On Windows most servers are launched through a `.cmd` shim (`npx`, `uvx`),
    // which only the command interpreter resolves.
    let mut process = if cfg!(windows) {
        let mut through = Command::new("cmd");
        through.arg("/C").arg(program);
        through
    } else {
        Command::new(program)
    };
    let mut child = process
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| anyhow::anyhow!("could not start `{}`: {e}", command.join(" ")))?;

    let outcome = (|| {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("the server has no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("the server has no stdout"))?;
        let (lines, received) = mpsc::channel::<Value>();
        // Anything on stdout that is not a JSON message is a banner: skipped.
        // The thread ends when the pipe closes or the receiver is gone.
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Ok(message) = serde_json::from_str::<Value>(&line) {
                    if lines.send(message).is_err() {
                        break;
                    }
                }
            }
        });
        let mut send = |message: &Value| -> anyhow::Result<()> {
            writeln!(stdin, "{message}")?;
            Ok(stdin.flush()?)
        };
        let mut receive = || -> anyhow::Result<Value> {
            received.recv_timeout(patience).map_err(|e| match e {
                mpsc::RecvTimeoutError::Timeout => {
                    anyhow::anyhow!("the server said nothing for {} seconds", patience.as_secs())
                }
                mpsc::RecvTimeoutError::Disconnected => {
                    anyhow::anyhow!("the server exited before it answered")
                }
            })
        };
        list_tools(&mut send, &mut receive)
    })();

    // Best effort: a server that has already exited cannot be killed, and that
    // is not the error worth reporting.
    let _ = child.kill();
    let _ = child.wait();
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    /// A scripted server: what it will say, in order, and what it was sent.
    struct Script {
        says: RefCell<VecDeque<Value>>,
        heard: RefCell<Vec<Value>>,
    }

    impl Script {
        fn saying(messages: Vec<Value>) -> Self {
            Script {
                says: RefCell::new(messages.into()),
                heard: RefCell::new(Vec::new()),
            }
        }
        fn run(&self) -> anyhow::Result<Vec<Value>> {
            let mut send = |message: &Value| -> anyhow::Result<()> {
                self.heard.borrow_mut().push(message.clone());
                Ok(())
            };
            let mut receive = || -> anyhow::Result<Value> {
                self.says
                    .borrow_mut()
                    .pop_front()
                    .ok_or_else(|| anyhow::anyhow!("the server said nothing more"))
            };
            list_tools(&mut send, &mut receive)
        }
        fn methods_heard(&self) -> Vec<String> {
            self.heard
                .borrow()
                .iter()
                .filter_map(|m| m.get("method").and_then(Value::as_str).map(str::to_string))
                .collect()
        }
    }

    fn welcome() -> Value {
        json!({ "jsonrpc": "2.0", "id": 1, "result": { "protocolVersion": PROTOCOL, "capabilities": { "tools": {} } } })
    }

    #[test]
    fn it_shakes_hands_then_lists() -> anyhow::Result<()> {
        let script = Script::saying(vec![
            welcome(),
            json!({ "jsonrpc": "2.0", "id": 2, "result": { "tools": [{ "name": "a" }, { "name": "b" }] } }),
        ]);
        let tools = script.run()?;
        assert_eq!(tools.len(), 2);
        assert_eq!(
            script.methods_heard(),
            ["initialize", "notifications/initialized", "tools/list"]
        );
        Ok(())
    }

    #[test]
    fn it_follows_the_cursor_to_the_last_page() -> anyhow::Result<()> {
        let script = Script::saying(vec![
            welcome(),
            json!({ "jsonrpc": "2.0", "id": 2, "result": { "tools": [{ "name": "a" }], "nextCursor": "page-2" } }),
            json!({ "jsonrpc": "2.0", "id": 3, "result": { "tools": [{ "name": "b" }] } }),
        ]);
        let names: Vec<String> = script
            .run()?
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_string))
            .collect();
        assert_eq!(names, ["a", "b"]);
        let asked_with = script
            .heard
            .borrow()
            .last()
            .and_then(|m| m.pointer("/params/cursor").cloned());
        assert_eq!(asked_with, Some(json!("page-2")));
        Ok(())
    }

    #[test]
    fn what_is_not_the_answer_goes_by() -> anyhow::Result<()> {
        let script = Script::saying(vec![
            json!({ "jsonrpc": "2.0", "method": "notifications/message", "params": { "data": "starting" } }),
            // A request of the server's own that happens to reuse the id.
            json!({ "jsonrpc": "2.0", "id": 1, "method": "roots/list" }),
            welcome(),
            json!({ "jsonrpc": "2.0", "id": 99, "result": {} }),
            json!({ "jsonrpc": "2.0", "id": 2, "result": { "tools": [{ "name": "a" }] } }),
        ]);
        assert_eq!(script.run()?.len(), 1);
        Ok(())
    }

    #[test]
    fn a_refusal_a_silence_and_an_endless_server_are_each_an_error() {
        let refused = Script::saying(vec![
            json!({ "jsonrpc": "2.0", "id": 1, "error": { "code": -32602, "message": "unsupported protocol" } }),
        ]);
        let said = refused
            .run()
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(said.contains("unsupported protocol"), "{said}");

        let silent = Script::saying(vec![welcome()]);
        assert!(silent.run().is_err());

        let pages = (0..=MOST_PAGES as u64)
            .map(|n| json!({ "jsonrpc": "2.0", "id": 2 + n, "result": { "tools": [], "nextCursor": "again" } }));
        let endless = Script::saying(std::iter::once(welcome()).chain(pages).collect());
        let said = endless
            .run()
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(said.contains("still paging"), "{said}");
    }
}
