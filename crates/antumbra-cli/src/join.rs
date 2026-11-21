//! `antumbra join`: ask to join a workspace someone else hosts, and collect
//! the token once one of its admins has let you in.
//!
//! Asking changes nothing about this machine's setup. It keeps one file,
//! `~/.antumbra/join.json`, the owner's alone: where it asked, the handle it
//! asked for, and the secret the token is collected with. The server is sent
//! the secret's digest and never the secret until collection.
//!
//! Collecting writes the workspace's token to a file of its own,
//! `~/.antumbra/workspaces/<name>/token.txt`, and leaves the token this
//! machine already uses where it is: joining a workspace does not move a
//! machine off its own store.

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Args;
use serde_json::{json, Value};

use antumbra_core::join;

#[derive(Args, Debug)]
pub struct JoinArgs {
    /// The server the workspace is on, as its address (`https://...`).
    pub server: Option<String>,
    /// The join code one of the workspace's admins gave you.
    #[arg(long, env = "ANTUMBRA_JOIN_CODE", hide_env_values = true)]
    pub code: Option<String>,
    /// The handle you will be known by there: 3 to 32 of a-z, 0-9, - and _.
    /// Not `default`: an account is yours.
    #[arg(long)]
    pub handle: Option<String>,
    /// The name you go by, shown to the workspace's members.
    #[arg(long)]
    pub name: Option<String>,
    /// A line to the workspace's admins, who decide the request.
    #[arg(long)]
    pub note: Option<String>,
    /// Ask how the request stands, and collect the token once it is approved.
    #[arg(long, conflicts_with_all = ["code", "handle", "name", "note", "forget"])]
    pub status: bool,
    /// Drop the request this machine is waiting on, without collecting it.
    #[arg(long, conflicts_with_all = ["code", "handle", "name", "note"])]
    pub forget: bool,
    /// Accept a plain `http://` address that is not this machine: a lab whose
    /// network is trusted. The code and the secret travel in cleartext.
    #[arg(long)]
    pub allow_http: bool,
}

/// One request to a server: the path, the body, and what came back as a
/// status and the answer's JSON. A seam, so the flow is tested with no server.
type Send<'a> = &'a dyn Fn(&str, &str, Value) -> anyhow::Result<(u16, Value)>;

/// What this machine keeps of a request it is waiting on.
#[derive(Debug, Clone, PartialEq)]
struct Waiting {
    server: String,
    handle: String,
    /// The secret the token is collected with. Never sent before then.
    secret: String,
    /// The request's id, once the server has answered with one.
    request: Option<String>,
}

impl Waiting {
    fn to_json(&self) -> Value {
        json!({
            "server": self.server,
            "handle": self.handle,
            "secret": self.secret,
            "request": self.request,
        })
    }

    fn from_json(kept: &Value) -> Option<Self> {
        let text = |name: &str| kept.get(name).and_then(Value::as_str).map(str::to_string);
        Some(Self {
            server: text("server")?,
            handle: text("handle")?,
            secret: text("secret")?,
            request: text("request"),
        })
    }
}

fn waiting_file(antumbra: &Path) -> PathBuf {
    antumbra.join("join.json")
}

fn read_waiting(antumbra: &Path) -> Option<Waiting> {
    let kept: Value =
        serde_json::from_str(&std::fs::read_to_string(waiting_file(antumbra)).ok()?).ok()?;
    Waiting::from_json(&kept)
}

fn keep_waiting(antumbra: &Path, waiting: &Waiting) -> anyhow::Result<()> {
    crate::setup::write_private(
        &waiting_file(antumbra),
        &serde_json::to_string_pretty(&waiting.to_json())?,
    )?;
    Ok(())
}

/// A secret nobody could guess: 32 bytes from the system, in hex.
fn new_secret() -> anyhow::Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes)
        .map_err(|e| anyhow::anyhow!("no randomness from the system: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// What a server said was wrong, or its status when it said nothing.
fn said(status: u16, answer: &Value) -> String {
    answer
        .get("error")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("the server answered {status}"))
}

/// A directory name for a workspace on a server: what a path can hold of both.
fn workspace_dir(antumbra: &Path, server: &str, workspace: &str) -> PathBuf {
    let plain = |s: &str| -> String {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    };
    let host = server
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/');
    antumbra
        .join("workspaces")
        .join(format!("{}__{}", plain(host), plain(workspace)))
}

/// Ask to join. Asking again for the same handle on the same server sends
/// the same secret, so the server answers with the request there already is.
fn ask(antumbra: &Path, args: &JoinArgs, send: Send<'_>) -> anyhow::Result<Vec<String>> {
    let (Some(server), Some(code), Some(handle)) = (&args.server, &args.code, &args.handle) else {
        anyhow::bail!(
            "to ask: antumbra join <server> --code <code> --handle <handle> [--name <name>] [--note <line>]\n\
             to see how a request stands: antumbra join --status"
        );
    };
    let server = server.trim().trim_end_matches('/').to_string();
    crate::setup::may_carry_the_token(&server, args.allow_http)?;
    // Said here before anything is sent: the server would say the same.
    let handle = join::handle_of(handle).map_err(|e| anyhow::anyhow!("{e}"))?;

    let held = read_waiting(antumbra);
    let waiting = match held {
        Some(held) if held.server == server && held.handle == handle => held,
        Some(held) => anyhow::bail!(
            "this machine is already waiting on a request for `{}` at {}. \
             See how it stands with `antumbra join --status`, or drop it with `antumbra join --forget`",
            held.handle,
            held.server
        ),
        None => Waiting {
            server: server.clone(),
            handle: handle.clone(),
            secret: new_secret()?,
            request: None,
        },
    };
    // Kept before it is sent: an answer lost on the way is asked for again
    // with the same secret, and is the same request.
    keep_waiting(antumbra, &waiting)?;
    let (status, answer) = send(
        &server,
        "/join/request",
        json!({
            "code": code.trim(),
            "handle": handle,
            "display": args.name,
            "note": args.note,
            "host": antumbra_http::host_id(),
            "secret_digest": join::digest(&waiting.secret),
        }),
    )?;
    if status != 200 {
        // Nothing was asked: nothing is waited on.
        if waiting.request.is_none() {
            let _ = std::fs::remove_file(waiting_file(antumbra));
        }
        anyhow::bail!("{}", said(status, &answer));
    }
    let request = answer
        .get("request")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("the server answered without a request id"))?;
    keep_waiting(
        antumbra,
        &Waiting {
            request: Some(request.to_string()),
            ..waiting
        },
    )?;
    let standing = answer
        .get("standing")
        .and_then(Value::as_str)
        .unwrap_or("pending");
    Ok(match standing {
        "approved" => vec![
            format!("You are let in as `{handle}`: that code asks no one."),
            "Collect your token with: antumbra join --status".to_string(),
        ],
        _ => vec![
            format!("Asked to join as `{handle}`. The workspace's admins decide."),
            "See how it stands, and collect your token once approved, with: antumbra join --status"
                .to_string(),
            "Nothing about this machine's setup has changed.".to_string(),
        ],
    })
}

/// Ask how the request stands; approved, collect the token and keep it.
fn status(antumbra: &Path, send: Send<'_>) -> anyhow::Result<Vec<String>> {
    let Some(waiting) = read_waiting(antumbra) else {
        anyhow::bail!("this machine is not waiting on a request. Ask with: antumbra join <server> --code <code> --handle <handle>");
    };
    let Some(request) = &waiting.request else {
        anyhow::bail!("the request was never answered by the server. Ask again with the same server and handle");
    };
    let (status, answer) = send(
        &waiting.server,
        "/join/collect",
        json!({ "request": request, "secret": waiting.secret }),
    )?;
    if status != 200 {
        anyhow::bail!("{}", said(status, &answer));
    }
    let done = || std::fs::remove_file(waiting_file(antumbra));
    match answer
        .get("standing")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "pending" => Ok(vec![format!(
            "Still waiting: no admin of the workspace has decided the request for `{}` yet.",
            waiting.handle
        )]),
        "denied" => {
            let _ = done();
            let reason = answer.get("reason").and_then(Value::as_str);
            Ok(vec![match reason {
                Some(reason) => format!("Refused: {reason}"),
                None => "Refused. The admin who decided gave no reason.".to_string(),
            }])
        }
        "expired" => {
            let _ = done();
            Ok(vec![
                "The request expired before it was decided or collected. Ask again with a new code.".to_string(),
            ])
        }
        "collected" => {
            let _ = done();
            Ok(vec![
                "The token for this request was already collected, and is given once.".to_string(),
                "If it did not reach this machine, ask one of the workspace's admins for a new code.".to_string(),
            ])
        }
        "approved" => {
            let field = |name: &str| {
                answer
                    .get(name)
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| {
                        anyhow::anyhow!("the server approved the request and sent no `{name}`")
                    })
            };
            let (token, workspace, user) = (field("token")?, field("workspace")?, field("user")?);
            let dir = workspace_dir(antumbra, &waiting.server, &workspace);
            let token_file = dir.join("token.txt");
            // The token first: it is given once.
            let caveat = crate::setup::write_private(&token_file, &format!("{token}\n"))?;
            crate::setup::write_private(
                &dir.join("workspace.json"),
                &serde_json::to_string_pretty(&json!({
                    "url": waiting.server,
                    "workspace": workspace,
                    "user": user,
                    "token_file": token_file,
                    "joined_at": chrono::Utc::now().to_rfc3339(),
                }))?,
            )?;
            let _ = done();
            let mut lines = vec![
                format!("You are {user} in {workspace} at {}.", waiting.server),
                format!("Your token for it is in {}", token_file.display()),
                "This machine's own setup is as it was: your private memories stay yours and stay in use.".to_string(),
                format!(
                    "To point this machine's agents at the workspace instead: antumbra setup hosted {} --token-file {}",
                    waiting.server,
                    token_file.display()
                ),
            ];
            lines.extend(caveat);
            Ok(lines)
        }
        other => anyhow::bail!(
            "the server answered with a standing this version does not know: {other:?}"
        ),
    }
}

fn forget(antumbra: &Path) -> anyhow::Result<Vec<String>> {
    match read_waiting(antumbra) {
        Some(waiting) => {
            std::fs::remove_file(waiting_file(antumbra))?;
            Ok(vec![format!(
                "Dropped the request for `{}` at {}. If it is approved it can no longer be collected from this machine.",
                waiting.handle, waiting.server
            )])
        }
        None => Ok(vec!["This machine is not waiting on a request.".to_string()]),
    }
}

/// The flow, with the network and the home given.
fn run_in(antumbra: &Path, args: &JoinArgs, send: Send<'_>) -> anyhow::Result<Vec<String>> {
    if args.forget {
        return forget(antumbra);
    }
    if args.status {
        return status(antumbra, send);
    }
    ask(antumbra, args, send)
}

pub fn run(args: JoinArgs) -> anyhow::Result<()> {
    let home = crate::machine::home().ok_or_else(|| anyhow::anyhow!("no home directory"))?;
    let agent = antumbra_http::agent(Duration::from_secs(30));
    let send = |server: &str, path: &str, body: Value| -> anyhow::Result<(u16, Value)> {
        let mut response = agent
            .post(format!("{server}{path}"))
            .header("content-type", "application/json")
            .send_json(body)
            .map_err(|e| anyhow::anyhow!("could not reach {server}: {e}"))?;
        let status = response.status().as_u16();
        let answer = response.body_mut().read_json().unwrap_or(Value::Null);
        Ok((status, answer))
    };
    for line in run_in(&home.join(".antumbra"), &args, &send)? {
        println!("{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn args(server: &str, handle: &str) -> JoinArgs {
        JoinArgs {
            server: Some(server.to_string()),
            code: Some("aj1_code".to_string()),
            handle: Some(handle.to_string()),
            name: Some("Bodil Lindqvist".to_string()),
            note: None,
            status: false,
            forget: false,
            allow_http: false,
        }
    }

    fn asking_status() -> JoinArgs {
        JoinArgs {
            status: true,
            ..args("", "")
        }
    }

    /// A server that takes one request and answers by its standing.
    struct Server {
        standing: RefCell<&'static str>,
        digest: RefCell<Option<String>>,
        asked: RefCell<Vec<Value>>,
    }

    impl Server {
        fn new() -> Self {
            Self {
                standing: RefCell::new("pending"),
                digest: RefCell::new(None),
                asked: RefCell::new(Vec::new()),
            }
        }

        fn send(&self, _server: &str, path: &str, body: Value) -> anyhow::Result<(u16, Value)> {
            self.asked
                .borrow_mut()
                .push(json!({ "path": path, "body": body.clone() }));
            match path {
                "/join/request" => {
                    *self.digest.borrow_mut() = body["secret_digest"].as_str().map(str::to_string);
                    Ok((
                        200,
                        json!({ "request": "r1", "standing": "pending", "handle": body["handle"] }),
                    ))
                }
                "/join/collect" => {
                    let shown = join::digest(body["secret"].as_str().unwrap_or_default());
                    if Some(shown) != *self.digest.borrow() {
                        return Ok((
                            403,
                            json!({ "error": "no such request, or that is not its secret" }),
                        ));
                    }
                    Ok((
                        200,
                        match *self.standing.borrow() {
                            "approved" => {
                                json!({ "standing": "approved", "workspace": "ws:crew", "user": "user:bodil", "token": "the.signed.token" })
                            }
                            "denied" => {
                                json!({ "standing": "denied", "reason": "not this quarter" })
                            }
                            other => json!({ "standing": other }),
                        },
                    ))
                }
                _ => Ok((404, Value::Null)),
            }
        }
    }

    /// Asked, waited, approved, collected: the token is kept in a file of its
    /// own, the request file is gone, and the secret was sent only to collect.
    #[test]
    fn a_request_is_asked_waited_on_and_collected() -> anyhow::Result<()> {
        let home = crate::agents::scratch::Scratch::new("join")?;
        let antumbra = home.path().join(".antumbra");
        let server = Server::new();
        let send = |s: &str, p: &str, b: Value| server.send(s, p, b);

        let said = run_in(&antumbra, &args("https://memory.example/", "Bodil"), &send)?;
        assert!(said[0].contains("`bodil`"), "{said:?}");
        let waiting = read_waiting(&antumbra).ok_or_else(|| anyhow::anyhow!("nothing kept"))?;
        assert_eq!(waiting.server, "https://memory.example");
        assert_eq!(waiting.request.as_deref(), Some("r1"));
        // The server was given the digest, not the secret.
        let first = server.asked.borrow()[0].to_string();
        assert!(first.contains(&join::digest(&waiting.secret)) && !first.contains(&waiting.secret));

        let said = run_in(&antumbra, &asking_status(), &send)?;
        assert!(said[0].contains("Still waiting"), "{said:?}");
        assert!(read_waiting(&antumbra).is_some());

        *server.standing.borrow_mut() = "approved";
        let said = run_in(&antumbra, &asking_status(), &send)?;
        assert!(said[0].contains("user:bodil in ws:crew"), "{said:?}");
        let token = antumbra.join("workspaces/memory.example__ws_crew/token.txt");
        assert_eq!(std::fs::read_to_string(&token)?.trim(), "the.signed.token");
        assert!(
            read_waiting(&antumbra).is_none(),
            "the request is done with"
        );
        // The machine's own token was never touched.
        assert!(!antumbra.join("token.txt").exists());
        Ok(())
    }

    /// A retry is the same request: the secret kept is the secret sent again.
    /// Another request is refused while one is waited on, until it is dropped.
    #[test]
    fn one_request_is_waited_on_at_a_time() -> anyhow::Result<()> {
        let home = crate::agents::scratch::Scratch::new("join-one")?;
        let antumbra = home.path().join(".antumbra");
        let server = Server::new();
        let send = |s: &str, p: &str, b: Value| server.send(s, p, b);
        run_in(&antumbra, &args("https://memory.example", "bodil"), &send)?;
        let first = server.digest.borrow().clone();
        run_in(&antumbra, &args("https://memory.example", "bodil"), &send)?;
        assert_eq!(*server.digest.borrow(), first, "the same secret");

        let Err(refused) = run_in(&antumbra, &args("https://other.example", "bodil"), &send) else {
            anyhow::bail!("a second request was made over the first");
        };
        assert!(refused.to_string().contains("already waiting"), "{refused}");
        let dropped = run_in(
            &antumbra,
            &JoinArgs {
                forget: true,
                ..args("", "")
            },
            &send,
        )?;
        assert!(dropped[0].contains("Dropped"), "{dropped:?}");
        run_in(&antumbra, &args("https://other.example", "bodil"), &send)?;
        Ok(())
    }

    /// What can be said before anything is sent is: a handle nobody may take,
    /// and an address the code would cross in cleartext. A refusal is told
    /// with its reason and ends the wait.
    #[test]
    fn what_cannot_be_asked_is_said_and_a_refusal_ends_the_wait() -> anyhow::Result<()> {
        let home = crate::agents::scratch::Scratch::new("join-refused")?;
        let antumbra = home.path().join(".antumbra");
        let server = Server::new();
        let send = |s: &str, p: &str, b: Value| server.send(s, p, b);

        for (server_at, handle, why) in [
            (
                "https://memory.example",
                "default",
                "not a handle anyone may take",
            ),
            ("https://memory.example", "x", "3 to 32"),
            ("http://memory.example", "bodil", "cleartext"),
        ] {
            let Err(refused) = run_in(&antumbra, &args(server_at, handle), &send) else {
                anyhow::bail!("{handle} at {server_at} was asked");
            };
            assert!(refused.to_string().contains(why), "{refused}");
        }
        assert!(server.asked.borrow().is_empty(), "nothing was sent");
        assert!(read_waiting(&antumbra).is_none());

        run_in(&antumbra, &args("https://memory.example", "bodil"), &send)?;
        *server.standing.borrow_mut() = "denied";
        let said = run_in(&antumbra, &asking_status(), &send)?;
        assert_eq!(said, ["Refused: not this quarter"]);
        assert!(read_waiting(&antumbra).is_none());
        let Err(none) = run_in(&antumbra, &asking_status(), &send) else {
            anyhow::bail!("a status was given with nothing waited on");
        };
        assert!(
            none.to_string().contains("not waiting on a request"),
            "{none}"
        );
        Ok(())
    }
}
