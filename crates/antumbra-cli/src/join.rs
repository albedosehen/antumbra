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
    /// Your email address, where the workspace has a control plane: it sends
    /// a link there, and the workspace's admins see your request once you
    /// have followed it.
    #[arg(long, conflicts_with_all = ["status", "forget"])]
    pub email: Option<String>,
    /// Drop the request this machine is waiting on, without collecting it.
    #[arg(long, conflicts_with_all = ["code", "handle", "name", "note"])]
    pub forget: bool,
    /// Give this machine's coding agent the tools of the workspaces it has
    /// joined: each is registered with Claude Code as an MCP server of its
    /// own, beside the machine's own, reading the workspace's token from its
    /// file.
    #[arg(long, conflicts_with_all = ["code", "handle", "name", "note", "status", "forget", "disconnect"])]
    pub connect: bool,
    /// Take those MCP servers back out.
    #[arg(long, conflicts_with_all = ["code", "handle", "name", "note", "status", "forget"])]
    pub disconnect: bool,
    /// Leave a workspace this machine joined: you are removed from it on its
    /// server, your agent's tools for it are taken out, and what this
    /// machine kept of it (its token among it) is deleted. What you shared
    /// there stays there.
    #[arg(long, conflicts_with_all = ["code", "handle", "name", "note", "status", "forget", "connect", "disconnect", "write"])]
    pub leave: bool,
    /// With `--leave`: drop only what this machine keeps, and ask nothing of
    /// the server. For a server that is gone, or that refuses.
    #[arg(long, requires = "leave")]
    pub local: bool,
    /// Where an agent's new memories go once this machine is in a workspace:
    /// `private` (your own store; the default), `workspace` (what is learned
    /// about the team's work goes to the workspace), or `ask` (your own
    /// store, and the agent asks when something looks like the team's).
    #[arg(long, value_name = "WHERE", conflicts_with_all = ["code", "handle", "name", "note", "status", "forget"])]
    pub write: Option<String>,
    /// The workspace `--connect`, `--disconnect` and `--write` are about, by
    /// its id or address, when this machine has joined more than one.
    #[arg(long)]
    pub to: Option<String>,
    /// Accept a plain `http://` address that is not this machine: a lab whose
    /// network is trusted. The code and the secret travel in cleartext.
    #[arg(long)]
    pub allow_http: bool,
}

/// One request to a server: the path, the body, and what came back as a
/// status and the answer's JSON. A seam, so the flow is tested with no server.
pub(crate) type Send<'a> = &'a dyn Fn(&str, &str, Value) -> anyhow::Result<(u16, Value)>;

/// What this machine keeps of a request it is waiting on.
#[derive(Debug, Clone, PartialEq)]
struct Waiting {
    server: String,
    handle: String,
    /// The secret the token is collected with. Never sent before then.
    secret: String,
    /// The request's id, once the server has answered with one.
    request: Option<String>,
    /// The control plane the request was made at and is collected from,
    /// where the workspace's own server signs nothing.
    through: Option<String>,
}

impl Waiting {
    fn to_json(&self) -> Value {
        json!({
            "server": self.server,
            "handle": self.handle,
            "secret": self.secret,
            "request": self.request,
            "through": self.through,
        })
    }

    fn from_json(kept: &Value) -> Option<Self> {
        let text = |name: &str| kept.get(name).and_then(Value::as_str).map(str::to_string);
        Some(Self {
            server: text("server")?,
            handle: text("handle")?,
            secret: text("secret")?,
            request: text("request"),
            through: text("through"),
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
pub(crate) fn new_secret() -> anyhow::Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes)
        .map_err(|e| anyhow::anyhow!("no randomness from the system: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// What a server said was wrong, or its status when it said nothing.
pub(crate) fn said(status: u16, answer: &Value) -> String {
    answer
        .get("error")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("the server answered {status}"))
}

/// A directory name for a workspace on a server: what a path can hold of both.
pub(crate) fn workspace_dir(antumbra: &Path, server: &str, workspace: &str) -> PathBuf {
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
            through: None,
        },
    };
    // Kept before it is sent: an answer lost on the way is asked for again
    // with the same secret, and is the same request.
    keep_waiting(antumbra, &waiting)?;
    let asking = json!({
        "code": code.trim(),
        "handle": handle,
        "display": args.name,
        "note": args.note,
        "host": antumbra_http::host_id(),
        "secret_digest": join::digest(&waiting.secret),
    });
    let nothing_asked = || {
        // Nothing was asked: nothing is waited on.
        if waiting.request.is_none() {
            let _ = std::fs::remove_file(waiting_file(antumbra));
        }
    };
    let (mut status, mut answer) = send(&server, "/join/request", asking.clone())?;
    // A workspace whose tokens a control plane signs says where that is: the
    // request is made there, with an address it verifies.
    let through = answer
        .get("control_plane")
        .and_then(Value::as_str)
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| status != 200 && !url.is_empty());
    if let Some(through) = &through {
        let Some(email) = args
            .email
            .as_deref()
            .map(str::trim)
            .filter(|e| !e.is_empty())
        else {
            nothing_asked();
            anyhow::bail!(
                "this workspace verifies an address before its admins see a request. Ask again with: --email <your address>"
            );
        };
        if let Err(why) = crate::setup::may_carry_the_token(through, args.allow_http) {
            nothing_asked();
            return Err(why);
        }
        let mut with_address = asking;
        with_address["email"] = json!(email);
        (status, answer) = send(through, "/join/request", with_address)?;
    }
    if status != 200 {
        nothing_asked();
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
            through: through.clone(),
            ..waiting
        },
    )?;
    let standing = answer
        .get("standing")
        .and_then(Value::as_str)
        .unwrap_or("pending");
    Ok(match standing {
        "unverified" => vec![
            format!(
                "A link was sent to {}. Follow it: the workspace's admins see your request to join as `{handle}` once you have.",
                args.email.as_deref().unwrap_or("your address").trim()
            ),
            "Then see how it stands, and collect your token once approved, with: antumbra join --status".to_string(),
            "Nothing about this machine's setup has changed.".to_string(),
        ],
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
        waiting.through.as_deref().unwrap_or(&waiting.server),
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
        "unverified" => Ok(vec![
            "The link sent to your address has not been followed yet: the workspace's admins see your request once it has.".to_string(),
            "If no link came, drop this with `antumbra join --forget` and ask again.".to_string(),
        ]),
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
                    "write": "private",
                    "joined_at": chrono::Utc::now().to_rfc3339(),
                }))?,
            )?;
            let _ = done();
            let mut lines = vec![
                format!("You are {user} in {workspace} at {}.", waiting.server),
                format!("Your token for it is in {}", token_file.display()),
                "This machine's own setup is as it was: your private memories stay yours and stay in use.".to_string(),
                "What the workspace holds comes with each prompt from now, labelled as the workspace's.".to_string(),
                "To give your agent the workspace's tools as well: antumbra join --connect".to_string(),
                "New memories go to your own store unless you say otherwise: antumbra join --write private|workspace|ask".to_string(),
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

/// Where new memories go, as `--write` names it.
const WRITES: [&str; 3] = ["private", "workspace", "ask"];

/// Register an MCP server with the machine's agent, or take it out: its name
/// and, to register, its definition as JSON. `Ok(false)` when the agent's
/// command is not on this machine. A seam, so the flow is tested without one.
pub(crate) type Register<'a> = &'a dyn Fn(&str, Option<&str>) -> anyhow::Result<bool>;

/// The name a workspace's MCP server is registered under.
fn server_name(workspace: &str) -> String {
    let plain: String = workspace
        .trim_start_matches("ws:")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    format!("antumbra-{}", plain.trim_matches('-'))
}

/// The joined workspaces `--to` names, or all of them, each with the file
/// that records it.
fn recorded(antumbra: &Path, to: Option<&str>) -> anyhow::Result<Vec<(PathBuf, Value)>> {
    let mut found = Vec::new();
    let dirs = std::fs::read_dir(antumbra.join("workspaces"))
        .map(|dirs| dirs.flatten().map(|d| d.path()).collect::<Vec<_>>())
        .unwrap_or_default();
    for dir in dirs {
        let file = dir.join("workspace.json");
        let Some(kept) = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        else {
            continue;
        };
        let field = |name: &str| kept.get(name).and_then(Value::as_str).unwrap_or_default();
        let named = to.is_none_or(|to| {
            let to = to.trim().trim_end_matches('/');
            field("workspace") == to
                || field("url").trim_end_matches('/') == to
                || field("workspace").trim_start_matches("ws:") == to
        });
        if named && !field("workspace").is_empty() {
            found.push((file, kept));
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    if found.is_empty() {
        anyhow::bail!(match to {
            Some(to) => format!("this machine has not joined `{to}`"),
            None => "this machine has joined no workspace. Ask to join one with: antumbra join <server> --code <code> --handle <handle>".to_string(),
        });
    }
    Ok(found)
}

fn keep(file: &Path, kept: &Value) -> anyhow::Result<()> {
    crate::setup::write_private(file, &serde_json::to_string_pretty(kept)?)?;
    Ok(())
}

/// Give the machine's agent each workspace's tools, or take them back.
fn connect(
    antumbra: &Path,
    to: Option<&str>,
    on: bool,
    program: &str,
    register: Register<'_>,
) -> anyhow::Result<Vec<String>> {
    let mut lines = Vec::new();
    for (file, mut kept) in recorded(antumbra, to)? {
        let field = |name: &str| {
            kept.get(name)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let (workspace, url, token_file) = (field("workspace"), field("url"), field("token_file"));
        let name = server_name(&workspace);
        if !on {
            let had = register(&name, None)?;
            if let Some(kept) = kept.as_object_mut() {
                kept.remove("server");
            }
            keep(&file, &kept)?;
            lines.push(match had {
                true => format!(
                    "{workspace}: its tools are taken back out (the MCP server \"{name}\")."
                ),
                false => format!(
                    "{workspace}: recorded as not connected; no agent's command was found to tell."
                ),
            });
            continue;
        }
        // A bridge of its own, reading the workspace's token from its file
        // and never the machine's own from the environment.
        let server = json!({
            "type": "stdio",
            "command": program,
            "args": ["mcp", "bridge", "--agent", "claude", "--surface", url, "--token-file", token_file, "--file-token-only"],
        })
        .to_string();
        if register(&name, Some(&server))? {
            kept["server"] = json!(name);
            keep(&file, &kept)?;
            lines.push(format!(
                "{workspace}: your agent has its tools as the MCP server \"{name}\", beside your own. Start a new session to see them."
            ));
        } else {
            lines.push(format!(
                "{workspace}: none of the agents setup connects was found on this machine. For Claude Code, once it is installed:"
            ));
            lines.push(format!(
                "  claude mcp add-json {name} --scope user '{server}'"
            ));
        }
    }
    Ok(lines)
}

/// Say where an agent's new memories go for each workspace named.
fn set_write(antumbra: &Path, to: Option<&str>, write: &str) -> anyhow::Result<Vec<String>> {
    let write = write.trim().to_ascii_lowercase();
    if !WRITES.contains(&write.as_str()) {
        anyhow::bail!("`{write}` is not where memories can go: private, workspace or ask");
    }
    let mut lines = Vec::new();
    for (file, mut kept) in recorded(antumbra, to)? {
        let workspace = kept
            .get("workspace")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let connected = kept.get("server").and_then(Value::as_str).is_some();
        kept["write"] = json!(write);
        keep(&file, &kept)?;
        lines.push(match write.as_str() {
            "workspace" if connected => format!("{workspace}: what is learned about the team's work goes to the workspace; what is yours stays in your own store."),
            "workspace" => format!("{workspace}: recorded, and it does nothing until your agent has the workspace's tools: antumbra join --connect"),
            "ask" => format!("{workspace}: new memories go to your own store, and your agent asks when something looks like the team's."),
            _ => format!("{workspace}: new memories go to your own store."),
        });
    }
    lines.push("An agent reads this when a session starts.".to_string());
    Ok(lines)
}

/// Register with Claude Code, by its own command.
fn with_claude(name: &str, server: Option<&str>) -> anyhow::Result<bool> {
    let Some(claude) = crate::setup::find_program("claude") else {
        return Ok(false);
    };
    let quiet = |args: &[&str]| {
        std::process::Command::new(&claude)
            .args(args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
    };
    // Out first either way: registering again replaces what was there.
    let _ = quiet(&["mcp", "remove", name, "--scope", "user"]);
    if let Some(server) = server {
        let out = std::process::Command::new(&claude)
            .args(["mcp", "add-json", name, "--scope", "user", server])
            .output()?;
        if !out.status.success() {
            anyhow::bail!(
                "registering the MCP server failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
    }
    Ok(true)
}

/// The bridge a server's definition runs, as its program and its arguments
/// for `agent`: the definition is Claude Code's, and names it.
fn bridge_for(server: &str, agent: &str) -> anyhow::Result<(String, Vec<String>)> {
    let server: Value = serde_json::from_str(server)?;
    let command = server["command"].as_str().unwrap_or("antumbra").to_string();
    let args = server["args"]
        .as_array()
        .map(|args| {
            args.iter()
                .filter_map(Value::as_str)
                .map(|arg| if arg == "claude" { agent } else { arg }.to_string())
                .collect()
        })
        .unwrap_or_default();
    Ok((command, args))
}

/// Register with Codex, by its own command.
fn with_codex(name: &str, server: Option<&str>) -> anyhow::Result<bool> {
    let Some(codex) = crate::setup::find_running("codex") else {
        return Ok(false);
    };
    let _ = std::process::Command::new(&codex)
        .args(["mcp", "remove", name])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if let Some(server) = server {
        let (command, args) = bridge_for(server, "codex")?;
        let out = std::process::Command::new(&codex)
            .args(["mcp", "add", name, "--"])
            .arg(command)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()?;
        if !out.status.success() {
            anyhow::bail!(
                "registering the MCP server with Codex failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
    }
    Ok(true)
}

/// Register with Copilot CLI, whose config is in `dir`: one entry in its
/// `mcp-config.json`, the rest of the file left as it is. `Ok(false)` when
/// Copilot CLI has no config on this machine.
fn with_copilot_at(dir: &Path, name: &str, server: Option<&str>) -> anyhow::Result<bool> {
    if !dir.is_dir() {
        return Ok(false);
    }
    let path = dir.join("mcp-config.json");
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let edited = match server {
        Some(server) => {
            let (command, args) = bridge_for(server, "copilot")?;
            let entry =
                json!({ "type": "local", "command": command, "args": args, "tools": ["*"] });
            // An entry of this name from before is replaced, not kept.
            let without = without_server(&text, name)?.unwrap_or(text.clone());
            crate::setup::settings::put(&without, "mcp-config.json", &["mcpServers"], name, &entry)?
                .or((without != text).then_some(without))
        }
        None => without_server(&text, name)?,
    };
    if let Some(edited) = edited {
        if path.exists() {
            std::fs::copy(
                &path,
                path.with_file_name(format!(
                    "mcp-config.json.antumbra-backup-{}",
                    chrono::Utc::now().format("%Y%m%d%H%M%S")
                )),
            )?;
        }
        std::fs::write(&path, edited)?;
    }
    Ok(true)
}

/// `text` (Copilot CLI's `mcp-config.json`) without the server `name`, or
/// nothing when it holds none. The file is written again from what it says,
/// so its spacing may change; a copy is kept beside it first.
fn without_server(text: &str, name: &str) -> anyhow::Result<Option<String>> {
    if text.trim().is_empty() {
        return Ok(None);
    }
    let mut config: Value = serde_json::from_str(text)
        .map_err(|e| anyhow::anyhow!("mcp-config.json is not JSON ({e}); left as it is"))?;
    let had = config
        .get_mut("mcpServers")
        .and_then(Value::as_object_mut)
        .is_some_and(|servers| servers.remove(name).is_some());
    Ok(had
        .then(|| serde_json::to_string_pretty(&config).map(|t| t + "\n"))
        .transpose()?)
}

/// Where Copilot CLI keeps its config: `COPILOT_HOME`, or `.copilot` in the
/// user's home.
fn copilot_home(antumbra: &Path) -> Option<PathBuf> {
    std::env::var("COPILOT_HOME")
        .ok()
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
        // `antumbra` is `.antumbra` in the user's home.
        .or_else(|| antumbra.parent().map(|home| home.join(".copilot")))
}

/// One request to a server as a member: the path, the member's token, the
/// body, and what came back. A seam, like [`Send`].
pub(crate) type SendAs<'a> = &'a dyn Fn(&str, &str, &str, Value) -> anyhow::Result<(u16, Value)>;

/// Leave one workspace: be removed on its server (unless `local`), take the
/// agent's tools for it out, and delete what this machine keeps of it.
fn leave(
    antumbra: &Path,
    to: Option<&str>,
    local: bool,
    register: Register<'_>,
    send_as: SendAs<'_>,
) -> anyhow::Result<Vec<String>> {
    let found = recorded(antumbra, to)?;
    let [(file, kept)] = found.as_slice() else {
        let names: Vec<&str> = found
            .iter()
            .filter_map(|(_, kept)| kept.get("workspace").and_then(Value::as_str))
            .collect();
        anyhow::bail!(
            "this machine has joined more than one workspace ({}). Say which to leave with --to",
            names.join(", ")
        );
    };
    let field = |name: &str| {
        kept.get(name)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let (workspace, url, user, token_file) = (
        field("workspace"),
        field("url"),
        field("user"),
        field("token_file"),
    );
    let mut lines = Vec::new();
    if !local {
        let token = std::fs::read_to_string(&token_file)
            .map(|t| t.trim().to_string())
            .unwrap_or_default();
        if token.is_empty() || user.is_empty() {
            anyhow::bail!(
                "this machine no longer holds a token for {workspace}, so its server cannot be told. To drop what the machine keeps of it: antumbra join --leave --local"
            );
        }
        let (status, answer) = send_as(&url, "/members/remove", &token, json!({ "user": user }))?;
        if status != 200 {
            anyhow::bail!(
                "{url} did not let you leave {workspace}: {}\nNothing on this machine was changed. To drop what the machine keeps of it anyway: antumbra join --leave --local",
                said(status, &answer)
            );
        }
        lines.push(format!(
            "You have left {workspace} at {url}. What you shared there stays there."
        ));
    }
    // The agent's tools for it, then the record and the token.
    let name = server_name(&workspace);
    let _ = register(&name, None);
    if let Some(dir) = file.parent() {
        std::fs::remove_dir_all(dir)?;
    }
    lines.push(match local {
        true => format!("This machine has dropped what it kept of {workspace}. You are still a member there until one of its admins removes you."),
        false => "This machine has dropped its token for it and your agent's tools for it.".to_string(),
    });
    Ok(lines)
}

/// A text file written whole, its directory made when it is not there.
fn write_whole(path: &Path, text: &str) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, text)?;
    Ok(())
}

/// A value as a file's text.
fn pretty(value: &Value) -> String {
    let mut text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    text.push('\n');
    text
}

/// Register with Gemini CLI, whose config is in `gemini`: an extension of
/// the workspace's own, beside the one setup wrote, holding its server and
/// nothing else. `Ok(false)` when setup has not connected Gemini CLI here.
fn with_gemini_at(gemini: &Path, name: &str, server: Option<&str>) -> anyhow::Result<bool> {
    let extensions = gemini.join("extensions");
    if !extensions.join(crate::agents::gemini::EXTENSION).is_dir() {
        return Ok(false);
    }
    let dir = extensions.join(name);
    match server {
        None => {
            if dir.is_dir() {
                std::fs::remove_dir_all(&dir)?;
            }
        }
        Some(server) => {
            let (command, args) = bridge_for(server, "gemini")?;
            write_whole(
                &dir.join("gemini-extension.json"),
                &pretty(&json!({
                    "name": name,
                    "version": env!("CARGO_PKG_VERSION"),
                    "description": "A workspace this machine joined, through Antumbra. Written by `antumbra join --connect`; `antumbra join --disconnect` takes it out.",
                    "mcpServers": { name: { "command": command, "args": args } },
                })),
            )?;
        }
    }
    Ok(true)
}

/// Register with goose, whose plugins are in `plugins`: a plugin of the
/// workspace's own holding its server. `Ok(false)` when setup has not
/// connected goose here.
fn with_goose_at(plugins: &Path, name: &str, server: Option<&str>) -> anyhow::Result<bool> {
    if !plugins.join(crate::agents::goose::PLUGIN).is_dir() {
        return Ok(false);
    }
    let dir = plugins.join(name);
    match server {
        None => {
            if dir.is_dir() {
                std::fs::remove_dir_all(&dir)?;
            }
        }
        Some(server) => {
            let (command, args) = bridge_for(server, "goose")?;
            write_whole(
                &dir.join("plugin.json"),
                &pretty(&json!({
                    "name": name,
                    "version": env!("CARGO_PKG_VERSION"),
                    "description": "A workspace this machine joined, through Antumbra. Written by `antumbra join --connect`.",
                })),
            )?;
            write_whole(
                &dir.join(".mcp.json"),
                &pretty(&json!({ "mcpServers": { name: { "command": command, "args": args } } })),
            )?;
        }
    }
    Ok(true)
}

/// Register with OpenCode, whose config is in `config`: a plugin file of the
/// workspace's own that registers its server the two ways OpenCode's two
/// release lines take one, as setup's own plugin registers the machine's.
/// `Ok(false)` when setup has not connected OpenCode here.
fn with_opencode_at(config: &Path, name: &str, server: Option<&str>) -> anyhow::Result<bool> {
    let plugins = config.join("plugins");
    if !plugins.join("antumbra.js").is_file() {
        return Ok(false);
    }
    let file = plugins.join(format!("{name}.js"));
    match server {
        None => {
            if file.is_file() {
                std::fs::remove_file(&file)?;
            }
        }
        Some(server) => {
            let (command, args) = bridge_for(server, "opencode")?;
            let mut line = vec![command];
            line.extend(args);
            // Strings as JavaScript: JSON's escapes are JavaScript's.
            let (name_js, line_js) = (json!(name).to_string(), json!(line).to_string());
            write_whole(
                &file,
                &format!(
                    "// A workspace this machine joined, through Antumbra. Written by `antumbra join --connect`;\n\
                     // `antumbra join --disconnect` takes it out.\n\
                     const NAME = {name_js}\n\
                     const SERVER = {{ type: \"local\", command: {line_js} }}\n\n\
                     export default {{\n  id: NAME,\n\n  // OpenCode 2.x.\n  async setup(ctx) {{\n    try {{\n      ctx.mcp.transform((servers) => {{\n        if (!servers.get(NAME)) servers.set(NAME, SERVER)\n      }})\n    }} catch {{}}\n  }},\n\n  // OpenCode 1.x.\n  async server() {{\n    return {{\n      config: async (config) => {{\n        config.mcp = {{ ...(config.mcp ?? {{}}) }}\n        if (!config.mcp[NAME]) config.mcp[NAME] = {{ ...SERVER, enabled: true }}\n      }},\n    }}\n  }},\n}}\n"
                ),
            )?;
        }
    }
    Ok(true)
}

/// Every agent told of a workspace's server, or asked to drop it, each with
/// how it went: Claude Code and Codex by their own commands, Copilot CLI by
/// an entry in its file, and Gemini CLI, goose and OpenCode by a bundle of
/// the workspace's own beside the one setup wrote them.
fn every_agent(
    antumbra: &Path,
    name: &str,
    server: Option<&str>,
) -> [(&'static str, anyhow::Result<bool>); 6] {
    // `antumbra` is `.antumbra` in the user's home.
    let home = antumbra.parent().unwrap_or(antumbra);
    let variable = |name: &str| std::env::var(name).ok();
    let nowhere = |_: &str| None;
    let ground = crate::agents::Ground {
        home,
        antumbra: crate::agents::Program::Named,
        env: &variable,
        find: &nowhere,
    };
    [
        ("Claude Code", with_claude(name, server)),
        ("Codex", with_codex(name, server)),
        (
            "Copilot CLI",
            match copilot_home(antumbra) {
                Some(dir) => with_copilot_at(&dir, name, server),
                None => Ok(false),
            },
        ),
        (
            "Gemini CLI",
            with_gemini_at(&crate::agents::gemini::home(&ground), name, server),
        ),
        (
            "goose",
            with_goose_at(&crate::agents::goose::plugins_dir(&ground), name, server),
        ),
        (
            "OpenCode",
            with_opencode_at(&crate::agents::opencode::config_dir(&ground), name, server),
        ),
    ]
}

/// Note which agents were told and which could not be, and say whether any
/// was: one agent that fails does not keep the others from being told.
fn tell_each<const N: usize>(
    results: [(&'static str, anyhow::Result<bool>); N],
    told: &std::cell::RefCell<std::collections::BTreeSet<&'static str>>,
    failed: &std::cell::RefCell<std::collections::BTreeSet<String>>,
) -> bool {
    let mut any = false;
    for (agent, result) in results {
        match result {
            Ok(true) => {
                told.borrow_mut().insert(agent);
                any = true;
            }
            Ok(false) => {}
            Err(why) => {
                failed
                    .borrow_mut()
                    .insert(format!("{agent} could not be told: {why}"));
            }
        }
    }
    any
}

/// The flow, with the network and the home given.
fn run_in(antumbra: &Path, args: &JoinArgs, send: Send<'_>) -> anyhow::Result<Vec<String>> {
    if let Some(write) = &args.write {
        return set_write(antumbra, args.to.as_deref(), write);
    }
    if args.connect || args.disconnect {
        let program = std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "antumbra".to_string());
        // Every agent on this machine that takes a server by name: Claude
        // Code and Codex by their own commands, Copilot CLI by its file.
        let told = std::cell::RefCell::new(std::collections::BTreeSet::new());
        let failed = std::cell::RefCell::new(std::collections::BTreeSet::new());
        let register = |name: &str, server: Option<&str>| {
            Ok(tell_each(
                every_agent(antumbra, name, server),
                &told,
                &failed,
            ))
        };
        let mut lines = connect(
            antumbra,
            args.to.as_deref(),
            args.connect,
            &program,
            &register,
        )?;
        let told: Vec<&str> = told.into_inner().into_iter().collect();
        // Taking a server out is asked of every agent found, and none says
        // whether it had it, so only a registration is reported by name.
        if args.connect && !told.is_empty() {
            lines.push(format!("Told: {}.", told.join(", ")));
        }
        lines.extend(failed.into_inner());
        return Ok(lines);
    }
    if args.forget {
        return forget(antumbra);
    }
    if args.status {
        return status(antumbra, send);
    }
    ask(antumbra, args, send)
}

/// One request to a server over the network, as [`Send`] takes it.
pub(crate) fn over_the_network(
    server: &str,
    path: &str,
    body: Value,
) -> anyhow::Result<(u16, Value)> {
    let agent = antumbra_http::agent(Duration::from_secs(30));
    let mut response = agent
        .post(format!("{server}{path}"))
        .header("content-type", "application/json")
        .send_json(body)
        .map_err(|e| anyhow::anyhow!("could not reach {server}: {e}"))?;
    let status = response.status().as_u16();
    let answer = response.body_mut().read_json().unwrap_or(Value::Null);
    Ok((status, answer))
}

pub fn run(args: JoinArgs) -> anyhow::Result<()> {
    let home = crate::machine::home().ok_or_else(|| anyhow::anyhow!("no home directory"))?;
    let antumbra = home.join(".antumbra");
    let lines = if args.leave {
        let register = |name: &str, server: Option<&str>| {
            Ok(every_agent(&antumbra, name, server)
                .into_iter()
                .any(|(_, told)| told.unwrap_or(false)))
        };
        let send_as =
            |server: &str, path: &str, token: &str, body: Value| -> anyhow::Result<(u16, Value)> {
                let mut response = antumbra_http::agent(Duration::from_secs(30))
                    .post(format!("{}{path}", server.trim_end_matches('/')))
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .send_json(body)
                    .map_err(|e| anyhow::anyhow!("could not reach {server}: {e}"))?;
                let status = response.status().as_u16();
                Ok((
                    status,
                    response.body_mut().read_json().unwrap_or(Value::Null),
                ))
            };
        leave(
            &antumbra,
            args.to.as_deref(),
            args.local,
            &register,
            &send_as,
        )?
    } else {
        run_in(&antumbra, &args, &over_the_network)?
    };
    for line in lines {
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
            email: None,
            leave: false,
            local: false,
            status: false,
            forget: false,
            connect: false,
            disconnect: false,
            write: None,
            to: None,
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

    fn joined_home(name: &str) -> anyhow::Result<(crate::agents::scratch::Scratch, PathBuf)> {
        let home = crate::agents::scratch::Scratch::new(name)?;
        let antumbra = home.path().join(".antumbra");
        for (dir, url, workspace) in [
            ("a__ws_crew", "https://team.example", "ws:crew"),
            ("b__ws_lab", "https://lab.example", "ws:The Lab"),
        ] {
            let at = antumbra.join("workspaces").join(dir);
            std::fs::create_dir_all(&at)?;
            std::fs::write(
                at.join("workspace.json"),
                json!({ "url": url, "workspace": workspace, "user": "user:bodil", "token_file": at.join("token.txt"), "write": "private" }).to_string(),
            )?;
        }
        Ok((home, antumbra))
    }

    fn recorded_as(antumbra: &Path, dir: &str) -> anyhow::Result<Value> {
        Ok(serde_json::from_str(&std::fs::read_to_string(
            antumbra.join("workspaces").join(dir).join("workspace.json"),
        )?)?)
    }

    /// Each workspace's tools are a server of its own, a bridge to the
    /// workspace with the workspace's token and never the machine's own; and
    /// they are taken back out by name.
    #[test]
    fn an_agent_is_given_each_workspaces_tools_and_they_are_taken_back() -> anyhow::Result<()> {
        let (_home, antumbra) = joined_home("join-connect")?;
        let calls = RefCell::new(Vec::new());
        let register = |name: &str, server: Option<&str>| {
            calls
                .borrow_mut()
                .push((name.to_string(), server.map(str::to_string)));
            Ok(true)
        };
        let said = connect(&antumbra, None, true, "/bin/antumbra", &register)?;
        assert_eq!(said.len(), 2, "{said:?}");
        let registered = calls.borrow().clone();
        assert_eq!(registered[0].0, "antumbra-crew");
        assert_eq!(registered[1].0, "antumbra-the-lab");
        let server: Value = serde_json::from_str(registered[0].1.as_deref().unwrap_or("{}"))?;
        assert_eq!(server["command"], "/bin/antumbra");
        let args: Vec<&str> = server["args"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert!(
            args.windows(2)
                .any(|w| w == ["--surface", "https://team.example"]),
            "{args:?}"
        );
        assert!(
            args.contains(&"--file-token-only") && args.contains(&"--token-file"),
            "{args:?}"
        );
        assert_eq!(
            recorded_as(&antumbra, "a__ws_crew")?["server"],
            "antumbra-crew"
        );

        // One of them, taken back.
        calls.borrow_mut().clear();
        connect(&antumbra, Some("crew"), false, "/bin/antumbra", &register)?;
        assert_eq!(*calls.borrow(), [("antumbra-crew".to_string(), None)]);
        assert!(recorded_as(&antumbra, "a__ws_crew")?
            .get("server")
            .is_none());
        assert_eq!(
            recorded_as(&antumbra, "b__ws_lab")?["server"],
            "antumbra-the-lab"
        );

        // No agent's command: the line to run is given, and nothing is recorded.
        let none = |_: &str, _: Option<&str>| Ok(false);
        let said = connect(&antumbra, Some("ws:crew"), true, "/bin/antumbra", &none)?;
        assert!(
            said.iter()
                .any(|l| l.contains("claude mcp add-json antumbra-crew")),
            "{said:?}"
        );
        assert!(recorded_as(&antumbra, "a__ws_crew")?
            .get("server")
            .is_none());
        Ok(())
    }

    /// Where new memories go is one of three, recorded for the workspace, and
    /// writing to a workspace the agent has no tools for is said to do nothing.
    #[test]
    fn where_new_memories_go_is_recorded_for_a_workspace() -> anyhow::Result<()> {
        let (_home, antumbra) = joined_home("join-write")?;
        assert!(set_write(&antumbra, None, "everywhere").is_err());
        let said = set_write(&antumbra, Some("crew"), "Workspace")?;
        assert!(
            said[0].contains("does nothing until your agent has the workspace's tools"),
            "{said:?}"
        );
        assert_eq!(recorded_as(&antumbra, "a__ws_crew")?["write"], "workspace");
        assert_eq!(recorded_as(&antumbra, "b__ws_lab")?["write"], "private");
        connect(&antumbra, Some("crew"), true, "/bin/antumbra", &|_, _| {
            Ok(true)
        })?;
        let said = set_write(&antumbra, Some("crew"), "workspace")?;
        assert!(said[0].contains("goes to the workspace"), "{said:?}");
        set_write(&antumbra, None, "ask")?;
        assert_eq!(recorded_as(&antumbra, "b__ws_lab")?["write"], "ask");
        assert!(set_write(&antumbra, Some("ws:nowhere"), "ask").is_err());
        Ok(())
    }

    /// Copilot CLI is told by one entry in its own file: the bridge under
    /// its own name, whatever else the file holds left there; told again, the
    /// entry is replaced; and taken out, only that entry goes.
    #[test]
    fn copilot_is_told_by_one_entry_in_its_file() -> anyhow::Result<()> {
        let home = crate::agents::scratch::Scratch::new("join-copilot")?;
        let dir = home.path().join(".copilot");
        let server = |url: &str| {
            json!({ "type": "stdio", "command": "/bin/antumbra",
                "args": ["mcp", "bridge", "--agent", "claude", "--surface", url, "--token-file", "/t", "--file-token-only"] })
            .to_string()
        };
        // No Copilot CLI on this machine: nothing is made for it.
        assert!(!with_copilot_at(
            &dir,
            "antumbra-crew",
            Some(&server("https://a.example"))
        )?);
        assert!(!dir.exists());

        std::fs::create_dir_all(&dir)?;
        let path = dir.join("mcp-config.json");
        std::fs::write(
            &path,
            "{ \"mcpServers\": { \"context7\": { \"type\": \"local\", \"command\": \"npx\" } } }",
        )?;
        assert!(with_copilot_at(
            &dir,
            "antumbra-crew",
            Some(&server("https://a.example"))
        )?);
        let read = || -> anyhow::Result<Value> {
            Ok(serde_json::from_str(&std::fs::read_to_string(&path)?)?)
        };
        let entry = read()?["mcpServers"]["antumbra-crew"].clone();
        assert_eq!(entry["command"], "/bin/antumbra");
        assert_eq!(entry["tools"], json!(["*"]));
        let args: Vec<&str> = entry["args"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert!(
            args.windows(2).any(|w| w == ["--agent", "copilot"]),
            "{args:?}"
        );
        assert!(
            args.contains(&"--file-token-only") && !args.contains(&"claude"),
            "{args:?}"
        );
        assert_eq!(read()?["mcpServers"]["context7"]["command"], "npx");

        // Told again with another address: one entry, the new one.
        assert!(with_copilot_at(
            &dir,
            "antumbra-crew",
            Some(&server("https://b.example"))
        )?);
        let text = std::fs::read_to_string(&path)?;
        assert!(
            text.contains("https://b.example") && !text.contains("https://a.example"),
            "{text}"
        );

        // Taken out: only that entry goes, and taking it out twice is quiet.
        assert!(with_copilot_at(&dir, "antumbra-crew", None)?);
        assert!(read()?["mcpServers"].get("antumbra-crew").is_none());
        assert_eq!(read()?["mcpServers"]["context7"]["command"], "npx");
        assert!(with_copilot_at(&dir, "antumbra-crew", None)?);

        // A file that is not JSON is left as it is, and said.
        std::fs::write(&path, "not json")?;
        assert!(
            with_copilot_at(&dir, "antumbra-crew", Some(&server("https://a.example"))).is_err()
        );
        assert_eq!(std::fs::read_to_string(&path)?, "not json");
        Ok(())
    }

    /// The bridge an agent runs names that agent, and is otherwise the one
    /// Claude Code is given.
    #[test]
    fn the_bridge_names_the_agent_it_is_for() -> anyhow::Result<()> {
        let server = json!({ "command": "/bin/antumbra", "args": ["mcp", "bridge", "--agent", "claude", "--surface", "https://a.example"] }).to_string();
        let (command, args) = bridge_for(&server, "codex")?;
        assert_eq!(command, "/bin/antumbra");
        assert_eq!(
            args,
            [
                "mcp",
                "bridge",
                "--agent",
                "codex",
                "--surface",
                "https://a.example"
            ]
        );
        Ok(())
    }

    /// An agent that cannot be told is said, and the others are told still.
    #[test]
    fn one_agent_failing_does_not_keep_the_others_from_being_told() {
        let told = RefCell::new(std::collections::BTreeSet::new());
        let failed = RefCell::new(std::collections::BTreeSet::new());
        let any = tell_each(
            [
                ("Claude Code", Ok(true)),
                ("Codex", Err(anyhow::anyhow!("codex not found in PATH"))),
                ("Copilot CLI", Ok(false)),
            ],
            &told,
            &failed,
        );
        assert!(any);
        assert_eq!(
            told.into_inner().into_iter().collect::<Vec<_>>(),
            ["Claude Code"]
        );
        let failed: Vec<String> = failed.into_inner().into_iter().collect();
        assert_eq!(failed, ["Codex could not be told: codex not found in PATH"]);
        let (told, failed) = (
            RefCell::new(Default::default()),
            RefCell::new(Default::default()),
        );
        assert!(!tell_each(
            [("Codex", Err(anyhow::anyhow!("no")))],
            &told,
            &failed
        ));
    }

    /// A workspace whose tokens a control plane signs: the request is made
    /// there with an address, the link has to be followed first, and the
    /// token is collected there and kept for the workspace's own server.
    #[test]
    fn a_workspace_with_a_control_plane_is_joined_through_it() -> anyhow::Result<()> {
        let home = crate::agents::scratch::Scratch::new("join-through")?;
        let antumbra = home.path().join(".antumbra");
        let standing = RefCell::new("unverified");
        let asked = RefCell::new(Vec::new());
        let send = |server: &str, path: &str, body: Value| -> anyhow::Result<(u16, Value)> {
            asked
                .borrow_mut()
                .push((server.to_string(), path.to_string(), body.clone()));
            Ok(match (server, path) {
                ("https://memory.example", _) => (
                    400,
                    json!({ "error": "ask to join there, with your email address", "control_plane": "https://accounts.example/" }),
                ),
                ("https://accounts.example", "/join/request") => {
                    (200, json!({ "request": "r9", "standing": "unverified" }))
                }
                ("https://accounts.example", "/join/collect") => match *standing.borrow() {
                    "approved" => (
                        200,
                        json!({ "standing": "approved", "workspace": "ws:crew", "user": "user:bodil", "token": "the.rs256.token" }),
                    ),
                    other => (200, json!({ "standing": other })),
                },
                _ => (404, json!({})),
            })
        };
        let server = "https://memory.example";

        // No address: said, and nothing is waited on.
        let Err(why) = run_in(&antumbra, &args(server, "Bodil"), &send) else {
            anyhow::bail!("asked without an address");
        };
        assert!(why.to_string().contains("--email"), "{why}");
        assert!(read_waiting(&antumbra).is_none());
        assert!(
            asked.borrow().iter().all(|(s, ..)| s == server),
            "the control plane was asked without an address"
        );

        let with_email = JoinArgs {
            email: Some(" bodil@example.com ".into()),
            ..args(server, "Bodil")
        };
        let said = run_in(&antumbra, &with_email, &send)?;
        assert!(
            said[0].contains("A link was sent to bodil@example.com"),
            "{said:?}"
        );
        let waiting = read_waiting(&antumbra).ok_or_else(|| anyhow::anyhow!("nothing kept"))?;
        assert_eq!(
            (
                waiting.server.as_str(),
                waiting.through.as_deref(),
                waiting.request.as_deref()
            ),
            (server, Some("https://accounts.example"), Some("r9"))
        );
        let (to, _, sent) = asked
            .borrow()
            .last()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("nothing sent"))?;
        assert_eq!(to, "https://accounts.example");
        assert_eq!(sent["email"], "bodil@example.com");
        assert!(
            sent.get("secret").is_none() && sent["secret_digest"] == join::digest(&waiting.secret)
        );

        // Not followed, then waiting on an admin: both keep the request.
        let said = run_in(&antumbra, &asking_status(), &send)?;
        assert!(said[0].contains("has not been followed"), "{said:?}");
        *standing.borrow_mut() = "pending";
        assert!(run_in(&antumbra, &asking_status(), &send)?[0].contains("Still waiting"));
        assert!(read_waiting(&antumbra).is_some());
        assert_eq!(
            asked
                .borrow()
                .last()
                .map(|(s, p, _)| (s.clone(), p.clone())),
            Some((
                "https://accounts.example".to_string(),
                "/join/collect".to_string()
            ))
        );

        // Approved: collected from the control plane, kept for the workspace's server.
        *standing.borrow_mut() = "approved";
        let said = run_in(&antumbra, &asking_status(), &send)?;
        assert!(
            said[0].contains("user:bodil in ws:crew at https://memory.example"),
            "{said:?}"
        );
        let dir = antumbra.join("workspaces/memory.example__ws_crew");
        assert_eq!(
            std::fs::read_to_string(dir.join("token.txt"))?.trim(),
            "the.rs256.token"
        );
        let kept: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("workspace.json"))?)?;
        assert_eq!(kept["url"], server);
        assert!(read_waiting(&antumbra).is_none());
        Ok(())
    }

    /// Leaving: the server is asked first, and a refusal changes nothing
    /// here; left, the agent's tools and the machine's record of the
    /// workspace go, and another workspace is untouched. `--local` asks no
    /// server.
    #[test]
    fn a_workspace_is_left_on_its_server_and_then_on_this_machine() -> anyhow::Result<()> {
        let (_home, antumbra) = joined_home("join-leave")?;
        for dir in ["a__ws_crew", "b__ws_lab"] {
            std::fs::write(
                antumbra.join("workspaces").join(dir).join("token.txt"),
                "the-token\n",
            )?;
        }
        let taken = RefCell::new(Vec::new());
        let register = |name: &str, server: Option<&str>| {
            taken
                .borrow_mut()
                .push((name.to_string(), server.is_none()));
            Ok(true)
        };
        let asked = RefCell::new(Vec::new());
        let refuse = RefCell::new(true);
        let send_as =
            |server: &str, path: &str, token: &str, body: Value| -> anyhow::Result<(u16, Value)> {
                asked.borrow_mut().push((
                    server.to_string(),
                    path.to_string(),
                    token.to_string(),
                    body,
                ));
                Ok(match *refuse.borrow() {
                    true => (400, json!({ "error": "a workspace keeps an owner" })),
                    false => (200, json!({ "removed": true })),
                })
            };
        // Two joined: which one has to be said.
        assert!(leave(&antumbra, None, false, &register, &send_as).is_err());
        assert!(asked.borrow().is_empty());

        // Refused: said, and nothing here is changed.
        let Err(why) = leave(&antumbra, Some("crew"), false, &register, &send_as) else {
            anyhow::bail!("left a workspace whose server refused");
        };
        assert!(
            why.to_string().contains("keeps an owner") && why.to_string().contains("--local"),
            "{why}"
        );
        assert!(antumbra
            .join("workspaces/a__ws_crew/workspace.json")
            .exists());
        assert!(taken.borrow().is_empty());

        *refuse.borrow_mut() = false;
        let said = leave(&antumbra, Some("crew"), false, &register, &send_as)?;
        assert!(said[0].contains("You have left ws:crew"), "{said:?}");
        let (server, path, token, body) = asked
            .borrow()
            .last()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("nothing asked"))?;
        assert_eq!(
            (server.as_str(), path.as_str(), token.as_str()),
            ("https://team.example", "/members/remove", "the-token")
        );
        assert_eq!(body, json!({ "user": "user:bodil" }));
        assert_eq!(*taken.borrow(), [("antumbra-crew".to_string(), true)]);
        assert!(!antumbra.join("workspaces/a__ws_crew").exists());
        assert!(antumbra
            .join("workspaces/b__ws_lab/workspace.json")
            .exists());

        // The other, on this machine only: no server is asked.
        let before = asked.borrow().len();
        let said = leave(&antumbra, None, true, &register, &send_as)?;
        assert!(said[0].contains("still a member"), "{said:?}");
        assert_eq!(asked.borrow().len(), before);
        assert!(!antumbra.join("workspaces/b__ws_lab").exists());
        Ok(())
    }

    /// Gemini CLI, goose and OpenCode are each told by a bundle of the
    /// workspace's own beside the one setup wrote: nothing is made where
    /// setup has not connected the agent, setup's own bundle is untouched,
    /// and taking it out removes only the workspace's.
    #[test]
    fn three_agents_are_told_by_a_bundle_of_the_workspaces_own() -> anyhow::Result<()> {
        let home = crate::agents::scratch::Scratch::new("join-bundles")?;
        let server = json!({ "type": "stdio", "command": "/bin/antumbra",
            "args": ["mcp", "bridge", "--agent", "claude", "--surface", "https://team.example", "--token-file", "/t", "--file-token-only"] })
        .to_string();
        let name = "antumbra-crew";
        let expects = |args: &Value, agent: &str| {
            let args: Vec<&str> = args
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            args.windows(2).any(|w| w == ["--agent", agent])
                && args.contains(&"--file-token-only")
                && !args.contains(&"claude")
        };

        // Gemini CLI.
        let gemini = home.path().join(".gemini");
        assert!(!with_gemini_at(&gemini, name, Some(&server))?);
        assert!(!gemini.exists());
        let own = gemini.join("extensions").join("antumbra");
        std::fs::create_dir_all(&own)?;
        std::fs::write(own.join("gemini-extension.json"), "{}")?;
        assert!(with_gemini_at(&gemini, name, Some(&server))?);
        let manifest: Value = serde_json::from_str(&std::fs::read_to_string(
            gemini
                .join("extensions")
                .join(name)
                .join("gemini-extension.json"),
        )?)?;
        assert_eq!(manifest["name"], name);
        assert_eq!(manifest["mcpServers"][name]["command"], "/bin/antumbra");
        assert!(
            expects(&manifest["mcpServers"][name]["args"], "gemini"),
            "{manifest}"
        );
        assert!(with_gemini_at(&gemini, name, None)?);
        assert!(!gemini.join("extensions").join(name).exists());
        assert_eq!(
            std::fs::read_to_string(own.join("gemini-extension.json"))?,
            "{}"
        );

        // goose.
        let plugins = home.path().join(".agents").join("plugins");
        assert!(!with_goose_at(&plugins, name, Some(&server))?);
        std::fs::create_dir_all(plugins.join("antumbra"))?;
        assert!(with_goose_at(&plugins, name, Some(&server))?);
        let plugin: Value = serde_json::from_str(&std::fs::read_to_string(
            plugins.join(name).join("plugin.json"),
        )?)?;
        let servers: Value = serde_json::from_str(&std::fs::read_to_string(
            plugins.join(name).join(".mcp.json"),
        )?)?;
        assert_eq!(plugin["name"], name);
        assert!(
            expects(&servers["mcpServers"][name]["args"], "goose"),
            "{servers}"
        );
        assert!(with_goose_at(&plugins, name, None)?);
        assert!(!plugins.join(name).exists() && plugins.join("antumbra").is_dir());

        // OpenCode.
        let config = home.path().join(".config").join("opencode");
        assert!(!with_opencode_at(&config, name, Some(&server))?);
        std::fs::create_dir_all(config.join("plugins"))?;
        std::fs::write(
            config.join("plugins").join("antumbra.js"),
            "// setup's own\n",
        )?;
        assert!(with_opencode_at(&config, name, Some(&server))?);
        let plugin = std::fs::read_to_string(config.join("plugins").join(format!("{name}.js")))?;
        assert!(
            plugin.contains("const NAME = \"antumbra-crew\""),
            "{plugin}"
        );
        assert!(
            plugin.contains("\"--agent\",\"opencode\"") && plugin.contains("\"--file-token-only\""),
            "{plugin}"
        );
        assert!(
            plugin.contains("servers.set(NAME, SERVER)")
                && plugin.contains("config.mcp[NAME] = { ...SERVER, enabled: true }"),
            "{plugin}"
        );
        assert!(with_opencode_at(&config, name, None)?);
        assert!(!config.join("plugins").join(format!("{name}.js")).exists());
        assert_eq!(
            std::fs::read_to_string(config.join("plugins").join("antumbra.js"))?,
            "// setup's own\n"
        );
        Ok(())
    }
}
