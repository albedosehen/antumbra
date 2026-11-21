//! `antumbra login`: sign this machine in again to a workspace its account
//! is in, where a control plane signs the workspace's tokens.
//!
//! For a lost token, or a second machine. `login <server> --email <address>`
//! asks the server where its control plane is and asks that to send a link;
//! the address follows it in a browser, which is given nothing; `login
//! --status` then collects the token here, with a secret this machine made
//! and never sent before. An account in several workspaces is asked which.
//! The token is kept where `antumbra join` keeps one, so the hooks and
//! everything else that reads a joined workspace find it.

use std::path::{Path, PathBuf};

use clap::Args;
use serde_json::{json, Value};

use antumbra_core::join;

use crate::join::{new_secret, said, workspace_dir, Send};

#[derive(Debug, Clone, Args)]
pub struct LoginArgs {
    /// The workspace's server, as `https://memory.example`.
    pub server: Option<String>,
    /// The address of your account.
    #[arg(long, conflicts_with_all = ["status", "forget"])]
    pub email: Option<String>,
    /// The workspace to sign in to, when your account is in more than one.
    #[arg(long)]
    pub workspace: Option<String>,
    /// Collect the token once you have followed the link.
    #[arg(long, conflicts_with = "forget")]
    pub status: bool,
    /// Drop the sign-in this machine is waiting on.
    #[arg(long)]
    pub forget: bool,
    /// Accept a plain `http://` address that is not this machine.
    #[arg(long)]
    pub allow_http: bool,
}

fn waiting_file(antumbra: &Path) -> PathBuf {
    antumbra.join("login.json")
}

fn read_waiting(antumbra: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(waiting_file(antumbra)).ok()?).ok()
}

fn ask(antumbra: &Path, args: &LoginArgs, send: Send<'_>) -> anyhow::Result<Vec<String>> {
    let (Some(server), Some(email)) = (
        args.server
            .as_deref()
            .map(|s| s.trim().trim_end_matches('/')),
        args.email
            .as_deref()
            .map(str::trim)
            .filter(|e| !e.is_empty()),
    ) else {
        anyhow::bail!(
            "to sign in: antumbra login <server> --email <your address> [--workspace <workspace>]\n\
             to collect the token once you followed the link: antumbra login --status"
        );
    };
    crate::setup::may_carry_the_token(server, args.allow_http)?;
    let (status, at) = send(server, "/join/where", json!({}))?;
    if status != 200 {
        anyhow::bail!("{}", said(status, &at));
    }
    let Some(through) = at
        .get("control_plane")
        .and_then(Value::as_str)
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty())
    else {
        anyhow::bail!(
            "{server} signs its own tokens and keeps no accounts to sign in to. For a new token, ask one of the workspace's admins for a join code: antumbra join {server} --code <code> --handle <handle>"
        );
    };
    crate::setup::may_carry_the_token(&through, args.allow_http)?;
    let secret = new_secret()?;
    let (status, answer) = send(
        &through,
        "/login/request",
        json!({
            "email": email,
            "secret_digest": join::digest(&secret),
            "workspace": args.workspace,
        }),
    )?;
    if status != 200 {
        anyhow::bail!("{}", said(status, &answer));
    }
    let request = answer
        .get("request")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("the control plane answered without a request id"))?;
    crate::setup::write_private(
        &waiting_file(antumbra),
        &serde_json::to_string_pretty(&json!({
            "server": server,
            "through": through,
            "secret": secret,
            "request": request,
        }))?,
    )?;
    Ok(vec![
        format!("If {email} has an account, a link was sent to it. Follow it in any browser."),
        "Then collect the token on this machine with: antumbra login --status".to_string(),
    ])
}

fn status(antumbra: &Path, args: &LoginArgs, send: Send<'_>) -> anyhow::Result<Vec<String>> {
    let waiting = read_waiting(antumbra);
    let field = |name: &str| {
        waiting
            .as_ref()
            .and_then(|w| w.get(name))
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let (Some(server), Some(through), Some(secret), Some(request)) = (
        field("server"),
        field("through"),
        field("secret"),
        field("request"),
    ) else {
        anyhow::bail!("this machine is not waiting on a sign-in. Ask with: antumbra login <server> --email <your address>");
    };
    let (status, answer) = send(
        &through,
        "/login/collect",
        json!({ "request": request, "secret": secret, "workspace": args.workspace }),
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
            "The link has not been followed yet, or the address has no account.".to_string(),
            "If no link came, drop this with `antumbra login --forget` and ask again.".to_string(),
        ]),
        "choose" => {
            let mut lines =
                vec!["Your account is in more than one workspace. Say which:".to_string()];
            for offered in answer
                .get("workspaces")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let text = |name: &str| offered.get(name).and_then(Value::as_str).unwrap_or("?");
                lines.push(format!(
                    "  antumbra login --status --workspace {}    (as {})",
                    text("workspace"),
                    text("user")
                ));
            }
            Ok(lines)
        }
        "expired" => {
            let _ = done();
            Ok(vec![
                "The link was followed too long ago. Ask again.".to_string()
            ])
        }
        "collected" => {
            let _ = done();
            Ok(vec![
                "The token for this sign-in was already collected, and is given once. Ask again."
                    .to_string(),
            ])
        }
        "approved" => {
            let text = |name: &str| {
                answer
                    .get(name)
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| anyhow::anyhow!("the control plane sent no `{name}`"))
            };
            let (token, workspace, user) = (text("token")?, text("workspace")?, text("user")?);
            let dir = workspace_dir(antumbra, &server, &workspace);
            let token_file = dir.join("token.txt");
            // The token first: it is given once.
            let caveat = crate::setup::write_private(&token_file, &format!("{token}\n"))?;
            // What this machine already keeps of the workspace stays: where
            // new memories go, and whether an agent has its tools.
            let record = dir.join("workspace.json");
            let mut kept: Value = std::fs::read_to_string(&record)
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok())
                .filter(Value::is_object)
                .unwrap_or_else(
                    || json!({ "write": "private", "joined_at": chrono::Utc::now().to_rfc3339() }),
                );
            kept["url"] = json!(server);
            kept["workspace"] = json!(workspace);
            kept["user"] = json!(user);
            kept["token_file"] = json!(token_file);
            crate::setup::write_private(&record, &serde_json::to_string_pretty(&kept)?)?;
            let _ = done();
            let mut lines = vec![
                format!("Signed in as {user} in {workspace} at {server}."),
                format!("Your token for it is in {}", token_file.display()),
            ];
            lines.extend(caveat);
            Ok(lines)
        }
        other => anyhow::bail!(
            "the control plane answered with a standing this version does not know: {other:?}"
        ),
    }
}

/// The flow, with the network and the home given.
fn run_in(antumbra: &Path, args: &LoginArgs, send: Send<'_>) -> anyhow::Result<Vec<String>> {
    if args.forget {
        return Ok(vec![match std::fs::remove_file(waiting_file(antumbra)) {
            Ok(()) => "Dropped the sign-in this machine was waiting on.".to_string(),
            Err(_) => "This machine is not waiting on a sign-in.".to_string(),
        }]);
    }
    if args.status {
        return status(antumbra, args, send);
    }
    ask(antumbra, args, send)
}

pub fn run(args: LoginArgs) -> anyhow::Result<()> {
    let home = crate::machine::home().ok_or_else(|| anyhow::anyhow!("no home directory"))?;
    for line in run_in(
        &home.join(".antumbra"),
        &args,
        &crate::join::over_the_network,
    )? {
        println!("{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn args() -> LoginArgs {
        LoginArgs {
            server: Some("https://memory.example/".into()),
            email: Some("dana@example.com".into()),
            workspace: None,
            status: false,
            forget: false,
            allow_http: false,
        }
    }

    /// A machine signs in through the control plane its server names: the
    /// secret is sent only when collecting, an account in two workspaces is
    /// asked which, and the token is kept for the workspace's server beside
    /// what the machine already kept of it.
    #[test]
    fn a_machine_signs_in_and_says_which_workspace() -> anyhow::Result<()> {
        let home = crate::agents::scratch::Scratch::new("login")?;
        let antumbra = home.path().join(".antumbra");
        let standing = RefCell::new("unverified");
        let asked = RefCell::new(Vec::new());
        let send = |server: &str, path: &str, body: Value| -> anyhow::Result<(u16, Value)> {
            asked
                .borrow_mut()
                .push((server.to_string(), path.to_string(), body.clone()));
            Ok(match (server, path) {
                ("https://memory.example", "/join/where") => (
                    200,
                    json!({ "signs_here": false, "control_plane": "https://accounts.example/" }),
                ),
                ("https://own.example", "/join/where") => {
                    (200, json!({ "signs_here": true, "control_plane": null }))
                }
                ("https://accounts.example", "/login/request") => {
                    (200, json!({ "request": "s1", "standing": "unverified" }))
                }
                ("https://accounts.example", "/login/collect") => {
                    match (*standing.borrow(), body["workspace"].as_str()) {
                        ("followed", None) => (
                            200,
                            json!({ "standing": "choose", "workspaces": [
                        { "workspace": "ws:first", "user": "user:a1" }, { "workspace": "ws:crew", "user": "user:dana" } ] }),
                        ),
                        ("followed", Some(ws)) => (
                            200,
                            json!({ "standing": "approved", "workspace": ws, "user": "user:dana", "token": "the.rs256.token" }),
                        ),
                        (other, _) => (200, json!({ "standing": other })),
                    }
                }
                _ => (404, json!({})),
            })
        };

        // A server that signs its own tokens has no accounts to sign in to.
        let own = LoginArgs {
            server: Some("https://own.example".into()),
            ..args()
        };
        let Err(why) = run_in(&antumbra, &own, &send) else {
            anyhow::bail!("signed in to a server with no accounts");
        };
        assert!(why.to_string().contains("join code"), "{why}");

        let said = run_in(&antumbra, &args(), &send)?;
        assert!(said[0].contains("dana@example.com"), "{said:?}");
        let kept = read_waiting(&antumbra).ok_or_else(|| anyhow::anyhow!("nothing kept"))?;
        let secret = kept["secret"].as_str().unwrap_or_default().to_string();
        let (to, _, sent) = asked
            .borrow()
            .last()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("nothing sent"))?;
        assert_eq!(to, "https://accounts.example");
        assert!(
            sent["secret_digest"] == join::digest(&secret) && !sent.to_string().contains(&secret)
        );

        let collecting = LoginArgs {
            status: true,
            email: None,
            server: None,
            ..args()
        };
        assert!(run_in(&antumbra, &collecting, &send)?[0].contains("not been followed"));
        *standing.borrow_mut() = "followed";
        let said = run_in(&antumbra, &collecting, &send)?;
        assert!(
            said.iter()
                .any(|l| l.contains("--workspace ws:crew") && l.contains("user:dana")),
            "{said:?}"
        );
        assert!(
            read_waiting(&antumbra).is_some(),
            "asking which spends nothing"
        );

        // What the machine already keeps of the workspace stays.
        let dir = antumbra.join("workspaces/memory.example__ws_crew");
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("workspace.json"), json!({ "url": "https://memory.example", "workspace": "ws:crew", "write": "ask", "server": "antumbra-crew" }).to_string())?;
        let said = run_in(
            &antumbra,
            &LoginArgs {
                workspace: Some("ws:crew".into()),
                ..collecting
            },
            &send,
        )?;
        assert!(
            said[0].contains("Signed in as user:dana in ws:crew at https://memory.example"),
            "{said:?}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("token.txt"))?.trim(),
            "the.rs256.token"
        );
        let record: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("workspace.json"))?)?;
        assert_eq!(
            (
                record["write"].clone(),
                record["server"].clone(),
                record["user"].clone()
            ),
            (json!("ask"), json!("antumbra-crew"), json!("user:dana"))
        );
        assert!(read_waiting(&antumbra).is_none());
        Ok(())
    }
}
