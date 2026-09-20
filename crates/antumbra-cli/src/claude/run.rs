//! `antumbra claude ...`: the edge, where the machine is read and the surface is
//! called. Everything it decides is decided in the modules beside it.

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};

use super::{
    bridge, brief, conventions, examine, mcp_lint, mcp_stdio, render, repository_root, rules,
    Inputs,
};
use crate::cli::ClaudeAction;

const SURFACE_TIMEOUT: Duration = Duration::from_secs(60);

fn home() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

fn project(dir: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    match dir {
        Some(dir) => Ok(dir),
        None => Ok(std::env::current_dir()?),
    }
}

/// One tool call over `POST {surface}/mcp/call`. A refusal comes back with the
/// surface's own words, since "400" tells a user nothing they can act on.
fn call_surface(
    agent: &ureq::Agent,
    surface: &str,
    token: Option<&str>,
    tool: &str,
    arguments: Value,
) -> anyhow::Result<Value> {
    let url = format!("{}/mcp/call", surface.trim_end_matches('/'));
    let mut request = agent.post(&url).header("content-type", "application/json");
    if let Some(token) = token {
        request = request.header("authorization", &format!("Bearer {token}"));
    }
    let mut response = request
        .send_json(json!({ "tool": tool, "arguments": arguments }))
        .map_err(|e| anyhow::anyhow!("{tool}: could not reach {url}: {e}"))?;
    let status = response.status();
    let body = response
        .body_mut()
        .read_json::<Value>()
        .unwrap_or(Value::Null);
    let refusal = body.get("error").map(|e| match e.as_str() {
        Some(text) => text.to_string(),
        None => e.to_string(),
    });
    match (status.is_success(), refusal) {
        (true, None) => Ok(body),
        (_, Some(said)) => anyhow::bail!("{tool}: {said}"),
        (false, None) => anyhow::bail!("{tool}: {url} answered {status}"),
    }
}

/// The `tools/list` answer to lint: a saved one, or a server asked just now.
fn listed_tools(
    from: Option<PathBuf>,
    command: &[String],
    patience: Duration,
) -> anyhow::Result<Vec<mcp_lint::Tool>> {
    let answer = match from {
        Some(path) if path.as_os_str() == "-" => serde_json::from_reader(std::io::stdin().lock())?,
        Some(path) => {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
            serde_json::from_str(&text)?
        }
        None => Value::Array(mcp_stdio::ask(command, patience)?),
    };
    mcp_lint::tools_in(&answer)
}

/// Run one `antumbra claude` action. Blocking: the caller keeps it off the
/// async runtime.
pub fn run(action: ClaudeAction) -> anyhow::Result<()> {
    match action {
        ClaudeAction::Doctor { dir } => {
            let report = examine(&Inputs::gather(home().as_deref(), &project(dir)?));
            println!("{}", render(&report));
            let missing = report.required_missing();
            if missing > 0 {
                anyhow::bail!("{missing} required setting(s) missing");
            }
        }
        ClaudeAction::Bridge {
            dir,
            dry_run,
            remove,
        } => {
            let project = project(dir)?;
            let root = repository_root(&project);
            let said = if remove {
                bridge::remove_bridges(&root, &project, dry_run)?
            } else {
                bridge::write_bridges(&root, &project, dry_run)?
            };
            if dry_run {
                println!("dry run: nothing written");
            }
            for line in said {
                println!("{line}");
            }
        }
        ClaudeAction::Brief { dir } => {
            let inputs = Inputs::gather_without_version(home().as_deref(), &project(dir)?);
            if let Some(text) = brief::brief(&examine(&inputs), &inputs.instructions, inputs.os) {
                println!("{text}");
            }
        }
        ClaudeAction::Remember {
            surface,
            token,
            dry_run,
        } => {
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .timeout_global(Some(SURFACE_TIMEOUT))
                .http_status_as_error(false)
                .build()
                .into();
            let call = |tool: &str, arguments: Value| {
                call_surface(&agent, &surface, token.as_deref(), tool, arguments)
            };
            if dry_run {
                println!("dry run: nothing written");
            }
            for line in conventions::remember(&call, &rules(), dry_run)? {
                println!("{line}");
            }
        }
        ClaudeAction::McpLint {
            server,
            from,
            timeout_secs,
            command,
        } => {
            let tools = listed_tools(from, &command, Duration::from_secs(timeout_secs))?;
            let findings = mcp_lint::lint(&tools);
            println!("{}", mcp_lint::render(&server, tools.len(), &findings));
            let failures = mcp_lint::failures(&findings);
            if failures > 0 {
                anyhow::bail!(
                    "{failures} tool(s) would break requests or cost the server its tools"
                );
            }
        }
    }
    Ok(())
}
