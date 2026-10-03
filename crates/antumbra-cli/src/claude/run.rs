//! `antumbra claude ...`: the edge, where the machine is read and the surface is
//! called. Everything it decides is decided in the modules beside it.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};

use super::{
    apply, auto_mode, bridge, brief, conventions, dependencies, examine, mcp_lint, mcp_stdio,
    reanchor, render, repository_root, skills, Inputs, Standing,
};
use crate::cli::ClaudeAction;

const SURFACE_TIMEOUT: Duration = Duration::from_secs(60);

/// A hook has seconds, not a minute, and is not worth waiting on.
const HOOK_TIMEOUT: Duration = Duration::from_secs(5);

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

/// An HTTP agent for the surface that hands back a refusal's body, not just its
/// status, so the surface's own words reach the user.
fn surface_agent() -> ureq::Agent {
    surface_agent_within(SURFACE_TIMEOUT)
}

fn surface_agent_within(patience: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(patience))
        .http_status_as_error(false)
        .build()
        .into()
}

/// A working tree's `origin` URL, when it has one.
fn origin_of(dir: &Path) -> Option<String> {
    std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["remote", "get-url", "origin"])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|url| !url.is_empty())
}

/// The newest `limit` merged pull requests of `repo` (a `host/org/name`
/// slug, which `gh` accepts as it is), as `gh pr list --json` writes them.
fn merged_pull_requests(repo: &str, limit: u32) -> anyhow::Result<String> {
    let out = std::process::Command::new("gh")
        .args(["pr", "list", "--repo", repo, "--state", "merged"])
        .args(["--limit", &limit.to_string(), "--json", reanchor::GH_FIELDS])
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| anyhow::anyhow!("could not run gh, the GitHub CLI: {e}"))?;
    if !out.status.success() {
        anyhow::bail!(
            "gh pr list failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8(out.stdout)?)
}

/// The project, and every repository directly under `repos`.
fn tree_dirs(project: &Path, repos: Option<&Path>) -> Vec<PathBuf> {
    let children = repos
        .and_then(|dir| std::fs::read_dir(dir).ok())
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join(".git").exists());
    std::iter::once(project.to_path_buf())
        .chain(children)
        .collect()
}

/// The project, and every repository directly under `repos`.
fn working_trees(project: &Path, repos: Option<&Path>) -> Vec<auto_mode::Seen> {
    let children = repos
        .and_then(|dir| std::fs::read_dir(dir).ok())
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join(".git").exists());
    std::iter::once((project.to_path_buf(), true))
        .chain(children.map(|path| (path, false)))
        .filter_map(|(dir, is_project)| auto_mode::seen(&origin_of(&dir)?, is_project))
        .collect()
}

/// The skills installed for the user and for the project, by the name each
/// declares (its directory's name when it declares none).
fn installed_skills(home: Option<&Path>, project: &Path) -> Vec<String> {
    let mut names: Vec<String> = home
        .into_iter()
        .chain(std::iter::once(project))
        .map(|base| base.join(".claude").join("skills"))
        .filter_map(|dir| std::fs::read_dir(dir).ok())
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let text = std::fs::read_to_string(entry.path().join("SKILL.md")).ok()?;
            skills::declared_name(&text).or_else(|| entry.file_name().into_string().ok())
        })
        .collect();
    names.sort();
    names.dedup();
    names
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
            let project = project(dir)?;
            let report = examine(&Inputs::gather(home().as_deref(), &project));
            println!("{}", render(&report));
            // Advisory: a drifted hook is worth knowing, not a missing setting.
            print!(
                "{}",
                super::hooks::render(&super::hooks::installed(home().as_deref(), &project))
            );
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
        ClaudeAction::Dependencies {
            dir,
            repos,
            surface,
            token,
            dry_run,
        } => {
            let trees: Vec<dependencies::Tree> = tree_dirs(&project(dir)?, repos.as_deref())
                .iter()
                .filter_map(|dir| {
                    let anchor = crate::gitctx::detect_in(dir)?;
                    let files = crate::gitctx::tracked_files(dir);
                    Some(dependencies::Tree::read(anchor, &files, |path| {
                        std::fs::read_to_string(dir.join(path)).ok()
                    }))
                })
                .collect();
            if trees.is_empty() {
                anyhow::bail!("no repository with an `origin` remote to read");
            }
            let agent = surface_agent();
            let call = |tool: &str, arguments: Value| {
                call_surface(&agent, &surface, token.as_deref(), tool, arguments)
            };
            if dry_run {
                println!("dry run: nothing recorded");
            }
            for line in dependencies::record(&call, &trees, dry_run)? {
                println!("{line}");
            }
        }
        ClaudeAction::Remember {
            surface,
            token,
            dry_run,
        } => {
            let agent = surface_agent();
            let call = |tool: &str, arguments: Value| {
                call_surface(&agent, &surface, token.as_deref(), tool, arguments)
            };
            if dry_run {
                println!("dry run: nothing written");
            }
            // The memories say what was checked against the verified release,
            // so a loss that ended by then is retired from them.
            let lasting = super::version::rules_for(super::VERIFIED_AGAINST);
            for line in conventions::remember(&call, &lasting, dry_run)? {
                println!("{line}");
            }
        }
        ClaudeAction::AutoModeEnv {
            dir,
            repos,
            surface,
            token,
            each,
        } => {
            let trees = working_trees(&project(dir)?, repos.as_deref());
            let agent = surface_agent();
            let call = |tool: &str, arguments: Value| {
                call_surface(&agent, &surface, token.as_deref(), tool, arguments)
            };
            // Memory is a second opinion. Without it the remotes still draft.
            let asked = auto_mode::remembered_repositories(&call).and_then(|remembered| {
                Ok((
                    remembered,
                    auto_mode::candidates(&call, &auto_mode::slots(), each)?,
                ))
            });
            let (remembered, memory) = match asked {
                Ok((remembered, found)) => (remembered, auto_mode::Memory::Asked(found)),
                Err(why) => (Vec::new(), auto_mode::Memory::NotAsked(why.to_string())),
            };
            let owners = auto_mode::owners(&trees, &remembered);
            println!("{}", auto_mode::render(&owners, &memory));
        }
        ClaudeAction::Apply { dir, file, dry_run } => {
            let home = home();
            let report = examine(&Inputs::gather(home.as_deref(), &project(dir)?));
            let path = match (file, home) {
                (Some(path), _) => path,
                (None, Some(home)) => home.join(".claude").join("settings.json"),
                (None, None) => {
                    anyhow::bail!("no home directory to find your settings in: give --file")
                }
            };
            let settings = apply::wanted(&report);
            let said = apply::apply(&path, &settings, dry_run)?;
            if dry_run && !settings.is_empty() {
                println!("dry run: nothing written");
            }
            for line in said {
                println!("{line}");
            }
            // The doctor asks for this one too, and this deliberately does not.
            if report.findings.iter().any(|finding| {
                finding.rule.id == "auto-mode-default"
                    && matches!(finding.standing, Standing::Missing { .. })
            }) {
                println!(
                    "left     permissions.defaultMode: set it to \"auto\" yourself if you want \
                     auto mode. Nothing here writes under `permissions`"
                );
            }
        }
        ClaudeAction::SkillUsed {
            name,
            surface,
            token,
        } => {
            let as_a_hook = name.is_none();
            let agent = surface_agent_within(if as_a_hook {
                HOOK_TIMEOUT
            } else {
                SURFACE_TIMEOUT
            });
            let call = |tool: &str, arguments: Value| {
                call_surface(&agent, &surface, token.as_deref(), tool, arguments)
            };
            let counted = match name {
                Some(name) => skills::record(&call, &name).map(|how| Some((name, how))),
                // Input that names no skill is not an error: most hook events do not.
                None => serde_json::from_reader::<_, Value>(std::io::stdin().lock())
                    .map_err(anyhow::Error::from)
                    .map(|input| skills::skill_in(&input))
                    .and_then(|found| match found {
                        Some(name) => skills::record(&call, &name).map(|how| Some((name, how))),
                        None => Ok(None),
                    }),
            };
            match (as_a_hook, counted) {
                // Fail open, and silently: a counter must not stop a session.
                (true, _) => {}
                (false, Ok(Some((name, skills::Recorded::First)))) => {
                    println!("{name}: first use counted");
                }
                (false, Ok(Some((name, skills::Recorded::Again)))) => {
                    println!("{name}: counted");
                }
                (false, Ok(None)) => {}
                (false, Err(why)) => return Err(why),
            }
        }
        ClaudeAction::Skills {
            dir,
            days,
            surface,
            token,
        } => {
            let agent = surface_agent();
            let call = |tool: &str, arguments: Value| {
                call_surface(&agent, &surface, token.as_deref(), tool, arguments)
            };
            let used = skills::usage(&call)?;
            let installed = installed_skills(home().as_deref(), &project(dir)?);
            let cutoff = (chrono::Utc::now() - chrono::Duration::days(days)).to_rfc3339();
            println!("{}", skills::render(&installed, &used, Some(&cutoff)));
        }
        ClaudeAction::Reanchor {
            dir,
            limit,
            days,
            dry_run,
            surface,
            token,
        } => {
            let project = project(dir)?;
            let repo = origin_of(&project)
                .and_then(|url| antumbra_core::repo_slug_from_remote(&url))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{} has no origin remote to name its repository by",
                        project.display()
                    )
                })?;
            let mut merges = reanchor::merges_in(&merged_pull_requests(&repo, limit)?)?;
            if let Some(days) = days {
                let cutoff = chrono::Utc::now() - chrono::Duration::days(days);
                merges = reanchor::merged_since(merges, cutoff);
            }
            let agent = surface_agent();
            let call = |tool: &str, arguments: Value| {
                call_surface(&agent, &surface, token.as_deref(), tool, arguments)
            };
            if dry_run {
                println!("dry run: nothing written");
            }
            for line in reanchor::report(&call, &repo, &merges, dry_run)? {
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
