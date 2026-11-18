//! `antumbra hook <agent> <event>`: one agent's hook, answered by the hook
//! scripts Claude Code runs.
//!
//! The three touchpoints (what a session opens with, what is recalled for a
//! prompt, what is kept when a turn ends) are scripts written to Claude
//! Code's contract: its field names on standard input, its JSON on standard
//! output, the server's address in its settings' `env`. Other agents have the
//! same three moments and a contract of their own for each. Writing the
//! scripts again per agent would be five copies to keep alike.
//!
//! So an agent's hook runs this command, which is three small translations
//! around the same scripts:
//!
//! 1. [`heard`]: what the agent handed its hook, in the names the scripts
//!    read.
//! 2. The script, run with the address, workspace and machine name
//!    `antumbra setup` recorded, which Claude Code's settings would have
//!    given it.
//! 3. [`answer`]: what the script said ([`Said`]), in the shape this agent
//!    reads.
//!
//! Like the scripts, it never fails the agent: whatever goes wrong, it says
//! so on standard error, prints nothing, and exits 0.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use clap::{Args, ValueEnum};
use serde_json::{json, Map, Value};

use super::{Agent, Reach};
#[cfg(test)]
use crate::claude::hooks;

/// The moments an agent's hook is run at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Event {
    /// A session opens: what Antumbra remembers that matters here.
    SessionStart,
    /// The user sent a prompt: what is recalled for it.
    Prompt,
    /// A turn ended: how it ended, and a reminder to store what was learned.
    Stop,
}

impl Event {
    /// The name the scripts know the moment by, which is Claude Code's.
    fn name(self) -> &'static str {
        match self {
            Event::SessionStart => "SessionStart",
            Event::Prompt => "UserPromptSubmit",
            Event::Stop => "Stop",
        }
    }

    /// The script that serves it, without its extension.
    fn script(self) -> &'static str {
        match self {
            Event::SessionStart => "antumbra-session-start",
            Event::Prompt => "antumbra-prompt-recall",
            Event::Stop => "antumbra-capture",
        }
    }

    /// As it is typed after the agent's name.
    pub fn word(self) -> &'static str {
        match self {
            Event::SessionStart => "session-start",
            Event::Prompt => "prompt",
            Event::Stop => "stop",
        }
    }
}

/// `antumbra hook <agent> <event>`.
#[derive(Args, Debug, Clone)]
pub struct HookArgs {
    /// The agent whose hook this is.
    #[arg(value_enum)]
    pub agent: Agent,
    /// The moment it was run at.
    #[arg(value_enum)]
    pub event: Event,
}

/// The first of `names` that `input` holds a value under.
fn held<'a>(input: &'a Value, names: &[&str]) -> Option<&'a Value> {
    names
        .iter()
        .find_map(|name| input.get(name))
        .filter(|v| !v.is_null())
}

/// A prompt without the blocks an agent put ahead of it. Gemini CLI, run for
/// one prompt, hands its prompt hook the prompt with the session-start hook's
/// context in front, and a recall on that asks about the context.
fn without_hook_context(prompt: &str) -> &str {
    let mut rest = prompt.trim_start();
    while let Some(after) = rest.strip_prefix("<hook_context>") {
        match after.find("</hook_context>") {
            Some(end) => rest = after[end + "</hook_context>".len()..].trim_start(),
            None => break,
        }
    }
    rest
}

/// What `agent` handed its hook for `event`, in the names the scripts read
/// (Claude Code's): each agent's own name for a field is taken, and the
/// moment is named as the scripts know it.
pub fn heard(agent: Agent, event: Event, input: &Value) -> Value {
    let mut out = Map::new();
    out.insert("hook_event_name".into(), json!(event.name()));
    let fields: [(&str, &[&str]); 7] = [
        ("session_id", &["session_id", "sessionId"]),
        ("cwd", &["cwd", "working_dir", "workingDir"]),
        ("prompt", &["prompt", "message"]),
        (
            "last_assistant_message",
            &[
                "last_assistant_message",
                "prompt_response",
                "lastAssistantMessage",
            ],
        ),
        ("stop_hook_active", &["stop_hook_active", "stopHookActive"]),
        ("transcript_path", &["transcript_path", "transcriptPath"]),
        ("prompt_id", &["prompt_id", "turn_id", "turnId"]),
    ];
    for (name, theirs) in fields {
        if let Some(value) = held(input, theirs) {
            out.insert(name.into(), value.clone());
        }
    }
    if agent == Agent::Gemini {
        if let Some(prompt) = out.get("prompt").and_then(Value::as_str) {
            let own = without_hook_context(prompt).to_string();
            out.insert("prompt".into(), Value::String(own));
        }
    }
    Value::Object(out)
}

/// What a script said, whichever of its ways it said it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Said {
    /// Nothing, which is most turns' end.
    Nothing,
    /// Context for the model: the bootstrap, or what was recalled.
    Context(String),
    /// A reminder to store what was learned before the turn ends.
    Reminder(String),
}

/// Read a script's standard output for `event`. A turn's end says its
/// reminder as context that keeps the turn going or, asked to, as a block;
/// both are the reminder. Anything that is not the scripts' JSON is nothing.
pub fn said(event: Event, stdout: &str) -> Said {
    let Ok(out) = serde_json::from_str::<Value>(stdout.trim()) else {
        return Said::Nothing;
    };
    let text = |v: Option<&Value>| {
        v.and_then(Value::as_str)
            .map(str::to_owned)
            .filter(|t| !t.trim().is_empty())
    };
    if let Some(context) = text(out.pointer("/hookSpecificOutput/additionalContext")) {
        return match event {
            Event::Stop => Said::Reminder(context),
            _ => Said::Context(context),
        };
    }
    if out.get("decision").and_then(Value::as_str) == Some("block") {
        if let Some(reason) = text(out.get("reason")) {
            return Said::Reminder(reason);
        }
    }
    Said::Nothing
}

/// What to print for `agent`: `what` in the shape its hooks are read in, or
/// nothing to print.
pub fn answer(agent: Agent, event: Event, what: &Said) -> Option<String> {
    let out = match (agent, event, what) {
        (_, _, Said::Nothing) => return None,
        // Gemini CLI: context under the event's own name; a turn's end is
        // denied with the reminder, which it sends the agent as a new prompt.
        (Agent::Gemini, Event::SessionStart, Said::Context(text)) => json!({
            "hookSpecificOutput": { "hookEventName": "SessionStart", "additionalContext": text }
        }),
        (Agent::Gemini, Event::Prompt, Said::Context(text)) => json!({
            "hookSpecificOutput": { "hookEventName": "BeforeAgent", "additionalContext": text }
        }),
        (Agent::Gemini, Event::Stop, Said::Reminder(text)) => {
            json!({ "decision": "deny", "reason": text })
        }
        // Codex: Claude Code's own shapes. A turn's end is blocked with the
        // reminder, which Codex sends the agent as a new prompt.
        (Agent::Codex, Event::SessionStart, Said::Context(text)) => json!({
            "hookSpecificOutput": { "hookEventName": "SessionStart", "additionalContext": text }
        }),
        (Agent::Codex, Event::Prompt, Said::Context(text)) => json!({
            "hookSpecificOutput": { "hookEventName": "UserPromptSubmit", "additionalContext": text }
        }),
        (Agent::Codex, Event::Stop, Said::Reminder(text)) => {
            json!({ "decision": "block", "reason": text })
        }
        // A reminder where a turn is not ending, or context where one is,
        // has no place to go.
        (Agent::Gemini | Agent::Codex, _, _) => return None,
    };
    Some(out.to_string())
}

/// How long the script may run before it is ended: longer than the scripts'
/// own budgets, and shorter than the agent's patience with its hook.
const PATIENCE: Duration = Duration::from_secs(20);

/// The most of a hook's input or a script's answer that is read.
const MOST: u64 = 4 * 1024 * 1024;

/// The command that runs a hook script on this platform.
fn interpreter(script: &Path) -> Command {
    if cfg!(windows) {
        let mut command = Command::new("powershell");
        command.args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
        ]);
        command.arg(script);
        command
    } else {
        let mut command = Command::new("bash");
        command.arg(script);
        command
    }
}

/// The script for `event` among the hooks setup wrote under `antumbra`.
fn script_path(antumbra: &Path, event: Event) -> PathBuf {
    let extension = if cfg!(windows) { "ps1" } else { "sh" };
    antumbra
        .join("hooks")
        .join(format!("{}.{extension}", event.script()))
}

/// Run `script` with `input` on its standard input and `env` added to its
/// environment, in `cwd` when that is a directory; its standard output, or
/// nothing when it ran past [`PATIENCE`].
fn run_script(
    script: &Path,
    input: &Value,
    env: &[(String, String)],
    cwd: Option<&Path>,
) -> anyhow::Result<String> {
    let mut command = interpreter(script);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    for (name, value) in env {
        command.env(name, value);
    }
    if let Some(cwd) = cwd.filter(|c| c.is_dir()) {
        command.current_dir(cwd);
    }
    let mut child = command
        .spawn()
        .map_err(|e| anyhow::anyhow!("{} could not be run: {e}", script.display()))?;
    if let Some(mut stdin) = child.stdin.take() {
        // A script that does not read its input closes the pipe; that is its
        // business, not an error.
        let _ = stdin.write_all(input.to_string().as_bytes());
    }
    let stdout = child.stdout.take();
    let reader = std::thread::spawn(move || {
        let mut text = Vec::new();
        if let Some(stdout) = stdout {
            let _ = stdout.take(MOST).read_to_end(&mut text);
        }
        text
    });
    let started = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if started.elapsed() > PATIENCE {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!(
                "{} ran past {} seconds and was ended",
                script.display(),
                PATIENCE.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    let text = reader
        .join()
        .map_err(|_| anyhow::anyhow!("reading {} failed", script.display()))?;
    Ok(String::from_utf8_lossy(&text).into_owned())
}

/// The most characters of context `agent` takes from a hook, where that is
/// less than the scripts allow for Claude Code. Codex moves a hook's context
/// to a file past some 2,500 tokens, and a bootstrap the agent has to go and
/// open is one it may not read.
fn context_limit(agent: Agent) -> Option<u32> {
    match agent {
        Agent::Codex => Some(7_000),
        Agent::Gemini => None,
    }
}

/// What the script is told beyond its input: where the server is, as Claude
/// Code's settings tell it, and that it is not Claude Code it runs for.
fn script_env(reach: &Reach, agent: Agent) -> Vec<(String, String)> {
    let mut env = vec![
        ("ANTUMBRA_URL".to_string(), reach.url.clone()),
        (
            "ANTUMBRA_TOKEN_FILE".to_string(),
            reach.token_file.display().to_string(),
        ),
        // The brief is what Claude Code lost with its telemetry off. It is
        // not about any other agent, and neither are the rules kept from it.
        ("ANTUMBRA_BRIEF".to_string(), "0".to_string()),
        // An agent that is not Claude Code may show a turn's-end reminder to
        // its user, so it is given the short one.
        ("ANTUMBRA_CAPTURE_REMINDER".to_string(), "brief".to_string()),
        ("ANTUMBRA_AGENT".to_string(), agent.name().to_string()),
    ];
    if let Some(limit) = context_limit(agent) {
        env.push(("ANTUMBRA_CONTEXT_LIMIT".to_string(), limit.to_string()));
    }
    if let Some(workspace) = &reach.workspace {
        env.push(("ANTUMBRA_WORKSPACE_ID".to_string(), workspace.clone()));
    }
    if let Some(host) = &reach.host {
        env.push(("ANTUMBRA_HOST_ID".to_string(), host.clone()));
    }
    if let Ok(this) = std::env::current_exe() {
        env.push(("ANTUMBRA_BIN".to_string(), this.display().to_string()));
    }
    env
}

fn serve(args: &HookArgs) -> anyhow::Result<Option<String>> {
    let home = crate::machine::home().ok_or_else(|| {
        anyhow::anyhow!("no home directory (neither USERPROFILE nor HOME is set)")
    })?;
    let reach = Reach::read(&home, |name| std::env::var(name).ok())?;
    let mut raw = Vec::new();
    std::io::stdin().take(MOST).read_to_end(&mut raw)?;
    let input: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
    let heard = heard(args.agent, args.event, &input);
    let script = script_path(&home.join(".antumbra"), args.event);
    if !script.is_file() {
        anyhow::bail!(
            "{} is not there; run antumbra setup again to write it",
            script.display()
        );
    }
    let cwd = heard.get("cwd").and_then(Value::as_str).map(PathBuf::from);
    let stdout = run_script(
        &script,
        &heard,
        &script_env(&reach, args.agent),
        cwd.as_deref(),
    )?;
    Ok(answer(args.agent, args.event, &said(args.event, &stdout)))
}

/// Serve one hook. Never fails the agent that ran it.
pub fn run(args: &HookArgs) {
    match serve(args) {
        Ok(Some(out)) => println!("{out}"),
        Ok(None) => {}
        Err(problem) => eprintln!(
            "antumbra hook {} {}: {problem}",
            args.agent.name(),
            args.event.word()
        ),
    }
}

/// Every script the adapter runs, by file name, for setup to write.
pub fn scripts() -> Vec<String> {
    let extension = if cfg!(windows) { "ps1" } else { "sh" };
    [Event::SessionStart, Event::Prompt, Event::Stop]
        .iter()
        .map(|event| format!("{}.{extension}", event.script()))
        .collect()
}

#[cfg(test)]
mod tests;
