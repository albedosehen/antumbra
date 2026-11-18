use std::path::PathBuf;

use serde_json::{json, Value};

use super::*;
use crate::agents::scratch::Scratch;

/// What Gemini CLI 0.68 hands each hook, as it was recorded from one, with
/// the paths shortened.
fn gemini(event: &str, more: Value) -> Value {
    let mut input = json!({
        "session_id": "c3de1ea0-e4aa-48b2-a988-1e559b411015",
        "transcript_path": "/home/me/.gemini/tmp/work/chats/session-2026-11-18T15-38-c3de1ea0.jsonl",
        "cwd": "/home/me/work",
        "hook_event_name": event,
        "timestamp": "2026-11-18T15:38:02.000Z",
    });
    if let (Some(input), Some(more)) = (input.as_object_mut(), more.as_object()) {
        input.extend(more.clone());
    }
    input
}

#[test]
fn a_hooks_input_is_heard_in_the_names_the_scripts_read() {
    let heard = heard(
        Agent::Gemini,
        Event::Stop,
        &gemini(
            "AfterAgent",
            json!({ "prompt": "Say hello.", "prompt_response": "Hello.", "stop_hook_active": false }),
        ),
    );
    assert_eq!(heard["hook_event_name"], "Stop");
    assert_eq!(heard["session_id"], "c3de1ea0-e4aa-48b2-a988-1e559b411015");
    assert_eq!(heard["cwd"], "/home/me/work");
    assert_eq!(heard["last_assistant_message"], "Hello.");
    assert_eq!(heard["stop_hook_active"], false);
    assert_eq!(
        heard["transcript_path"],
        "/home/me/.gemini/tmp/work/chats/session-2026-11-18T15-38-c3de1ea0.jsonl"
    );
    // Its own name for the moment is not passed on: the scripts know theirs.
    assert!(heard.get("timestamp").is_none());

    let start = super::heard(
        Agent::Gemini,
        Event::SessionStart,
        &gemini("SessionStart", json!({ "source": "startup" })),
    );
    assert_eq!(start["hook_event_name"], "SessionStart");
    // An input that is not an object is heard as the moment alone.
    let none = super::heard(Agent::Gemini, Event::Prompt, &Value::Null);
    assert_eq!(none, json!({ "hook_event_name": "UserPromptSubmit" }));
}

#[test]
fn a_prompt_is_recalled_on_without_the_context_put_ahead_of_it() {
    // Run for one prompt, Gemini CLI hands the prompt hook the session-start
    // hook's context in front of what the user typed.
    let prompt = "<hook_context># Antumbra session bootstrap (host=bench)\n\nBuilds with bun.</hook_context>\n\nSay hello to the sandbox.";
    let heard = heard(
        Agent::Gemini,
        Event::Prompt,
        &gemini("BeforeAgent", json!({ "prompt": prompt })),
    );
    assert_eq!(heard["prompt"], "Say hello to the sandbox.");
    assert_eq!(heard["hook_event_name"], "UserPromptSubmit");

    assert_eq!(without_hook_context("plain"), "plain");
    assert_eq!(
        without_hook_context("<hook_context>a</hook_context><hook_context>b</hook_context> ask"),
        "ask"
    );
    // A block that never closes is the user's own text.
    assert_eq!(
        without_hook_context("<hook_context>never closed"),
        "<hook_context>never closed"
    );
}

#[test]
fn what_a_script_said_is_read_whichever_way_it_said_it() {
    let context = r#"{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"what is remembered"}}"#;
    assert_eq!(
        said(Event::SessionStart, context),
        Said::Context("what is remembered".into())
    );
    assert_eq!(
        said(Event::Prompt, &format!("\n{context}\r\n")),
        Said::Context("what is remembered".into())
    );
    // A turn's end says its reminder as context that keeps the turn going,
    // or as a block when asked to. Both are the reminder.
    let kept_going = r#"{"hookSpecificOutput":{"hookEventName":"Stop","additionalContext":"store what was learned"}}"#;
    assert_eq!(
        said(Event::Stop, kept_going),
        Said::Reminder("store what was learned".into())
    );
    let blocked = r#"{"decision":"block","reason":"store what was learned"}"#;
    assert_eq!(
        said(Event::Stop, blocked),
        Said::Reminder("store what was learned".into())
    );
    for nothing in ["", "  \n", "not json", "{}", r#"{"decision":"block"}"#] {
        assert_eq!(said(Event::Stop, nothing), Said::Nothing, "{nothing:?}");
    }
    let empty = r#"{"hookSpecificOutput":{"additionalContext":"  "}}"#;
    assert_eq!(said(Event::Prompt, empty), Said::Nothing);
}

#[test]
fn gemini_is_answered_in_its_own_shape() -> anyhow::Result<()> {
    let parsed = |text: Option<String>| -> anyhow::Result<Value> {
        let text = text.ok_or_else(|| anyhow::anyhow!("nothing was answered"))?;
        Ok(serde_json::from_str(&text)?)
    };
    let context = Said::Context("what is remembered".into());
    assert_eq!(
        parsed(answer(Agent::Gemini, Event::SessionStart, &context))?,
        json!({ "hookSpecificOutput": { "hookEventName": "SessionStart", "additionalContext": "what is remembered" } })
    );
    assert_eq!(
        parsed(answer(Agent::Gemini, Event::Prompt, &context))?,
        json!({ "hookSpecificOutput": { "hookEventName": "BeforeAgent", "additionalContext": "what is remembered" } })
    );
    // A turn's end is denied with the reminder, which Gemini CLI sends the
    // agent as a new prompt.
    assert_eq!(
        parsed(answer(
            Agent::Gemini,
            Event::Stop,
            &Said::Reminder("store what was learned".into())
        ))?,
        json!({ "decision": "deny", "reason": "store what was learned" })
    );
    assert_eq!(answer(Agent::Gemini, Event::Stop, &Said::Nothing), None);
    // Context has no place at a turn's end, nor a reminder at its start.
    assert_eq!(answer(Agent::Gemini, Event::Stop, &context), None);
    assert_eq!(
        answer(
            Agent::Gemini,
            Event::SessionStart,
            &Said::Reminder("x".into())
        ),
        None
    );
    Ok(())
}

#[test]
fn every_script_the_adapter_runs_is_bundled() {
    let names = scripts();
    assert_eq!(names.len(), 3);
    for name in names {
        assert!(hooks::bundled(&name).is_some(), "{name} is not bundled");
    }
}

/// A script for this platform's shell that answers as the session-start hook
/// does, with what it was told and handed in its context.
fn echo_script(dir: &std::path::Path) -> std::io::Result<PathBuf> {
    if cfg!(windows) {
        let path = dir.join("echo.ps1");
        std::fs::write(
            &path,
            concat!(
                "$in = [Console]::In.ReadToEnd()\n",
                "$ctx = \"url=$($env:ANTUMBRA_URL) brief=$($env:ANTUMBRA_BRIEF) agent=$($env:ANTUMBRA_AGENT) host=$($env:ANTUMBRA_HOST_ID) cwd=$((Get-Location).Path) in=\" + ($in -replace '[\"{}]', '')\n",
                "@{ hookSpecificOutput = @{ hookEventName = 'SessionStart'; additionalContext = $ctx } } | ConvertTo-Json -Compress\n",
            ),
        )?;
        Ok(path)
    } else {
        let path = dir.join("echo.sh");
        std::fs::write(
            &path,
            concat!(
                "input=$(cat)\n",
                "printf '{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\"url=%s brief=%s agent=%s host=%s cwd=%s in=%s\"}}' \\\n",
                "  \"$ANTUMBRA_URL\" \"$ANTUMBRA_BRIEF\" \"$ANTUMBRA_AGENT\" \"$ANTUMBRA_HOST_ID\" \"$(pwd)\" \"$(printf '%s' \"$input\" | tr -d '\"{}')\"\n",
            ),
        )?;
        Ok(path)
    }
}

#[test]
fn the_script_is_run_where_the_session_is_with_what_setup_recorded() -> anyhow::Result<()> {
    let scratch = Scratch::new("hook-run")?;
    let script = echo_script(scratch.path())?;
    let session = scratch.path().join("project");
    std::fs::create_dir_all(&session)?;
    let reach = Reach {
        url: "https://memory.example".into(),
        workspace: Some("ws:team".into()),
        token_file: scratch.path().join("token.txt"),
        token_given: None,
        host: Some("bench".into()),
    };
    let env = script_env(&reach, Agent::Gemini);
    assert!(env.contains(&("ANTUMBRA_WORKSPACE_ID".to_string(), "ws:team".to_string())));
    // The token's file is named; the token is never put in the environment.
    assert!(env.iter().all(|(name, _)| name != "ANTUMBRA_TOKEN"));

    let heard = json!({ "hook_event_name": "SessionStart", "session_id": "s-1" });
    let stdout = run_script(&script, &heard, &env, Some(&session))?;
    let Said::Context(context) = said(Event::SessionStart, &stdout) else {
        anyhow::bail!("the script's answer was not read as context: {stdout:?}");
    };
    assert!(context.contains("url=https://memory.example"), "{context}");
    // The brief is about Claude Code, and is off for every other agent.
    assert!(context.contains("brief=0"), "{context}");
    assert!(context.contains("agent=gemini"), "{context}");
    assert!(context.contains("host=bench"), "{context}");
    assert!(context.contains("session_id:s-1"), "{context}");
    let folder = session
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    assert!(context.contains(&format!("{folder} in=")), "{context}");

    // A directory that is not there is not a reason to fail the hook.
    let gone = scratch.path().join("gone");
    let stdout = run_script(&script, &heard, &env, Some(&gone))?;
    assert!(matches!(
        said(Event::SessionStart, &stdout),
        Said::Context(_)
    ));
    Ok(())
}

#[test]
fn a_script_that_is_not_there_is_an_error_and_not_a_panic() -> anyhow::Result<()> {
    let scratch = Scratch::new("hook-missing")?;
    let path = script_path(scratch.path(), Event::Prompt);
    assert!(path
        .display()
        .to_string()
        .contains("antumbra-prompt-recall."));
    assert!(!path.exists());
    Ok(())
}
