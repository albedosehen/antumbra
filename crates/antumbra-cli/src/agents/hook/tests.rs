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

/// What Codex 0.170 hands its `Stop` hook, as it was recorded from one.
fn codex_stop() -> Value {
    json!({
        "session_id": "01a1ef66-b699-7f61-9cbf-d6dc80999ae6",
        "turn_id": "01a1ef66-b703-77c3-80e2-a8131f700837",
        "transcript_path": null,
        "cwd": "/home/me/work",
        "hook_event_name": "Stop",
        "model": "gpt-5.5",
        "permission_mode": "default",
        "stop_hook_active": false,
        "last_assistant_message": "It builds with bun.",
    })
}

#[test]
fn codex_is_heard_and_answered_as_claude_code_is() -> anyhow::Result<()> {
    let heard = heard(Agent::Codex, Event::Stop, &codex_stop());
    assert_eq!(heard["hook_event_name"], "Stop");
    assert_eq!(heard["last_assistant_message"], "It builds with bun.");
    // Its turn is the id the server judges a turn's recalls by.
    assert_eq!(heard["prompt_id"], "01a1ef66-b703-77c3-80e2-a8131f700837");
    // A transcript it does not have is not handed on as one.
    assert!(heard.get("transcript_path").is_none());

    let parsed = |text: Option<String>| -> anyhow::Result<Value> {
        let text = text.ok_or_else(|| anyhow::anyhow!("nothing was answered"))?;
        Ok(serde_json::from_str(&text)?)
    };
    let context = Said::Context("what is remembered".into());
    assert_eq!(
        parsed(answer(Agent::Codex, Event::SessionStart, &context))?,
        json!({ "hookSpecificOutput": { "hookEventName": "SessionStart", "additionalContext": "what is remembered" } })
    );
    assert_eq!(
        parsed(answer(Agent::Codex, Event::Prompt, &context))?,
        json!({ "hookSpecificOutput": { "hookEventName": "UserPromptSubmit", "additionalContext": "what is remembered" } })
    );
    assert_eq!(
        parsed(answer(
            Agent::Codex,
            Event::Stop,
            &Said::Reminder("store what was learned".into())
        ))?,
        json!({ "decision": "block", "reason": "store what was learned" })
    );
    assert_eq!(answer(Agent::Codex, Event::Stop, &context), None);
    Ok(())
}

#[test]
fn codex_is_given_a_smaller_bootstrap_and_gemini_the_whole_one() {
    let reach = Reach {
        url: "https://memory.example".into(),
        workspace: None,
        token_file: PathBuf::from("token.txt"),
        token_given: None,
        host: None,
    };
    let limit = |agent| {
        script_env(&reach, agent)
            .into_iter()
            .find(|(name, _)| name == "ANTUMBRA_CONTEXT_LIMIT")
            .map(|(_, value)| value)
    };
    assert_eq!(limit(Agent::Codex).as_deref(), Some("7000"));
    // goose is handed the bootstrap again with every turn.
    assert_eq!(limit(Agent::Goose).as_deref(), Some("4000"));
    assert_eq!(limit(Agent::Gemini), None);
}

/// Lines of a Copilot CLI 1.0 session's event log, as one was recorded, cut
/// to what is read.
const COPILOT_EVENTS: &str = concat!(
    r#"{"type":"user.message","data":{"content":"Say hello."},"id":"1","timestamp":"t","parentId":null}"#,
    "\n",
    r#"{"type":"assistant.message","data":{"messageId":"m1","model":"gpt","content":"First answer.","toolRequests":[]},"id":"2","timestamp":"t","parentId":"1"}"#,
    "\n",
    r#"{"type":"assistant.turn_end","data":{"turnId":"0"},"id":"3","timestamp":"t","parentId":"2"}"#,
    "\n",
    r#"{"type":"assistant.message","data":{"messageId":"m2","model":"gpt","content":"It builds with bun.","toolRequests":[]},"id":"4","timestamp":"t","parentId":"3"}"#,
    "\n",
    r#"{"type":"hook.start","data":{"hookType":"agentStop","input":{"sessionId":"s"}},"id":"5","timestamp":"t","parentId":"4"}"#,
    "\n",
);

#[test]
fn copilots_last_answer_is_read_from_its_event_log() {
    assert_eq!(
        last_in_events(COPILOT_EVENTS).as_deref(),
        Some("It builds with bun.")
    );
    // A first line cut in two, as the end of a long log starts with, is
    // passed over.
    let cut = format!("ssage\",\"data\":{{\"content\":\"lost\"}}}}\n{COPILOT_EVENTS}");
    assert_eq!(last_in_events(&cut).as_deref(), Some("It builds with bun."));
    // A log in a shape this does not know says nothing, and that is all.
    assert_eq!(last_in_events(""), None);
    assert_eq!(
        last_in_events("{\"type\":\"assistant.message\",\"data\":{}}\n"),
        None
    );
    assert_eq!(last_in_events("not json\n"), None);
}

#[test]
fn copilot_is_heard_in_its_own_names_and_answered_in_its_own_shape() -> anyhow::Result<()> {
    // What Copilot CLI 1.0 hands `agentStop`, as it was recorded from one.
    let stop = json!({
        "sessionId": "5b8043e9-514d-4193-af43-dc98b702296f",
        "timestamp": 1_795_018_354_785_u64,
        "cwd": "/home/me/work",
        "transcriptPath": "/home/me/.copilot/session-state/5b8043e9/events.jsonl",
        "stopReason": "end_turn",
        "stop_hook_active": false,
    });
    let heard = heard(Agent::Copilot, Event::Stop, &stop);
    assert_eq!(heard["session_id"], "5b8043e9-514d-4193-af43-dc98b702296f");
    assert_eq!(
        heard["transcript_path"],
        "/home/me/.copilot/session-state/5b8043e9/events.jsonl"
    );
    assert_eq!(heard["stop_hook_active"], false);
    // It gives the log and not the answer; the answer is read from the log.
    assert!(heard.get("last_assistant_message").is_none());

    let parsed = |text: Option<String>| -> anyhow::Result<Value> {
        let text = text.ok_or_else(|| anyhow::anyhow!("nothing was answered"))?;
        Ok(serde_json::from_str(&text)?)
    };
    let context = Said::Context("what is remembered".into());
    for event in [Event::SessionStart, Event::Prompt] {
        assert_eq!(
            parsed(answer(Agent::Copilot, event, &context))?,
            json!({ "additionalContext": "what is remembered" })
        );
    }
    assert_eq!(
        parsed(answer(
            Agent::Copilot,
            Event::Stop,
            &Said::Reminder("store what was learned".into())
        ))?,
        json!({ "decision": "block", "reason": "store what was learned" })
    );
    assert_eq!(answer(Agent::Copilot, Event::Stop, &context), None);
    Ok(())
}

#[test]
fn the_end_of_a_long_log_is_all_that_is_read() -> anyhow::Result<()> {
    let scratch = Scratch::new("hook-tail")?;
    let path = scratch.path().join("events.jsonl");
    let filler = format!(
        "{}\n",
        r#"{"type":"tool.execution_complete","data":{"output":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}}"#
    );
    let mut text = String::from(
        "{\"type\":\"assistant.message\",\"data\":{\"content\":\"an early answer\"}}\n",
    );
    while (text.len() as u64) < TAIL + 4096 {
        text.push_str(&filler);
    }
    text.push_str("{\"type\":\"assistant.message\",\"data\":{\"content\":\"the last answer\"}}\n");
    std::fs::write(&path, &text)?;
    let read = tail(&path).ok_or_else(|| anyhow::anyhow!("the log was not read"))?;
    assert!(read.len() as u64 <= TAIL);
    assert!(!read.contains("an early answer"));
    assert_eq!(last_in_events(&read).as_deref(), Some("the last answer"));
    // No log is no answer, not an error.
    assert_eq!(tail(&scratch.path().join("gone.jsonl")), None);
    Ok(())
}

#[test]
fn goose_is_heard_in_its_own_names_and_reads_only_a_turns_end() -> anyhow::Result<()> {
    // What goose 1.60 hands its hooks, as it was recorded from one.
    let prompt = json!({ "event": "UserPromptSubmit", "session_id": "20261118_1",
        "matcher_context": "Say hello.", "message": "Say hello.",
        "working_dir": "/home/me/work" });
    let heard = heard(Agent::Goose, Event::Prompt, &prompt);
    assert_eq!(heard["hook_event_name"], "UserPromptSubmit");
    assert_eq!(heard["prompt"], "Say hello.");
    assert_eq!(heard["cwd"], "/home/me/work");
    assert_eq!(heard["session_id"], "20261118_1");
    let stop = json!({ "event": "Stop", "session_id": "20261118_1", "matcher_context": null,
        "last_assistant_message": "It builds with bun.", "working_dir": "/home/me/work" });
    let heard = super::heard(Agent::Goose, Event::Stop, &stop);
    assert_eq!(heard["last_assistant_message"], "It builds with bun.");

    // goose drops what a session-start or prompt hook prints: nothing is
    // printed for them, and the context goes to the file it keeps in mind.
    let context = Said::Context("what is remembered".into());
    assert_eq!(answer(Agent::Goose, Event::SessionStart, &context), None);
    assert_eq!(answer(Agent::Goose, Event::Prompt, &context), None);
    let blocked = answer(
        Agent::Goose,
        Event::Stop,
        &Said::Reminder("store what was learned".into()),
    )
    .ok_or_else(|| anyhow::anyhow!("a turn's end was not answered"))?;
    assert_eq!(
        serde_json::from_str::<Value>(&blocked)?,
        json!({ "decision": "block", "reason": "store what was learned" })
    );
    Ok(())
}

#[test]
fn opencode_is_answered_through_its_plugin() -> anyhow::Result<()> {
    // The plugin hands the hook what it knows in the scripts' own names.
    let heard = heard(
        Agent::Opencode,
        Event::Stop,
        &json!({ "session_id": "ses_1", "cwd": "/home/me/work", "last_assistant_message": "Done." }),
    );
    assert_eq!(heard["session_id"], "ses_1");
    assert_eq!(heard["last_assistant_message"], "Done.");

    let parsed = |text: Option<String>| -> anyhow::Result<Value> {
        let text = text.ok_or_else(|| anyhow::anyhow!("nothing was answered"))?;
        Ok(serde_json::from_str(&text)?)
    };
    let context = Said::Context("what is remembered".into());
    for event in [Event::SessionStart, Event::Prompt] {
        assert_eq!(
            parsed(answer(Agent::Opencode, event, &context))?,
            json!({ "context": "what is remembered" })
        );
    }
    assert_eq!(
        parsed(answer(
            Agent::Opencode,
            Event::Stop,
            &Said::Reminder("store what was learned".into())
        ))?,
        json!({ "reminder": "store what was learned" })
    );
    assert_eq!(answer(Agent::Opencode, Event::Stop, &Said::Nothing), None);
    Ok(())
}
