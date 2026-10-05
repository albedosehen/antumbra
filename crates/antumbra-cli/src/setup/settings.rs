//! Wiring Antumbra into Claude Code's `settings.json`, as text.
//!
//! The same rule as `antumbra claude apply`: the file is the user's, so it is
//! never read into `serde_json` and written back (that would re-sort every
//! key and lose the formatting). The text is scanned for the byte spans that
//! matter and edited by insertion; every other byte stays where it was. What
//! setup adds, and the limits on it:
//!
//! - **Hooks: adding only.** An event that already runs a script of the same
//!   name, wherever it lives, is left alone, so a hand-wired setup is never
//!   doubled. Otherwise the hook goes in first in that event's list, ahead of
//!   the user's own.
//! - **`env`: only the `ANTUMBRA_*` names setup owns**, inserted, or replaced
//!   when they are already there (a re-run that moves from a local server to
//!   a hosted one has to change `ANTUMBRA_URL`). A name that is the user's
//!   choice rather than the server's (`ANTUMBRA_HOST_ID`, stamped on what this
//!   machine writes) is only added when absent. No other name is touched.
//! - **`autoMemoryEnabled`: added when absent, never changed.** A user who set
//!   it either way has answered already.
//!
//! [`wire`] re-reads its own result and checks it against the original: the
//! same top-level keys with the same values (but the three above), every hook
//! the user had still there in order, every env name it did not own
//! unchanged. Anything else is an error, and nothing is written.

use serde_json::Value;

/// One hook to run on one event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hook {
    /// The Claude Code event, `SessionStart`, `UserPromptSubmit`, ...
    pub event: &'static str,
    /// The script's file name, for finding one already wired.
    pub script: &'static str,
    /// The command Claude Code runs.
    pub command: String,
    /// Seconds before Claude Code gives up on it.
    pub timeout: u32,
}

/// What setup wants the settings to hold.
#[derive(Debug, Clone, Default)]
pub struct Wiring {
    pub hooks: Vec<Hook>,
    /// `ANTUMBRA_*` names and values, set whatever they held.
    pub env: Vec<(String, String)>,
    /// `ANTUMBRA_*` names and values, added only when the name is absent.
    pub env_default: Vec<(String, String)>,
    /// Turn Claude Code's own file memory off, when the file does not say.
    pub auto_memory_off: bool,
}

/// What [`wire`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Wired {
    /// The new text of the file.
    pub text: String,
    /// Events a hook was added to.
    pub added: Vec<&'static str>,
    /// Events that already ran a script of that name: (event, the command).
    pub already: Vec<(&'static str, String)>,
    /// Env names written (added or changed).
    pub env_set: Vec<String>,
    /// `autoMemoryEnabled` was added as `false`.
    pub auto_memory_set: bool,
    /// `autoMemoryEnabled` was already in the file, with this value.
    pub auto_memory_was: Option<bool>,
}

impl Wired {
    /// Whether the file changes at all.
    pub fn changed(&self) -> bool {
        !self.added.is_empty() || !self.env_set.is_empty() || self.auto_memory_set
    }
}

/// One member of an object: its key, where the key starts, and the value's
/// span (end exclusive).
struct Member {
    key: String,
    at: usize,
    start: usize,
    end: usize,
}

/// An object's direct members, and the index of its `}`. `open` is the index
/// of its `{`.
fn members(text: &str, open: usize) -> anyhow::Result<(Vec<Member>, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = open + 1;
    loop {
        i = skip_ws(bytes, i);
        match bytes.get(i) {
            Some(b'}') => return Ok((out, i)),
            Some(b',') => {
                i += 1;
                continue;
            }
            Some(b'"') => {}
            _ => anyhow::bail!("unexpected text in an object at byte {i}"),
        }
        let at = i;
        let key_end = string_end(bytes, i)?;
        let key: String = serde_json::from_str(&text[i..key_end])?;
        i = skip_ws(bytes, key_end);
        if bytes.get(i) != Some(&b':') {
            anyhow::bail!("a key without a value at byte {i}");
        }
        let start = skip_ws(bytes, i + 1);
        let end = value_end(bytes, start)?;
        out.push(Member {
            key,
            at,
            start,
            end,
        });
        i = end;
    }
}

/// The spans of an array's direct elements, and the index of its `]`.
fn elements(text: &str, open: usize) -> anyhow::Result<(Vec<(usize, usize)>, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = open + 1;
    loop {
        i = skip_ws(bytes, i);
        match bytes.get(i) {
            Some(b']') => return Ok((out, i)),
            Some(b',') => {
                i += 1;
                continue;
            }
            Some(_) => {}
            None => anyhow::bail!("an array that never closes"),
        }
        let end = value_end(bytes, i)?;
        out.push((i, end));
        i = end;
    }
}

fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while matches!(bytes.get(i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        i += 1;
    }
    i
}

/// The index just past the string that opens at `i`.
fn string_end(bytes: &[u8], i: usize) -> anyhow::Result<usize> {
    let mut j = i + 1;
    while let Some(&b) = bytes.get(j) {
        match b {
            b'\\' => j += 2,
            b'"' => return Ok(j + 1),
            _ => j += 1,
        }
    }
    anyhow::bail!("a string that never closes")
}

/// The index just past the value that starts at `i`.
fn value_end(bytes: &[u8], i: usize) -> anyhow::Result<usize> {
    match bytes.get(i) {
        Some(b'"') => string_end(bytes, i),
        Some(b'{' | b'[') => {
            let mut depth = 0usize;
            let mut j = i;
            while let Some(&b) = bytes.get(j) {
                match b {
                    b'"' => {
                        j = string_end(bytes, j)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            anyhow::bail!("a value that never closes")
        }
        Some(_) => {
            let mut j = i;
            while let Some(&b) = bytes.get(j) {
                if matches!(b, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    break;
                }
                j += 1;
            }
            Ok(j)
        }
        None => anyhow::bail!("a value is missing at the end"),
    }
}

/// The whitespace the line holding byte `at` starts with.
fn indent_at(text: &str, at: usize) -> String {
    let start = text[..at].rfind('\n').map_or(0, |n| n + 1);
    let line = &text[start..];
    line[..line.len() - line.trim_start().len()].to_string()
}

/// The indentation a container's entries use: the first entry's line, or one
/// step (two spaces) past the container's own line when it is empty or all on
/// one line.
fn entry_indent(text: &str, open: usize, first: Option<usize>) -> String {
    match first {
        Some(at) if text[open..at].contains('\n') => indent_at(text, at),
        _ => format!("{}  ", indent_at(text, open)),
    }
}

/// Insert `entry` (already formatted) as the first entry of the container
/// whose opening bracket is at `open` and closing one at `close`, with
/// `first` the start of its current first entry.
fn insert_first(
    text: &str,
    open: usize,
    close: usize,
    first: Option<usize>,
    entry: &str,
) -> String {
    let indent = entry_indent(text, open, first);
    let mut out = String::with_capacity(text.len() + entry.len() + 16);
    out.push_str(&text[..=open]);
    match first {
        Some(at) => {
            out.push('\n');
            out.push_str(&indent);
            out.push_str(entry);
            out.push(',');
            if text[open + 1..at].contains('\n') {
                out.push_str(&text[open + 1..]);
            } else {
                out.push('\n');
                out.push_str(&indent);
                out.push_str(&text[at..]);
            }
        }
        None => {
            out.push('\n');
            out.push_str(&indent);
            out.push_str(entry);
            out.push('\n');
            out.push_str(&indent_at(text, open));
            out.push_str(&text[close..]);
        }
    }
    out
}

/// The root object's `{`.
fn root_open(text: &str) -> anyhow::Result<usize> {
    let at = skip_ws(text.as_bytes(), 0);
    match text.as_bytes().get(at) {
        Some(b'{') => Ok(at),
        _ => anyhow::bail!("settings.json does not hold a JSON object"),
    }
}

fn find<'a>(found: &'a [Member], key: &str) -> Option<&'a Member> {
    found.iter().find(|m| m.key == key)
}

/// Where an object's first member starts, the place an insertion goes.
fn first_at(found: &[Member]) -> Option<usize> {
    found.first().map(|m| m.at)
}

/// The scripts an event's hooks already run, as (file name, command).
fn wired_scripts(event: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for matcher in event.as_array().into_iter().flatten() {
        for hook in matcher
            .get("hooks")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(command) = hook.get("command").and_then(Value::as_str) else {
                continue;
            };
            for word in command.split(|c: char| c.is_whitespace() || c == '"' || c == '\'') {
                let name = word.rsplit(['/', '\\']).next().unwrap_or(word);
                if name.ends_with(".ps1") || name.ends_with(".sh") {
                    out.push((name.to_string(), command.to_string()));
                }
            }
        }
    }
    out
}

/// The stem a script is known by, whichever shell it is for:
/// `antumbra-capture.ps1` and `antumbra-capture.sh` are the same hook.
fn stem(name: &str) -> &str {
    name.trim_end_matches(".ps1").trim_end_matches(".sh")
}

/// `text` (a settings file, or empty for none) with `wiring` applied.
pub fn wire(text: &str, wiring: &Wiring) -> anyhow::Result<Wired> {
    let original_text = if text.trim().is_empty() { "{}\n" } else { text };
    let original: Value = serde_json::from_str(original_text).map_err(|e| {
        anyhow::anyhow!(
            "settings.json is not valid JSON ({e}); fix it or move it aside, then run setup again"
        )
    })?;
    if !original.is_object() {
        anyhow::bail!("settings.json does not hold a JSON object");
    }
    let mut text = original_text.to_string();
    let mut wired = Wired::default();

    // autoMemoryEnabled, added when absent.
    match original.get("autoMemoryEnabled") {
        Some(v) => wired.auto_memory_was = v.as_bool(),
        None if wiring.auto_memory_off => {
            let open = root_open(&text)?;
            let (found, close) = members(&text, open)?;
            let first = first_at(&found);
            text = insert_first(&text, open, close, first, "\"autoMemoryEnabled\": false");
            wired.auto_memory_set = true;
        }
        None => {}
    }

    // env: the names setup owns, set; the defaults, added when absent.
    let defaults = wiring
        .env_default
        .iter()
        .filter(|(name, _)| original.get("env").and_then(|e| e.get(name)).is_none());
    // Each insertion goes first, so they go in backwards to read in order.
    let env: Vec<&(String, String)> = wiring.env.iter().chain(defaults).collect();
    for (name, value) in env.into_iter().rev() {
        let open = root_open(&text)?;
        let (found, close) = members(&text, open)?;
        let literal = Value::String(value.clone()).to_string();
        match find(&found, "env") {
            Some(env) if text.as_bytes()[env.start] == b'{' => {
                let env_open = env.start;
                let (vars, env_close) = members(&text, env_open)?;
                match find(&vars, name) {
                    Some(var) => {
                        if text[var.start..var.end] != literal {
                            text = format!("{}{literal}{}", &text[..var.start], &text[var.end..]);
                            wired.env_set.push(name.clone());
                        }
                    }
                    None => {
                        let first = first_at(&vars);
                        let entry = format!("{}: {literal}", Value::String(name.clone()));
                        text = insert_first(&text, env_open, env_close, first, &entry);
                        wired.env_set.push(name.clone());
                    }
                }
            }
            Some(_) => anyhow::bail!("settings.json has an \"env\" that is not an object"),
            None => {
                let first = first_at(&found);
                let entry = format!(
                    "\"env\": {{\n{}  {}: {literal}\n{}}}",
                    entry_indent(&text, open, first),
                    Value::String(name.clone()),
                    entry_indent(&text, open, first)
                );
                text = insert_first(&text, open, close, first, &entry);
                wired.env_set.push(name.clone());
            }
        }
    }

    // Hooks, adding only, backwards for the same reason.
    for hook in wiring.hooks.iter().rev() {
        let current: Value = serde_json::from_str(&text)?;
        let event_value = current.get("hooks").and_then(|h| h.get(hook.event));
        if let Some(event_value) = event_value {
            if let Some((_, command)) = wired_scripts(event_value)
                .into_iter()
                .find(|(name, _)| stem(name) == stem(hook.script))
            {
                wired.already.push((hook.event, command));
                continue;
            }
        }
        let open = root_open(&text)?;
        let (found, close) = members(&text, open)?;
        match find(&found, "hooks") {
            Some(hooks) if text.as_bytes()[hooks.start] == b'{' => {
                let hooks_open = hooks.start;
                let (events, hooks_close) = members(&text, hooks_open)?;
                match find(&events, hook.event) {
                    Some(list) if text.as_bytes()[list.start] == b'[' => {
                        let arr_open = list.start;
                        let (items, arr_close) = elements(&text, arr_open)?;
                        let first = items.first().map(|(s, _)| *s);
                        let indent = entry_indent(&text, arr_open, first);
                        text = insert_first(
                            &text,
                            arr_open,
                            arr_close,
                            first,
                            &matcher_text(hook, &indent),
                        );
                    }
                    Some(_) => anyhow::bail!("settings.json's hooks.{} is not a list", hook.event),
                    None => {
                        let first = first_at(&events);
                        let indent = entry_indent(&text, hooks_open, first);
                        let entry = event_text(hook, &indent);
                        text = insert_first(&text, hooks_open, hooks_close, first, &entry);
                    }
                }
            }
            Some(_) => anyhow::bail!("settings.json has a \"hooks\" that is not an object"),
            None => {
                let first = first_at(&found);
                let indent = entry_indent(&text, open, first);
                let entry = format!(
                    "\"hooks\": {{\n{indent}  {}\n{indent}}}",
                    event_text(hook, &format!("{indent}  "))
                );
                text = insert_first(&text, open, close, first, &entry);
            }
        }
        wired.added.push(hook.event);
    }

    check(&original, &text, wiring)?;
    wired.added.reverse();
    wired.already.reverse();
    wired.env_set.reverse();
    wired.text = text;
    Ok(wired)
}

/// One hook's matcher, written the way people write it by hand: `type`,
/// `command`, `timeout`, in that order. `indent` is the line it starts on.
fn matcher_text(hook: &Hook, indent: &str) -> String {
    format!(
        "{{\n{indent}  \"hooks\": [\n{indent}    {{ \"type\": \"command\", \"command\": {}, \"timeout\": {} }}\n{indent}  ]\n{indent}}}",
        Value::String(hook.command.clone()),
        hook.timeout
    )
}

/// An event's member, `\"Event\": [ matcher ]`, at `indent`.
fn event_text(hook: &Hook, indent: &str) -> String {
    format!(
        "{}: [\n{indent}  {}\n{indent}]",
        Value::String(hook.event.to_string()),
        matcher_text(hook, &format!("{indent}  "))
    )
}

/// The edited text holds everything the original did, and only adds what
/// `wiring` asked for.
fn check(original: &Value, text: &str, wiring: &Wiring) -> anyhow::Result<()> {
    let edited: Value = serde_json::from_str(text).map_err(|e| {
        anyhow::anyhow!("the edit did not produce valid JSON ({e}); nothing was written")
    })?;
    let (Some(before), Some(after)) = (original.as_object(), edited.as_object()) else {
        anyhow::bail!("the edit lost the settings object; nothing was written");
    };
    for (key, value) in before {
        if matches!(key.as_str(), "hooks" | "env") {
            continue;
        }
        if after.get(key) != Some(value) {
            anyhow::bail!("the edit changed \"{key}\"; nothing was written");
        }
    }
    let owned: Vec<&str> = wiring
        .env
        .iter()
        .chain(&wiring.env_default)
        .map(|(n, _)| n.as_str())
        .collect();
    if let Some(env) = before.get("env").and_then(Value::as_object) {
        for (name, value) in env {
            if owned.contains(&name.as_str()) {
                continue;
            }
            if after.get("env").and_then(|e| e.get(name)) != Some(value) {
                anyhow::bail!("the edit changed env.{name}; nothing was written");
            }
        }
    }
    if let Some(hooks) = before.get("hooks").and_then(Value::as_object) {
        for (event, list) in hooks {
            let had = list.as_array().cloned().unwrap_or_default();
            let has = after
                .get("hooks")
                .and_then(|h| h.get(event))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if has.len() < had.len() || has[has.len() - had.len()..] != had[..] {
                anyhow::bail!(
                    "the edit changed the {event} hooks already there; nothing was written"
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wiring() -> Wiring {
        Wiring {
            hooks: vec![
                Hook {
                    event: "SessionStart",
                    script: "antumbra-session-start.sh",
                    command: "bash \"/home/me/.antumbra/hooks/antumbra-session-start.sh\"".into(),
                    timeout: 10,
                },
                Hook {
                    event: "Stop",
                    script: "antumbra-capture.sh",
                    command: "bash \"/home/me/.antumbra/hooks/antumbra-capture.sh\"".into(),
                    timeout: 5,
                },
            ],
            env: vec![
                ("ANTUMBRA_URL".into(), "http://127.0.0.1:8081".into()),
                ("ANTUMBRA_WORKSPACE_ID".into(), "ws:default".into()),
            ],
            env_default: vec![("ANTUMBRA_HOST_ID".into(), "laptop".into())],
            auto_memory_off: true,
        }
    }

    fn parsed(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn an_empty_file_gets_everything() {
        let wired = wire("", &wiring()).unwrap();
        let v = parsed(&wired.text);
        assert_eq!(v["autoMemoryEnabled"], false);
        assert_eq!(v["env"]["ANTUMBRA_URL"], "http://127.0.0.1:8081");
        assert_eq!(
            v["hooks"]["SessionStart"][0]["hooks"][0]["command"],
            "bash \"/home/me/.antumbra/hooks/antumbra-session-start.sh\""
        );
        assert_eq!(v["hooks"]["Stop"][0]["hooks"][0]["timeout"], 5);
        assert_eq!(wired.added, vec!["SessionStart", "Stop"]);
        assert!(wired.auto_memory_set);
        assert!(wired.changed());
    }

    #[test]
    fn the_users_own_keys_order_and_formatting_survive() {
        let text = r#"{
    "theme": "dark",
    "permissions": {
        "allow": ["Bash(git status)"]
    },
    "env": {
        "MY_VAR": "1",
        "ANTUMBRA_URL": "http://old:8081"
    },
    "hooks": {
        "Stop": [
            {
                "hooks": [{ "type": "command", "command": "echo mine" }]
            }
        ]
    },
    "model": "opus"
}
"#;
        let wired = wire(text, &wiring()).unwrap();
        let v = parsed(&wired.text);
        // Theirs, untouched and in place.
        assert!(wired.text.contains("    \"theme\": \"dark\",\n"));
        assert!(wired
            .text
            .contains("        \"allow\": [\"Bash(git status)\"]"));
        assert!(wired.text.trim_end().ends_with("\"model\": \"opus\"\n}"));
        assert_eq!(v["env"]["MY_VAR"], "1");
        // Ours, set: a stale URL is replaced.
        assert_eq!(v["env"]["ANTUMBRA_URL"], "http://127.0.0.1:8081");
        assert_eq!(v["env"]["ANTUMBRA_WORKSPACE_ID"], "ws:default");
        // Our Stop hook goes in ahead of theirs; theirs stays.
        assert_eq!(v["hooks"]["Stop"].as_array().unwrap().len(), 2);
        assert_eq!(v["hooks"]["Stop"][1]["hooks"][0]["command"], "echo mine");
        // A new event inside their hooks object, indented like its neighbors.
        assert!(wired.text.contains("        \"SessionStart\": ["));
        assert_eq!(
            wired.env_set,
            vec!["ANTUMBRA_URL", "ANTUMBRA_WORKSPACE_ID", "ANTUMBRA_HOST_ID"]
        );
    }

    #[test]
    fn a_host_id_the_user_chose_is_kept() {
        let wired = wire(r#"{ "env": { "ANTUMBRA_HOST_ID": "windows" } }"#, &wiring()).unwrap();
        assert_eq!(parsed(&wired.text)["env"]["ANTUMBRA_HOST_ID"], "windows");
        assert!(!wired.env_set.contains(&"ANTUMBRA_HOST_ID".to_string()));
        let wired = wire("{}", &wiring()).unwrap();
        assert_eq!(parsed(&wired.text)["env"]["ANTUMBRA_HOST_ID"], "laptop");
    }

    #[test]
    fn a_hook_already_wired_by_name_is_not_doubled() {
        // Hand-wired on Windows: the PowerShell twin of the bash script.
        let text = r#"{
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "powershell -NoProfile -File \"C:\\Users\\me\\.claude\\antumbra-session-start.ps1\"" }]}]
  }
}"#;
        let wired = wire(text, &wiring()).unwrap();
        let v = parsed(&wired.text);
        assert_eq!(v["hooks"]["SessionStart"].as_array().unwrap().len(), 1);
        assert_eq!(wired.already.len(), 1);
        assert_eq!(wired.already[0].0, "SessionStart");
        assert_eq!(wired.added, vec!["Stop"]);
    }

    #[test]
    fn running_it_twice_changes_nothing_the_second_time() {
        let once = wire("{}", &wiring()).unwrap();
        let twice = wire(&once.text, &wiring()).unwrap();
        assert_eq!(once.text, twice.text);
        assert!(!twice.changed());
        assert_eq!(twice.already.len(), 2);
    }

    #[test]
    fn an_answer_about_auto_memory_is_kept() {
        let wired = wire(r#"{ "autoMemoryEnabled": true }"#, &wiring()).unwrap();
        assert_eq!(parsed(&wired.text)["autoMemoryEnabled"], true);
        assert_eq!(wired.auto_memory_was, Some(true));
        assert!(!wired.auto_memory_set);

        let mut w = wiring();
        w.auto_memory_off = false;
        let wired = wire("{}", &w).unwrap();
        assert!(parsed(&wired.text).get("autoMemoryEnabled").is_none());
    }

    #[test]
    fn one_line_objects_and_escaped_strings_are_handled() {
        let text = r#"{"note": "a \"quoted\" } brace", "env": {}, "hooks": {"Stop": []}}"#;
        let wired = wire(text, &wiring()).unwrap();
        let v = parsed(&wired.text);
        assert_eq!(v["note"], "a \"quoted\" } brace");
        assert_eq!(v["env"]["ANTUMBRA_WORKSPACE_ID"], "ws:default");
        assert_eq!(v["hooks"]["Stop"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn a_file_that_is_not_json_is_refused() {
        let err = wire("{ not json", &wiring()).unwrap_err().to_string();
        assert!(err.contains("not valid JSON"), "{err}");
        assert!(wire("[]", &wiring()).is_err());
        assert!(wire(r#"{"hooks": []}"#, &wiring()).is_err());
    }
}
