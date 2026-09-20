//! `antumbra claude apply` (ADR-0021): write the settings lines the doctor
//! prints, and nothing else.
//!
//! The doctor deliberately does not edit the agent's settings, and this keeps
//! the reason it gave. The file is the user's: its key order is theirs, its
//! formatting is theirs, and reading it into `serde_json` and writing it back
//! would re-sort every key (this workspace does not enable `preserve_order`,
//! and even with it the whitespace would go). So nothing here round-trips the
//! file. The text is parsed only to decide what is missing, and the edit is a
//! textual insertion into the `env` object, matching the indentation already
//! there. Every other byte is left exactly as it was.
//!
//! Three limits hold it in:
//!
//! - **Only `env`, and only known names.** The names come from the compiled
//!   matrix, never from an argument. Nothing under `permissions` is ever
//!   touched, not even `permissions.defaultMode`, which the doctor asks for:
//!   that object is what grants the agent its permissions, and a tool that
//!   writes it on its own is a tool to distrust.
//! - **Adding only.** A name the file already sets is left alone, whatever its
//!   value. If the user set it to `0` deliberately, that is their answer.
//! - **A backup, and a check that rolls back.** After writing, the result is
//!   re-read and compared with what was there: it must differ in exactly the
//!   names that were added and nowhere else, or the backup is restored.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{EnvSetting, Report, Standing};

/// The environment settings the report says are missing, in the matrix's order.
/// A rule the doctor reports as missing but that no `env` name settles (the
/// default permission mode) is not among them.
pub fn wanted(report: &Report) -> Vec<EnvSetting> {
    report
        .findings
        .iter()
        .filter(|finding| matches!(finding.standing, Standing::Missing { .. }))
        .filter_map(|finding| finding.rule.env)
        .collect()
}

/// The braces of a top-level object value: `(index of `{`, index of `}`)`.
///
/// A scan, not a parse, because the byte offsets are the point. Only a string
/// at depth 1 that is followed by `:` counts as a key, so a value that happens
/// to read `"env"`, or an `env` nested inside another object, is not mistaken
/// for the one meant.
fn top_level_object(text: &str, key: &str) -> Option<(usize, usize)> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut string_at = 0usize;
    let mut last_key: Option<&str> = None;
    let mut open: Option<usize> = None;

    for (i, c) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
                if depth == 1 {
                    let is_key = text[i + 1..].trim_start().starts_with(':');
                    last_key = is_key.then(|| &text[string_at + 1..i]);
                }
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                string_at = i;
            }
            '{' | '[' => {
                depth += 1;
                if c == '{' && depth == 2 && open.is_none() && last_key == Some(key) {
                    open = Some(i);
                }
            }
            '}' | ']' => {
                depth = depth.saturating_sub(1);
                if depth == 1 {
                    match open {
                        Some(at) => return Some((at, i)),
                        None => last_key = None,
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// The whitespace a line starts with.
fn indent_of(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// The indentation of the line holding byte `at`.
fn indent_at(text: &str, at: usize) -> &str {
    let start = text[..at].rfind('\n').map_or(0, |n| n + 1);
    indent_of(&text[start..])
}

/// The indentation the entries inside an object already use, or `None` when it
/// holds none to copy.
fn inner_indent(inside: &str) -> Option<&str> {
    inside
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(indent_of)
}

fn one_line(indent: &str, setting: &EnvSetting) -> String {
    format!(
        "{indent}{}: {}",
        Value::String(setting.name.to_string()),
        Value::String(setting.value.to_string())
    )
}

/// The file's text with `settings` added to its top-level `env` object, which
/// is created when it is absent. Only an insertion: nothing already there is
/// rewritten.
pub fn with_env(text: &str, settings: &[EnvSetting]) -> anyhow::Result<String> {
    if settings.is_empty() {
        return Ok(text.to_string());
    }
    match top_level_object(text, "env") {
        Some((open, close)) => {
            let inside = &text[open + 1..close];
            let outer = indent_at(text, close).to_string();
            let indent = inner_indent(inside)
                .map(str::to_string)
                .unwrap_or_else(|| format!("{outer}  "));
            let lines: Vec<String> = settings
                .iter()
                .map(|setting| one_line(&indent, setting))
                .collect();
            let mut out = String::with_capacity(text.len() + 64 * settings.len());
            out.push_str(&text[..=open]);
            if inside.trim().is_empty() {
                // An empty object has nothing to keep, so it is written out.
                out.push('\n');
                out.push_str(&lines.join(",\n"));
                out.push('\n');
                out.push_str(&outer);
            } else {
                for line in &lines {
                    out.push('\n');
                    out.push_str(line);
                    out.push(',');
                }
                out.push_str(inside);
            }
            out.push_str(&text[close..]);
            Ok(out)
        }
        None => {
            let root = text
                .find('{')
                .ok_or_else(|| anyhow::anyhow!("this is not a JSON object"))?;
            let inside = &text[root + 1..];
            let outer = indent_at(text, root).to_string();
            let indent = inner_indent(inside)
                .map(str::to_string)
                .unwrap_or_else(|| format!("{outer}  "));
            let deeper = format!("{indent}  ");
            let block = settings
                .iter()
                .map(|setting| one_line(&deeper, setting))
                .collect::<Vec<_>>()
                .join(",\n");
            let has_more = inside.trim_start().starts_with(|c| c != '}');
            let mut out = String::with_capacity(text.len() + 64 * settings.len() + 16);
            out.push_str(&text[..=root]);
            out.push('\n');
            out.push_str(&indent);
            out.push_str("\"env\": {\n");
            out.push_str(&block);
            out.push('\n');
            out.push_str(&indent);
            out.push('}');
            if has_more {
                out.push(',');
                out.push_str(&text[root + 1..]);
            } else {
                out.push('\n');
                out.push_str(inside.trim_start_matches([' ', '\t', '\n', '\r']));
            }
            Ok(out)
        }
    }
}

fn env_of(json: &Value) -> serde_json::Map<String, Value> {
    json.get("env")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

/// Confirm the rewrite changed exactly what it meant to: the same document,
/// plus these names in `env`, and nothing else anywhere. The check that makes a
/// textual edit safe to run on someone's settings.
pub fn only_added(before: &str, after: &str, added: &[EnvSetting]) -> anyhow::Result<()> {
    let was: Value = serde_json::from_str(before)
        .map_err(|e| anyhow::anyhow!("the file was not JSON to begin with: {e}"))?;
    let mut now: Value = serde_json::from_str(after)
        .map_err(|e| anyhow::anyhow!("the rewrite is not valid JSON: {e}"))?;
    let mut env = env_of(&now);
    for setting in added {
        match env.remove(setting.name) {
            Some(Value::String(value)) if value == setting.value => {}
            Some(other) => anyhow::bail!("{} was written as {other}", setting.name),
            None => anyhow::bail!("{} is not in the result", setting.name),
        }
    }
    match now.as_object_mut() {
        // An `env` this created and then emptied never existed: drop it, so the
        // comparison is against the document as it was.
        Some(root) if env.is_empty() && was.get("env").is_none() => {
            root.remove("env");
        }
        Some(root) => {
            root.insert("env".to_string(), Value::Object(env));
        }
        None => anyhow::bail!("the result is not a JSON object"),
    }
    anyhow::ensure!(
        now == was,
        "the rewrite changed more than the names it added"
    );
    Ok(())
}

/// Where a backup of `path` goes: beside it, stamped, so an earlier one is
/// never overwritten.
pub fn backup_path(path: &Path, now: chrono::DateTime<chrono::Utc>) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".antumbra-{}.bak", now.format("%Y%m%dT%H%M%SZ")));
    path.with_file_name(name)
}

/// Add `settings` to `path`, backing it up first and rolling back if the result
/// is not exactly the file plus those names. With `dry_run`, nothing is written
/// and the rewrite is still checked.
pub fn apply(path: &Path, settings: &[EnvSetting], dry_run: bool) -> anyhow::Result<Vec<String>> {
    let mut said = Vec::new();
    if settings.is_empty() {
        said.push("nothing to add: every setting the doctor asks for is already there".into());
        return Ok(said);
    }
    // A file that is not there yet is the common case on a new machine.
    let before = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            said.push(format!("{} does not exist yet; writing it", path.display()));
            "{\n}\n".to_string()
        }
        Err(e) => return Err(anyhow::anyhow!("read {}: {e}", path.display())),
    };
    // Refuse a file the agent itself would refuse: an edit to something this
    // does not understand is exactly what must not happen.
    serde_json::from_str::<Value>(&before)
        .map_err(|e| anyhow::anyhow!("{} is not valid JSON ({e}); left alone", path.display()))?;

    let after = with_env(&before, settings)?;
    only_added(&before, &after, settings)?;
    said.extend(
        settings
            .iter()
            .map(|s| format!("add      \"{}\": \"{}\"", s.name, s.value)),
    );
    if dry_run {
        return Ok(said);
    }

    let backup = backup_path(path, chrono::Utc::now());
    if path.exists() {
        std::fs::copy(path, &backup)
            .map_err(|e| anyhow::anyhow!("back up to {}: {e}", backup.display()))?;
        said.push(format!("backup   {}", backup.display()));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, &after).map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))?;

    // Read back what is on disk, not what was meant to be: a short write, or
    // anything else between here and the file, is caught and undone.
    let written = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read back {}: {e}", path.display()))?;
    if let Err(why) = only_added(&before, &written, settings) {
        std::fs::write(path, &before)
            .map_err(|e| anyhow::anyhow!("{why}; and restoring {} failed: {e}", path.display()))?;
        anyhow::bail!("{why}; {} is back as it was", path.display());
    }
    said.push(format!("wrote    {}", path.display()));
    Ok(said)
}

#[cfg(test)]
mod tests {
    use super::*;

    const POWERSHELL: EnvSetting = EnvSetting {
        name: "CLAUDE_CODE_USE_POWERSHELL_TOOL",
        value: "1",
    };
    const PROTOCOL: EnvSetting = EnvSetting {
        name: "MCP_PROTOCOL_NEGOTIATION",
        value: "auto",
    };

    /// Add, then check the result really is the document plus those names.
    fn added(text: &str, settings: &[EnvSetting]) -> anyhow::Result<String> {
        let after = with_env(text, settings)?;
        only_added(text, &after, settings)?;
        Ok(after)
    }

    #[test]
    fn the_top_level_env_is_found_and_a_lookalike_is_not() {
        let text = r#"{
  "hooks": { "env": { "NOT": "this one" } },
  "note": "env",
  "env": { "REAL": "1" }
}"#;
        let Some((open, close)) = top_level_object(text, "env") else {
            panic!("the top-level env was not found in {text}");
        };
        assert_eq!(&text[open + 1..close], r#" "REAL": "1" "#);
        assert_eq!(top_level_object(text, "nothing"), None);
        // An `env` that is not an object is not somewhere to add a name.
        assert_eq!(top_level_object(r#"{"env": ["a"]}"#, "env"), None);
        // Nor is the first object inside such an array: the key is spent on the
        // array, and what is in it belongs to the array.
        assert_eq!(top_level_object(r#"{"env": [{"A": "1"}]}"#, "env"), None);
        // A document whose root is an array has no top-level keys at all, so a
        // value that happens to read `env` must not adopt the object after it.
        assert_eq!(top_level_object(r#"["env", {"A": "1"}]"#, "env"), None);
    }

    #[test]
    fn a_name_is_added_and_every_other_byte_is_left_alone() -> anyhow::Result<()> {
        let before = "{\n  \"$schema\": \"x\",\n  \"env\": {\n    \"ZZZ\": \"keep\",\n    \"AAA\": \"keep\"\n  },\n  \"permissions\": { \"deny\": [\"Read(**)\"] }\n}\n";
        let after = added(before, &[POWERSHELL])?;
        assert!(
            after
                .contains("    \"CLAUDE_CODE_USE_POWERSHELL_TOOL\": \"1\",\n    \"ZZZ\": \"keep\""),
            "{after}"
        );
        // The user's order is theirs: ZZZ still precedes AAA.
        let (z, a) = (after.find("ZZZ"), after.find("AAA"));
        assert!(z < a && z.is_some(), "{after}");
        // Everything that was there is still there, byte for byte.
        for kept in [
            "\"$schema\": \"x\"",
            "\"permissions\": { \"deny\": [\"Read(**)\"] }",
        ] {
            assert!(after.contains(kept), "{kept} is gone from {after}");
        }
        assert_eq!(
            after.len(),
            before.len() + one_line("    ", &POWERSHELL).len() + 2
        );
        Ok(())
    }

    #[test]
    fn the_indentation_already_there_is_the_indentation_used() -> anyhow::Result<()> {
        let tabs = "{\n\t\"env\": {\n\t\t\"A\": \"1\"\n\t}\n}";
        assert!(added(tabs, &[PROTOCOL])?.contains("\n\t\t\"MCP_PROTOCOL_NEGOTIATION\": \"auto\","));
        let wide = "{\n    \"env\": {\n        \"A\": \"1\"\n    }\n}";
        assert!(added(wide, &[PROTOCOL])?.contains("\n        \"MCP_PROTOCOL_NEGOTIATION\""));
        Ok(())
    }

    #[test]
    fn an_empty_or_missing_env_is_written_out() -> anyhow::Result<()> {
        let empty = added("{\n  \"env\": {},\n  \"other\": 1\n}", &[POWERSHELL])?;
        assert!(
            empty.contains("\"env\": {\n    \"CLAUDE_CODE_USE_POWERSHELL_TOOL\": \"1\"\n  },"),
            "{empty}"
        );

        let none = added("{\n  \"other\": 1\n}", &[POWERSHELL, PROTOCOL])?;
        assert!(none.contains("\"env\": {\n"), "{none}");
        assert!(none.contains("\"other\": 1"), "{none}");

        let bare = added("{}", &[POWERSHELL])?;
        assert_eq!(
            serde_json::from_str::<Value>(&bare)?,
            serde_json::json!({ "env": { "CLAUDE_CODE_USE_POWERSHELL_TOOL": "1" } })
        );
        Ok(())
    }

    #[test]
    fn several_names_are_added_in_the_order_given() -> anyhow::Result<()> {
        let after = added(
            "{\n  \"env\": {\n    \"A\": \"1\"\n  }\n}",
            &[POWERSHELL, PROTOCOL],
        )?;
        let (first, second) = (
            after.find("CLAUDE_CODE_USE_POWERSHELL_TOOL"),
            after.find("MCP_PROTOCOL_NEGOTIATION"),
        );
        assert!(first < second && first.is_some(), "{after}");
        Ok(())
    }

    #[test]
    fn the_check_refuses_a_rewrite_that_changed_anything_else() {
        let before = "{\n  \"env\": { \"A\": \"1\" },\n  \"keep\": true\n}";
        let cases = [
            // A name dropped.
            r#"{ "env": { "CLAUDE_CODE_USE_POWERSHELL_TOOL": "1" } }"#,
            // Another value changed.
            r#"{ "env": { "A": "2", "CLAUDE_CODE_USE_POWERSHELL_TOOL": "1" }, "keep": true }"#,
            // Something else added.
            r#"{ "env": { "A": "1", "CLAUDE_CODE_USE_POWERSHELL_TOOL": "1" }, "keep": true, "extra": 1 }"#,
            // The name written with the wrong value.
            r#"{ "env": { "A": "1", "CLAUDE_CODE_USE_POWERSHELL_TOOL": "0" }, "keep": true }"#,
            // Not written at all.
            r#"{ "env": { "A": "1" }, "keep": true }"#,
            // Not JSON.
            "{ not json",
        ];
        for after in cases {
            assert!(
                only_added(before, after, &[POWERSHELL]).is_err(),
                "accepted: {after}"
            );
        }
        // The one that is right.
        let good =
            r#"{ "env": { "A": "1", "CLAUDE_CODE_USE_POWERSHELL_TOOL": "1" }, "keep": true }"#;
        assert!(only_added(before, good, &[POWERSHELL]).is_ok());
    }

    #[test]
    fn nothing_under_permissions_is_ever_written() -> anyhow::Result<()> {
        let before = "{\n  \"permissions\": {\n    \"defaultMode\": \"manual\",\n    \"allow\": [\"Bash\"]\n  }\n}";
        let after = added(before, &[POWERSHELL])?;
        assert!(after.contains("\"defaultMode\": \"manual\""), "{after}");
        assert!(after.contains("\"allow\": [\"Bash\"]"), "{after}");
        let json: Value = serde_json::from_str(&after)?;
        assert_eq!(
            json.pointer("/permissions/defaultMode"),
            Some(&Value::String("manual".into()))
        );
        Ok(())
    }

    #[test]
    fn a_backup_is_stamped_so_an_earlier_one_survives() -> anyhow::Result<()> {
        let path = Path::new("/home/someone/.claude/settings.json");
        let at = |s: &str| -> anyhow::Result<chrono::DateTime<chrono::Utc>> {
            Ok(chrono::DateTime::parse_from_rfc3339(s)?.into())
        };
        let first = backup_path(path, at("2026-09-20T07:00:00Z")?);
        let second = backup_path(path, at("2026-09-20T08:30:00Z")?);
        assert_ne!(first, second);
        assert_eq!(first.parent(), path.parent());
        assert!(
            first.to_string_lossy().contains("settings.json.antumbra-"),
            "{first:?}"
        );
        Ok(())
    }
}
