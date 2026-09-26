//! Whether the Antumbra hooks Claude Code runs are the ones this build ships.
//!
//! The hooks are copied into place by hand, usually into `~/.claude`, and a
//! copy edited there drifts from the repository without anyone noticing. A
//! fix made in place never reaches the repository, and one made in the
//! repository never reaches the machine. Measured once already: the per-prompt
//! recall ran for days with a time budget and UTF-8 handling the repository
//! did not have. So the doctor finds every hook script the settings run, and
//! compares each one it knows by name with the copy bundled into this build.
//! Line endings are not a difference.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// Every hook this build ships, by file name.
const BUNDLED: &[(&str, &str)] = &[
    (
        "antumbra-session-start.ps1",
        include_str!("../../../../scripts/hooks/antumbra-session-start.ps1"),
    ),
    (
        "antumbra-session-start.sh",
        include_str!("../../../../scripts/hooks/antumbra-session-start.sh"),
    ),
    (
        "antumbra-prompt-recall.ps1",
        include_str!("../../../../scripts/hooks/antumbra-prompt-recall.ps1"),
    ),
    (
        "antumbra-prompt-recall.sh",
        include_str!("../../../../scripts/hooks/antumbra-prompt-recall.sh"),
    ),
    (
        "antumbra-capture.ps1",
        include_str!("../../../../scripts/hooks/antumbra-capture.ps1"),
    ),
    (
        "antumbra-capture.sh",
        include_str!("../../../../scripts/hooks/antumbra-capture.sh"),
    ),
    (
        "antumbra-mcp-headers.ps1",
        include_str!("../../../../scripts/hooks/antumbra-mcp-headers.ps1"),
    ),
    (
        "antumbra-mcp-headers.sh",
        include_str!("../../../../scripts/hooks/antumbra-mcp-headers.sh"),
    ),
    (
        "strip-attribution.ps1",
        include_str!("../../../../scripts/hooks/strip-attribution.ps1"),
    ),
    (
        "strip-attribution.sh",
        include_str!("../../../../scripts/hooks/strip-attribution.sh"),
    ),
];

/// How an installed hook compares with the bundled one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookState {
    Same,
    Differs,
    /// The settings run it, and there is no file there.
    Missing,
}

/// One hook script the settings run that this build knows by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledHook {
    pub name: String,
    pub path: PathBuf,
    pub state: HookState,
}

/// The words of a command: whitespace-separated, and a double-quoted run kept
/// whole, so a quoted path with spaces is one word.
fn words(command: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut quoted = false;
    for c in command.chars() {
        match c {
            '"' => quoted = !quoted,
            c if c.is_whitespace() && !quoted => {
                if !word.is_empty() {
                    out.push(std::mem::take(&mut word));
                }
            }
            c => word.push(c),
        }
    }
    if !word.is_empty() {
        out.push(word);
    }
    out
}

/// Every script a settings file's hooks run: each word of each hook command
/// that ends in `.ps1` or `.sh`, in the order the settings give them.
pub fn scripts_in(settings: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let Some(events) = settings.get("hooks").and_then(Value::as_object) else {
        return out;
    };
    for matchers in events.values().filter_map(Value::as_array) {
        for hook in matchers
            .iter()
            .filter_map(|m| m.get("hooks").and_then(Value::as_array))
            .flatten()
        {
            let Some(command) = hook.get("command").and_then(Value::as_str) else {
                continue;
            };
            for word in words(command) {
                if (word.ends_with(".ps1") || word.ends_with(".sh")) && !out.contains(&word) {
                    out.push(word);
                }
            }
        }
    }
    out
}

/// The file name of a script path written with either separator.
fn file_name(script: &str) -> &str {
    script.rsplit(['/', '\\']).next().unwrap_or(script)
}

fn same_text(a: &str, b: &str) -> bool {
    a.replace("\r\n", "\n") == b.replace("\r\n", "\n")
}

/// Compare every script in `scripts` that this build knows by name with the
/// bundled copy. A relative path is resolved against `base`; `read` gives a
/// file's text, or `None` when there is none.
pub fn compare(
    scripts: &[String],
    base: &Path,
    read: impl Fn(&Path) -> Option<String>,
) -> Vec<InstalledHook> {
    let mut out = Vec::new();
    for script in scripts {
        let name = file_name(script);
        let Some((_, bundled)) = BUNDLED.iter().find(|(n, _)| *n == name) else {
            continue;
        };
        let path = Path::new(script);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            base.join(path)
        };
        let state = match read(&path) {
            None => HookState::Missing,
            Some(text) if same_text(&text, bundled) => HookState::Same,
            Some(_) => HookState::Differs,
        };
        out.push(InstalledHook {
            name: name.to_string(),
            path,
            state,
        });
    }
    out
}

/// The hooks the user's and the project's settings run, compared.
pub fn installed(home: Option<&Path>, project: &Path) -> Vec<InstalledHook> {
    let mut sources: Vec<(PathBuf, PathBuf)> = Vec::new();
    if let Some(home) = home {
        sources.push((
            home.join(".claude").join("settings.json"),
            home.to_path_buf(),
        ));
    }
    for file in ["settings.json", "settings.local.json"] {
        sources.push((project.join(".claude").join(file), project.to_path_buf()));
    }
    let read = |p: &Path| std::fs::read_to_string(p).ok();
    let mut out: Vec<InstalledHook> = Vec::new();
    for (settings, base) in sources {
        let Some(value) = read(&settings).and_then(|t| serde_json::from_str::<Value>(&t).ok())
        else {
            continue;
        };
        for hook in compare(&scripts_in(&value), &base, read) {
            if !out.iter().any(|h| h.path == hook.path) {
                out.push(hook);
            }
        }
    }
    out
}

/// What the doctor prints about the hooks; nothing when the settings run none
/// of the bundled ones.
pub fn render(hooks: &[InstalledHook]) -> String {
    if hooks.is_empty() {
        return String::new();
    }
    let mut out = String::from("\nAntumbra hooks:\n");
    for h in hooks {
        let said = match h.state {
            HookState::Same => "the same as this build's",
            HookState::Differs => {
                "differs from this build's: a change made there that the repository lacks, or one the repository has that it lacks"
            }
            HookState::Missing => "is run by the settings and is not there",
        };
        out.push_str(&format!("  {} at {}: {said}\n", h.name, h.path.display()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn bundled(name: &str) -> &'static str {
        BUNDLED.iter().find(|(n, _)| *n == name).unwrap().1
    }

    #[test]
    fn every_script_a_hook_command_runs_is_found() {
        let settings = serde_json::json!({
            "hooks": {
                "UserPromptSubmit": [{ "hooks": [{ "type": "command",
                    "command": "powershell -NoProfile -File \"C:\\Users\\me\\My Hooks\\antumbra-prompt-recall.ps1\"" }]}],
                "Stop": [{ "hooks": [{ "type": "command",
                    "command": "bash ./scripts/hooks/antumbra-capture.sh" }]}],
                "PreCompact": [{ "hooks": [{ "type": "command",
                    "command": "bash ./scripts/hooks/antumbra-capture.sh" }]}],
                "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "git status" }]}]
            }
        });
        let mut found = scripts_in(&settings);
        found.sort();
        assert_eq!(
            found,
            vec![
                "./scripts/hooks/antumbra-capture.sh".to_string(),
                "C:\\Users\\me\\My Hooks\\antumbra-prompt-recall.ps1".to_string(),
            ]
        );
        assert!(scripts_in(&serde_json::json!({})).is_empty());
    }

    #[test]
    fn an_installed_hook_is_compared_with_the_bundled_one() {
        let base = Path::new("/project");
        let files: HashMap<PathBuf, String> = [
            (
                base.join("same/antumbra-capture.sh"),
                bundled("antumbra-capture.sh").replace('\n', "\r\n"),
            ),
            (
                base.join("edited/antumbra-prompt-recall.ps1"),
                format!(
                    "{}\n# edited in place\n",
                    bundled("antumbra-prompt-recall.ps1")
                ),
            ),
            (base.join("mine/my-own-hook.sh"), "echo mine".to_string()),
        ]
        .into_iter()
        .collect();
        let scripts = vec![
            "same/antumbra-capture.sh".to_string(),
            "edited/antumbra-prompt-recall.ps1".into(),
            "gone/antumbra-session-start.ps1".into(),
            "mine/my-own-hook.sh".into(),
        ];
        let hooks = compare(&scripts, base, |p| files.get(p).cloned());
        let states: Vec<(&str, HookState)> =
            hooks.iter().map(|h| (h.name.as_str(), h.state)).collect();
        // Line endings are not a difference, and a hook this build does not
        // ship is not judged.
        assert_eq!(
            states,
            vec![
                ("antumbra-capture.sh", HookState::Same),
                ("antumbra-prompt-recall.ps1", HookState::Differs),
                ("antumbra-session-start.ps1", HookState::Missing),
            ]
        );
        let said = render(&hooks);
        assert!(said.contains("antumbra-prompt-recall.ps1 at"));
        assert!(said.contains("differs from this build's"));
        assert!(render(&[]).is_empty());
    }

    #[test]
    fn a_quoted_path_with_spaces_is_one_word() {
        assert_eq!(
            words(r#"pwsh -File "C:\a b\x.ps1" -X"#),
            vec!["pwsh", "-File", r"C:\a b\x.ps1", "-X"]
        );
    }
}
