//! Copilot CLI: a hooks file of Antumbra's own, and one server in
//! `mcp-config.json`.
//!
//! Copilot CLI keeps its config in `~/.copilot` (or where `COPILOT_HOME`
//! says): MCP servers in `mcp-config.json`, and hooks in any `*.json` under
//! `hooks/`. Copilot in VS Code reads the same two places.
//!
//! - **The hooks** are a file of Antumbra's own, `hooks/antumbra.json`, with
//!   `sessionStart`, `userPromptSubmitted` and `agentStop`. Copilot CLI runs
//!   a user's own hooks without asking.
//! - **The server** is one entry, `mcpServers.antumbra`, put into
//!   `mcp-config.json` as text: the rest of the file is left byte for byte,
//!   with a backup beside it.
//!
//! The file is edited here, where Codex's is edited by `codex mcp add`,
//! because the `copilot` on a PATH is not always Copilot CLI: VS Code puts a
//! launcher of that name there, and what that does when it is run by a
//! program is not something to find out on a person's machine.
//!
//! Copilot CLI puts an MCP server's instructions in its system prompt only
//! for servers it has on a list of its own, so the bridge's are not read
//! there. The hooks say what they say.

use std::path::PathBuf;

use serde_json::{json, Value};

use super::hook::Event;
use super::{keep, kept, Agent, Finding, Ground, Kept, Report, Shell};
use crate::setup::settings;

/// The name the server is registered under, and the hooks file's.
pub const NAME: &str = "antumbra";

/// Where Copilot CLI keeps its config: `COPILOT_HOME`, or `.copilot` in the
/// user's home.
pub fn home(ground: &Ground<'_>) -> PathBuf {
    (ground.env)("COPILOT_HOME")
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .map_or_else(|| ground.home.join(".copilot"), PathBuf::from)
}

/// Each hook: Copilot CLI's event, the moment, and how long it waits, in
/// seconds.
const HOOKS: [(&str, Event, u32); 3] = [
    ("sessionStart", Event::SessionStart, 15),
    ("userPromptSubmitted", Event::Prompt, 25),
    ("agentStop", Event::Stop, 15),
];

/// The hooks file. Each hook gives its command twice, for the shell Copilot
/// CLI runs it in on this platform or another.
fn hooks_file(ground: &Ground<'_>) -> Kept {
    let mut events = serde_json::Map::new();
    for (theirs, event, timeout) in HOOKS {
        let line = |shell| {
            format!(
                "{} hook {} {}",
                ground.antumbra.line(shell),
                Agent::Copilot.name(),
                event.word()
            )
        };
        events.insert(
            theirs.to_string(),
            json!([{
                "type": "command",
                "bash": line(Shell::Posix),
                "powershell": line(Shell::PowerShell),
                "timeoutSec": timeout,
            }]),
        );
    }
    let file = json!({ "version": 1, "hooks": events });
    // A value built here always prints; the fallback is its compact form.
    let mut text = serde_json::to_string_pretty(&file).unwrap_or_else(|_| file.to_string());
    text.push('\n');
    Kept {
        path: "antumbra.json",
        text,
    }
}

/// The server's entry: the bridge, and every one of its tools.
fn server(ground: &Ground<'_>) -> Value {
    json!({
        "type": "local",
        "command": ground.antumbra.file(),
        "args": ["mcp", "bridge", "--agent", Agent::Copilot.name()],
        "tools": ["*"],
    })
}

pub fn connect(ground: &Ground<'_>, report: &dyn Report) -> anyhow::Result<()> {
    let dir = home(ground);
    keep(
        &dir.join("hooks"),
        &[hooks_file(ground)],
        "Copilot CLI: Antumbra's hooks (antumbra.json, three hooks)",
        report,
    )?;

    let path = dir.join("mcp-config.json");
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    match settings::put(
        &text,
        "mcp-config.json",
        &["mcpServers"],
        NAME,
        &server(ground),
    )? {
        None => report.ok(&format!(
            "Copilot CLI: the MCP server \"antumbra\" is already in {}",
            path.display()
        )),
        Some(_) if report.dry_run() => report.would(&format!(
            "add the MCP server \"antumbra\" to {}",
            path.display()
        )),
        Some(edited) => {
            std::fs::create_dir_all(&dir)?;
            if path.exists() {
                let backup = path.with_file_name(format!(
                    "mcp-config.json.antumbra-backup-{}",
                    chrono::Utc::now().format("%Y%m%d%H%M%S")
                ));
                std::fs::copy(&path, &backup)?;
                report.info(&format!(
                    "backed up Copilot CLI's MCP servers to {}",
                    backup.display()
                ));
            }
            std::fs::write(&path, edited)?;
            report.ok(&format!(
                "Copilot CLI: MCP server \"antumbra\" added to {}",
                path.display()
            ));
        }
    }
    Ok(())
}

pub fn check(ground: &Ground<'_>) -> Vec<Finding> {
    let dir = home(ground);
    let mut found = kept(
        &dir.join("hooks"),
        &[hooks_file(ground)],
        "Copilot CLI: Antumbra's hooks",
    );
    let path = dir.join("mcp-config.json");
    let config: Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null);
    let bridged = config
        .pointer(&format!("/mcpServers/{NAME}/args"))
        .and_then(Value::as_array)
        .is_some_and(|args| args.iter().any(|a| a == "bridge"));
    if bridged {
        found.push(Finding::good(
            "Copilot CLI: the MCP server \"antumbra\" is registered",
        ));
    } else {
        found.push(Finding::bad(
            format!(
                "Copilot CLI: the MCP server \"antumbra\" is not in {}",
                path.display()
            ),
            "run setup again",
        ));
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::scratch::Scratch;
    use crate::agents::tests::Told;
    use crate::agents::Program;

    fn nothing(_: &str) -> Option<String> {
        None
    }

    fn nowhere(_: &str) -> Option<PathBuf> {
        None
    }

    #[test]
    fn the_hooks_are_a_file_of_their_own_with_a_command_for_each_shell() -> anyhow::Result<()> {
        let home = Scratch::new("copilot-hooks")?;
        let at = home.path().join("my tools").join("antumbra");
        let ground = Ground {
            home: home.path(),
            antumbra: Program::At(at.clone()),
            env: &nothing,
            find: &nowhere,
        };
        let file = hooks_file(&ground);
        assert_eq!(file.path, "antumbra.json");
        let hooks: Value = serde_json::from_str(&file.text)?;
        assert_eq!(hooks["version"], 1);
        for (event, word) in [
            ("sessionStart", "session-start"),
            ("userPromptSubmitted", "prompt"),
            ("agentStop", "stop"),
        ] {
            let hook = &hooks["hooks"][event][0];
            assert_eq!(hook["type"], "command");
            let quoted = format!("\"{}\" hook copilot {word}", at.display());
            assert_eq!(hook["bash"], quoted);
            assert_eq!(hook["powershell"], format!("& {quoted}"));
            assert!(hook["timeoutSec"]
                .as_u64()
                .is_some_and(|s| (10..=30).contains(&s)));
        }
        Ok(())
    }

    #[test]
    fn the_server_is_added_beside_the_users_own_and_never_twice() -> anyhow::Result<()> {
        let home = Scratch::new("copilot-connect")?;
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        let dir = home.path().join(".copilot");
        std::fs::create_dir_all(&dir)?;
        let theirs = "{\n  \"mcpServers\": {\n    \"context7\": { \"type\": \"local\", \"command\": \"npx\", \"args\": [\"-y\", \"some-mcp\"], \"tools\": [\"*\"] }\n  }\n}\n";
        std::fs::write(dir.join("mcp-config.json"), theirs)?;

        // A dry run says what it would do and does none of it.
        let dry = Told {
            dry_run: true,
            ..Told::default()
        };
        connect(&ground, &dry)?;
        assert!(!dir.join("hooks").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join("mcp-config.json"))?,
            theirs
        );
        assert!(check(&ground).iter().any(|f| f.fix.is_some()));

        let told = Told::default();
        connect(&ground, &told)?;
        let config: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("mcp-config.json"))?)?;
        assert_eq!(
            config["mcpServers"]["antumbra"],
            json!({ "type": "local", "command": "antumbra",
                    "args": ["mcp", "bridge", "--agent", "copilot"], "tools": ["*"] })
        );
        assert_eq!(config["mcpServers"]["context7"]["command"], "npx");
        assert!(dir.join("hooks").join("antumbra.json").is_file());
        let said = told.said.borrow().join("\n");
        assert!(
            said.contains("backed up Copilot CLI's MCP servers"),
            "{said}"
        );
        assert!(
            check(&ground).iter().all(|f| f.fix.is_none()),
            "{:?}",
            check(&ground)
        );

        let again = Told::default();
        connect(&ground, &again)?;
        let said = again.said.borrow().join("\n");
        assert!(said.contains("already in"), "{said}");
        assert!(!said.contains("backed up"), "{said}");
        Ok(())
    }

    #[test]
    fn the_config_is_where_copilot_home_says() -> anyhow::Result<()> {
        let home = Scratch::new("copilot-home")?;
        let elsewhere = home.path().join("elsewhere");
        let named = elsewhere.display().to_string();
        let env = move |name: &str| (name == "COPILOT_HOME").then(|| named.clone());
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &env,
            find: &nowhere,
        };
        assert_eq!(super::home(&ground), elsewhere);
        Ok(())
    }
}
