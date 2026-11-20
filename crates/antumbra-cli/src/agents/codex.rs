//! Codex: its own command for the MCP server, and three hooks in `hooks.json`.
//!
//! Codex reads MCP servers from `config.toml` and hooks from `hooks.json`,
//! both in `~/.codex` (or where `CODEX_HOME` says). Its hooks are Claude
//! Code's in everything that matters here: the same event names, the same
//! fields on standard input, the same JSON back.
//!
//! - **The server** is registered with Codex's own `codex mcp add`, which
//!   edits `config.toml` as Codex does and replaces an entry of the same
//!   name. Without `codex` on the PATH, setup prints the command, and the
//!   check says what to install without counting it against the setup.
//! - **The hooks** are added to `hooks.json` as setup adds Claude Code's to
//!   its settings: as text, adding only, with a backup, and never twice.
//!
//! One thing is Codex's own to decide, and nothing here goes around it: a
//! hook it has not seen before is skipped until its user approves it, in
//! Codex, under `/hooks`. Until then a session has Antumbra's tools and none
//! of the three touchpoints, so setup says so.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::hook::Event;
use super::{Agent, Finding, Ground, Report, Shell};
use crate::setup::settings::{self, Hook, Wiring};

/// The name the server is registered under.
pub const SERVER: &str = "antumbra";

/// Where Codex keeps its config: `CODEX_HOME`, or `.codex` in the user's home.
pub fn home(ground: &Ground<'_>) -> PathBuf {
    (ground.env)("CODEX_HOME")
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .map_or_else(|| ground.home.join(".codex"), PathBuf::from)
}

/// Codex runs a hook's command in PowerShell on Windows and in the user's
/// shell everywhere else.
fn shell() -> Shell {
    if cfg!(windows) {
        Shell::PowerShell
    } else {
        Shell::Posix
    }
}

/// Each hook: Codex's event, the moment, the words that mark the hook in a
/// command already there, and how long Codex waits for it, in seconds.
const HOOKS: [(&str, Event, &str, u32); 3] = [
    (
        "SessionStart",
        Event::SessionStart,
        "hook codex session-start",
        15,
    ),
    ("UserPromptSubmit", Event::Prompt, "hook codex prompt", 25),
    ("Stop", Event::Stop, "hook codex stop", 15),
];

fn hooks(ground: &Ground<'_>) -> Vec<Hook> {
    HOOKS
        .into_iter()
        .map(|(event, moment, mark, timeout)| Hook {
            event,
            script: mark,
            command: format!(
                "{} hook {} {}",
                ground.antumbra.line(shell()),
                Agent::Codex.name(),
                moment.word()
            ),
            timeout,
            tool: None,
            waits: true,
        })
        .collect()
}

/// The arguments that register the bridge, after the program's name.
fn registration(ground: &Ground<'_>) -> Vec<String> {
    ["mcp", "add", SERVER, "--"]
        .into_iter()
        .map(str::to_owned)
        .chain([ground.antumbra.file()])
        .chain(
            ["mcp", "bridge", "--agent", Agent::Codex.name()]
                .into_iter()
                .map(str::to_owned),
        )
        .collect()
}

/// Whether `config` (the text of `config.toml`) holds the server, run as the
/// bridge: its table, with the bridge's arguments before the next table.
fn registered(config: &str) -> bool {
    let header = format!("[mcp_servers.{SERVER}]");
    let Some(at) = config.lines().position(|line| line.trim() == header) else {
        return false;
    };
    config
        .lines()
        .skip(at + 1)
        .take_while(|line| !line.trim_start().starts_with('['))
        .any(|line| line.contains("\"bridge\""))
}

/// Where an organization's requirements for Codex are on this machine: a
/// file Codex reads above every config of the user's. `env` gives the value
/// of an environment variable, or nothing.
pub fn requirements(env: &dyn Fn(&str) -> Option<String>) -> PathBuf {
    if cfg!(windows) {
        env("ProgramData")
            .map(|dir| dir.trim().to_string())
            .filter(|dir| !dir.is_empty())
            .map_or_else(|| PathBuf::from(r"C:\ProgramData"), PathBuf::from)
            .join("OpenAI")
            .join("Codex")
            .join("requirements.toml")
    } else {
        PathBuf::from("/etc/codex/requirements.toml")
    }
}

/// What `requirements`, the text of an organization's requirements file at
/// `path`, stops of what setup wrote for Codex. Three things in it do, by
/// Codex's own pages (read on 2026-11-20): `allow_managed_hooks_only = true`
/// has Codex ignore every hook but the organization's, `hooks = false` under
/// `[features]` turns hooks off, and an `mcp_servers` table is the list of
/// the servers a user may enable, so one it does not name stays disabled.
///
/// The file is TOML, read here a line at a time for those three, as
/// `config.toml` is read for the server's table, and nothing else in it is
/// judged. A list that mentions the server anywhere is taken to name it: how
/// an entry is matched against a server's command is Codex's to decide.
pub fn held_back(requirements: &str, path: &Path) -> Vec<Finding> {
    let mut table = String::new();
    let (mut managed_only, mut hooks_off, mut listed) = (false, false, false);
    let lists = |name: &str| name == "mcp_servers" || name.starts_with("mcp_servers.");
    for line in requirements.lines() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if let Some(header) = line.strip_prefix('[') {
            table = header
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim()
                .to_string();
            listed |= lists(&table);
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().trim_matches('"');
        let key = if table.is_empty() {
            key.to_string()
        } else {
            format!("{table}.{key}")
        };
        match (key.as_str(), value.trim()) {
            ("allow_managed_hooks_only", value) => managed_only = value == "true",
            ("features.hooks", value) => hooks_off = value == "false",
            (key, _) => listed |= lists(key),
        }
    }
    let theirs = "that file is your organization's: ask whoever manages it";
    let mut found = Vec::new();
    if managed_only {
        found.push(Finding::bad(
            format!(
                "Codex: {} has it run only your organization's hooks (allow_managed_hooks_only), so Antumbra's three do not run",
                path.display()
            ),
            theirs,
        ));
    }
    if hooks_off {
        found.push(Finding::bad(
            format!(
                "Codex: {} turns hooks off (features.hooks = false), so Antumbra's three do not run",
                path.display()
            ),
            theirs,
        ));
    }
    if listed && !requirements.contains(SERVER) {
        found.push(Finding::bad(
            format!(
                "Codex: {} lists the MCP servers that may be enabled (mcp_servers), and \"{SERVER}\" is not one, so Codex leaves it disabled",
                path.display()
            ),
            format!("{theirs}, to list it as [mcp_servers.{SERVER}]"),
        ));
    }
    found
}

pub fn connect(ground: &Ground<'_>, report: &dyn Report) -> anyhow::Result<()> {
    let dir = home(ground);

    // The hooks.
    let path = dir.join("hooks.json");
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let wiring = Wiring {
        hooks: hooks(ground),
        file: "hooks.json",
        ..Wiring::default()
    };
    let wired = settings::wire(&text, &wiring)?;
    let added = wired.changed();
    if !added {
        report.ok(&format!(
            "Codex: {} already runs Antumbra's hooks",
            path.display()
        ));
    } else if report.dry_run() {
        report.would(&format!(
            "add hooks on {} to {}",
            wired.added.join(", "),
            path.display()
        ));
    } else {
        std::fs::create_dir_all(&dir)?;
        if path.exists() {
            let backup = path.with_file_name(format!(
                "hooks.json.antumbra-backup-{}",
                chrono::Utc::now().format("%Y%m%d%H%M%S")
            ));
            std::fs::copy(&path, &backup)?;
            report.info(&format!("backed up Codex's hooks to {}", backup.display()));
        }
        std::fs::write(&path, &wired.text)?;
        report.ok(&format!(
            "Codex: hooks on {} added to {}",
            wired.added.join(", "),
            path.display()
        ));
    }

    // The MCP server, by Codex's own command.
    let args = registration(ground);
    let by_hand = format!("codex {}", args.join(" "));
    let config = std::fs::read_to_string(dir.join("config.toml")).unwrap_or_default();
    match (ground.find)(Agent::Codex.program()) {
        _ if registered(&config) && config.contains(&ground.antumbra.file()) => {
            report.ok("Codex: the MCP server \"antumbra\" is already registered");
        }
        None => {
            report.warn(
                "the codex command is not on your PATH, so the MCP server is not registered. Once it is:",
            );
            report.info(&by_hand);
        }
        Some(_) if report.dry_run() => {
            report.would("register the MCP server \"antumbra\" with Codex");
        }
        Some(codex) => {
            let out = Command::new(codex)
                .args(&args)
                .stdin(Stdio::null())
                .output()?;
            if out.status.success() {
                report.ok("Codex: MCP server \"antumbra\" registered");
            } else {
                report.warn(&format!(
                    "registering the MCP server with Codex failed: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                ));
                report.info(&by_hand);
            }
        }
    }

    if added {
        report.warn(
            "Codex skips a hook until you approve it. Start Codex, type /hooks, and approve Antumbra's three; until then a session has the tools and none of the memory around them.",
        );
    }
    Ok(())
}

pub fn check(ground: &Ground<'_>) -> Vec<Finding> {
    let dir = home(ground);
    let mut found = Vec::new();
    let path = dir.join("hooks.json");
    let hooks_file: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(serde_json::Value::Null);
    let missing: Vec<&str> = HOOKS
        .into_iter()
        .filter(|(event, _, mark, _)| {
            !hooks_file
                .get("hooks")
                .and_then(|h| h.get(event))
                .is_some_and(|e| e.to_string().contains(mark))
        })
        .map(|(event, ..)| event)
        .collect();
    if missing.is_empty() {
        found.push(Finding::good(format!(
            "Codex: {} runs Antumbra's three hooks (Codex runs them once you have approved them, under /hooks)",
            path.display()
        )));
    } else {
        found.push(Finding::bad(
            format!(
                "Codex: {} does not run Antumbra's hook on {}",
                path.display(),
                missing.join(", ")
            ),
            "run setup again to add it",
        ));
    }
    let config = dir.join("config.toml");
    if registered(&std::fs::read_to_string(&config).unwrap_or_default()) {
        found.push(Finding::good(
            "Codex: the MCP server \"antumbra\" is registered",
        ));
    } else if (ground.find)(Agent::Codex.program()).is_none() {
        // Only `codex mcp add` registers it, so without the program setup
        // has nothing to run, and running it again would change nothing.
        found.push(Finding::waiting(
            format!(
                "Codex: the MCP server \"antumbra\" is not in {}, because the codex command that registers it is not on your PATH",
                config.display()
            ),
            "install Codex, then run setup again",
        ));
    } else {
        found.push(Finding::bad(
            format!(
                "Codex: the MCP server \"antumbra\" is not in {}",
                config.display()
            ),
            "run setup again",
        ));
    }
    found
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

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
    fn an_organizations_requirements_that_stop_the_hooks_or_the_server_are_said() {
        let path = Path::new("/etc/codex/requirements.toml");
        let said = |text: &str| -> Vec<String> {
            held_back(text, path)
                .into_iter()
                .map(|found| {
                    // Each is the organization's to change, and fails the check.
                    assert!(!found.works, "{found:?}");
                    assert!(
                        found
                            .fix
                            .as_deref()
                            .is_some_and(|fix| fix.contains("your organization's")),
                        "{found:?}"
                    );
                    found.said
                })
                .collect()
        };
        // Nothing of ours in it.
        assert!(said("").is_empty());
        assert!(said(
            "approval_policy = \"on-request\"\n\n[features]\nhooks = true\n\n[hooks]\nmanaged_dir = \"/enterprise/hooks\"\n"
        )
        .is_empty());
        // Only the organization's hooks.
        let found = said("# policy\nallow_managed_hooks_only = true  # ours alone\n");
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("allow_managed_hooks_only"), "{found:?}");
        assert!(
            found[0].contains("requirements.toml"),
            "the file is named: {found:?}"
        );
        // Written out as off, or commented out, or another table's key of
        // the same name: none of them is the requirement.
        for not_it in [
            "allow_managed_hooks_only = false\n",
            "# allow_managed_hooks_only = true\n",
            "[tools]\nallow_managed_hooks_only = true\n",
            "[other]\nhooks = false\n",
        ] {
            assert!(said(not_it).is_empty(), "{not_it}");
        }
        // Hooks off, under its table or as a dotted key.
        for off in ["[features]\nhooks = false\n", "features.hooks = false\n"] {
            let found = said(off);
            assert_eq!(found.len(), 1, "{off}");
            assert!(found[0].contains("features.hooks"), "{found:?}");
        }
        // A list of servers that does not name ours, in each way TOML writes
        // one, an empty one included.
        for list in [
            "[mcp_servers.docs]\nidentity = { command = \"codex-mcp\" }\n",
            "[mcp_servers]\n",
            "mcp_servers = {}\n",
            "mcp_servers.docs.identity = { url = \"https://example.com/mcp\" }\n",
        ] {
            let found = said(list);
            assert_eq!(found.len(), 1, "{list}");
            assert!(found[0].contains("mcp_servers"), "{found:?}");
        }
        // One that names it is not judged further.
        assert!(said(
            "[mcp_servers.docs]\nidentity = { command = \"codex-mcp\" }\n\n[mcp_servers.antumbra]\nidentity = { command = \"antumbra\" }\n"
        )
        .is_empty());
        // All three at once are three things said.
        assert_eq!(
            said("allow_managed_hooks_only = true\n[features]\nhooks = false\n[mcp_servers]\n")
                .len(),
            3
        );
    }

    #[test]
    fn the_requirements_are_where_codex_reads_them() {
        let none = |_: &str| None;
        let path = requirements(&none);
        assert!(path.ends_with("requirements.toml"));
        if cfg!(windows) {
            assert_eq!(
                path,
                PathBuf::from(r"C:\ProgramData\OpenAI\Codex\requirements.toml")
            );
            let moved = |name: &str| (name == "ProgramData").then(|| r"D:\Data".to_string());
            assert_eq!(
                requirements(&moved),
                PathBuf::from(r"D:\Data\OpenAI\Codex\requirements.toml")
            );
        } else {
            assert_eq!(path, PathBuf::from("/etc/codex/requirements.toml"));
        }
    }

    #[test]
    fn the_hooks_are_added_beside_the_users_own_and_never_twice() -> anyhow::Result<()> {
        let home = Scratch::new("codex-hooks")?;
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        let dir = home.path().join(".codex");
        std::fs::create_dir_all(&dir)?;
        let theirs = r#"{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command", "command": "notify-send done", "timeout": 5 } ] }
    ]
  }
}
"#;
        std::fs::write(dir.join("hooks.json"), theirs)?;

        let told = Told::default();
        connect(&ground, &told)?;
        let after: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("hooks.json"))?)?;
        for (event, word) in [
            ("SessionStart", "session-start"),
            ("UserPromptSubmit", "prompt"),
            ("Stop", "stop"),
        ] {
            let hook = &after["hooks"][event][0]["hooks"][0];
            assert_eq!(hook["command"], format!("antumbra hook codex {word}"));
            // Seconds, as Codex counts them.
            assert!(hook["timeout"]
                .as_u64()
                .is_some_and(|s| (10..=30).contains(&s)));
        }
        // The user's own hook is still there, after Antumbra's.
        assert_eq!(
            after["hooks"]["Stop"][1]["hooks"][0]["command"],
            "notify-send done"
        );
        let said = told.said.borrow().join("\n");
        assert!(said.contains("backed up Codex's hooks"), "{said}");
        // Codex's own gate is said, since nothing here goes around it.
        assert!(
            said.contains("warn Codex skips a hook until you approve it"),
            "{said}"
        );
        // Without codex on the PATH, the command to run is given.
        assert!(
            said.contains("info codex mcp add antumbra -- antumbra mcp bridge --agent codex"),
            "{said}"
        );
        assert!(check(&ground)
            .iter()
            .any(|f| f.fix.is_none() && f.said.contains("three hooks")));
        // The server waits on the program, which does not fail the check.
        assert!(check(&ground)
            .iter()
            .any(|f| f.fix.is_some() && f.works && f.said.contains("is not in")));

        let again = Told::default();
        connect(&ground, &again)?;
        let said = again.said.borrow().join("\n");
        assert!(said.contains("already runs Antumbra's hooks"), "{said}");
        assert!(!said.contains("Codex skips a hook"), "{said}");
        let twice: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("hooks.json"))?)?;
        assert_eq!(twice, after);
        Ok(())
    }

    #[test]
    fn a_server_setup_could_not_register_without_codex_does_not_fail_the_check(
    ) -> anyhow::Result<()> {
        let home = Scratch::new("codex-missing")?;
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        // Named with `--agent`, on a machine without Codex: the hooks are
        // written, and the server is left for `codex mcp add`.
        connect(&ground, &Told::default())?;
        let config = home.path().join(".codex").join("config.toml");
        assert!(!config.exists());

        // Setup again would do no more, so it is said with what to install
        // and the check passes.
        let found = check(&ground);
        assert!(found.iter().all(|f| f.works), "{found:?}");
        let waits: Vec<&Finding> = found.iter().filter(|f| f.fix.is_some()).collect();
        let [waiting] = waits.as_slice() else {
            anyhow::bail!("one thing waits on Codex, and the check found {found:?}");
        };
        assert!(
            waiting
                .said
                .contains("the codex command that registers it is not on your PATH"),
            "{waiting:?}"
        );
        assert_eq!(
            waiting.fix.as_deref(),
            Some("install Codex, then run setup again")
        );

        // With Codex installed, setup can register it, and the check says so
        // and fails. The check only looks for the program and never runs it.
        let on_path = |name: &str| (name == "codex").then(|| PathBuf::from("codex"));
        let installed = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &on_path,
        };
        let found = check(&installed);
        assert!(
            found.iter().any(|f| !f.works
                && f.said.contains("is not in")
                && f.fix.as_deref() == Some("run setup again")),
            "{found:?}"
        );

        // A server registered while Codex was there is found without it.
        std::fs::write(
            &config,
            "[mcp_servers.antumbra]\ncommand = \"antumbra\"\nargs = [\"mcp\", \"bridge\", \"--agent\", \"codex\"]\n",
        )?;
        let found = check(&ground);
        assert!(found.iter().all(|f| f.fix.is_none()), "{found:?}");
        Ok(())
    }

    #[test]
    fn a_dry_run_writes_nothing() -> anyhow::Result<()> {
        let home = Scratch::new("codex-dry")?;
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        let told = Told {
            dry_run: true,
            ..Told::default()
        };
        connect(&ground, &told)?;
        assert!(told.said.borrow()[0]
            .starts_with("would add hooks on SessionStart, UserPromptSubmit, Stop"));
        assert!(!home.path().join(".codex").exists());
        Ok(())
    }

    #[test]
    fn the_server_is_found_in_the_config_codex_writes() {
        let named = "model = \"gpt\"\n\n[mcp_servers.antumbra]\ncommand = \"antumbra\"\nargs = [\"mcp\", \"bridge\", \"--agent\", \"codex\"]\n";
        assert!(registered(named));
        // Another server of that name, or the name under another table.
        assert!(!registered(
            "[mcp_servers.antumbra]\ncommand = \"something-else\"\n"
        ));
        assert!(!registered(
            "[mcp_servers.other]\nargs = [\"mcp\", \"bridge\"]\n\n[mcp_servers.antumbra]\ncommand = \"x\"\n"
        ));
        assert!(!registered(""));
    }

    #[test]
    fn the_registration_names_the_bridge_and_where_antumbra_is() -> anyhow::Result<()> {
        let home = Scratch::new("codex-register")?;
        let at = home.path().join("my tools").join("antumbra");
        let ground = Ground {
            home: home.path(),
            antumbra: Program::At(at.clone()),
            env: &nothing,
            find: &nowhere,
        };
        // Arguments, each its own: a path with a space needs no quoting.
        assert_eq!(
            registration(&ground),
            vec![
                "mcp".to_string(),
                "add".into(),
                "antumbra".into(),
                "--".into(),
                at.display().to_string(),
                "mcp".into(),
                "bridge".into(),
                "--agent".into(),
                "codex".into(),
            ]
        );
        let elsewhere = home.path().join("codex-home");
        let named = elsewhere.display().to_string();
        let env = move |name: &str| (name == "CODEX_HOME").then(|| named.clone());
        let moved = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &env,
            find: &nowhere,
        };
        assert_eq!(super::home(&moved), elsewhere);
        Ok(())
    }
}
