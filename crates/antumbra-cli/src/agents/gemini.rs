//! Gemini CLI: an extension of Antumbra's own.
//!
//! Gemini CLI reads MCP servers and hooks from its `settings.json`, and also
//! from each extension under `~/.gemini/extensions`, which is a directory
//! holding a manifest (`gemini-extension.json`, with the extension's MCP
//! servers) and a `hooks/hooks.json`. An extension placed there by hand is
//! loaded like one its `extensions install` command put there: it is listed,
//! its server is connected, its hooks run.
//!
//! So Antumbra is an extension, and the user's `settings.json` is never
//! edited. The directory is Antumbra's own: connecting again replaces it, and
//! `gemini extensions uninstall antumbra` takes it away.
//!
//! - **The server** is the bridge, `antumbra mcp bridge --agent gemini`.
//! - **The hooks** run `antumbra hook gemini <event>`: `SessionStart` for
//!   what a session opens with, `BeforeAgent` for what is recalled for a
//!   prompt, `AfterAgent` for how the turn ended and the reminder to store.
//!   A timeout here is in milliseconds, where Claude Code's is in seconds.
//!
//! Two things are Gemini CLI's own to decide, and setup leaves them so:
//!
//! - **Trust in a folder.** In a folder its user has not trusted, Gemini CLI
//!   connects no MCP server at all, the user's own included.
//! - **Approval of a tool.** It asks before an Antumbra tool runs, and an
//!   extension's server cannot be marked trusted. A policy file of the user's
//!   allows them without asking.
//!
//! Started by Unity Gateway (`ug gemini`), Gemini CLI has another home, and
//! loads the extensions there and none of these. So on a machine `ug` is on,
//! the same directory is written there too.

use std::path::PathBuf;

use serde_json::json;

use super::hook::Event;
use super::{gateway, keep, kept, Agent, Finding, Ground, Kept, Report, Shell};

/// The extension's name, which is also its directory's and its server's.
pub const EXTENSION: &str = "antumbra";

/// Where Gemini CLI keeps its config: `.gemini` in the user's home, or in the
/// directory `GEMINI_CLI_HOME` names.
pub fn home(ground: &Ground<'_>) -> PathBuf {
    (ground.env)("GEMINI_CLI_HOME")
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .map_or_else(|| ground.home.to_path_buf(), PathBuf::from)
        .join(".gemini")
}

/// The extension's directory.
pub fn extension(ground: &Ground<'_>) -> PathBuf {
    home(ground).join("extensions").join(EXTENSION)
}

/// Every directory a Gemini CLI on this machine loads the extension from,
/// each with what setup calls it: Gemini CLI's own, and, on a machine Unity
/// Gateway is on, the one in the home `ug gemini` starts it with.
fn places(ground: &Ground<'_>) -> Vec<(PathBuf, &'static str)> {
    let own = extension(ground);
    let theirs = gateway::gemini_home(ground)
        .join(".gemini")
        .join("extensions")
        .join(EXTENSION);
    let both = gateway::present(ground) && theirs != own;
    std::iter::once((own, "Gemini CLI: the antumbra extension"))
        .chain(both.then_some((
            theirs,
            "Gemini CLI started by Unity Gateway: the antumbra extension",
        )))
        .collect()
}

/// Gemini CLI runs a hook's command in PowerShell on Windows and in bash
/// everywhere else.
fn shell() -> Shell {
    if cfg!(windows) {
        Shell::PowerShell
    } else {
        Shell::Posix
    }
}

/// The hooks, each with the name it is listed under and how long Gemini CLI
/// waits for it, in milliseconds. Longer than the scripts' own budgets by
/// what starting this program and the script's shell costs.
const HOOKS: [(&str, Event, &str, u32); 3] = [
    (
        "SessionStart",
        Event::SessionStart,
        "antumbra-session-start",
        15_000,
    ),
    ("BeforeAgent", Event::Prompt, "antumbra-recall", 25_000),
    ("AfterAgent", Event::Stop, "antumbra-capture", 15_000),
];

/// The extension's files, for an `antumbra` run as `ground` says.
pub fn files(ground: &Ground<'_>) -> Vec<Kept> {
    let manifest = json!({
        "name": EXTENSION,
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Antumbra: a memory that carries across sessions, agents and machines. Written by `antumbra setup`; run it again to update.",
        "mcpServers": {
            EXTENSION: {
                "command": ground.antumbra.file(),
                "args": ["mcp", "bridge", "--agent", Agent::Gemini.name()],
            }
        }
    });
    let mut events = serde_json::Map::new();
    for (theirs, event, name, timeout) in HOOKS {
        let command = format!(
            "{} hook {} {}",
            ground.antumbra.line(shell()),
            Agent::Gemini.name(),
            event.word()
        );
        events.insert(
            theirs.to_string(),
            json!([{
                "matcher": "*",
                "hooks": [{ "name": name, "type": "command", "command": command, "timeout": timeout }]
            }]),
        );
    }
    let pretty = |value: &serde_json::Value| {
        // A value built here is always printable; the fallback is its
        // compact form, which Gemini CLI reads as well.
        let mut text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
        text.push('\n');
        text
    };
    vec![
        Kept {
            path: "gemini-extension.json",
            text: pretty(&manifest),
        },
        Kept {
            path: "hooks/hooks.json",
            text: pretty(&json!({ "hooks": events })),
        },
    ]
}

pub fn connect(ground: &Ground<'_>, report: &dyn Report) -> anyhow::Result<()> {
    let files = files(ground);
    for (dir, what) in places(ground) {
        keep(
            &dir,
            &files,
            &format!("{what} (the MCP bridge and three hooks)"),
            report,
        )?;
    }
    report.info(
        "Gemini CLI connects no MCP server in a folder you have not trusted, and asks before an Antumbra tool runs.",
    );
    Ok(())
}

pub fn check(ground: &Ground<'_>) -> Vec<Finding> {
    let files = files(ground);
    places(ground)
        .into_iter()
        .flat_map(|(dir, what)| kept(&dir, &files, what))
        .collect()
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
    fn the_extension_carries_the_bridge_and_the_three_hooks() -> anyhow::Result<()> {
        let home = Scratch::new("gemini-files")?;
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        let files = files(&ground);
        let manifest: Value = serde_json::from_str(&files[0].text)?;
        assert_eq!(files[0].path, "gemini-extension.json");
        assert_eq!(manifest["name"], "antumbra");
        assert_eq!(manifest["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(
            manifest["mcpServers"]["antumbra"],
            json!({ "command": "antumbra", "args": ["mcp", "bridge", "--agent", "gemini"] })
        );
        let hooks: Value = serde_json::from_str(&files[1].text)?;
        assert_eq!(files[1].path, "hooks/hooks.json");
        for (event, word) in [
            ("SessionStart", "session-start"),
            ("BeforeAgent", "prompt"),
            ("AfterAgent", "stop"),
        ] {
            let hook = &hooks["hooks"][event][0]["hooks"][0];
            assert_eq!(hook["type"], "command");
            assert_eq!(hook["command"], format!("antumbra hook gemini {word}"));
            // Milliseconds, and more than the script's own budget.
            assert!(hook["timeout"].as_u64().is_some_and(|ms| ms >= 15_000));
        }
        Ok(())
    }

    #[test]
    fn an_antumbra_that_is_not_on_the_path_is_run_by_where_it_is() -> anyhow::Result<()> {
        let home = Scratch::new("gemini-path")?;
        let at = home.path().join("my tools").join("antumbra");
        let ground = Ground {
            home: home.path(),
            antumbra: Program::At(at.clone()),
            env: &nothing,
            find: &nowhere,
        };
        let files = files(&ground);
        let manifest: Value = serde_json::from_str(&files[0].text)?;
        // A program, taken apart from its arguments: no quoting.
        assert_eq!(
            manifest["mcpServers"]["antumbra"]["command"],
            at.display().to_string()
        );
        let hooks: Value = serde_json::from_str(&files[1].text)?;
        let command = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        // A command line: quoted, and in PowerShell a quoted path runs after `&`.
        let quoted = format!("\"{}\" hook gemini session-start", at.display());
        if cfg!(windows) {
            assert_eq!(command, format!("& {quoted}"));
        } else {
            assert_eq!(command, quoted);
        }
        Ok(())
    }

    #[test]
    fn connecting_writes_the_directory_and_again_changes_nothing() -> anyhow::Result<()> {
        let home = Scratch::new("gemini-connect")?;
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        let dir = home
            .path()
            .join(".gemini")
            .join("extensions")
            .join("antumbra");

        // A dry run says what it would do and does none of it.
        let dry = Told {
            dry_run: true,
            ..Told::default()
        };
        connect(&ground, &dry)?;
        assert!(
            dry.said.borrow()[0].starts_with("would write"),
            "{:?}",
            dry.said
        );
        assert!(!dir.exists());
        assert!(check(&ground)
            .iter()
            .any(|f| f.fix.is_some() && f.said.contains("missing")));

        let told = Told::default();
        connect(&ground, &told)?;
        assert!(
            told.said.borrow()[0].contains("written to"),
            "{:?}",
            told.said
        );
        assert!(dir.join("gemini-extension.json").is_file());
        assert!(dir.join("hooks").join("hooks.json").is_file());
        assert!(check(&ground).iter().all(|f| f.fix.is_none()));

        let again = Told::default();
        connect(&ground, &again)?;
        assert!(
            again.said.borrow()[0].contains("already in"),
            "{:?}",
            again.said
        );

        // A file edited in place, or written by an earlier build, is said
        // with what to do. It still works, so it does not fail the check.
        std::fs::write(dir.join("hooks").join("hooks.json"), "{}")?;
        let found = check(&ground);
        assert!(
            found.iter().any(|f| f.fix.is_some()
                && f.works
                && f.said.contains("hooks/hooks.json")
                && f.said.contains("differs")),
            "{found:?}"
        );
        // One that is gone does.
        std::fs::remove_file(dir.join("gemini-extension.json"))?;
        let found = check(&ground);
        assert!(
            found
                .iter()
                .any(|f| !f.works && f.said.contains("gemini-extension.json missing")),
            "{found:?}"
        );
        Ok(())
    }

    #[test]
    fn with_unity_gateway_here_the_extension_is_also_in_the_home_ug_starts_it_with(
    ) -> anyhow::Result<()> {
        let home = Scratch::new("gemini-gateway")?;
        let ug = |name: &str| (name == "ug").then(|| PathBuf::from(name));
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &ug,
        };
        let own = home
            .path()
            .join(".gemini")
            .join("extensions")
            .join("antumbra");
        let theirs = home
            .path()
            .join(".ucode")
            .join(".gemini-home")
            .join(".gemini")
            .join("extensions")
            .join("antumbra");

        let told = Told::default();
        connect(&ground, &told)?;
        let said = told.said.borrow().clone();
        assert!(
            said[0].starts_with("ok Gemini CLI: the antumbra extension")
                && said[0].contains("written to"),
            "{said:?}"
        );
        assert!(
            said[1].starts_with("ok Gemini CLI started by Unity Gateway")
                && said[1].contains("written to"),
            "{said:?}"
        );
        // The same files, both places.
        for file in ["gemini-extension.json", "hooks/hooks.json"] {
            assert_eq!(
                std::fs::read_to_string(own.join(file))?,
                std::fs::read_to_string(theirs.join(file))?,
                "{file}"
            );
        }
        assert!(check(&ground).iter().all(|f| f.fix.is_none()));

        // ug's copy gone, as after it was installed later than setup ran: the
        // check says which, and running setup again writes it.
        std::fs::remove_dir_all(&theirs)?;
        let found = check(&ground);
        assert!(
            found.iter().any(|f| !f.works
                && f.said.starts_with("Gemini CLI started by Unity Gateway")
                && f.said.contains("missing")),
            "{found:?}"
        );
        assert!(
            found
                .iter()
                .any(|f| f.fix.is_none() && f.said.starts_with("Gemini CLI: ")),
            "{found:?}"
        );

        // Without ug on this machine, its directory is not written to.
        let plain = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        connect(&plain, &Told::default())?;
        assert!(!theirs.exists());
        assert!(check(&plain).iter().all(|f| f.fix.is_none()));
        Ok(())
    }

    #[test]
    fn a_setup_run_from_inside_ug_gemini_writes_both_homes_once() -> anyhow::Result<()> {
        let home = Scratch::new("gemini-inside")?;
        // What `ug gemini` gives the session, and so a setup run from it.
        let theirs = home.path().join(".ucode").join(".gemini-home");
        let named = theirs.display().to_string();
        let inside = move |name: &str| (name == "GEMINI_CLI_HOME").then(|| named.clone());
        let ug = |name: &str| (name == "ug").then(|| PathBuf::from(name));

        // Read as it stands, the variable names ug's home as Gemini CLI's own:
        // one place, written once.
        let as_it_stands = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &inside,
            find: &ug,
        };
        assert_eq!(places(&as_it_stands).len(), 1);

        // Read as setup reads it, it is ug's: the user's own home is written
        // as well.
        let outside = gateway::outside(home.path(), &inside);
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &outside,
            find: &ug,
        };
        connect(&ground, &Told::default())?;
        for dir in [home.path().join(".gemini"), theirs.join(".gemini")] {
            assert!(
                dir.join("extensions")
                    .join("antumbra")
                    .join("gemini-extension.json")
                    .is_file(),
                "{}",
                dir.display()
            );
        }
        Ok(())
    }

    #[test]
    fn the_config_is_where_gemini_cli_home_says() -> anyhow::Result<()> {
        let home = Scratch::new("gemini-home")?;
        let elsewhere = home.path().join("elsewhere");
        let named = elsewhere.display().to_string();
        let env = move |name: &str| (name == "GEMINI_CLI_HOME").then(|| named.clone());
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &env,
            find: &nowhere,
        };
        assert_eq!(
            extension(&ground),
            elsewhere
                .join(".gemini")
                .join("extensions")
                .join("antumbra")
        );
        Ok(())
    }
}
