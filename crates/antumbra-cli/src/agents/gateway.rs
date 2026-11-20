//! Unity Gateway: the same agents, started by `ug`.
//!
//! Unity Gateway is Databricks' launcher for coding agents. `ug claude`,
//! `ug gemini` and the rest sign in to a workspace and start the agent
//! against that workspace's gateway. What it writes it keeps in `~/.ucode`,
//! and for two of the agents here that is their whole config directory, which
//! it points the agent at for the session. Such an agent reads nothing setup
//! wrote where the agent keeps its config on its own.
//!
//! An agent at a time, as its source and its README have it
//! (github.com/databricks/unity-gateway, read on 2026-11-20):
//!
//! - **Gemini CLI** is started with `GEMINI_CLI_HOME` naming
//!   `~/.ucode/.gemini-home`, so the extensions it loads are the ones under
//!   `.gemini/extensions` there ([`gemini_home`]).
//! - **OpenCode** is started with `XDG_CONFIG_HOME` naming
//!   `~/.ucode/opencode-xdg`, so the plugins it loads are the ones under
//!   `opencode/plugins` there ([`opencode_config_home`]).
//! - **Claude Code** is started with `--settings` naming
//!   `~/.claude/ucode-settings.json`, over the user's own settings, which it
//!   still reads. The exception is a launch that relays a Claude
//!   subscription: that one leaves the user's settings out
//!   (`--setting-sources project,local`) and hands the user's MCP servers
//!   back with `--mcp-config`, so the server is there and the hooks are not
//!   ([`ClaudeLaunch`]). `ug` merges a `--settings` its caller passes into
//!   its own, the hooks of both kept, so Antumbra's are passed that way
//!   ([`connect_claude`]).
//! - **Codex** and **Copilot CLI** are started from their own directories,
//!   and need nothing more than setup writes there.
//!
//! So on a machine `ug` is on ([`present`]), setup writes Antumbra's
//! extension and plugin a second time, where `ug` will have the agent look,
//! and a settings file of Antumbra's own for Claude Code. Nothing of Unity
//! Gateway's is edited: each is a directory or a file that is Antumbra's
//! whole, beside what `ug` keeps.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{keep, kept, Finding, Ground, Kept, Report};

/// Where Unity Gateway keeps what it writes, in the user's home. It reads no
/// variable for it.
const DIR: &str = ".ucode";

/// The settings file of Antumbra's own that a `ug claude` launch is given,
/// in `~/.antumbra`.
pub const CLAUDE_SETTINGS: &str = "claude-settings.json";

/// What setup says it wrote there.
const CLAUDE_WHAT: &str =
    "Claude Code started by Unity Gateway: settings of Antumbra's own (its hooks, and where the server is)";

fn dir(ground: &Ground<'_>) -> PathBuf {
    ground.home.join(DIR)
}

/// Whether Unity Gateway is on this machine: its program is on the PATH,
/// under either name it goes by, or it has saved a workspace here.
pub fn present(ground: &Ground<'_>) -> bool {
    ["ug", "ucode"]
        .into_iter()
        .any(|name| (ground.find)(name).is_some())
        || dir(ground).join("state.json").is_file()
}

/// The directory `ug gemini` gives Gemini CLI as its home, so its config is
/// in `.gemini` there.
pub fn gemini_home(ground: &Ground<'_>) -> PathBuf {
    dir(ground).join(".gemini-home")
}

/// The directory `ug opencode` gives OpenCode as its XDG config home, so its
/// config is in `opencode` there.
pub fn opencode_config_home(ground: &Ground<'_>) -> PathBuf {
    dir(ground).join("opencode-xdg")
}

/// Whether `path` is `dir` or inside it. Windows names a file in any case.
fn within(path: &Path, dir: &Path) -> bool {
    let same = |a: &std::ffi::OsStr, b: &std::ffi::OsStr| {
        if cfg!(windows) {
            a.eq_ignore_ascii_case(b)
        } else {
            a == b
        }
    };
    let mut path = path.components();
    dir.components().all(|part| {
        path.next()
            .is_some_and(|have| same(have.as_os_str(), part.as_os_str()))
    })
}

/// An environment variable as a shell outside Unity Gateway has it. `ug`
/// starts Gemini CLI and OpenCode with a variable that moves their config
/// into `~/.ucode`, and a setup an agent runs from such a session inherits
/// it. Read as it stands, it would have setup take ug's copy for the agent's
/// own directory, and look for goose's config under OpenCode's. So a
/// `GEMINI_CLI_HOME` or an `XDG_CONFIG_HOME` that names a place inside
/// `~/.ucode` is read as not set: it is ug's, for the agent it started. What
/// the user's own value was, where they had one, the session no longer says.
pub fn outside<'a>(
    home: &'a Path,
    env: &'a dyn Fn(&str) -> Option<String>,
) -> impl Fn(&str) -> Option<String> + 'a {
    move |name| {
        let value = env(name)?;
        let moved = matches!(name, "GEMINI_CLI_HOME" | "XDG_CONFIG_HOME")
            && within(Path::new(value.trim()), &home.join(DIR));
        (!moved).then_some(value)
    }
}

/// How `ug claude` starts Claude Code, as far as what setup wired goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeLaunch {
    /// Over the user's own settings, which are read as usual: the hooks and
    /// the server setup wired are there.
    OverUserSettings,
    /// With the user's own settings left out: the server is handed back, and
    /// the hooks are not.
    WithoutUserSettings,
}

/// The settings file `ug claude` passes Claude Code. It is in `~/.claude`
/// wherever Claude Code's own config directory has been moved to.
pub fn claude_settings(ground: &Ground<'_>) -> PathBuf {
    ground.home.join(".claude").join("ucode-settings.json")
}

/// How `ug claude` starts Claude Code, from the settings it passes
/// ([`claude_settings`]), or nothing when those are not a gateway's.
///
/// A launch that relays a Claude subscription keeps Claude Code's own
/// sign-in as the credential, so `ug` writes no `apiKeyHelper` for it and
/// points Claude Code at a relay on this machine. That is the launch it
/// starts without the user's settings.
pub fn claude_launch(settings: &Value) -> Option<ClaudeLaunch> {
    let gateway = settings
        .pointer("/env/ANTHROPIC_BASE_URL")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|url| !url.is_empty())?;
    let helper = settings
        .get("apiKeyHelper")
        .and_then(Value::as_str)
        .is_some_and(|helper| !helper.trim().is_empty());
    let on_this_machine = gateway
        .split_once("://")
        .is_some_and(|(_, rest)| crate::setup::is_this_machine(crate::setup::host_of(rest)));
    Some(if !helper && on_this_machine {
        ClaudeLaunch::WithoutUserSettings
    } else {
        ClaudeLaunch::OverUserSettings
    })
}

/// How Claude Code is started through `ug` with Antumbra's settings.
fn claude_command(antumbra: &Path) -> String {
    format!(
        "ug claude --settings \"{}\"",
        antumbra.join(CLAUDE_SETTINGS).display()
    )
}

fn claude_files(text: &str) -> [Kept; 1] {
    [Kept {
        path: CLAUDE_SETTINGS,
        text: text.to_string(),
    }]
}

/// Write `text`, Claude Code settings that hold what setup wires and nothing
/// else, into `antumbra` (`~/.antumbra`), and say when a `ug claude` launch
/// needs to be given them.
pub fn connect_claude(
    antumbra: &Path,
    text: &str,
    launch: Option<ClaudeLaunch>,
    report: &dyn Report,
) -> anyhow::Result<()> {
    keep(antumbra, &claude_files(text), CLAUDE_WHAT, report)?;
    match launch {
        Some(ClaudeLaunch::WithoutUserSettings) => report.warn(&format!(
            "`ug claude` relays a Claude subscription here, and starts Claude Code without your own settings, so without Antumbra's hooks. Start it with: {}",
            claude_command(antumbra)
        )),
        _ => report.info(&format!(
            "`ug claude` reads your own settings too, so it needs nothing more. A launch that leaves them out (one that relays a Claude subscription) takes Antumbra's with: {}",
            claude_command(antumbra)
        )),
    }
    Ok(())
}

/// What is there of it, read back, and whether a `ug claude` launch needs it.
pub fn check_claude(antumbra: &Path, text: &str, launch: Option<ClaudeLaunch>) -> Vec<Finding> {
    let mut found = kept(
        antumbra,
        &claude_files(text),
        "Claude Code started by Unity Gateway: settings of Antumbra's own",
    );
    match launch {
        Some(ClaudeLaunch::WithoutUserSettings) => found.push(Finding::waiting(
            "`ug claude` relays a Claude subscription here, and starts Claude Code without your own settings, so without Antumbra's hooks",
            format!("start it with: {}", claude_command(antumbra)),
        )),
        Some(ClaudeLaunch::OverUserSettings) => found.push(Finding::good(
            "`ug claude` starts Claude Code with your own settings read as usual, where Antumbra's hooks are",
        )),
        None => {}
    }
    found
}

#[cfg(test)]
mod tests {
    use serde_json::json;

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
    fn its_program_or_a_saved_workspace_is_unity_gateway_on_this_machine() -> anyhow::Result<()> {
        let home = Scratch::new("gateway-present")?;
        let plain = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        assert!(!present(&plain));
        // Its directory alone does not say so: what it holds is ug's to make.
        std::fs::create_dir_all(home.path().join(".ucode"))?;
        assert!(!present(&plain));
        // Either name it goes by, on the PATH.
        for name in ["ug", "ucode"] {
            let on_path = |asked: &str| (asked == name).then(|| PathBuf::from(asked));
            let installed = Ground {
                home: home.path(),
                antumbra: Program::Named,
                env: &nothing,
                find: &on_path,
            };
            assert!(present(&installed), "{name}");
        }
        // Or the workspace it saved.
        std::fs::write(home.path().join(".ucode").join("state.json"), "{}")?;
        assert!(present(&plain));
        Ok(())
    }

    #[test]
    fn a_variable_ug_set_for_its_agent_is_not_the_users() -> anyhow::Result<()> {
        let home = Scratch::new("gateway-outside")?;
        let ucode = home.path().join(".ucode");
        let theirs = ucode.join(".gemini-home").display().to_string();
        let xdg = ucode.join("opencode-xdg").display().to_string();
        let set = |name: &str| match name {
            "GEMINI_CLI_HOME" => Some(theirs.clone()),
            "XDG_CONFIG_HOME" => Some(format!(" {xdg} ")),
            // Another variable that names a place there is nobody's to drop.
            "CODEX_HOME" => Some(theirs.clone()),
            _ => None,
        };
        let env = outside(home.path(), &set);
        assert_eq!(env("GEMINI_CLI_HOME"), None);
        assert_eq!(env("XDG_CONFIG_HOME"), None);
        assert_eq!(env("CODEX_HOME"), Some(theirs.clone()));
        assert_eq!(env("OPENCODE_CONFIG_DIR"), None);

        // One the user set, to a place of their own, stands.
        let own = home.path().join("elsewhere").display().to_string();
        let set = |name: &str| (name == "GEMINI_CLI_HOME").then(|| own.clone());
        let env = outside(home.path(), &set);
        assert_eq!(env("GEMINI_CLI_HOME"), Some(own.clone()));
        // A directory that only starts with ug's name is not inside it.
        let beside = home.path().join(".ucode-old").display().to_string();
        let set = |name: &str| (name == "XDG_CONFIG_HOME").then(|| beside.clone());
        let env = outside(home.path(), &set);
        assert_eq!(env("XDG_CONFIG_HOME"), Some(beside.clone()));
        Ok(())
    }

    #[test]
    fn a_relayed_subscription_is_the_launch_without_the_users_settings() {
        // The gateway itself, with the helper that signs in to it.
        let direct = json!({
            "apiKeyHelper": "ug auth-token --host https://workspace.example",
            "env": { "ANTHROPIC_BASE_URL": "https://workspace.example/ai-gateway/anthropic" }
        });
        assert_eq!(claude_launch(&direct), Some(ClaudeLaunch::OverUserSettings));
        // A relay on this machine, and Claude Code's own sign-in.
        for relay in [
            "http://127.0.0.1:51234",
            "http://localhost:51234/",
            "http://[::1]:51234",
        ] {
            let relayed = json!({ "env": { "ANTHROPIC_BASE_URL": relay } });
            assert_eq!(
                claude_launch(&relayed),
                Some(ClaudeLaunch::WithoutUserSettings),
                "{relay}"
            );
            let blank = json!({ "apiKeyHelper": " ", "env": { "ANTHROPIC_BASE_URL": relay } });
            assert_eq!(
                claude_launch(&blank),
                Some(ClaudeLaunch::WithoutUserSettings),
                "{relay}"
            );
        }
        // A proxy on this machine that a helper signs in to is not a relay.
        let proxied = json!({
            "apiKeyHelper": "ug auth-token",
            "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:51234" }
        });
        assert_eq!(
            claude_launch(&proxied),
            Some(ClaudeLaunch::OverUserSettings)
        );
        // Settings that name no gateway are not ug's for one.
        for other in [json!({}), json!({ "env": {} }), Value::Null] {
            assert_eq!(claude_launch(&other), None, "{other}");
        }
    }

    #[test]
    fn the_settings_are_written_whole_and_the_relayed_launch_is_said() -> anyhow::Result<()> {
        let home = Scratch::new("gateway-claude")?;
        let antumbra = home.path().join(".antumbra");
        let text = "{\n  \"hooks\": {}\n}\n";
        let file = antumbra.join(CLAUDE_SETTINGS);
        let command = format!("ug claude --settings \"{}\"", file.display());

        // A dry run says what it would do and does none of it.
        let dry = Told {
            dry_run: true,
            ..Told::default()
        };
        connect_claude(&antumbra, text, None, &dry)?;
        assert!(
            dry.said.borrow()[0].starts_with("would write"),
            "{:?}",
            dry.said
        );
        assert!(!file.exists());
        assert!(check_claude(&antumbra, text, None)
            .iter()
            .any(|f| !f.works && f.said.contains("claude-settings.json missing")));

        // A launch over the user's settings needs nothing: it is said how the
        // other kind is given them.
        let told = Told::default();
        connect_claude(&antumbra, text, Some(ClaudeLaunch::OverUserSettings), &told)?;
        assert_eq!(std::fs::read_to_string(&file)?, text);
        let said = told.said.borrow().clone();
        assert!(said[0].contains("written to"), "{said:?}");
        assert!(
            said[1].starts_with("info ") && said[1].contains(&command),
            "{said:?}"
        );
        let found = check_claude(&antumbra, text, Some(ClaudeLaunch::OverUserSettings));
        assert!(found.iter().all(|f| f.fix.is_none()), "{found:?}");

        // The relayed launch is a warning, with the command, and the check
        // says it without failing: nothing setup writes changes it.
        let again = Told::default();
        connect_claude(
            &antumbra,
            text,
            Some(ClaudeLaunch::WithoutUserSettings),
            &again,
        )?;
        let said = again.said.borrow().clone();
        assert!(said[0].contains("already in"), "{said:?}");
        assert!(
            said[1].starts_with("warn ") && said[1].contains(&command),
            "{said:?}"
        );
        let found = check_claude(&antumbra, text, Some(ClaudeLaunch::WithoutUserSettings));
        assert!(
            found.iter().any(|f| f.works
                && f.said.contains("without your own settings")
                && f.fix.as_deref().is_some_and(|fix| fix.contains(&command))),
            "{found:?}"
        );

        // Settings this build would write differently are said, and work.
        let found = check_claude(&antumbra, "{}\n", None);
        assert!(
            found
                .iter()
                .any(|f| f.works && f.fix.is_some() && f.said.contains("differs")),
            "{found:?}"
        );
        Ok(())
    }
}
