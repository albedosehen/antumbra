//! goose: a plugin of Antumbra's own, and a file goose keeps in mind.
//!
//! goose loads plugins from `~/.agents/plugins`, each a directory that may
//! hold hooks (`hooks/hooks.json`) and MCP servers (`.mcp.json`). So Antumbra
//! is a plugin there, `antumbra`, with the bridge as its server and three
//! hooks, and nothing of the user's is edited to connect it.
//!
//! goose's hooks differ from the other agents' in one thing that shapes the
//! rest: **what a hook prints never reaches the model.** A prompt hook's
//! output is dropped, and a session-start hook can only show its user a
//! line. What goose does read on every turn is a file its config names,
//! `GOOSE_MOIM_MESSAGE_FILE`, which it puts in front of the model as what is
//! on its mind. So:
//!
//! - **The hooks write that file.** At a session's start the bootstrap is
//!   kept for the session; for each prompt the file becomes that bootstrap
//!   and what was recalled for the prompt. goose runs a prompt's hook before
//!   the turn, so the model reads what belongs to the prompt in front of it.
//! - **Setup names the file in goose's `config.yaml`**, by adding one line,
//!   and only when the config names none: a file a person chose is theirs,
//!   and setup then says recall is left out.
//! - **A turn's end** is the one place a hook's answer is read: a block, with
//!   the reminder, which goose hands the agent before it may stop.
//!
//! The file is one for the machine, not one a session: two goose sessions at
//! once each write it, and a turn may read what the other's prompt recalled.
//! It is the user's own memory either way, and the next prompt rewrites it.
//!
//! goose runs a hook's command with `sh -c` on every platform, Windows too,
//! where that needs an `sh` on the PATH (Git for Windows has one).

use std::path::{Path, PathBuf};

use serde_json::json;

use super::hook::{Event, Said};
use super::{keep, kept, Agent, Finding, Ground, Kept, Program, Report};

/// The plugin's name, which is also its directory's and its server's.
pub const PLUGIN: &str = "antumbra";

/// The line in goose's config that names the file it keeps in mind.
const KEY: &str = "GOOSE_MOIM_MESSAGE_FILE";

/// The value of an environment variable that is set to something.
fn named(ground: &Ground<'_>, name: &str) -> Option<PathBuf> {
    (ground.env)(name)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Where goose keeps its config. Under `GOOSE_PATH_ROOT` when that is set;
/// else `%APPDATA%\Block\goose\config` on Windows, and `goose` in the XDG
/// config directory (`~/.config`) everywhere else.
pub fn config_dir(ground: &Ground<'_>) -> PathBuf {
    if let Some(root) = named(ground, "GOOSE_PATH_ROOT") {
        return root.join("config");
    }
    if cfg!(windows) {
        named(ground, "APPDATA")
            .unwrap_or_else(|| ground.home.join("AppData").join("Roaming"))
            .join("Block")
            .join("goose")
            .join("config")
    } else {
        named(ground, "XDG_CONFIG_HOME")
            .unwrap_or_else(|| ground.home.join(".config"))
            .join("goose")
    }
}

/// Where goose looks for a user's plugins: `.agents/plugins` in the home, or
/// under `GOOSE_PATH_ROOT`.
pub fn plugins_dir(ground: &Ground<'_>) -> PathBuf {
    named(ground, "GOOSE_PATH_ROOT")
        .unwrap_or_else(|| ground.home.to_path_buf())
        .join(".agents")
        .join("plugins")
}

/// Where the hooks keep what goose is to have in mind.
fn mind_dir(home: &Path) -> PathBuf {
    home.join(".antumbra").join("goose")
}

/// The file goose's config is pointed at.
pub fn top_of_mind(home: &Path) -> PathBuf {
    mind_dir(home).join("top-of-mind.md")
}

/// `antumbra` as `sh -c` runs it, on every platform. A Windows path is
/// written with forward slashes, which Windows takes and `sh` does not read
/// as escapes.
fn in_sh(program: &Program) -> String {
    match program {
        Program::Named => "antumbra".to_string(),
        Program::At(path) => format!("\"{}\"", path.display().to_string().replace('\\', "/")),
    }
}

/// Each hook: goose's event, the moment, and how long goose waits, in seconds.
const HOOKS: [(&str, Event, u32); 3] = [
    ("SessionStart", Event::SessionStart, 15),
    ("UserPromptSubmit", Event::Prompt, 25),
    ("Stop", Event::Stop, 15),
];

/// The plugin's files.
pub fn files(ground: &Ground<'_>) -> Vec<Kept> {
    let pretty = |value: &serde_json::Value| {
        // A value built here always prints; the fallback is its compact form.
        let mut text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
        text.push('\n');
        text
    };
    let mut events = serde_json::Map::new();
    for (theirs, event, timeout) in HOOKS {
        let command = format!(
            "{} hook {} {}",
            in_sh(&ground.antumbra),
            Agent::Goose.name(),
            event.word()
        );
        events.insert(
            theirs.to_string(),
            json!([{ "hooks": [{ "type": "command", "command": command, "timeout": timeout }] }]),
        );
    }
    vec![
        Kept {
            path: "plugin.json",
            text: pretty(&json!({
                "name": PLUGIN,
                "version": env!("CARGO_PKG_VERSION"),
                "description": "Antumbra: a memory that carries across sessions, agents and machines. Written by `antumbra setup`; run it again to update.",
            })),
        },
        Kept {
            path: ".mcp.json",
            text: pretty(&json!({
                "mcpServers": {
                    PLUGIN: {
                        "command": ground.antumbra.file(),
                        "args": ["mcp", "bridge", "--agent", Agent::Goose.name()],
                    }
                }
            })),
        },
        Kept {
            path: "hooks/hooks.json",
            text: pretty(&json!({ "hooks": events })),
        },
    ]
}

/// The file `config` (goose's `config.yaml`) names to keep in mind, when it
/// names one: the value of its top-level [`KEY`], without its quotes.
fn kept_in_mind(config: &str) -> Option<String> {
    config.lines().find_map(|line| {
        let value = line
            .strip_prefix(KEY)?
            .trim_start()
            .strip_prefix(':')?
            .trim();
        let unquoted = value
            .strip_prefix('\'')
            .and_then(|v| v.strip_suffix('\''))
            .map(|v| v.replace("''", "'"))
            .or_else(|| {
                value
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .map(|v| v.replace("\\\\", "\\"))
            })
            .unwrap_or_else(|| value.to_string());
        Some(unquoted).filter(|v| !v.is_empty())
    })
}

/// `config` with one more line, naming `file` as what goose keeps in mind.
/// The value is single-quoted, where YAML reads a backslash as itself.
fn naming(config: &str, file: &Path) -> String {
    let mut out = config.to_string();
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&format!(
        "{KEY}: '{}'\n",
        file.display().to_string().replace('\'', "''")
    ));
    out
}

/// Whether a line can be added to `config` as it is laid out: empty, or a
/// mapping written a key a line, which is how goose writes it.
fn takes_a_line(config: &str) -> bool {
    let first = config
        .lines()
        .map(str::trim_end)
        .find(|line| !line.trim_start().is_empty() && !line.trim_start().starts_with('#'));
    match first {
        None => true,
        Some(line) => {
            !line.starts_with(['{', '[', '-', ' ', '\t'])
                && line.split_once(':').is_some_and(|(key, _)| !key.is_empty())
        }
    }
}

pub fn connect(ground: &Ground<'_>, report: &dyn Report) -> anyhow::Result<()> {
    keep(
        &plugins_dir(ground).join(PLUGIN),
        &files(ground),
        "goose: the antumbra plugin (the MCP bridge and three hooks)",
        report,
    )?;

    // The file goose keeps in mind, named in its config when it names none.
    let ours = top_of_mind(ground.home);
    let path = config_dir(ground).join("config.yaml");
    let config = std::fs::read_to_string(&path).unwrap_or_default();
    match kept_in_mind(&config) {
        Some(theirs) if Path::new(&theirs) == ours => report.ok(&format!(
            "goose: {} already names Antumbra's file as what it keeps in mind",
            path.display()
        )),
        Some(theirs) => report.warn(&format!(
            "goose already keeps a file of yours in mind ({theirs}), so what Antumbra recalls for a session and a prompt is left out of goose's context; its tools and the reminder at a turn's end work. To have both, remove {KEY} from {} and run setup again",
            path.display()
        )),
        None if !takes_a_line(&config) => {
            report.warn(&format!(
                "{} is not laid out a key a line, so it was left as it is. For what Antumbra recalls to reach goose, add:",
                path.display()
            ));
            report.info(naming("", &ours).trim_end());
        }
        None if report.dry_run() => report.would(&format!(
            "name Antumbra's file in {} as what goose keeps in mind ({KEY})",
            path.display()
        )),
        None => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if path.exists() {
                let backup = path.with_file_name(format!(
                    "config.yaml.antumbra-backup-{}",
                    chrono::Utc::now().format("%Y%m%d%H%M%S")
                ));
                std::fs::copy(&path, &backup)?;
                report.info(&format!("backed up goose's config to {}", backup.display()));
            }
            std::fs::write(&path, naming(&config, &ours))?;
            report.ok(&format!(
                "goose: {} names Antumbra's file as what it keeps in mind ({KEY})",
                path.display()
            ));
        }
    }
    if cfg!(windows) && (ground.find)("sh").is_none() {
        report.warn(
            "goose runs its hooks with sh, and there is no sh on your PATH: its hooks will not run until there is (Git for Windows has one). Its tools work either way.",
        );
    }
    Ok(())
}

pub fn check(ground: &Ground<'_>) -> Vec<Finding> {
    let mut found = kept(
        &plugins_dir(ground).join(PLUGIN),
        &files(ground),
        "goose: the antumbra plugin",
    );
    let ours = top_of_mind(ground.home);
    let path = config_dir(ground).join("config.yaml");
    match kept_in_mind(&std::fs::read_to_string(&path).unwrap_or_default()) {
        Some(theirs) if Path::new(&theirs) == ours => found.push(Finding::good(
            "goose: its config names Antumbra's file as what it keeps in mind",
        )),
        Some(theirs) => found.push(Finding::good(format!(
            "goose: it keeps a file of yours in mind ({theirs}), so what Antumbra recalls is left out of its context"
        ))),
        None => found.push(Finding::bad(
            format!(
                "goose: {} does not name Antumbra's file ({KEY}), so what is recalled does not reach it",
                path.display()
            ),
            "run setup again to add the line",
        )),
    }
    if cfg!(windows) && (ground.find)("sh").is_none() {
        found.push(Finding::bad(
            "goose: there is no sh on your PATH, and goose runs its hooks with it",
            "install Git for Windows, or put another sh on the PATH",
        ));
    }
    found
}

/// A session's id as part of a file's name: its letters, digits, `_` and
/// `-`, so an id carrying a path cannot name a file anywhere else.
fn file_name_of(session: Option<&str>) -> String {
    let safe: String = session
        .unwrap_or_default()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        .take(80)
        .collect();
    if safe.is_empty() {
        "default".to_string()
    } else {
        safe
    }
}

/// How long a session's bootstrap is kept after it was written.
const KEPT_FOR: std::time::Duration = std::time::Duration::from_secs(7 * 24 * 3600);

/// Write `text` to `path` whole: beside it first, then in its place, so goose
/// never reads half of it.
fn write_whole(path: &Path, text: &str) -> std::io::Result<()> {
    let beside = path.with_extension("md.writing");
    std::fs::write(&beside, text)?;
    std::fs::rename(&beside, path)
}

/// Keep what a hook's script said where goose will read it. At a session's
/// start the bootstrap is kept for the session, and bootstraps a week old
/// are cleared away. For a prompt, the file goose keeps in mind becomes the
/// session's bootstrap and what was recalled for the prompt, or is emptied
/// when there is neither, so nothing from an earlier prompt is read as this
/// one's.
pub fn keep_in_mind(
    home: &Path,
    event: Event,
    session: Option<&str>,
    said: &Said,
) -> std::io::Result<()> {
    let dir = mind_dir(home);
    let start = dir.join(format!("start-{}.md", file_name_of(session)));
    let context = match said {
        Said::Context(text) => text.as_str(),
        _ => "",
    };
    match event {
        Event::SessionStart => {
            std::fs::create_dir_all(&dir)?;
            for entry in std::fs::read_dir(&dir)?.flatten() {
                let old = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|at| at.elapsed().ok())
                    .is_some_and(|age| age > KEPT_FOR);
                let ours = entry.file_name().to_string_lossy().starts_with("start-");
                if old && ours {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
            std::fs::write(&start, context)?;
            write_whole(&top_of_mind(home), context)
        }
        Event::Prompt => {
            std::fs::create_dir_all(&dir)?;
            let opened = std::fs::read_to_string(&start).unwrap_or_default();
            let parts: Vec<&str> = [opened.trim(), context.trim()]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect();
            write_whole(&top_of_mind(home), &parts.join("\n\n"))
        }
        Event::Stop => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::agents::scratch::Scratch;
    use crate::agents::tests::Told;

    fn nowhere(_: &str) -> Option<PathBuf> {
        None
    }

    fn sh(name: &str) -> Option<PathBuf> {
        (name == "sh").then(|| PathBuf::from("/usr/bin/sh"))
    }

    /// A ground whose goose lives under `root`, as `GOOSE_PATH_ROOT` puts it.
    fn rooted<'a>(
        home: &'a Path,
        env: &'a dyn Fn(&str) -> Option<String>,
        find: &'a dyn Fn(&str) -> Option<PathBuf>,
    ) -> Ground<'a> {
        Ground {
            home,
            antumbra: Program::Named,
            env,
            find,
        }
    }

    #[test]
    fn the_plugin_carries_the_bridge_and_the_three_hooks() -> anyhow::Result<()> {
        let home = Scratch::new("goose-files")?;
        let none = |_: &str| None;
        let at = home.path().join("my tools").join("antumbra");
        let ground = Ground {
            home: home.path(),
            antumbra: Program::At(at.clone()),
            env: &none,
            find: &nowhere,
        };
        let files = files(&ground);
        let paths: Vec<&str> = files.iter().map(|f| f.path).collect();
        assert_eq!(paths, vec!["plugin.json", ".mcp.json", "hooks/hooks.json"]);
        let servers: Value = serde_json::from_str(&files[1].text)?;
        assert_eq!(
            servers["mcpServers"]["antumbra"],
            json!({ "command": at.display().to_string(), "args": ["mcp", "bridge", "--agent", "goose"] })
        );
        let hooks: Value = serde_json::from_str(&files[2].text)?;
        // `sh -c` on every platform: quoted, with forward slashes.
        let quoted = format!("\"{}\"", at.display().to_string().replace('\\', "/"));
        for (event, word) in [
            ("SessionStart", "session-start"),
            ("UserPromptSubmit", "prompt"),
            ("Stop", "stop"),
        ] {
            let hook = &hooks["hooks"][event][0]["hooks"][0];
            assert_eq!(hook["command"], format!("{quoted} hook goose {word}"));
            assert!(!hook["command"].as_str().unwrap_or_default().contains('\\'));
        }
        Ok(())
    }

    #[test]
    fn goose_is_found_where_each_platform_and_a_path_root_put_it() -> anyhow::Result<()> {
        let home = Scratch::new("goose-paths")?;
        let root = home.path().join("root");
        let named = root.display().to_string();
        let env = move |name: &str| (name == "GOOSE_PATH_ROOT").then(|| named.clone());
        let ground = rooted(home.path(), &env, &nowhere);
        assert_eq!(config_dir(&ground), root.join("config"));
        assert_eq!(plugins_dir(&ground), root.join(".agents").join("plugins"));

        let none = |_: &str| None;
        let plain = rooted(home.path(), &none, &nowhere);
        assert_eq!(
            plugins_dir(&plain),
            home.path().join(".agents").join("plugins")
        );
        let config = config_dir(&plain);
        if cfg!(windows) {
            assert!(config.ends_with(Path::new("Block").join("goose").join("config")));
        } else {
            assert_eq!(config, home.path().join(".config").join("goose"));
        }
        Ok(())
    }

    #[test]
    fn the_file_goose_keeps_in_mind_is_read_from_its_config() {
        assert_eq!(kept_in_mind(""), None);
        assert_eq!(kept_in_mind("GOOSE_PROVIDER: openai\n"), None);
        assert_eq!(
            kept_in_mind("GOOSE_PROVIDER: openai\nGOOSE_MOIM_MESSAGE_FILE: /home/me/notes.md\n")
                .as_deref(),
            Some("/home/me/notes.md")
        );
        assert_eq!(
            kept_in_mind("GOOSE_MOIM_MESSAGE_FILE: 'C:\\Users\\me\\it''s.md'\n").as_deref(),
            Some("C:\\Users\\me\\it's.md")
        );
        assert_eq!(
            kept_in_mind("GOOSE_MOIM_MESSAGE_FILE: \"C:\\\\Users\\\\me\\\\n.md\"\n").as_deref(),
            Some("C:\\Users\\me\\n.md")
        );
        // The name inside another mapping is not the setting.
        assert_eq!(
            kept_in_mind("extensions:\n  GOOSE_MOIM_MESSAGE_FILE: nested\n"),
            None
        );
        // What is written is read back as it was.
        let path = Path::new("C:\\Users\\me o'brien\\.antumbra\\goose\\top-of-mind.md");
        let written = naming("GOOSE_PROVIDER: openai", path);
        assert!(written.starts_with("GOOSE_PROVIDER: openai\nGOOSE_MOIM_MESSAGE_FILE: '"));
        assert_eq!(
            kept_in_mind(&written).as_deref(),
            Some(path.display().to_string().as_str())
        );
        assert!(takes_a_line(""));
        assert!(takes_a_line(
            "# goose\n\nGOOSE_PROVIDER: openai\nextensions:\n  x:\n    enabled: true\n"
        ));
        assert!(!takes_a_line("{GOOSE_PROVIDER: openai}\n"));
        assert!(!takes_a_line("- a list\n"));
    }

    #[test]
    fn connecting_writes_the_plugin_and_one_line_of_config() -> anyhow::Result<()> {
        let home = Scratch::new("goose-connect")?;
        let root = home.path().join("root");
        let named = root.display().to_string();
        let env = move |name: &str| (name == "GOOSE_PATH_ROOT").then(|| named.clone());
        let ground = rooted(home.path(), &env, &sh);
        let config = root.join("config").join("config.yaml");
        std::fs::create_dir_all(root.join("config"))?;
        let theirs = "GOOSE_PROVIDER: openai\nextensions:\n  developer:\n    enabled: true\n";
        std::fs::write(&config, theirs)?;

        let dry = Told {
            dry_run: true,
            ..Told::default()
        };
        connect(&ground, &dry)?;
        assert_eq!(std::fs::read_to_string(&config)?, theirs);
        assert!(!root.join(".agents").exists());

        let told = Told::default();
        connect(&ground, &told)?;
        let plugin = root.join(".agents").join("plugins").join("antumbra");
        assert!(plugin.join(".mcp.json").is_file());
        assert!(plugin.join("hooks").join("hooks.json").is_file());
        let after = std::fs::read_to_string(&config)?;
        // Their config, then the one line.
        assert!(after.starts_with(theirs), "{after}");
        assert_eq!(after.lines().count(), theirs.lines().count() + 1);
        assert_eq!(
            kept_in_mind(&after).map(PathBuf::from),
            Some(top_of_mind(home.path()))
        );
        let said = told.said.borrow().join("\n");
        assert!(said.contains("backed up goose's config"), "{said}");
        assert!(
            check(&ground).iter().all(|f| f.fix.is_none()),
            "{:?}",
            check(&ground)
        );

        let again = Told::default();
        connect(&ground, &again)?;
        assert_eq!(std::fs::read_to_string(&config)?, after);
        let said = again.said.borrow().join("\n");
        assert!(said.contains("already names Antumbra's file"), "{said}");
        Ok(())
    }

    #[test]
    fn a_file_the_person_already_keeps_in_mind_is_left_theirs() -> anyhow::Result<()> {
        let home = Scratch::new("goose-theirs")?;
        let root = home.path().join("root");
        let named = root.display().to_string();
        let env = move |name: &str| (name == "GOOSE_PATH_ROOT").then(|| named.clone());
        let ground = rooted(home.path(), &env, &sh);
        std::fs::create_dir_all(root.join("config"))?;
        let theirs = "GOOSE_MOIM_MESSAGE_FILE: /home/me/notes.md\n";
        std::fs::write(root.join("config").join("config.yaml"), theirs)?;
        let told = Told::default();
        connect(&ground, &told)?;
        assert_eq!(
            std::fs::read_to_string(root.join("config").join("config.yaml"))?,
            theirs
        );
        let said = told.said.borrow().join("\n");
        assert!(
            said.contains("warn goose already keeps a file of yours in mind (/home/me/notes.md)"),
            "{said}"
        );
        // It is their choice, and the check does not call it a fault.
        assert!(check(&ground).iter().all(|f| f.fix.is_none()));
        Ok(())
    }

    #[test]
    fn what_was_recalled_is_kept_where_goose_reads_it() -> anyhow::Result<()> {
        let home = Scratch::new("goose-mind")?;
        let file = top_of_mind(home.path());
        let bootstrap = Said::Context("# Antumbra session bootstrap\n\nBuilds with bun.".into());
        keep_in_mind(
            home.path(),
            Event::SessionStart,
            Some("20261118_1"),
            &bootstrap,
        )?;
        assert_eq!(
            std::fs::read_to_string(&file)?,
            "# Antumbra session bootstrap\n\nBuilds with bun."
        );

        let recalled =
            Said::Context("## Antumbra recall\n\n- deploys on Fridays are frozen".into());
        keep_in_mind(home.path(), Event::Prompt, Some("20261118_1"), &recalled)?;
        assert_eq!(
            std::fs::read_to_string(&file)?,
            "# Antumbra session bootstrap\n\nBuilds with bun.\n\n## Antumbra recall\n\n- deploys on Fridays are frozen"
        );

        // A prompt nothing was recalled for keeps the bootstrap, and none of
        // the last prompt's recall.
        keep_in_mind(
            home.path(),
            Event::Prompt,
            Some("20261118_1"),
            &Said::Nothing,
        )?;
        assert_eq!(
            std::fs::read_to_string(&file)?,
            "# Antumbra session bootstrap\n\nBuilds with bun."
        );

        // Another session has its own bootstrap, or none.
        keep_in_mind(home.path(), Event::Prompt, Some("20261118_2"), &recalled)?;
        assert_eq!(
            std::fs::read_to_string(&file)?,
            "## Antumbra recall\n\n- deploys on Fridays are frozen"
        );
        // A turn's end changes nothing here.
        keep_in_mind(
            home.path(),
            Event::Stop,
            Some("20261118_2"),
            &Said::Reminder("x".into()),
        )?;
        assert!(std::fs::read_to_string(&file)?.contains("deploys on Fridays"));
        // An id that carries a path names a file in the same place.
        assert_eq!(file_name_of(Some("../../etc/passwd")), "etcpasswd");
        assert_eq!(file_name_of(Some("")), "default");
        assert_eq!(file_name_of(None), "default");
        Ok(())
    }
}
