//! Coding agents other than Claude Code.
//!
//! Claude Code is given the server's address, a helper that reads the token
//! for each connection, and hook scripts with the address in its settings'
//! `env`. No other agent has all three, and each has its own config file and
//! its own shape for a hook's input and answer. What they share is here: where
//! the server is and how to reach it, as `antumbra setup` left it
//! ([`Reach`]); the bridge that carries an agent's MCP connection there
//! ([`bridge`]); and the command that answers an agent's hook with the hook
//! scripts Claude Code runs ([`hook`]).
//!
//! What differs is one module an agent: where its config lives, and what
//! Antumbra writes there. Where an agent reads a directory or a file of
//! Antumbra's own, that is all its module writes: connecting again replaces
//! it whole, and nothing a person wrote is edited ([`gemini`], [`opencode`],
//! [`goose`]'s plugin, [`copilot`]'s hooks). Where an agent reads only its
//! user's files, its module adds to them as setup adds to Claude Code's
//! settings: with the agent's own command, or as text, adding only, with a
//! backup, and never twice ([`codex`], [`copilot`]'s servers, one line of
//! [`goose`]'s config). `antumbra setup` connects every agent it finds
//! ([`connect`]), and `antumbra setup check` reads back what it wrote
//! ([`check`]).

use std::path::{Path, PathBuf};

use serde_json::Value;

pub mod bridge;
pub mod codex;
pub mod copilot;
pub mod gemini;
pub mod goose;
pub mod hook;
pub mod opencode;

/// A coding agent `antumbra setup` connects besides Claude Code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Agent {
    /// Gemini CLI.
    Gemini,
    /// Codex.
    Codex,
    /// GitHub Copilot CLI, and Copilot in VS Code, which reads the same files.
    Copilot,
    /// goose.
    Goose,
    /// OpenCode, 1.x and 2.x.
    Opencode,
}

impl Agent {
    /// Every agent, in the order setup connects them.
    pub const ALL: [Agent; 5] = [
        Agent::Gemini,
        Agent::Codex,
        Agent::Copilot,
        Agent::Goose,
        Agent::Opencode,
    ];

    /// The name it is chosen by (`--agent`), recorded under, and named to the
    /// server with.
    pub fn name(self) -> &'static str {
        match self {
            Agent::Gemini => "gemini",
            Agent::Codex => "codex",
            Agent::Copilot => "copilot",
            Agent::Goose => "goose",
            Agent::Opencode => "opencode",
        }
    }

    /// What a person calls it.
    pub fn title(self) -> &'static str {
        match self {
            Agent::Gemini => "Gemini CLI",
            Agent::Codex => "Codex",
            Agent::Copilot => "Copilot CLI",
            Agent::Goose => "goose",
            Agent::Opencode => "OpenCode",
        }
    }

    /// The program a person starts it with, which is how setup finds it.
    pub fn program(self) -> &'static str {
        match self {
            Agent::Gemini => "gemini",
            Agent::Codex => "codex",
            Agent::Copilot => "copilot",
            Agent::Goose => "goose",
            Agent::Opencode => "opencode",
        }
    }

    /// The agent `name` chooses, in any case.
    pub fn named(name: &str) -> Option<Agent> {
        Agent::ALL
            .into_iter()
            .find(|agent| agent.name().eq_ignore_ascii_case(name.trim()))
    }
}

/// The shell an agent runs a hook's command in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shell {
    /// `sh -c`, `bash -c`.
    Posix,
    /// `powershell -Command`, `pwsh -Command`.
    PowerShell,
}

/// How something outside this process runs `antumbra`: by its name where that
/// is on the PATH, which survives an upgrade that moves it, else by where it
/// is now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Program {
    Named,
    At(PathBuf),
}

impl Program {
    /// This program, given whether its name is on the PATH.
    pub fn this(on_path: bool) -> Self {
        if on_path {
            return Program::Named;
        }
        match std::env::current_exe() {
            Ok(path) => Program::At(path),
            Err(_) => Program::Named,
        }
    }

    /// For a config that takes a program apart from its arguments.
    pub fn file(&self) -> String {
        match self {
            Program::Named => "antumbra".to_string(),
            Program::At(path) => path.display().to_string(),
        }
    }

    /// For a command line `shell` reads. A path is quoted, since it may hold
    /// a space, and PowerShell runs a quoted path only after `&`.
    pub fn line(&self, shell: Shell) -> String {
        match (self, shell) {
            (Program::Named, _) => "antumbra".to_string(),
            (Program::At(path), Shell::Posix) => format!("\"{}\"", path.display()),
            (Program::At(path), Shell::PowerShell) => format!("& \"{}\"", path.display()),
        }
    }
}

/// What connecting an agent, or checking one, stands on.
pub struct Ground<'a> {
    /// The user's home directory.
    pub home: &'a Path,
    /// How the agent runs `antumbra`.
    pub antumbra: Program,
    /// The value of an environment variable, or nothing: an agent's own
    /// variable for where its config lives.
    pub env: &'a dyn Fn(&str) -> Option<String>,
    /// Where a program is on the PATH, or nothing: how an agent is found,
    /// and how its own command is run.
    pub find: &'a dyn Fn(&str) -> Option<PathBuf>,
}

/// Where connecting an agent says what it did, as setup's other steps do.
pub trait Report {
    /// Whether this run changes nothing and only says what it would.
    fn dry_run(&self) -> bool;
    fn ok(&self, said: &str);
    fn would(&self, said: &str);
    fn warn(&self, said: &str);
    fn info(&self, said: &str);
}

/// One thing `antumbra setup check` found about an agent's connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// What was found.
    pub said: String,
    /// What to do about it, when it is not as it should be.
    pub fix: Option<String>,
}

impl Finding {
    fn good(said: impl Into<String>) -> Self {
        Finding {
            said: said.into(),
            fix: None,
        }
    }

    fn bad(said: impl Into<String>, fix: impl Into<String>) -> Self {
        Finding {
            said: said.into(),
            fix: Some(fix.into()),
        }
    }
}

/// One file Antumbra keeps in an agent's config: where, relative to the
/// directory the agent gives it, and its whole text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kept {
    pub path: &'static str,
    pub text: String,
}

/// Write `files` under `dir`, each whole, and say so once. A file already
/// holding its text is left as it is, so its time does not move.
fn keep(dir: &Path, files: &[Kept], what: &str, report: &dyn Report) -> anyhow::Result<()> {
    let differs = |file: &Kept| {
        std::fs::read_to_string(dir.join(file.path))
            .map(|have| have.replace("\r\n", "\n") != file.text)
            .unwrap_or(true)
    };
    if !files.iter().any(differs) {
        report.ok(&format!("{what} already in {}", dir.display()));
        return Ok(());
    }
    if report.dry_run() {
        report.would(&format!("write {what} to {}", dir.display()));
        return Ok(());
    }
    for file in files {
        let path = dir.join(file.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &file.text)?;
    }
    report.ok(&format!("{what} written to {}", dir.display()));
    Ok(())
}

/// What `files` under `dir` are, against what this build writes: each one
/// there and the same, or what to do about it.
fn kept(dir: &Path, files: &[Kept], what: &str) -> Vec<Finding> {
    let mut found = Vec::new();
    let mut missing = Vec::new();
    let mut changed = Vec::new();
    for file in files {
        match std::fs::read_to_string(dir.join(file.path)) {
            Ok(have) if have.replace("\r\n", "\n") == file.text => {}
            Ok(_) => changed.push(file.path),
            Err(_) => missing.push(file.path),
        }
    }
    if missing.is_empty() && changed.is_empty() {
        found.push(Finding::good(format!("{what} in {}", dir.display())));
    }
    if !missing.is_empty() {
        found.push(Finding::bad(
            format!(
                "{what}: {} missing from {}",
                missing.join(", "),
                dir.display()
            ),
            "run setup again to write it",
        ));
    }
    if !changed.is_empty() {
        found.push(Finding::bad(
            format!(
                "{what}: {} in {} differs from what this build writes",
                changed.join(", "),
                dir.display()
            ),
            "run setup again to write this build's",
        ));
    }
    found
}

/// Whether `agent` is on this machine: its program is on the PATH, or its
/// config directory is there.
pub fn present(agent: Agent, ground: &Ground<'_>) -> bool {
    (ground.find)(agent.program()).is_some()
        || match agent {
            Agent::Gemini => gemini::home(ground).is_dir(),
            Agent::Codex => codex::home(ground).is_dir(),
            Agent::Copilot => copilot::home(ground).is_dir(),
            Agent::Goose => goose::config_dir(ground).is_dir(),
            Agent::Opencode => opencode::config_dir(ground).is_dir(),
        }
}

/// Connect `agent`: write what it needs to reach the server through the
/// bridge and to run its hooks through `antumbra hook`.
pub fn connect(agent: Agent, ground: &Ground<'_>, report: &dyn Report) -> anyhow::Result<()> {
    match agent {
        Agent::Gemini => gemini::connect(ground, report),
        Agent::Codex => codex::connect(ground, report),
        Agent::Copilot => copilot::connect(ground, report),
        Agent::Goose => goose::connect(ground, report),
        Agent::Opencode => opencode::connect(ground, report),
    }
}

/// What is there of `agent`'s connection, read back.
pub fn check(agent: Agent, ground: &Ground<'_>) -> Vec<Finding> {
    match agent {
        Agent::Gemini => gemini::check(ground),
        Agent::Codex => codex::check(ground),
        Agent::Copilot => copilot::check(ground),
        Agent::Goose => goose::check(ground),
        Agent::Opencode => opencode::check(ground),
    }
}

/// How a command an agent runs reaches the server: what `antumbra setup`
/// recorded in `~/.antumbra/setup.json`, with the environment's `ANTUMBRA_*`
/// names winning over it, as they do in the hook scripts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reach {
    /// The server's address, without a trailing slash or `/mcp`.
    pub url: String,
    /// The workspace the token grants, where setup recorded it.
    pub workspace: Option<String>,
    /// The file the token is read from, each time it is needed.
    pub token_file: PathBuf,
    /// The token the environment held when this was read. It wins over the
    /// file's for as long as the process runs, as it does in the hook scripts.
    pub token_given: Option<String>,
    /// The name this machine's writes are stamped with, where one is known.
    pub host: Option<String>,
}

impl Reach {
    /// Read it for the user whose home is `home`. `env` gives the value of an
    /// environment variable, or nothing.
    pub fn read(home: &Path, env: impl Fn(&str) -> Option<String>) -> anyhow::Result<Self> {
        let antumbra = home.join(".antumbra");
        let state: Value = std::fs::read_to_string(antumbra.join("setup.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or(Value::Null);
        let said = |name: &str| {
            env(name)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let kept = |key: &str| {
            state
                .get(key)
                .and_then(Value::as_str)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let url = said("ANTUMBRA_URL")
            .or_else(|| kept("url"))
            .ok_or_else(|| {
                anyhow::anyhow!("Antumbra has not been set up on this machine. Run: antumbra setup")
            })?;
        Ok(Reach {
            url: url
                .trim_end_matches('/')
                .trim_end_matches("/mcp")
                .to_string(),
            workspace: said("ANTUMBRA_WORKSPACE_ID").or_else(|| kept("workspace")),
            token_file: said("ANTUMBRA_TOKEN_FILE")
                .or_else(|| kept("token_file"))
                .map(PathBuf::from)
                .unwrap_or_else(|| antumbra.join("token.txt")),
            token_given: said("ANTUMBRA_TOKEN"),
            host: said("ANTUMBRA_HOST_ID")
                .or_else(|| kept("host"))
                .map(|h| h.to_lowercase()),
        })
    }

    /// The token as it is now: the one the environment gave, else the file's.
    /// The file is read for each use, so a token replaced under a running
    /// agent is the one its next call presents.
    pub fn token(&self) -> anyhow::Result<String> {
        if let Some(token) = &self.token_given {
            return Ok(token.clone());
        }
        let token = std::fs::read_to_string(&self.token_file)
            .map(|t| t.trim().to_string())
            .unwrap_or_default();
        if token.is_empty() {
            anyhow::bail!(
                "no token in {}. Run: antumbra setup",
                self.token_file.display()
            );
        }
        Ok(token)
    }
}

/// A directory of a test's own, removed when the test is done with it.
#[cfg(test)]
pub(crate) mod scratch {
    use std::path::{Path, PathBuf};

    pub struct Scratch(PathBuf);

    /// Counts the directories made, so tests that name theirs alike and run
    /// side by side still get one each.
    static MADE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    impl Scratch {
        /// A new, empty directory named for `what`, this process and how many
        /// came before it, so two tests, or two runs at once, never share one.
        pub fn new(what: &str) -> std::io::Result<Self> {
            let nth = MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "antumbra-agents-{what}-{}-{nth}",
                std::process::id()
            ));
            if dir.exists() {
                std::fs::remove_dir_all(&dir)?;
            }
            std::fs::create_dir_all(&dir)?;
            Ok(Scratch(dir))
        }

        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // Best effort: a directory left behind is in the temp directory.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::RefCell;

    use super::*;

    /// A report that keeps what it was told, each line under the way it was
    /// said.
    #[derive(Default)]
    pub struct Told {
        pub dry_run: bool,
        pub said: RefCell<Vec<String>>,
    }

    impl Report for Told {
        fn dry_run(&self) -> bool {
            self.dry_run
        }
        fn ok(&self, said: &str) {
            self.said.borrow_mut().push(format!("ok {said}"));
        }
        fn would(&self, said: &str) {
            self.said.borrow_mut().push(format!("would {said}"));
        }
        fn warn(&self, said: &str) {
            self.said.borrow_mut().push(format!("warn {said}"));
        }
        fn info(&self, said: &str) {
            self.said.borrow_mut().push(format!("info {said}"));
        }
    }

    fn home_with(state: &str) -> anyhow::Result<scratch::Scratch> {
        let scratch = scratch::Scratch::new("reach")?;
        let antumbra = scratch.path().join(".antumbra");
        std::fs::create_dir_all(&antumbra)?;
        std::fs::write(antumbra.join("setup.json"), state)?;
        Ok(scratch)
    }

    #[test]
    fn what_setup_recorded_is_how_the_server_is_reached() -> anyhow::Result<()> {
        let home = home_with(
            r#"{"mode":"hosted","url":"https://memory.example/mcp/","workspace":"ws:team","host":"Laptop-7"}"#,
        )?;
        let reach = Reach::read(home.path(), |_| None)?;
        assert_eq!(reach.url, "https://memory.example");
        assert_eq!(reach.workspace.as_deref(), Some("ws:team"));
        assert_eq!(reach.host.as_deref(), Some("laptop-7"));
        assert_eq!(
            reach.token_file,
            home.path().join(".antumbra").join("token.txt")
        );
        Ok(())
    }

    #[test]
    fn the_environment_wins_over_what_setup_recorded() -> anyhow::Result<()> {
        let home = home_with(r#"{"url":"https://memory.example","workspace":"ws:team"}"#)?;
        let reach = Reach::read(home.path(), |name| match name {
            "ANTUMBRA_URL" => Some("http://127.0.0.1:8081/".into()),
            "ANTUMBRA_HOST_ID" => Some(" Bench ".into()),
            _ => None,
        })?;
        assert_eq!(reach.url, "http://127.0.0.1:8081");
        assert_eq!(reach.workspace.as_deref(), Some("ws:team"));
        assert_eq!(reach.host.as_deref(), Some("bench"));
        Ok(())
    }

    #[test]
    fn a_machine_never_set_up_says_what_to_run() -> anyhow::Result<()> {
        let scratch = scratch::Scratch::new("reach-none")?;
        let Err(problem) = Reach::read(scratch.path(), |_| None) else {
            anyhow::bail!("a home with no setup was read as one");
        };
        assert!(problem.to_string().contains("antumbra setup"), "{problem}");
        Ok(())
    }

    #[test]
    fn the_token_is_read_when_asked_for_and_a_missing_one_is_said() -> anyhow::Result<()> {
        let home = home_with(r#"{"url":"https://memory.example"}"#)?;
        let reach = Reach::read(home.path(), |_| None)?;
        let Err(problem) = reach.token() else {
            anyhow::bail!("a token was read from a file that is not there");
        };
        assert!(problem.to_string().contains("no token in"), "{problem}");
        std::fs::write(&reach.token_file, "first\n")?;
        assert_eq!(reach.token()?, "first");
        std::fs::write(&reach.token_file, "second\n")?;
        assert_eq!(reach.token()?, "second");
        // One the environment gave wins over the file, as in the hook scripts.
        let given = Reach::read(home.path(), |name| {
            (name == "ANTUMBRA_TOKEN").then(|| " from-env ".to_string())
        })?;
        assert_eq!(given.token()?, "from-env");
        Ok(())
    }
}
