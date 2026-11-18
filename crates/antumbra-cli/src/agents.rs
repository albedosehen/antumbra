//! Coding agents other than Claude Code.
//!
//! Claude Code is given the server's address, a helper that reads the token
//! for each connection, and hook scripts with the address in its settings'
//! `env`. No other agent has all three, and each has its own config file and
//! its own shape for a hook's input and answer. What they share is here: where
//! the server is and how to reach it, as `antumbra setup` left it
//! ([`Reach`]), and the bridge that carries an agent's MCP connection there
//! ([`bridge`]).

use std::path::{Path, PathBuf};

use serde_json::Value;

pub mod bridge;

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
mod tests {
    use super::*;

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
