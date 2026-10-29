//! Where `antumbra tui` finds its server and token when it is not told: the
//! places the agent's own connection is configured, so a developer who ran
//! `antumbra setup`, or wired Claude Code by hand, types `antumbra tui` and
//! nothing else.
//!
//! The server: `--surface`, else `ANTUMBRA_URL`, else Claude Code's settings
//! (`env.ANTUMBRA_URL` in `settings.json`, under `CLAUDE_CONFIG_DIR` or
//! `~/.claude`), else what `antumbra setup` recorded (`~/.antumbra/setup.json`),
//! else a stack on this machine.
//!
//! The token: `--token`, else every token on hand, with an operator's
//! preferred so the console can act when one can: `ANTUMBRA_TOKEN`, the file
//! `ANTUMBRA_TOKEN_FILE` names, `~/.antumbra/token.txt`, and those two in
//! Claude Code's settings. A hook token in the environment and an operator's
//! token saved to the file then give the console the operator's.

use std::path::PathBuf;

use anyhow::{bail, Result};
use base64::Engine;
use serde_json::Value;

/// The server when nothing names one: a stack on this machine.
pub const DEFAULT_SURFACE: &str = "http://127.0.0.1:8081";

/// What `Found::from` says for [`DEFAULT_SURFACE`].
pub const DEFAULT_FROM: &str = "the default: nothing names a server";

/// A setting, and where it was found, which the connect panel shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    pub value: String,
    pub from: String,
}

/// An environment variable's value by name.
pub type Lookup = Box<dyn Fn(&str) -> Option<String>>;

/// What the search reads: the environment and the home directory. Given, so
/// tests need not touch the real ones.
pub struct Places {
    pub env: Lookup,
    pub home: Option<PathBuf>,
}

impl Places {
    /// This process's environment and home.
    pub fn here() -> Self {
        Self {
            env: Box::new(|name| std::env::var(name).ok()),
            home: std::env::var_os("USERPROFILE")
                .or_else(|| std::env::var_os("HOME"))
                .map(PathBuf::from),
        }
    }

    fn var(&self, name: &str) -> Option<String> {
        filled((self.env)(name))
    }

    /// A variable from the `env` block of Claude Code's settings.
    fn claude_var(&self, name: &str) -> Option<String> {
        let dir = self
            .var("CLAUDE_CONFIG_DIR")
            .map(PathBuf::from)
            .or_else(|| self.home.as_ref().map(|h| h.join(".claude")))?;
        let settings: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("settings.json")).ok()?).ok()?;
        filled(settings.get("env")?.get(name)?.as_str().map(str::to_string))
    }

    /// The server `antumbra setup` recorded.
    fn setup_url(&self) -> Option<String> {
        let path = self.home.as_ref()?.join(".antumbra").join("setup.json");
        let record: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        filled(record.get("url")?.as_str().map(str::to_string))
    }
}

fn filled(value: Option<String>) -> Option<String> {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn read_token(path: &str) -> Option<String> {
    filled(std::fs::read_to_string(path).ok())
}

/// The server to read.
pub fn surface(given: Option<String>, places: &Places) -> Found {
    let found = |value: String, from: &str| Found {
        value: value.trim_end_matches('/').to_string(),
        from: from.to_string(),
    };
    if let Some(url) = filled(given) {
        return found(url, "--surface");
    }
    if let Some(url) = places.var("ANTUMBRA_URL") {
        return found(url, "ANTUMBRA_URL");
    }
    if let Some(url) = places.claude_var("ANTUMBRA_URL") {
        return found(url, "Claude Code's settings");
    }
    if let Some(url) = places.setup_url() {
        return found(url, "antumbra setup");
    }
    found(DEFAULT_SURFACE.to_string(), DEFAULT_FROM)
}

/// The token to present.
pub fn token(given: Option<String>, places: &Places) -> Result<Found> {
    if let Some(token) = filled(given) {
        return Ok(Found {
            value: token,
            from: "--token".into(),
        });
    }
    let mut found: Vec<Found> = Vec::new();
    let mut offer = |value: Option<String>, from: &str| {
        if let Some(value) = filled(value) {
            if !found.iter().any(|f| f.value == value) {
                found.push(Found {
                    value,
                    from: from.to_string(),
                });
            }
        }
    };
    offer(places.var("ANTUMBRA_TOKEN"), "ANTUMBRA_TOKEN");
    if let Some(path) = places.var("ANTUMBRA_TOKEN_FILE") {
        offer(read_token(&path), "ANTUMBRA_TOKEN_FILE");
    }
    if let Some(home) = &places.home {
        let path = home.join(".antumbra").join("token.txt");
        offer(read_token(&path.to_string_lossy()), "~/.antumbra/token.txt");
    }
    offer(
        places.claude_var("ANTUMBRA_TOKEN"),
        "ANTUMBRA_TOKEN in Claude Code's settings",
    );
    if let Some(path) = places.claude_var("ANTUMBRA_TOKEN_FILE") {
        offer(
            read_token(&path),
            "ANTUMBRA_TOKEN_FILE in Claude Code's settings",
        );
    }
    if found.is_empty() {
        bail!(
            "no token: run `antumbra setup`, which saves one to ~/.antumbra/token.txt, or give one with --token"
        );
    }
    let pick = found
        .iter()
        .position(|f| is_operator(&f.value))
        .unwrap_or(0);
    Ok(found.swap_remove(pick))
}

/// Whether `token` says it is an operator's. Read, not verified: it only
/// chooses among the tokens on hand, and the server decides what each may do.
pub fn is_operator(token: &str) -> bool {
    token
        .split('.')
        .nth(1)
        .and_then(|payload| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(payload.trim_end_matches('='))
                .ok()
        })
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|claims| claims.get("operator")?.as_bool())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests;
