//! Choosing which of your own memories go to a workspace you joined.
//!
//! Nothing a person remembers privately reaches a team by itself. They pick,
//! from a store that may hold thousands, and what they pick is copied. This
//! module is the part of that with no store in it: what a person may decide
//! about a memory, what is held back until they ask for it, and why.

use serde::{Deserialize, Serialize};

use crate::keyed::{CLAUDE_CODE_RULE, SKILL_USE};

/// What a person decided about one memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Choice {
    /// To be brought to the workspace.
    Chosen,
    /// Looked at and left out, this time.
    Declined,
    /// Never to be offered or brought, whatever else changes.
    Never,
    /// Brought: a copy is in the workspace.
    Brought,
}

impl Choice {
    pub fn as_str(self) -> &'static str {
        match self {
            Choice::Chosen => "chosen",
            Choice::Declined => "declined",
            Choice::Never => "never",
            Choice::Brought => "brought",
        }
    }
}

impl std::str::FromStr for Choice {
    type Err = crate::AntumbraError;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "chosen" => Ok(Choice::Chosen),
            "declined" => Ok(Choice::Declined),
            "never" => Ok(Choice::Never),
            "brought" => Ok(Choice::Brought),
            other => Err(crate::AntumbraError::other(format!(
                "unknown choice `{other}` (chosen, declined, never or brought)"
            ))),
        }
    }
}

/// Why a memory is held back from what is offered: it is not a memory of the
/// person's work, or it looks like it holds something that must not travel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Held {
    /// A counter or a rule the agent's hooks keep, or a note between the
    /// person's own machines: of no use to anyone else.
    Housekeeping,
    /// Text that looks like a key, a token or a password.
    LooksSecret,
}

impl Held {
    pub fn as_str(self) -> &'static str {
        match self {
            Held::Housekeeping => "housekeeping",
            Held::LooksSecret => "looks_secret",
        }
    }

    /// What to tell the person.
    pub fn says(self) -> &'static str {
        match self {
            Held::Housekeeping => "kept by your agent's hooks, not a memory of your work",
            Held::LooksSecret => "looks like it holds a key, a token or a password",
        }
    }
}

/// Whether `content` is a hooks' counter or rule, or (by its compartment's
/// name) a handoff between the person's machines.
pub fn housekeeping(content: &str, compartment_name: Option<&str>) -> bool {
    content.starts_with(SKILL_USE)
        || content.starts_with(CLAUDE_CODE_RULE)
        || compartment_name == Some(crate::handoff::COMPARTMENT_NAME)
}

/// Prefixes that open a credential of a known kind.
const SECRET_PREFIXES: &[&str] = &[
    "-----BEGIN",
    "ghp_",
    "gho_",
    "ghs_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "sk-",
    "AKIA",
    "ASIA",
    "AIza",
    "glpat-",
    "ck1",
    "aj1_",
];

/// Words that, followed by `=` or `:` and a value, name a credential.
const SECRET_NAMES: &[&str] = &[
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "api-key",
    "access_key",
    "private_key",
    "client_secret",
    "auth_token",
    "bearer",
];

/// Whether a run of text could be a credential by its shape alone: long, with
/// no spaces, and mixing letters and digits the way a random value does and a
/// word, a path or a commit hash in prose rarely does.
fn random_looking(word: &str) -> bool {
    let word = word.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    if word.len() < 32 {
        return false;
    }
    let tokenish = word
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+' | '/' | '='));
    let (digits, uppers, lowers) = word.chars().fold((0, 0, 0), |(d, u, l), c| {
        (
            d + usize::from(c.is_ascii_digit()),
            u + usize::from(c.is_ascii_uppercase()),
            l + usize::from(c.is_ascii_lowercase()),
        )
    });
    // A path or a URL is long and unbroken too; so is a 40-character commit
    // hash, which is hex and all one case.
    let pathlike = word.contains('/') && !word.contains('+') && !word.ends_with('=');
    let hex = word.chars().all(|c| c.is_ascii_hexdigit());
    tokenish
        && !pathlike
        && !(hex && word.len() <= 40)
        && digits >= 4
        && (uppers >= 2 || hex)
        && lowers >= 2
}

/// Whether `content` looks like it holds a key, a token or a password. This
/// errs toward holding back: a memory wrongly held is one more click for the
/// person, and a secret wrongly offered is in front of a team.
pub fn looks_secret(content: &str) -> bool {
    // A signed token: three base64url parts, the first a JSON header.
    if content.contains("eyJ")
        && content
            .split_whitespace()
            .any(|w| w.matches('.').count() >= 2 && w.contains("eyJ") && w.len() >= 40)
    {
        return true;
    }
    for word in content.split(|c: char| {
        c.is_whitespace() || matches!(c, '"' | '\'' | '`' | ',' | ';' | '(' | ')' | '<' | '>')
    }) {
        if word.is_empty() {
            continue;
        }
        let known = SECRET_PREFIXES
            .iter()
            .any(|p| word.starts_with(p) && (word.len() >= p.len() + 12 || *p == "-----BEGIN"));
        if known || random_looking(word) {
            return true;
        }
    }
    let lower = content.to_ascii_lowercase();
    SECRET_NAMES.iter().any(|name| {
        lower.match_indices(name).any(|(at, _)| {
            let rest = lower[at + name.len()..].trim_start();
            let value = rest
                .strip_prefix('=')
                .or_else(|| rest.strip_prefix(':'))
                .map(str::trim_start)
                .unwrap_or_default();
            // A name with a value after it, and the value is not a word about
            // the secret ("password: rotated last week").
            let first = value.split_whitespace().next().unwrap_or_default();
            first.len() >= 8 && first.chars().any(|c| c.is_ascii_digit()) && !value.starts_with('<')
        })
    })
}

/// Why `content` is held back from what is offered, if it is.
pub fn held(content: &str, compartment_name: Option<&str>) -> Option<Held> {
    if housekeeping(content, compartment_name) {
        return Some(Held::Housekeeping);
    }
    looks_secret(content).then_some(Held::LooksSecret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_are_held_back_by_their_shape() -> anyhow::Result<()> {
        for secret in [
            "the deploy key is ghp_aB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0dE3fG6",
            "AKIAIOSFODNN7EXAMPLE is the access key for the bucket",
            "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk",
            "set API_KEY=9f8a7b6c5d4e3f2a1b0c in the env file",
            "the staging password: Tr0ub4dor&3xyz",
            "token eyJhbGciOiJIUzI1NiJ9.eyJ0ZW5hbnQiOiJ3czphIn0.c2lnbmF0dXJlLXBhcnQ",
            "the webhook secret is Zk3Lm9Qp2Rs5Tv8Wx1Yb4Nc7Df0Gh3Jk6",
            "join with aj1_4bc388f24cf44a0bb160c3a62083f8ae",
        ] {
            assert!(looks_secret(secret), "offered: {secret}");
        }
        Ok(())
    }

    #[test]
    fn ordinary_memories_are_not_held_back() -> anyhow::Result<()> {
        for plain in [
            "Rotated the webhook signing secret after it appeared in a build log.",
            "The password policy changed: twelve characters at least.",
            "release/v0.5.1 is at 742fb02 after the merge of the join requests.",
            "merge commit 3c492ca5f0e1d2b3a4c5d6e7f8091a2b3c4d5e6f landed the profiles",
            "The file lives at crates/antumbra-mcp/src/http/dashboard/manage.js in the tree.",
            "See https://github.com/acme/billing/blob/main/docs/runbook-for-failed-charges.md",
            "Bearer tokens are read from the Authorization header, never from the path.",
            "Use the sk- prefix convention only for test fixtures.",
        ] {
            assert!(!looks_secret(plain), "held back: {plain}");
        }
        Ok(())
    }

    #[test]
    fn the_hooks_own_memories_are_housekeeping() -> anyhow::Result<()> {
        assert_eq!(
            held("[skill-use:dataviz] dataviz", None),
            Some(Held::Housekeeping)
        );
        assert_eq!(
            held("[claude-code:agents-md] Read AGENTS.md", None),
            Some(Held::Housekeeping)
        );
        assert_eq!(
            held("Finish the migration check", Some("handoff")),
            Some(Held::Housekeeping)
        );
        assert_eq!(held("Migrations run in two steps.", Some("default")), None);
        assert_eq!(
            held("password=hunter2hunter2", None),
            Some(Held::LooksSecret)
        );
        Ok(())
    }

    #[test]
    fn a_choice_reads_as_it_is_written() -> anyhow::Result<()> {
        for choice in [
            Choice::Chosen,
            Choice::Declined,
            Choice::Never,
            Choice::Brought,
        ] {
            assert_eq!(choice.as_str().parse::<Choice>()?, choice);
        }
        assert!("maybe".parse::<Choice>().is_err());
        Ok(())
    }
}
