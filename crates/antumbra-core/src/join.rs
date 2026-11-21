//! Asking to join a workspace someone else hosts.
//!
//! An admin of the workspace makes a **join code** and gives it to the person.
//! The person asks with it, choosing the handle they will be known by. The
//! workspace's admins see the request and decide. Approved, the person
//! collects a token, once, by showing the secret they asked with.
//!
//! Nothing here is a credential the server keeps in the clear: a code and a
//! request's secret are each kept as their SHA-256, so a read of the store
//! gives neither.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{AntumbraError, Result, TenantId, UserId};

/// How long a join code lasts when its maker does not say.
pub const CODE_DAYS: i64 = 14;
/// How many requests one code admits when its maker does not say.
pub const CODE_USES: u32 = 1;
/// How long an approval waits to be collected before it lapses.
pub const COLLECT_DAYS: i64 = 7;
/// How long after a denial the same handle may ask again.
pub const DENIAL_HOURS: i64 = 24;
/// The most requests a workspace holds undecided at once.
pub const PENDING_AT_MOST: usize = 100;
/// The longest line a person may write to the admins, in characters.
pub const MAX_NOTE_CHARS: usize = 500;

/// Handles nobody may take: the defaults a local setup runs as, the system's
/// own identities, and names that would read as an office.
pub const RESERVED_HANDLES: &[&str] = &[
    "default",
    "github",
    "dashboard",
    "system",
    "operator",
    "admin",
    "owner",
    "root",
    "antumbra",
    "any",
    "local",
    "me",
];

/// The SHA-256 of a code or a secret, in hex: what is kept of either.
pub fn digest(secret: &str) -> String {
    Sha256::digest(secret.trim().as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `raw` as a handle, or why it cannot be one: 3 to 32 of `a-z`, `0-9`, `-`
/// and `_`, starting with a letter or a digit, and none of
/// [`RESERVED_HANDLES`]. Given in any case, kept in lower.
pub fn handle_of(raw: &str) -> Result<String> {
    let handle = raw.trim().trim_start_matches("user:").to_ascii_lowercase();
    let well_formed = (3..=32).contains(&handle.len())
        && handle
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        && handle.starts_with(|c: char| c.is_ascii_alphanumeric());
    if !well_formed {
        return Err(AntumbraError::rejected(
            "a handle is 3 to 32 of a-z, 0-9, - and _, starting with a letter or a digit",
        ));
    }
    if RESERVED_HANDLES.contains(&handle.as_str()) {
        return Err(AntumbraError::rejected(format!(
            "`{handle}` is not a handle anyone may take; choose your own"
        )));
    }
    Ok(handle)
}

/// The user id a handle is in a workspace.
pub fn user_of(handle: &str) -> UserId {
    UserId::new(format!("user:{handle}"))
}

/// A join code as its workspace's admins see it. The code itself is shown
/// once, to its maker; what is kept is its digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinCode {
    /// The code's digest, which is also what names it to revoke. (`key`,
    /// not `id`: a row's `id` is the engine's own.)
    pub key: String,
    pub tenant: TenantId,
    /// What its maker called it, to tell one from another.
    pub label: Option<String>,
    pub created_by: UserId,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub uses_left: u32,
    /// Whether a request made with it waits for an admin. A code that does
    /// not ask is an invitation: whoever holds it is let in.
    pub needs_approval: bool,
    pub revoked: bool,
}

impl JoinCode {
    /// Whether a request may still be made with it at `now`.
    pub fn admits(&self, now: DateTime<Utc>) -> bool {
        !self.revoked && self.uses_left > 0 && self.expires_at > now
    }
}

/// Where a request stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Standing {
    Pending,
    Approved,
    Denied,
    /// Approved, and its token collected: the person is a member.
    Collected,
}

impl Standing {
    pub fn as_str(self) -> &'static str {
        match self {
            Standing::Pending => "pending",
            Standing::Approved => "approved",
            Standing::Denied => "denied",
            Standing::Collected => "collected",
        }
    }
}

/// One person's request to join one workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinRequest {
    pub key: String,
    pub tenant: TenantId,
    pub handle: String,
    pub display: Option<String>,
    /// What they wrote to the admins.
    pub note: Option<String>,
    /// The machine they asked from, as it named itself.
    pub host: Option<String>,
    /// The digest of the secret they will collect with.
    pub secret: String,
    pub standing: Standing,
    pub created_at: DateTime<Utc>,
    /// When an undecided request stops being one.
    pub expires_at: DateTime<Utc>,
    pub decided_at: Option<DateTime<Utc>>,
    pub decided_by: Option<UserId>,
    /// What the admin who decided said, shown to the person.
    pub reason: Option<String>,
}

impl JoinRequest {
    /// The user id the person will be.
    pub fn user(&self) -> UserId {
        user_of(&self.handle)
    }

    /// Whether it still waits for a decision at `now`.
    pub fn waits(&self, now: DateTime<Utc>) -> bool {
        self.standing == Standing::Pending && self.expires_at > now
    }

    /// Whether an approval has waited too long to be collected.
    pub fn lapsed(&self, now: DateTime<Utc>) -> bool {
        self.standing == Standing::Approved
            && self
                .decided_at
                .is_some_and(|at| at + Duration::days(COLLECT_DAYS) < now)
    }
}

/// What a person is told when they ask how their request stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    /// Still waiting for an admin.
    Pending,
    /// Approved: they are now this member of this workspace, and are to be
    /// given a token for it. Said once.
    Admitted { tenant: TenantId, user: UserId },
    /// Refused, with what the admin said.
    Denied { reason: Option<String> },
    /// Nobody decided before it expired, or an approval was never collected.
    Expired,
    /// Collected already: the token was given and is not given twice.
    AlreadyCollected,
}

/// A note to the admins, trimmed; `None` when blank. Refused when too long
/// or holding a control character other than a line break.
pub fn note_of(raw: &str) -> Result<Option<String>> {
    let note = raw.trim();
    if note.is_empty() {
        return Ok(None);
    }
    if note.chars().count() > MAX_NOTE_CHARS {
        return Err(AntumbraError::rejected(format!(
            "a note is at most {MAX_NOTE_CHARS} characters"
        )));
    }
    if note.chars().any(|c| c.is_control() && c != '\n') {
        return Err(AntumbraError::rejected(
            "a note holds no control characters",
        ));
    }
    Ok(Some(note.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_handle_is_plain_and_nobody_takes_a_default_or_an_office() -> anyhow::Result<()> {
        assert_eq!(handle_of("  Dana-W  ")?, "dana-w");
        assert_eq!(handle_of("user:eli_2")?, "eli_2");
        assert_eq!(user_of("dana").as_str(), "user:dana");
        for bad in [
            "ab",
            "",
            "-dash",
            "has space",
            "émile",
            "a/b",
            &"x".repeat(33),
            "user:",
        ] {
            assert!(handle_of(bad).is_err(), "{bad:?} passed");
        }
        for taken in [
            "default",
            "user:default",
            "GitHub",
            "admin",
            "operator",
            "root",
        ] {
            let refused = handle_of(taken)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(
                refused.contains("not a handle anyone may take"),
                "{taken}: {refused}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_digest_is_of_the_secret_and_gives_nothing_back() -> anyhow::Result<()> {
        let d = digest("aj1-secret");
        assert_eq!(d.len(), 64);
        assert_eq!(d, digest("  aj1-secret\n"));
        assert_ne!(d, digest("aj1-secreT"));
        assert!(!d.contains("secret"));
        Ok(())
    }

    #[test]
    fn a_code_admits_until_it_is_spent_expired_or_revoked() -> anyhow::Result<()> {
        let now = Utc::now();
        let code = JoinCode {
            key: digest("c"),
            tenant: TenantId::new("ws:t"),
            label: None,
            created_by: UserId::new("user:a"),
            created_at: now,
            expires_at: now + Duration::days(1),
            uses_left: 1,
            needs_approval: true,
            revoked: false,
        };
        assert!(code.admits(now));
        assert!(!code.admits(now + Duration::days(2)));
        assert!(!JoinCode {
            uses_left: 0,
            ..code.clone()
        }
        .admits(now));
        assert!(!JoinCode {
            revoked: true,
            ..code
        }
        .admits(now));
        Ok(())
    }

    #[test]
    fn a_note_is_short_and_plain() -> anyhow::Result<()> {
        assert_eq!(note_of("  hello\nthere ")?.as_deref(), Some("hello\nthere"));
        assert_eq!(note_of("   ")?, None);
        assert!(note_of(&"n".repeat(MAX_NOTE_CHARS + 1)).is_err());
        assert!(note_of("bell\u{7}").is_err());
        Ok(())
    }
}
