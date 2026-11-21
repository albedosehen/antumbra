//! Magic-link authentication for hosted onboarding: a passwordless login. The control
//! plane signs a short-lived one-time link carrying the email, sends it via a
//! [`Mailer`] seam, and on click verifies the signature → a [`VerifiedIdentity`]
//! (`email:<addr>`). The sign + verify are pure (testable offline); only the
//! send crosses the network, behind the seam. The real SMTP/email-API mailer is
//! a separate, feature-gated concern.

use std::time::Duration;

use jsonwebtoken::{
    decode, encode, get_current_timestamp, Algorithm, DecodingKey, EncodingKey, Header, Validation,
};
use serde::{Deserialize, Serialize};

use antumbra_core::{AntumbraError, Result};

use crate::VerifiedIdentity;

/// Sends a one-time login link to an address. The real impl is SMTP / an email
/// API (a feature-gated concern); tests use a capturing fake.
pub trait Mailer: Send + Sync {
    fn send_link(&self, to: &str, link: &str) -> Result<()>;
}

/// The one-time link token's claims: the email being verified, plus expiry, plus
/// the invite code a *signup* link carries (a login link omits it). `jti` is the
/// link's unique id, the key the single-use ledger consumes on verify -- without
/// it a leaked login link could be replayed for fresh tokens until expiry.
#[derive(Serialize, Deserialize)]
struct MagicClaims {
    sub: String,
    exp: u64,
    jti: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inv: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    join: Option<JoinAsk>,
}

/// What a link that asks to join a workspace carries: the request that is
/// made when the link is followed, and not before.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinAsk {
    /// The id the request will have, told to the asker when the link is sent.
    pub request: String,
    pub code: String,
    pub handle: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    /// The SHA-256 of the secret the token will be collected with.
    pub secret_digest: String,
}

/// A deliberately-conservative shape check (not full RFC 5322): exactly one `@`,
/// a non-empty local part, and a dotted, non-empty domain, no whitespace. It
/// guards against obviously-bad input before signing a link -- not a guarantee
/// of deliverability (the mailer is the real check).
fn is_plausible_email(email: &str) -> bool {
    if email.len() > 254 || email.chars().any(char::is_whitespace) {
        return false;
    }
    let mut parts = email.split('@');
    let (Some(local), Some(domain), None) = (parts.next(), parts.next(), parts.next()) else {
        return false; // zero, or more than one, `@`
    };
    !local.is_empty()
        && !domain.is_empty()
        && domain.contains('.')
        && !domain.starts_with('.')
        && !domain.ends_with('.')
}

fn mint_token(
    secret: &[u8],
    email: &str,
    invite: Option<&str>,
    join: Option<&JoinAsk>,
    ttl: Duration,
) -> Result<String> {
    let claims = MagicClaims {
        sub: email.to_string(),
        exp: get_current_timestamp() + ttl.as_secs(),
        jti: uuid::Uuid::new_v4().simple().to_string(),
        inv: invite.map(str::to_string),
        join: join.cloned(),
    };
    encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret),
    )
    .map_err(|e| AntumbraError::other(format!("mint magic token: {e}")))
}

/// Verify a link token's signature and expiry → its claims.
fn verify_token(secret: &[u8], token: &str) -> Result<MagicClaims> {
    let mut validation = Validation::new(Algorithm::HS256);
    validation.set_required_spec_claims(&["exp"]);
    validation.validate_aud = false;
    let data = decode::<MagicClaims>(token, &DecodingKey::from_secret(secret), &validation)
        .map_err(|e| AntumbraError::rejected(format!("invalid magic link: {e}")))?;
    Ok(data.claims)
}

/// A verified (signature + expiry) magic link, before the single-use check:
/// who it authenticates, the invite a signup link carried, and the `jti` /
/// `expires_at` pair the consumer writes through the single-use ledger.
pub struct VerifiedLink {
    pub identity: VerifiedIdentity,
    pub invite: Option<String>,
    /// The request a link that asks to join carries.
    pub join: Option<JoinAsk>,
    pub jti: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// A magic-link issuer/verifier over a [`Mailer`].
pub struct MagicLink<'a> {
    secret: &'a [u8],
    ttl: Duration,
    base_url: &'a str,
    mailer: &'a dyn Mailer,
}

impl<'a> MagicLink<'a> {
    /// `secret` signs the link token (separate from the RS256 issuer key);
    /// `base_url` is the control plane's external URL the link points back to.
    pub fn new(secret: &'a [u8], ttl: Duration, base_url: &'a str, mailer: &'a dyn Mailer) -> Self {
        Self {
            secret,
            ttl,
            base_url,
            mailer,
        }
    }

    /// Email a one-time LOGIN link (no invite) for an existing account.
    pub fn request(&self, email: &str) -> Result<()> {
        self.send(email, None, None)
    }

    /// Email a one-time link that, followed, asks to join a workspace as
    /// `ask` says: the address is verified by the following.
    pub fn request_join(&self, email: &str, ask: &JoinAsk) -> Result<()> {
        self.send(email, None, Some(ask))
    }

    /// Email a one-time SIGNUP link carrying `invite`, for a new account.
    pub fn request_signup(&self, email: &str, invite: &str) -> Result<()> {
        self.send(email, Some(invite), None)
    }

    fn send(&self, email: &str, invite: Option<&str>, join: Option<&JoinAsk>) -> Result<()> {
        if !is_plausible_email(email) {
            return Err(AntumbraError::rejected("not a valid email address"));
        }
        let token = mint_token(self.secret, email, invite, join, self.ttl)?;
        let link = format!(
            "{}/magic/verify?token={token}",
            self.base_url.trim_end_matches('/')
        );
        self.mailer.send_link(email, &link)
    }

    /// Verify a clicked link's token (signature + expiry) → the [`VerifiedLink`].
    /// This is the pure half; the caller still owes the link's `jti` to the
    /// single-use ledger ([`crate::authenticate`] does both).
    pub fn verify(&self, token: &str) -> Result<VerifiedLink> {
        let claims = verify_token(self.secret, token)?;
        let expires_at =
            chrono::DateTime::from_timestamp(claims.exp.min(i64::MAX as u64) as i64, 0)
                .ok_or_else(|| AntumbraError::other("magic link exp out of range"))?;
        Ok(VerifiedLink {
            identity: VerifiedIdentity::new(format!("email:{}", claims.sub))
                .with(Some(claims.sub), None),
            invite: claims.inv,
            join: claims.join,
            jti: claims.jti,
            expires_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Captures the links it would have sent, so a test can follow them.
    struct FakeMailer(Mutex<Vec<String>>);

    impl FakeMailer {
        /// Every link sent, in order. The fake only ever pushes to the list, so
        /// a poisoned lock holds a whole list and is read as is.
        fn links(&self) -> Vec<String> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl Mailer for FakeMailer {
        fn send_link(&self, _to: &str, link: &str) -> Result<()> {
            // Read as `links` is: a push cannot leave the list half-written.
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(link.to_string());
            Ok(())
        }
    }

    /// The one-time token a sent link carries.
    fn token_in(link: &str) -> Result<String> {
        link.split("token=")
            .nth(1)
            .map(str::to_string)
            .ok_or_else(|| AntumbraError::other("the link carries no token"))
    }

    #[test]
    fn request_emails_a_link_that_verifies_to_the_email_identity() -> Result<()> {
        let mailer = FakeMailer(Mutex::new(Vec::new()));
        let ml = MagicLink::new(
            b"magic-secret",
            Duration::from_secs(900),
            "https://app.example/",
            &mailer,
        );
        ml.request("ada@x.com")?;
        let link = mailer.links()[0].clone();
        assert!(link.starts_with("https://app.example/magic/verify?token="));

        let verified = ml.verify(&token_in(&link)?)?;
        assert_eq!(verified.identity.subject, "email:ada@x.com");
        assert_eq!(verified.identity.email.as_deref(), Some("ada@x.com"));
        assert_eq!(verified.invite, None, "a login link carries no invite");
        assert!(!verified.jti.is_empty(), "every link gets a unique id");
        assert!(
            verified.expires_at > chrono::Utc::now(),
            "expiry decodes to the future"
        );
        Ok(())
    }

    #[test]
    fn a_signup_link_round_trips_its_invite() -> Result<()> {
        let mailer = FakeMailer(Mutex::new(Vec::new()));
        let ml = MagicLink::new(b"s", Duration::from_secs(900), "https://app", &mailer);
        ml.request_signup("new@x.com", "invite-123")?;
        let link = mailer.links()[0].clone();
        let verified = ml.verify(&token_in(&link)?)?;
        assert_eq!(verified.identity.subject, "email:new@x.com");
        assert_eq!(verified.invite.as_deref(), Some("invite-123"));
        Ok(())
    }

    #[test]
    fn two_links_for_the_same_email_get_distinct_ids() -> Result<()> {
        let mailer = FakeMailer(Mutex::new(Vec::new()));
        let ml = MagicLink::new(b"s", Duration::from_secs(900), "https://app", &mailer);
        ml.request("ada@x.com")?;
        ml.request("ada@x.com")?;
        let links = mailer.links();
        let a = ml.verify(&token_in(&links[0])?)?;
        let b = ml.verify(&token_in(&links[1])?)?;
        assert_ne!(a.jti, b.jti);
        Ok(())
    }

    #[test]
    fn a_tampered_or_foreign_token_is_rejected() -> Result<()> {
        let mailer = FakeMailer(Mutex::new(Vec::new()));
        let ml = MagicLink::new(
            b"magic-secret",
            Duration::from_secs(900),
            "https://app",
            &mailer,
        );
        assert!(ml.verify("not-a-token").is_err());

        // A link signed with a different secret does not verify here.
        let other = MagicLink::new(
            b"other-secret",
            Duration::from_secs(900),
            "https://app",
            &mailer,
        );
        other.request("a@b.com")?;
        let sent = mailer
            .links()
            .last()
            .cloned()
            .ok_or_else(|| AntumbraError::other("no link was sent"))?;
        let foreign = token_in(&sent)?;
        assert!(ml.verify(&foreign).is_err());
        Ok(())
    }

    #[test]
    fn a_non_email_request_is_rejected() -> Result<()> {
        let mailer = FakeMailer(Mutex::new(Vec::new()));
        let ml = MagicLink::new(b"s", Duration::from_secs(60), "https://app", &mailer);
        for bad in [
            "notanemail",
            "",
            "@x.com",
            "a@",
            "a@b@c.com",
            "a@b",
            "a b@x.com",
            "a@.com",
            "a@x.",
        ] {
            assert!(ml.request(bad).is_err(), "{bad:?} should be rejected");
        }
        for good in ["ada@x.com", "a.b+tag@sub.example.org"] {
            assert!(is_plausible_email(good), "{good:?} should pass");
        }
        Ok(())
    }
}
