//! Magic-link authentication (ADR-0016): a passwordless login. The control
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

/// The one-time link token's claims: the email being verified, plus expiry.
#[derive(Serialize, Deserialize)]
struct MagicClaims {
    sub: String,
    exp: u64,
}

fn mint_token(secret: &[u8], email: &str, ttl: Duration) -> Result<String> {
    let claims = MagicClaims {
        sub: email.to_string(),
        exp: get_current_timestamp() + ttl.as_secs(),
    };
    encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret),
    )
    .map_err(|e| AntumbraError::other(format!("mint magic token: {e}")))
}

fn verify_token(secret: &[u8], token: &str) -> Result<String> {
    let mut validation = Validation::new(Algorithm::HS256);
    validation.set_required_spec_claims(&["exp"]);
    validation.validate_aud = false;
    let data = decode::<MagicClaims>(token, &DecodingKey::from_secret(secret), &validation)
        .map_err(|e| AntumbraError::other(format!("invalid magic link: {e}")))?;
    Ok(data.claims.sub)
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

    /// Email a one-time login link for `email`.
    pub fn request(&self, email: &str) -> Result<()> {
        if !email.contains('@') {
            return Err(AntumbraError::other("not an email address"));
        }
        let token = mint_token(self.secret, email, self.ttl)?;
        let link = format!(
            "{}/magic/verify?token={token}",
            self.base_url.trim_end_matches('/')
        );
        self.mailer.send_link(email, &link)
    }

    /// Verify a clicked link's token → the verified identity (`email:<addr>`).
    pub fn verify(&self, token: &str) -> Result<VerifiedIdentity> {
        let email = verify_token(self.secret, token)?;
        Ok(VerifiedIdentity::new(format!("email:{email}")).with(Some(email), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Captures the links it would have sent, so a test can follow them.
    struct FakeMailer(Mutex<Vec<String>>);
    impl Mailer for FakeMailer {
        fn send_link(&self, _to: &str, link: &str) -> Result<()> {
            self.0.lock().unwrap().push(link.to_string());
            Ok(())
        }
    }

    fn token_in(link: &str) -> String {
        link.split("token=").nth(1).unwrap().to_string()
    }

    #[test]
    fn request_emails_a_link_that_verifies_to_the_email_identity() {
        let mailer = FakeMailer(Mutex::new(Vec::new()));
        let ml = MagicLink::new(
            b"magic-secret",
            Duration::from_secs(900),
            "https://app.example/",
            &mailer,
        );
        ml.request("ada@x.com").unwrap();
        let link = mailer.0.lock().unwrap()[0].clone();
        assert!(link.starts_with("https://app.example/magic/verify?token="));

        let id = ml.verify(&token_in(&link)).unwrap();
        assert_eq!(id.subject, "email:ada@x.com");
        assert_eq!(id.email.as_deref(), Some("ada@x.com"));
    }

    #[test]
    fn a_tampered_or_foreign_token_is_rejected() {
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
        other.request("a@b.com").unwrap();
        let foreign = token_in(&mailer.0.lock().unwrap().last().unwrap().clone());
        assert!(ml.verify(&foreign).is_err());
    }

    #[test]
    fn a_non_email_request_is_rejected() {
        let mailer = FakeMailer(Mutex::new(Vec::new()));
        let ml = MagicLink::new(b"s", Duration::from_secs(60), "https://app", &mailer);
        assert!(ml.request("notanemail").is_err());
    }
}
