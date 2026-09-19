//! The App's own identity: a short-lived RS256 JWT signed with its private
//! key, which the API exchanges for an installation token scoped to one
//! organization's repositories. The private key is parsed once at startup and
//! never printed.

use chrono::{DateTime, Utc};
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use serde::Serialize;

/// How long an App token lives. GitHub refuses anything past ten minutes; a
/// minute of headroom covers clock skew on either side.
pub const APP_JWT_TTL_SECS: i64 = 9 * 60;
/// Issued-at is backdated by this much so a receiver with a slow clock does
/// not see a token from the future.
const CLOCK_SKEW_SECS: i64 = 60;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AppError {
    #[error("the GitHub App id is empty")]
    EmptyId,
    #[error("the GitHub App private key is not a PEM RSA key: {0}")]
    Key(String),
    #[error("cannot sign the GitHub App token: {0}")]
    Sign(String),
}

/// The App id and its parsed private key.
pub struct AppCredentials {
    app_id: String,
    key: EncodingKey,
}

// Hand-written so the key never reaches a log or a panic message.
impl std::fmt::Debug for AppCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppCredentials")
            .field("app_id", &self.app_id)
            .field("key", &"<redacted>")
            .finish()
    }
}

#[derive(Serialize)]
struct Claims<'a> {
    iat: i64,
    exp: i64,
    iss: &'a str,
}

impl AppCredentials {
    /// From the App id and its private key (PEM, PKCS#1 or PKCS#8).
    pub fn new(app_id: impl Into<String>, private_key_pem: &[u8]) -> Result<Self, AppError> {
        let app_id = app_id.into();
        let app_id = app_id.trim().to_string();
        if app_id.is_empty() {
            return Err(AppError::EmptyId);
        }
        let key =
            EncodingKey::from_rsa_pem(private_key_pem).map_err(|e| AppError::Key(e.to_string()))?;
        Ok(Self { app_id, key })
    }

    pub fn app_id(&self) -> &str {
        &self.app_id
    }

    /// A fresh App token valid from a minute ago until [`APP_JWT_TTL_SECS`]
    /// from `now`.
    pub fn jwt(&self, now: DateTime<Utc>) -> Result<String, AppError> {
        let now = now.timestamp();
        let claims = Claims {
            iat: now - CLOCK_SKEW_SECS,
            exp: now + APP_JWT_TTL_SECS,
            iss: &self.app_id,
        };
        encode(&Header::new(Algorithm::RS256), &claims, &self.key)
            .map_err(|e| AppError::Sign(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{APP_PRIVATE_KEY_PEM, APP_PUBLIC_KEY_PEM};
    use jsonwebtoken::{decode, DecodingKey, Validation};
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Seen {
        iat: i64,
        exp: i64,
        iss: String,
    }

    #[test]
    fn the_app_token_verifies_with_the_public_key_and_carries_the_claims() {
        let app = AppCredentials::new(" 12345 ", APP_PRIVATE_KEY_PEM).unwrap();
        assert_eq!(app.app_id(), "12345");
        let now = Utc::now();
        let token = app.jwt(now).unwrap();
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(&["12345"]);
        let seen = decode::<Seen>(
            &token,
            &DecodingKey::from_rsa_pem(APP_PUBLIC_KEY_PEM).unwrap(),
            &validation,
        )
        .unwrap()
        .claims;
        assert_eq!(seen.iss, "12345");
        assert_eq!(seen.iat, now.timestamp() - CLOCK_SKEW_SECS);
        assert_eq!(seen.exp, now.timestamp() + APP_JWT_TTL_SECS);
        assert!(seen.exp - seen.iat <= 10 * 60, "GitHub's ten-minute cap");
        assert!(
            !format!("{app:?}").contains("PRIVATE"),
            "the key never prints"
        );
    }

    #[test]
    fn a_bad_key_or_empty_id_is_refused() {
        assert_eq!(
            AppCredentials::new("", APP_PRIVATE_KEY_PEM).unwrap_err(),
            AppError::EmptyId
        );
        assert!(matches!(
            AppCredentials::new("1", b"not a pem").unwrap_err(),
            AppError::Key(_)
        ));
    }
}
