//! JWT bearer-token authentication for the networked MCP surface (ADR-0015).
//!
//! The networked server is multi-tenant **per request**: a client proves its
//! identity with a signed JWT whose claims carry the `tenant` and `user`. The
//! server verifies the signature (and expiry) and binds those claims as `$auth`
//! — the same `(tenant, user)` the engine's record-access `SIGNIN` expects.
//! There is no token-to-identity lookup table: the verified token *is* the
//! identity, so a leaked token grants exactly its claimed scope and nothing
//! wider. The signer (your auth service) holds the key; this server only
//! verifies.

use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use serde::Deserialize;

/// The verified identity carried by a request's bearer token — what becomes
/// `$auth.tenant` / `$auth.user` for the session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identity {
    pub tenant: String,
    pub user: String,
}

/// The claims the server requires. `tenant` and `user` become `$auth`; `exp` is
/// validated by the library. Any other claims in the token are ignored.
#[derive(Debug, Deserialize)]
struct Claims {
    tenant: String,
    user: String,
}

/// Why a token was rejected. Kept coarse on purpose — the wire response should
/// not reveal which check failed.
#[derive(Debug)]
pub enum AuthError {
    /// No `Authorization: Bearer <jwt>` header, or it was malformed.
    Missing,
    /// Signature, expiry, audience, or required claims rejected.
    Invalid(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::Missing => write!(f, "missing or malformed bearer token"),
            // The detail is for server-side logs only; the wire response is a
            // fixed string (see `http::unauthorized`), so it never leaks.
            AuthError::Invalid(reason) => write!(f, "invalid bearer token: {reason}"),
        }
    }
}

impl std::error::Error for AuthError {}

/// Verifies bearer JWTs against a configured key and extracts the [`Identity`].
pub struct JwtVerifier {
    key: DecodingKey,
    validation: Validation,
}

impl JwtVerifier {
    /// HS256 with a shared secret (symmetric — the simplest deployment; the
    /// signer and verifier share the secret).
    pub fn hs256(secret: &[u8]) -> Self {
        let mut validation = Validation::new(Algorithm::HS256);
        // Expiry is mandatory: a non-expiring memory token is a standing key.
        validation.set_required_spec_claims(&["exp"]);
        // Audience is opt-in (see `with_audience`); without it, don't reject a
        // token merely for carrying an `aud` claim meant for another verifier.
        validation.validate_aud = false;
        Self {
            key: DecodingKey::from_secret(secret),
            validation,
        }
    }

    /// RS256 with a PEM-encoded RSA public key (asymmetric — the signer holds
    /// the private key, this server only ever verifies).
    pub fn rs256_pem(pem: &[u8]) -> anyhow::Result<Self> {
        let key = DecodingKey::from_rsa_pem(pem)?;
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_required_spec_claims(&["exp"]);
        validation.validate_aud = false;
        Ok(Self { key, validation })
    }

    /// Require an audience (this server's identifier) so a token minted for a
    /// different service is rejected even if signed by the same key.
    #[must_use]
    pub fn with_audience(mut self, aud: &str) -> Self {
        self.validation.validate_aud = true;
        self.validation.set_audience(&[aud]);
        self
    }

    /// Verify a raw token string and return its identity.
    pub fn verify(&self, token: &str) -> Result<Identity, AuthError> {
        let data = decode::<Claims>(token, &self.key, &self.validation)
            .map_err(|e| AuthError::Invalid(e.to_string()))?;
        let c = data.claims;
        if c.tenant.trim().is_empty() || c.user.trim().is_empty() {
            return Err(AuthError::Invalid("empty tenant/user claim".into()));
        }
        Ok(Identity {
            tenant: c.tenant,
            user: c.user,
        })
    }

    /// Extract and verify from an `Authorization` header value
    /// (`Bearer <jwt>`, case-insensitive scheme).
    pub fn verify_header(&self, header: Option<&str>) -> Result<Identity, AuthError> {
        let raw = header.ok_or(AuthError::Missing)?;
        let token = raw
            .strip_prefix("Bearer ")
            .or_else(|| raw.strip_prefix("bearer "))
            .ok_or(AuthError::Missing)?;
        self.verify(token.trim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{encode, get_current_timestamp, EncodingKey, Header};
    use serde::Serialize;

    #[derive(Serialize)]
    struct EncClaims {
        tenant: String,
        user: String,
        exp: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        aud: Option<String>,
    }

    const SECRET: &[u8] = b"a-test-signing-secret-not-for-prod";

    fn mint(tenant: &str, user: &str, exp: u64, aud: Option<&str>) -> String {
        let claims = EncClaims {
            tenant: tenant.into(),
            user: user.into(),
            exp,
            aud: aud.map(Into::into),
        };
        encode(&Header::new(Algorithm::HS256), &claims, &EncodingKey::from_secret(SECRET)).unwrap()
    }

    fn future() -> u64 {
        get_current_timestamp() + 3600
    }

    #[test]
    fn valid_token_yields_identity() {
        let v = JwtVerifier::hs256(SECRET);
        let id = v.verify(&mint("ws:1", "user:a", future(), None)).unwrap();
        assert_eq!(
            id,
            Identity {
                tenant: "ws:1".into(),
                user: "user:a".into()
            }
        );
    }

    #[test]
    fn wrong_secret_is_rejected() {
        let other = JwtVerifier::hs256(b"a-different-secret");
        let err = other.verify(&mint("ws:1", "user:a", future(), None));
        assert!(matches!(err, Err(AuthError::Invalid(_))));
    }

    #[test]
    fn expired_token_is_rejected() {
        let v = JwtVerifier::hs256(SECRET);
        // Well past the library's default 60s expiry leeway.
        let past = get_current_timestamp() - 7200;
        let err = v.verify(&mint("ws:1", "user:a", past, None));
        assert!(matches!(err, Err(AuthError::Invalid(_))));
    }

    #[test]
    fn missing_claim_is_rejected() {
        // A token with no tenant/user claim fails to deserialize into Claims.
        #[derive(Serialize)]
        struct Thin {
            exp: u64,
        }
        let token = encode(
            &Header::new(Algorithm::HS256),
            &Thin { exp: future() },
            &EncodingKey::from_secret(SECRET),
        )
        .unwrap();
        let v = JwtVerifier::hs256(SECRET);
        assert!(matches!(v.verify(&token), Err(AuthError::Invalid(_))));
    }

    #[test]
    fn empty_claim_is_rejected() {
        let v = JwtVerifier::hs256(SECRET);
        assert!(matches!(
            v.verify(&mint("ws:1", "", future(), None)),
            Err(AuthError::Invalid(_))
        ));
    }

    #[test]
    fn audience_must_match_when_required() {
        let v = JwtVerifier::hs256(SECRET).with_audience("antumbra");
        assert!(v.verify(&mint("ws:1", "user:a", future(), Some("antumbra"))).is_ok());
        assert!(matches!(
            v.verify(&mint("ws:1", "user:a", future(), Some("other-service"))),
            Err(AuthError::Invalid(_))
        ));
    }

    #[test]
    fn header_parsing() {
        let v = JwtVerifier::hs256(SECRET);
        let token = mint("ws:1", "user:a", future(), None);
        assert!(v.verify_header(Some(&format!("Bearer {token}"))).is_ok());
        assert!(v.verify_header(Some(&format!("bearer {token}"))).is_ok());
        assert!(matches!(v.verify_header(Some(&token)), Err(AuthError::Missing)));
        assert!(matches!(v.verify_header(None), Err(AuthError::Missing)));
    }
}
