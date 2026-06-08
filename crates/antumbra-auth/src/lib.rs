//! The JWT token contract for the antumbra networked surface (ADR-0015/0016).
//!
//! The shared seam between the **issuer** (the offline mint, or the hosted
//! control plane) and the **verifier** (the MCP server). A client proves its
//! identity with a signed JWT whose claims carry the `tenant` and `user`; the
//! server verifies the signature (and expiry) and binds those claims as `$auth`
//! binding the same `(tenant, user)` the engine's record-access `SIGNIN` expects.
//! There is no token-to-identity lookup table: the verified token *is* the
//! identity, so a leaked token grants exactly its claimed scope and nothing
//! wider. The signer holds the key (a symmetric secret for HS256, or the auth
//! service's RSA private key for RS256); the server only ever verifies.

use jsonwebtoken::{
    decode, encode, get_current_timestamp, Algorithm, DecodingKey, EncodingKey, Header, Validation,
};
use serde::{Deserialize, Serialize};

/// The verified identity carried by a request's bearer token: what becomes
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

/// The claims a minted token carries (the inverse of [`Claims`]). `aud` is set
/// only by the RS256 (hosted) mint so a token can be scoped to one server; the
/// HS256 mint omits it (the offline verifier opts out of audience by default).
#[derive(Debug, Serialize)]
struct MintClaims<'a> {
    tenant: &'a str,
    user: &'a str,
    exp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    aud: Option<&'a str>,
}

/// Mint a long-lived, scope-bound HS256 token for a non-interactive client (a
/// lifecycle hook), valid for `ttl` from now. The token **is** the identity (no
/// lookup table), so it grants exactly `(tenant, user)` and nothing wider, and it
/// verifies through the same [`JwtVerifier::verify`] path. `ttl` is finite on
/// purpose: a hook token is long-lived, not a non-expiring standing key (the
/// verifier requires `exp`). This is the offline / self-hosted mint, signed with
/// the server's own symmetric secret; an RS256 deployment mints via its auth
/// service's private key instead.
pub fn mint_hs256(
    secret: &[u8],
    tenant: &str,
    user: &str,
    ttl: std::time::Duration,
) -> Result<String, AuthError> {
    if tenant.trim().is_empty() || user.trim().is_empty() {
        return Err(AuthError::Invalid("empty tenant/user".into()));
    }
    let claims = MintClaims {
        tenant,
        user,
        exp: get_current_timestamp() + ttl.as_secs(),
        aud: None,
    };
    encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret),
    )
    .map_err(|e| AuthError::Invalid(e.to_string()))
}

/// Mint a scope-bound **RS256** token signed with the auth service's RSA private
/// key: the asymmetric, hosted counterpart to [`mint_hs256`] (ADR-0015/0016).
/// The control plane (the issuer) holds the private key and mints on signup /
/// login; the MCP server only ever verifies, with the matching public key
/// ([`JwtVerifier::rs256_pem`]). Pass the server's `audience` so a token minted
/// for a different service is rejected even under the same key. `ttl` is finite
/// (the verifier requires `exp`).
pub fn mint_rs256(
    private_pem: &[u8],
    tenant: &str,
    user: &str,
    ttl: std::time::Duration,
    audience: Option<&str>,
) -> Result<String, AuthError> {
    if tenant.trim().is_empty() || user.trim().is_empty() {
        return Err(AuthError::Invalid("empty tenant/user".into()));
    }
    let claims = MintClaims {
        tenant,
        user,
        exp: get_current_timestamp() + ttl.as_secs(),
        aud: audience,
    };
    let key =
        EncodingKey::from_rsa_pem(private_pem).map_err(|e| AuthError::Invalid(e.to_string()))?;
    encode(&Header::new(Algorithm::RS256), &claims, &key)
        .map_err(|e| AuthError::Invalid(e.to_string()))
}

/// Why a token was rejected. Kept coarse on purpose: the wire response should
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
    /// HS256 with a shared secret (symmetric, the simplest deployment; the
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

    /// RS256 with a PEM-encoded RSA public key (asymmetric, the signer holds
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
        encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(SECRET),
        )
        .unwrap()
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

    // A throwaway RSA-2048 keypair for the asymmetric mint↔verify round-trip:
    // the control plane (issuer) signs with the private key, the MCP server
    // verifies with the public key. Test-only, never a real credential.
    const RS_PRIV: &[u8] = b"-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDMRax89LGXOtI6
2/wNDH9gV2AvSTetDqsPIAX0JsCVRWt/XqWm/F/mJrfbgsmjlLVWy8PJWl0+hdtu
wo93naG4DyWJW+8AEX8AL1HjrPBimWe3+qXBFxvPVygNqT2PvGYmO7X+jtHXEKvR
aVcOnOByukLaXuSQVaRa40VFjBIU0WKBlU72apuPAtoB9taZ6roz7rJ8iTGE4DYt
0LflF5LjQb6IjANfEAPcZICt0ufzD6OgG+1peuhqgwZErWfQgZ9eOOT2O9SdEB5z
nc/l1LR/F94WhOO29JTGohVN46gdcEkS9/FGBBi5/4p3L74LcpJPAnIKWv2jpD2n
q0NVpW13AgMBAAECggEAUHJ4FdYAQsDFnqyYPUNYvsZqePTq2lrWf2RrM9Y3LhJi
3YyWzIbD9c31xpthcezU5dPlzVyrMD5jRuGUwtTvpZ9BdzEflPVPAPGh3Hp1ST+F
G2247ax+JU/71DV8qyjVSeVmLVRty7cjE5vaz0R1GHnGbl3EwhsYWTr8QwGA9XU0
qWwDnBcBDpSuqM3S3FgbfE6xKzOtEIsmtwwVHi112qlWcSXFrSSFwm7l5/0huURV
6dAyeugDUWvZTjTZ/+liCsmpEbw4ftxeMLMONaDvQIQN2fgBvvGCBwJI/8PnV3vG
nGI//3ECD9ru1pjsqLngcA8ugpoT+FpN4IG5WLJ94QKBgQDkvIUbmLn9TkrvVgki
FHoOl3Y1uVYqYnYhaKc1GigMU2VJkW5NP6m19KEw13vkdc+KwZEzzxZIN1LmwCPc
TfFlqjCFs7/Urtvvnl3bnw1o2KrhXiR1JQrcoLDzwkid8Q6mEQ66MiEgDkBjLaOu
rkWOGByBgwujTDEkxsbv9XlJvwKBgQDknqvkppDjEoozBfH+S13n7qebJ3z6hAgN
fOgqUN8FTRO7gHebE0wCz9AVbXMvduv0oAiW77pmYUIl4otoWc3M/klcRTB1BOix
Eu5vD/oCu9ZHoCMO57yFOIa+YNjryer9NLm9ow/jTC5BUYYn5/jQJGnkNjWZM57q
Wh8Vy+4aSQKBgQDQNCFdG0nAnnFrJX8uvEDV41xATrF15yXsBxycI3Dst0RtEKm8
OwS5kTDgCmTFcc82WDdZV1jK50DYtXBu6aufhKiiKxmj+H5NwHNio4ZLN11jwpOg
5dTbOpGXb/M1gOR6mPA038hzK0XEgRiKuiqpypy37pa7T3E0LpOKfICodQKBgQDI
OkemDFPc7EHpig11gCCQny5f7ufAqJ484eacGRQamnTrxQn74Zyy4bsG6UL2kRr6
tqaPOwpv3EKI167tB6n9HcC2dUqJUnFRlJkK4F1Aw65aMOBDj6ZGr0kjt8KET+Xl
OaZrdkLV+cSRJItwq/P4p8uuOeQbd2B5M9EB0AeLMQKBgGQVVr40SZlQmMntXszm
0otGkCBvq5kMmhPFWt+zM8ZeAjf9lOIiYdrDNxje2HpK+6nzfLLwTwdwh31bWcvV
NYpgHMsZXpcBVVAT3Nm6eP8OaPsCB/tzMfRWN5GlrqoqprsIl88tSnpmvQ0KVS9l
nDEh0mKr23w08IOzqgciZR8L
-----END PRIVATE KEY-----
";
    const RS_PUB: &[u8] = b"-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzEWsfPSxlzrSOtv8DQx/
YFdgL0k3rQ6rDyAF9CbAlUVrf16lpvxf5ia324LJo5S1VsvDyVpdPoXbbsKPd52h
uA8liVvvABF/AC9R46zwYplnt/qlwRcbz1coDak9j7xmJju1/o7R1xCr0WlXDpzg
crpC2l7kkFWkWuNFRYwSFNFigZVO9mqbjwLaAfbWmeq6M+6yfIkxhOA2LdC35ReS
40G+iIwDXxAD3GSArdLn8w+joBvtaXroaoMGRK1n0IGfXjjk9jvUnRAec53P5dS0
fxfeFoTjtvSUxqIVTeOoHXBJEvfxRgQYuf+Kdy++C3KSTwJyClr9o6Q9p6tDVaVt
dwIDAQAB
-----END PUBLIC KEY-----
";

    #[test]
    fn rs256_mint_verifies_against_the_server_public_key() {
        use std::time::Duration;
        let token = mint_rs256(
            RS_PRIV,
            "ws:1",
            "user:a",
            Duration::from_secs(3600),
            Some("antumbra"),
        )
        .unwrap();
        // The MCP server verifies the issuer's token with the matching public key.
        let v = JwtVerifier::rs256_pem(RS_PUB)
            .unwrap()
            .with_audience("antumbra");
        assert_eq!(
            v.verify(&token).unwrap(),
            Identity {
                tenant: "ws:1".into(),
                user: "user:a".into()
            }
        );
        // A token minted for a different audience is rejected under the same key.
        let mismatched = mint_rs256(
            RS_PRIV,
            "ws:1",
            "user:a",
            Duration::from_secs(3600),
            Some("other"),
        )
        .unwrap();
        assert!(matches!(v.verify(&mismatched), Err(AuthError::Invalid(_))));
        // An empty scope is refused at mint time.
        assert!(mint_rs256(RS_PRIV, "", "user:a", Duration::from_secs(60), None).is_err());
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
        assert!(v
            .verify(&mint("ws:1", "user:a", future(), Some("antumbra")))
            .is_ok());
        assert!(matches!(
            v.verify(&mint("ws:1", "user:a", future(), Some("other-service"))),
            Err(AuthError::Invalid(_))
        ));
    }

    #[test]
    fn a_minted_token_verifies_to_its_identity() {
        let token = mint_hs256(
            SECRET,
            "ws:1",
            "user:a",
            std::time::Duration::from_secs(3600),
        )
        .unwrap();
        let id = JwtVerifier::hs256(SECRET).verify(&token).unwrap();
        assert_eq!(
            id,
            Identity {
                tenant: "ws:1".into(),
                user: "user:a".into()
            }
        );
    }

    #[test]
    fn a_minted_token_is_rejected_by_a_different_secret() {
        let token = mint_hs256(
            SECRET,
            "ws:1",
            "user:a",
            std::time::Duration::from_secs(3600),
        )
        .unwrap();
        let other = JwtVerifier::hs256(b"a-different-secret");
        assert!(matches!(other.verify(&token), Err(AuthError::Invalid(_))));
    }

    #[test]
    fn minting_an_empty_scope_is_refused() {
        assert!(matches!(
            mint_hs256(SECRET, "ws:1", "  ", std::time::Duration::from_secs(60)),
            Err(AuthError::Invalid(_))
        ));
    }

    #[test]
    fn header_parsing() {
        let v = JwtVerifier::hs256(SECRET);
        let token = mint("ws:1", "user:a", future(), None);
        assert!(v.verify_header(Some(&format!("Bearer {token}"))).is_ok());
        assert!(v.verify_header(Some(&format!("bearer {token}"))).is_ok());
        assert!(matches!(
            v.verify_header(Some(&token)),
            Err(AuthError::Missing)
        ));
        assert!(matches!(v.verify_header(None), Err(AuthError::Missing)));
    }
}
