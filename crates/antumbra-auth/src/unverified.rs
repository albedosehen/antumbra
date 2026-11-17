//! A token's claims, read without verifying it.
//!
//! The server verifies every token; nothing here does. These readers only
//! choose and describe tokens already on hand: the console picks the
//! operator's among the tokens it finds, setup names the workspace a token
//! grants so a hosted user is never asked for an id their token carries, and
//! says when it expires. What a token *may do* is the server's decision, made
//! on the signature. Two binaries decoded the second segment their own way;
//! this is the one way.

use base64::Engine;
use serde_json::Value;

/// The claims in `token`'s second segment, unverified: `None` for anything
/// that is not three dot-separated parts with base64url JSON in the middle.
pub fn claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// The workspace (tenant) the token grants.
pub fn tenant(token: &str) -> Option<String> {
    claims(token)?
        .get("tenant")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// The token's own id (`jti`), which a revocation names.
pub fn jti(token: &str) -> Option<String> {
    claims(token)?
        .get("jti")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// When the token expires, as seconds since the epoch.
pub fn expires(token: &str) -> Option<i64> {
    claims(token)?.get("exp").and_then(Value::as_i64)
}

/// Whether the token says it is its server's operator's, which the console
/// needs to act on the substrate. Read, not verified: it only chooses among
/// the tokens on hand, and the server decides what each may do.
pub fn is_operator(token: &str) -> bool {
    claims(token)
        .and_then(|c| c.get("operator").and_then(Value::as_bool))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    fn token(claims: &str) -> String {
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        format!(
            "{}.{}.{}",
            engine.encode(r#"{"alg":"HS256","typ":"JWT"}"#),
            engine.encode(claims),
            engine.encode("signature")
        )
    }

    #[test]
    fn the_claims_are_the_second_segment_decoded() -> Result<()> {
        let t = token(r#"{"tenant":"ws:acme","user":"user:dana","exp":1893456000}"#);
        let Some(read) = claims(&t) else {
            anyhow::bail!("a well-formed token has claims");
        };
        if read["user"] != "user:dana" {
            anyhow::bail!("the claims were misread: {read}");
        }
        if tenant(&t).as_deref() != Some("ws:acme") || expires(&t) != Some(1_893_456_000) {
            anyhow::bail!("the tenant and expiry come from their claims");
        }
        Ok(())
    }

    #[test]
    fn only_a_token_that_says_so_is_an_operators() -> Result<()> {
        assert!(is_operator(&token(
            r#"{"tenant":"ws:acme","user":"user:dana","operator":true}"#
        )));
        assert!(!is_operator(&token(
            r#"{"tenant":"ws:acme","user":"user:dana","operator":"yes"}"#
        )));
        assert!(!is_operator(&token(
            r#"{"tenant":"ws:acme","user":"user:dana"}"#
        )));
        Ok(())
    }

    #[test]
    fn what_is_not_a_token_has_no_claims() -> Result<()> {
        for not_a_token in ["", "a.b.c", "one-part", "a..c", &token("not json")] {
            assert!(claims(not_a_token).is_none(), "{not_a_token:?}");
            assert!(tenant(not_a_token).is_none(), "{not_a_token:?}");
            assert!(expires(not_a_token).is_none(), "{not_a_token:?}");
            assert!(!is_operator(not_a_token), "{not_a_token:?}");
        }
        // Padding, which some encoders add, is tolerated.
        let padded = format!("{}==", token(r#"{"tenant":"ws:p"}"#));
        assert_eq!(tenant(&padded).as_deref(), Some("ws:p"));
        Ok(())
    }
}
