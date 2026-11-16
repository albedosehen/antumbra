//! The bearer token: telling one apart from a sign-in link, and reading the
//! workspace it grants.
//!
//! The server verifies the token; nothing here does. The claims are read only
//! to name the workspace in the agent's settings, so a hosted user is never
//! asked for an id their token already carries; the reading is
//! [`antumbra_auth::unverified`]'s, the same the console does.

use serde_json::Value;

pub use antumbra_auth::unverified::{expires, is_operator, tenant as workspace};

/// Whether `text` has a JWT's shape: three base64url parts.
pub fn looks_like_jwt(text: &str) -> bool {
    let parts: Vec<&str> = text.split('.').collect();
    parts.len() == 3
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'=')
        })
}

/// What the user pasted: a token, or a sign-in link to trade for one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pasted {
    Token(String),
    Link(String),
}

/// Reads what was pasted, or says what it should have been.
pub fn read_pasted(text: &str) -> anyhow::Result<Pasted> {
    let text = text.trim().trim_matches('"').trim_matches('\'');
    let text = text.strip_prefix("Bearer ").unwrap_or(text).trim();
    if text.starts_with("https://") || text.starts_with("http://") {
        return Ok(Pasted::Link(text.to_string()));
    }
    if looks_like_jwt(text) {
        return Ok(Pasted::Token(text.to_string()));
    }
    anyhow::bail!(
        "that is neither a token (three parts separated by dots) nor a sign-in link (https://...)"
    )
}

/// The token in a sign-in link's answer, `{"token": "..."}`.
pub fn from_signin(body: &Value) -> Option<String> {
    body.get("token")
        .and_then(Value::as_str)
        .filter(|t| looks_like_jwt(t))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use base64::Engine;

    use super::*;

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
    fn the_workspace_comes_from_the_tenant_claim() {
        let t = token(r#"{"tenant":"ws:acme","user":"user:dana","exp":1893456000}"#);
        assert!(looks_like_jwt(&t));
        assert_eq!(workspace(&t).as_deref(), Some("ws:acme"));
        assert_eq!(expires(&t), Some(1_893_456_000));
        assert_eq!(workspace("a.b.c"), None);
    }

    #[test]
    fn only_a_token_that_says_so_is_an_operators() {
        assert!(is_operator(&token(
            r#"{"tenant":"ws:acme","user":"user:dana","operator":true}"#
        )));
        assert!(!is_operator(&token(
            r#"{"tenant":"ws:acme","user":"user:dana"}"#
        )));
        assert!(!is_operator("a.b.c"));
    }

    #[test]
    fn pasted_text_is_a_token_or_a_link() {
        let t = token(r#"{"tenant":"ws:x"}"#);
        assert_eq!(
            read_pasted(&format!("  {t}\n")).unwrap(),
            Pasted::Token(t.clone())
        );
        assert_eq!(
            read_pasted(&format!("Bearer {t}")).unwrap(),
            Pasted::Token(t.clone())
        );
        assert_eq!(
            read_pasted("https://control.example.com/magic/verify?token=abc").unwrap(),
            Pasted::Link("https://control.example.com/magic/verify?token=abc".into())
        );
        assert!(read_pasted("hunter2").is_err());
        assert!(!looks_like_jwt("a.b"));
        assert!(!looks_like_jwt("a..c"));
    }

    #[test]
    fn a_signin_answer_carries_the_token() {
        let t = token(r#"{"tenant":"ws:x"}"#);
        assert_eq!(from_signin(&serde_json::json!({ "token": t })), Some(t));
        assert_eq!(
            from_signin(&serde_json::json!({ "status": "link sent" })),
            None
        );
        assert_eq!(from_signin(&serde_json::json!({ "token": "nope" })), None);
    }
}
