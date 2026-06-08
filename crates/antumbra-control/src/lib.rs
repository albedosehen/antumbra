//! The hosted-onboarding control plane: the **issuer** side of the
//! `antumbra-auth` token contract. It gates signup behind invite codes,
//! provisions a fresh tenant for each new account, and issues the RS256 JWT the
//! MCP server verifies (with its matching public key). The offline tier is
//! unaffected: it keeps minting HS256 tokens via the CLI.
//!
//! This module is the pure flow over the store; it takes an already-[`VerifiedIdentity`]
//! (the OAuth / magic-link verification lives behind a separate provider seam and
//! the HTTP surface). So the full signup→token→login path is testable offline,
//! with no external auth service.

use std::time::Duration;

use chrono::Utc;

use antumbra_auth::mint_rs256;
use antumbra_core::{AntumbraError, Result, TenantId, UserId};
use antumbra_store::repo::account::{self, Account};
use antumbra_store::repo::{invite, principal};
use antumbra_store::Store;

mod magic;
pub use magic::{MagicLink, Mailer};

/// A verified external identity (the output of an OAuth callback or a magic-link
/// click). `subject` is the stable, canonical login key (e.g. `github:12345` or
/// `email:a@b.com`); `email` / `display` are provenance.
#[derive(Debug, Clone)]
pub struct VerifiedIdentity {
    pub subject: String,
    pub email: Option<String>,
    pub display: Option<String>,
}

impl VerifiedIdentity {
    pub fn new(subject: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            email: None,
            display: None,
        }
    }

    #[must_use]
    pub fn with(mut self, email: Option<String>, display: Option<String>) -> Self {
        self.email = email;
        self.display = display;
        self
    }
}

/// The token issuer: the RSA private key it signs with, the audience (the MCP
/// server's identifier), and the token lifetime. The control plane holds the
/// private key; the MCP server only verifies with the matching public key.
pub struct Issuer {
    private_pem: Vec<u8>,
    audience: String,
    ttl: Duration,
}

impl Issuer {
    pub fn new(private_pem: Vec<u8>, audience: impl Into<String>, ttl: Duration) -> Self {
        Self {
            private_pem,
            audience: audience.into(),
            ttl,
        }
    }

    fn mint(&self, tenant: &str, user: &str) -> Result<String> {
        mint_rs256(
            &self.private_pem,
            tenant,
            user,
            self.ttl,
            Some(&self.audience),
        )
        .map_err(|e| AntumbraError::other(format!("mint token: {e}")))
    }
}

/// A fresh, unguessable invite code for an operator to hand out. (Persist it with
/// [`antumbra_store::repo::invite::mint`].)
pub fn new_invite_code() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Sign up a verified identity: redeem the invite, provision a **new** tenant and
/// its owner user, record the account, and issue a token. Errors if the invite is
/// unknown/expired (consumed on success) or the identity is already registered.
pub async fn signup(
    store: &Store,
    identity: &VerifiedIdentity,
    invite_code: &str,
    issuer: &Issuer,
) -> Result<String> {
    if account::get_by_subject(store, &identity.subject)
        .await?
        .is_some()
    {
        return Err(AntumbraError::other(
            "identity already registered; log in instead",
        ));
    }
    let now = Utc::now();
    // Gate first: consume the invite (single-use) before doing any provisioning.
    invite::redeem(store, invite_code, now).await?;

    // Each signup gets its own isolated workspace, owned by its first user.
    let tenant = TenantId::new(format!("ws:{}", uuid::Uuid::new_v4().simple()));
    let user = UserId::new("user:owner");
    principal::provision(store, &tenant, &user).await?;
    account::create(
        store,
        &Account {
            subject: identity.subject.clone(),
            tenant: tenant.as_str().to_string(),
            user: user.as_str().to_string(),
            email: identity.email.clone(),
            display: identity.display.clone(),
            created_at: now,
        },
    )
    .await?;

    issuer.mint(tenant.as_str(), user.as_str())
}

/// Log an already-registered identity in: look up its account and issue a token.
/// Errors if the identity has no account (it must sign up first).
pub async fn login(store: &Store, identity: &VerifiedIdentity, issuer: &Issuer) -> Result<String> {
    let account = account::get_by_subject(store, &identity.subject)
        .await?
        .ok_or_else(|| AntumbraError::other("no account for this identity; sign up first"))?;
    issuer.mint(&account.tenant, &account.user)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_auth::JwtVerifier;
    use antumbra_store::EMBED_DIM;

    // The throwaway RSA-2048 keypair: the control plane signs with the private
    // key; the MCP-side verifier uses the public key. Test-only.
    const RS_PRIV: &[u8] = include_bytes!("../tests/test_jwt_priv.pem");
    const RS_PUB: &[u8] = include_bytes!("../tests/test_jwt_pub.pem");

    const AUD: &str = "antumbra";

    fn issuer() -> Issuer {
        Issuer::new(RS_PRIV.to_vec(), AUD, Duration::from_secs(3600))
    }

    /// Verify an issued token the same way the MCP server would, returning
    /// `(tenant, user)`.
    fn verify(token: &str) -> (String, String) {
        let id = JwtVerifier::rs256_pem(RS_PUB)
            .unwrap()
            .with_audience(AUD)
            .verify(token)
            .unwrap();
        (id.tenant, id.user)
    }

    #[tokio::test]
    async fn invite_signup_provisions_a_tenant_and_issues_a_verifiable_token() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        invite::mint(&store, "good-code", None).await.unwrap();
        let id = VerifiedIdentity::new("github:42").with(Some("a@b.com".into()), None);

        let token = signup(&store, &id, "good-code", &issuer()).await.unwrap();
        let (tenant, user) = verify(&token);
        assert!(tenant.starts_with("ws:"), "a fresh workspace");
        assert_eq!(user, "user:owner");

        // Login returns a token for the SAME tenant/user.
        let login_token = login(&store, &id, &issuer()).await.unwrap();
        assert_eq!(verify(&login_token), (tenant, user));
    }

    #[tokio::test]
    async fn signup_requires_a_valid_unused_invite() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let id = VerifiedIdentity::new("github:1");
        // No invite minted → rejected.
        assert!(signup(&store, &id, "nope", &issuer()).await.is_err());

        // A code is single-use: the second signup on it fails.
        invite::mint(&store, "one-shot", None).await.unwrap();
        signup(&store, &id, "one-shot", &issuer()).await.unwrap();
        let other = VerifiedIdentity::new("github:2");
        assert!(signup(&store, &other, "one-shot", &issuer()).await.is_err());
    }

    // The whole passwordless path, offline: request a magic link → follow it →
    // verify → invite-signup → an RS256 token the server would accept.
    #[tokio::test]
    async fn magic_link_then_invite_signup_issues_a_verifiable_token() {
        use std::sync::Mutex;
        struct Capture(Mutex<Vec<String>>);
        impl Mailer for Capture {
            fn send_link(&self, _to: &str, link: &str) -> Result<()> {
                self.0.lock().unwrap().push(link.to_string());
                Ok(())
            }
        }

        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        invite::mint(&store, "inv", None).await.unwrap();

        let mailer = Capture(Mutex::new(Vec::new()));
        let ml = MagicLink::new(b"magic", Duration::from_secs(600), "https://app", &mailer);
        ml.request("ada@x.com").unwrap();
        let token = {
            let links = mailer.0.lock().unwrap();
            links[0].split("token=").nth(1).unwrap().to_string()
        };
        let identity = ml.verify(&token).unwrap();
        assert_eq!(identity.subject, "email:ada@x.com");

        let jwt = signup(&store, &identity, "inv", &issuer()).await.unwrap();
        let (tenant, user) = verify(&jwt);
        assert!(tenant.starts_with("ws:"));
        assert_eq!(user, "user:owner");
    }

    #[tokio::test]
    async fn duplicate_signup_and_loginless_identity_are_rejected() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let id = VerifiedIdentity::new("github:7");
        // Login before signup → no account.
        assert!(login(&store, &id, &issuer()).await.is_err());

        invite::mint(&store, "c1", None).await.unwrap();
        invite::mint(&store, "c2", None).await.unwrap();
        signup(&store, &id, "c1", &issuer()).await.unwrap();
        // Same identity signing up again → already registered (and c2 untouched).
        assert!(signup(&store, &id, "c2", &issuer()).await.is_err());
        assert!(invite::get(&store, "c2").await.unwrap().is_some());
    }
}
