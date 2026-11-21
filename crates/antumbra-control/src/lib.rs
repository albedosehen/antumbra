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
use antumbra_core::join::{self as joining, Answer};
use antumbra_core::{AntumbraError, Result, TenantId, UserId};
use antumbra_store::repo::account::{self, Account, Membership};
use antumbra_store::repo::join as store_join;
use antumbra_store::repo::{invite, magic_use, principal};
use antumbra_store::Store;

mod magic;
pub use magic::{JoinAsk, MagicLink, Mailer, VerifiedLink};

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
        return Err(AntumbraError::rejected(
            "identity already registered; log in instead",
        ));
    }
    let now = Utc::now();
    // Gate first: consume the invite (single-use) before doing any provisioning.
    invite::redeem(store, invite_code, now).await?;

    // Each signup gets its own isolated workspace, owned by its first user, and
    // a user id of its own: a user id names one user in one workspace, so a
    // rule that matches on the user alone never matches another signup's rows.
    let tenant = TenantId::new(format!("ws:{}", uuid::Uuid::new_v4().simple()));
    let user = UserId::new(format!("user:{}", uuid::Uuid::new_v4().simple()));
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
    // The first of the workspaces this account is in; joining another adds
    // a row beside it.
    account::add_membership(
        store,
        &Membership {
            subject: identity.subject.clone(),
            tenant: tenant.as_str().to_string(),
            user: user.as_str().to_string(),
            joined_at: now,
        },
    )
    .await?;

    issuer.mint(tenant.as_str(), user.as_str())
}

/// The workspaces a registered identity is in, the first joined first.
/// Errors if the identity has no account.
pub async fn workspaces(store: &Store, identity: &VerifiedIdentity) -> Result<Vec<Membership>> {
    let account = account::get_by_subject(store, &identity.subject)
        .await?
        .ok_or_else(|| AntumbraError::rejected("no account for this identity; sign up first"))?;
    account::memberships(store, &account).await
}

/// Log an already-registered identity in: look up its account and issue a token.
/// Errors if the identity has no account (it must sign up first).
pub async fn login(store: &Store, identity: &VerifiedIdentity, issuer: &Issuer) -> Result<String> {
    let account = account::get_by_subject(store, &identity.subject)
        .await?
        .ok_or_else(|| AntumbraError::rejected("no account for this identity; sign up first"))?;
    issuer.mint(&account.tenant, &account.user)
}

/// A fresh id for a request to join, made before the request is.
pub fn new_request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// What following a link came to.
#[derive(Debug)]
pub enum Followed {
    /// A signup or a login: the bearer token.
    Token(String),
    /// A request to join a workspace, now made with its address verified. It
    /// waits for the workspace's admins unless its code needed none.
    Asked {
        request: String,
        workspace: String,
        handle: String,
        standing: String,
    },
}

/// Complete a clicked magic link: verify it, consume its single-use id, then
/// sign up (when the link carried an invite), ask to join (when it carried a
/// request), or log in (when it carried neither).
///
/// The id is consumed up front, so a link dies on its first click regardless of
/// the outcome -- a replayed link cannot probe for a different result, and two
/// concurrent clicks cannot both win the race.
pub async fn follow(
    store: &Store,
    magic: &MagicLink<'_>,
    token: &str,
    issuer: &Issuer,
) -> Result<Followed> {
    let link = magic.verify(token)?;
    // Opportunistic ledger hygiene: entries for links that can no longer
    // verify are dead weight.
    magic_use::sweep_expired(store, Utc::now()).await?;
    if !magic_use::consume(store, &link.jti, link.expires_at).await? {
        return Err(AntumbraError::rejected("magic link already used"));
    }
    if let Some(ask) = link.join {
        let request = store_join::ask(
            store,
            store_join::Asking {
                id: &ask.request,
                code: &ask.code,
                handle: &ask.handle,
                display: ask.display,
                note: ask.note,
                host: ask.host,
                secret_digest: &ask.secret_digest,
                email: link.identity.email.clone(),
            },
            Utc::now(),
        )
        .await?;
        return Ok(Followed::Asked {
            request: request.key,
            workspace: request.tenant.as_str().to_string(),
            handle: request.handle,
            standing: request.standing.as_str().to_string(),
        });
    }
    match link.invite {
        Some(code) => signup(store, &link.identity, &code, issuer).await,
        None => login(store, &link.identity, issuer).await,
    }
    .map(Followed::Token)
}

/// [`follow`] for a link that signs up or logs in: the bearer token. A link
/// that asks to join is refused here; it is followed with [`follow`].
pub async fn authenticate(
    store: &Store,
    magic: &MagicLink<'_>,
    token: &str,
    issuer: &Issuer,
) -> Result<String> {
    match follow(store, magic, token, issuer).await? {
        Followed::Token(token) => Ok(token),
        Followed::Asked { .. } => Err(AntumbraError::rejected(
            "this link asks to join a workspace and signs no one in",
        )),
    }
}

/// Send the link that, followed, asks to join the workspace `ask.code` is
/// for. What can be refused without the workspace's admins is refused here,
/// before anything is sent: a code that admits no one, a handle that cannot be
/// one, a note too long. The request itself is made when the link is
/// followed, so admins are never shown an address nobody verified.
pub async fn ask_to_join(
    store: &Store,
    magic: &MagicLink<'_>,
    email: &str,
    mut ask: JoinAsk,
) -> Result<()> {
    if store_join::admits(store, &ask.code, Utc::now())
        .await?
        .is_none()
    {
        return Err(AntumbraError::rejected(store_join::NO_SUCH_CODE));
    }
    ask.handle = joining::handle_of(&ask.handle)?;
    ask.display = antumbra_core::profile::display_of(ask.display.as_deref().unwrap_or_default())?;
    ask.note = joining::note_of(ask.note.as_deref().unwrap_or_default())?;
    ask.host = ask.host.as_deref().and_then(machine_named);
    let digest = ask.secret_digest.trim().to_ascii_lowercase();
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(AntumbraError::rejected(
            "`secret_digest` is the SHA-256 of your secret, in hex",
        ));
    }
    ask.secret_digest = digest;
    magic.request_join(email, &ask)
}

/// A machine's name as it gave it, kept when it is short and plain.
fn machine_named(raw: &str) -> Option<String> {
    let name = raw.trim();
    (!name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
    .then(|| name.to_string())
}

/// How a request made through a link stands, for the person who can show its
/// secret.
#[derive(Debug)]
pub enum Collected {
    /// The link was not followed yet, or there is no such request: one
    /// answer for both.
    Unverified,
    /// As the workspace's server would say it, and not admitted.
    Standing(Answer),
    /// Admitted: they are this member of this workspace, with a token for it.
    Admitted {
        workspace: String,
        user: String,
        token: String,
    },
}

/// Collect a request to join that was made through a link. Approved, the
/// person becomes a member, their account is recorded as being in the
/// workspace (and is made, when this is the first the address is seen), and a
/// token is signed for them there. Said once.
pub async fn collect_join(
    store: &Store,
    request: &str,
    secret: &str,
    issuer: &Issuer,
) -> Result<Collected> {
    let digest = joining::digest(secret);
    let Some(held) = store_join::request(store, request.trim())
        .await?
        .filter(|held| held.secret == digest)
    else {
        return Ok(Collected::Unverified);
    };
    // Only a request whose address was verified is this issuer's to sign for.
    let Some(email) = held.email.clone() else {
        return Ok(Collected::Unverified);
    };
    let now = Utc::now();
    let (tenant, user) = match store_join::collect(store, &held.key, &digest, now).await? {
        Answer::Admitted { tenant, user } => (tenant, user),
        other => return Ok(Collected::Standing(other)),
    };
    let subject = format!("email:{email}");
    if account::get_by_subject(store, &subject).await?.is_none() {
        // The first this address is seen: its account is made with the
        // workspace it joined.
        account::create(
            store,
            &Account {
                subject: subject.clone(),
                tenant: tenant.as_str().to_string(),
                user: user.as_str().to_string(),
                email: Some(email),
                display: held.display.clone(),
                created_at: now,
            },
        )
        .await?;
    }
    account::add_membership(
        store,
        &Membership {
            subject,
            tenant: tenant.as_str().to_string(),
            user: user.as_str().to_string(),
            joined_at: now,
        },
    )
    .await?;
    Ok(Collected::Admitted {
        token: issuer.mint(tenant.as_str(), user.as_str())?,
        workspace: tenant.as_str().to_string(),
        user: user.as_str().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use antumbra_auth::JwtVerifier;
    use antumbra_store::testing::mem_store;

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
    fn verify(token: &str) -> Result<(String, String)> {
        let id = JwtVerifier::rs256_pem(RS_PUB)
            .map_err(|e| AntumbraError::other(format!("read the public key: {e}")))?
            .with_audience(AUD)
            .verify(token)
            .map_err(|e| AntumbraError::other(format!("verify the issued token: {e}")))?;
        Ok((id.tenant, id.user))
    }

    /// Captures the links it would have sent, so a test can follow them.
    struct Capture(Mutex<Vec<String>>);

    impl Capture {
        /// Every link sent, in order. The fake only ever pushes to the list, so
        /// a poisoned lock holds a whole list and is read as is.
        fn links(&self) -> Vec<String> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl Mailer for Capture {
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

    #[tokio::test]
    async fn invite_signup_provisions_a_tenant_and_issues_a_verifiable_token() -> Result<()> {
        let store = mem_store().await?;
        invite::mint(&store, "good-code", None).await?;
        let id = VerifiedIdentity::new("github:42").with(Some("a@b.com".into()), None);

        let token = signup(&store, &id, "good-code", &issuer()).await?;
        let (tenant, user) = verify(&token)?;
        assert!(tenant.starts_with("ws:"), "a fresh workspace");
        assert!(user.starts_with("user:") && user != "user:owner", "{user}");

        // Login returns a token for the SAME tenant/user.
        let login_token = login(&store, &id, &issuer()).await?;
        assert_eq!(verify(&login_token)?, (tenant.clone(), user.clone()));

        // Another signup is another workspace and another user id: a user id
        // names one user in one workspace.
        invite::mint(&store, "second", None).await?;
        let other = VerifiedIdentity::new("github:43");
        let (tenant2, user2) = verify(&signup(&store, &other, "second", &issuer()).await?)?;
        assert_ne!(tenant2, tenant);
        assert_ne!(user2, user, "every signup's first user had been user:owner");

        // Each account is in the workspace it was made with, and that one
        // only; an identity with no account is in none and is told so.
        let mine = workspaces(&store, &id).await?;
        assert_eq!(mine.len(), 1);
        assert_eq!(
            (mine[0].tenant.clone(), mine[0].user.clone()),
            (tenant, user)
        );
        let theirs = workspaces(&store, &other).await?;
        assert_eq!(theirs.len(), 1);
        assert_eq!(theirs[0].tenant, tenant2);
        assert!(workspaces(&store, &VerifiedIdentity::new("github:44"))
            .await
            .is_err());
        Ok(())
    }

    #[tokio::test]
    async fn signup_requires_a_valid_unused_invite() -> Result<()> {
        let store = mem_store().await?;
        let id = VerifiedIdentity::new("github:1");
        // No invite minted → rejected.
        assert!(signup(&store, &id, "nope", &issuer()).await.is_err());

        // A code is single-use: the second signup on it fails.
        invite::mint(&store, "one-shot", None).await?;
        signup(&store, &id, "one-shot", &issuer()).await?;
        let other = VerifiedIdentity::new("github:2");
        assert!(signup(&store, &other, "one-shot", &issuer()).await.is_err());
        Ok(())
    }

    // The whole passwordless path, offline: request a magic link → follow it →
    // verify → invite-signup → an RS256 token the server would accept.
    #[tokio::test]
    async fn magic_link_then_invite_signup_issues_a_verifiable_token() -> Result<()> {
        let store = mem_store().await?;
        invite::mint(&store, "inv", None).await?;

        let mailer = Capture(Mutex::new(Vec::new()));
        let ml = MagicLink::new(b"magic", Duration::from_secs(600), "https://app", &mailer);
        // A signup link carries the invite; `authenticate` then provisions.
        ml.request_signup("ada@x.com", "inv")?;
        let token = token_in(&mailer.links()[0])?;
        let jwt = authenticate(&store, &ml, &token, &issuer()).await?;
        let (tenant, user) = verify(&jwt)?;
        assert!(tenant.starts_with("ws:"));
        assert!(user.starts_with("user:") && user != "user:owner", "{user}");

        // A second click of a *login* link for the same email logs in (no invite,
        // no second invite needed) to the SAME workspace.
        ml.request("ada@x.com")?;
        let login_token = token_in(&mailer.links()[1])?;
        let jwt2 = authenticate(&store, &ml, &login_token, &issuer()).await?;
        assert_eq!(verify(&jwt2)?, (tenant, user));

        // Every link is single-use: replaying either consumed token is refused
        // as a deliberate rejection, not an internal fault.
        for replayed in [&token, &login_token] {
            let Err(err) = authenticate(&store, &ml, replayed, &issuer()).await else {
                return Err(AntumbraError::other("a replayed link was accepted"));
            };
            assert!(err.is_rejection(), "replay must be a rejection: {err}");
            assert!(err.to_string().contains("already used"));
        }
        // The pure verify still passes (signature + expiry are intact); it is
        // the ledger that kills the replay.
        assert!(ml.verify(&token).is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn duplicate_signup_and_loginless_identity_are_rejected() -> Result<()> {
        let store = mem_store().await?;
        let id = VerifiedIdentity::new("github:7");
        // Login before signup → no account.
        assert!(login(&store, &id, &issuer()).await.is_err());

        invite::mint(&store, "c1", None).await?;
        invite::mint(&store, "c2", None).await?;
        signup(&store, &id, "c1", &issuer()).await?;
        // Same identity signing up again → already registered (and c2 untouched).
        assert!(signup(&store, &id, "c2", &issuer()).await.is_err());
        assert!(invite::get(&store, "c2").await?.is_some());
        Ok(())
    }

    /// Joining with a verified address: nothing is asked of a workspace's
    /// admins until the link is followed; then the request stands with the
    /// address on it; approved, it is collected once for a token the server
    /// verifies, and the account is in the workspace it joined. A second
    /// workspace is a second membership of the same account.
    #[tokio::test]
    async fn a_verified_address_joins_a_workspace_and_then_another() -> Result<()> {
        use antumbra_core::join::{JoinCode, Standing};
        let store = mem_store().await?;
        let now = Utc::now();
        let admin = UserId::new("user:ada");
        for (tenant, code) in [("ws:crew", "crew-code"), ("ws:lab", "lab-code")] {
            let tenant = TenantId::new(tenant);
            principal::provision(&store, &tenant, &admin).await.ok();
            store_join::make_code(
                &store,
                &JoinCode {
                    key: joining::digest(code),
                    tenant,
                    label: None,
                    created_by: admin.clone(),
                    created_at: now,
                    expires_at: now + chrono::Duration::days(14),
                    uses_left: 5,
                    needs_approval: true,
                    revoked: false,
                },
            )
            .await?;
        }
        let mailer = Capture(Mutex::new(Vec::new()));
        let ml = MagicLink::new(b"magic", Duration::from_secs(600), "https://app", &mailer);
        let ask = |request: &str, code: &str, handle: &str| JoinAsk {
            request: request.into(),
            code: code.into(),
            handle: handle.into(),
            display: Some("Bodil Lindqvist".into()),
            note: None,
            host: Some("laptop".into()),
            secret_digest: joining::digest("my-secret"),
        };
        let token_of = |link: &str| -> Result<String> {
            link.split("token=")
                .nth(1)
                .map(str::to_string)
                .ok_or_else(|| AntumbraError::other("the link carries no token"))
        };

        // Refused before anything is sent: a code that admits no one, a
        // handle that cannot be one.
        assert!(
            ask_to_join(&store, &ml, "bodil@example.com", ask("r0", "nope", "bodil"))
                .await
                .is_err()
        );
        assert!(ask_to_join(
            &store,
            &ml,
            "bodil@example.com",
            ask("r0", "crew-code", "admin")
        )
        .await
        .is_err());
        assert!(mailer.links().is_empty());

        ask_to_join(
            &store,
            &ml,
            "bodil@example.com",
            ask("r1", "crew-code", "Bodil"),
        )
        .await?;
        // The link is sent and not followed: the admins have nothing, and
        // collecting says so in the words it has for a request that is not.
        let crew = TenantId::new("ws:crew");
        assert!(store_join::requests(&store, &crew).await?.is_empty());
        assert!(matches!(
            collect_join(&store, "r1", "my-secret", &issuer()).await?,
            Collected::Unverified
        ));

        let link = token_of(&mailer.links()[0])?;
        let Followed::Asked {
            request,
            workspace,
            handle,
            standing,
        } = follow(&store, &ml, &link, &issuer()).await?
        else {
            return Err(AntumbraError::other(
                "a link that asks to join signed someone in",
            ));
        };
        assert_eq!(
            (
                request.as_str(),
                workspace.as_str(),
                handle.as_str(),
                standing.as_str()
            ),
            ("r1", "ws:crew", "bodil", "pending")
        );
        let held = store_join::requests(&store, &crew).await?;
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].email.as_deref(), Some("bodil@example.com"));
        assert_eq!(held[0].standing, Standing::Pending);
        // Followed once: the link is spent, and signs no one in.
        assert!(follow(&store, &ml, &link, &issuer()).await.is_err());

        // Waiting, with the wrong secret, then approved and collected once.
        assert!(matches!(
            collect_join(&store, "r1", "my-secret", &issuer()).await?,
            Collected::Standing(Answer::Pending)
        ));
        assert!(matches!(
            collect_join(&store, "r1", "a-guess", &issuer()).await?,
            Collected::Unverified
        ));
        store_join::decide(&store, &crew, &admin, "r1", true, None, Utc::now()).await?;
        let Collected::Admitted {
            workspace,
            user,
            token,
        } = collect_join(&store, "r1", "my-secret", &issuer()).await?
        else {
            return Err(AntumbraError::other("an approved request was not admitted"));
        };
        assert_eq!(
            (workspace.as_str(), user.as_str()),
            ("ws:crew", "user:bodil")
        );
        assert_eq!(
            verify(&token)?,
            ("ws:crew".to_string(), "user:bodil".to_string())
        );
        assert!(matches!(
            collect_join(&store, "r1", "my-secret", &issuer()).await?,
            Collected::Standing(Answer::AlreadyCollected)
        ));

        // The address had no account: it has one now, in the workspace joined,
        // and logging in signs for it.
        let id = VerifiedIdentity::new("email:bodil@example.com");
        let mine = workspaces(&store, &id).await?;
        assert_eq!(mine.len(), 1);
        assert_eq!(mine[0].tenant, "ws:crew");
        assert_eq!(verify(&login(&store, &id, &issuer()).await?)?.0, "ws:crew");

        // A second workspace, as another user there: a second membership.
        ask_to_join(
            &store,
            &ml,
            "bodil@example.com",
            ask("r2", "lab-code", "bodil-l"),
        )
        .await?;
        follow(&store, &ml, &token_of(&mailer.links()[1])?, &issuer()).await?;
        store_join::decide(
            &store,
            &TenantId::new("ws:lab"),
            &admin,
            "r2",
            true,
            None,
            Utc::now(),
        )
        .await?;
        assert!(matches!(
            collect_join(&store, "r2", "my-secret", &issuer()).await?,
            Collected::Admitted { .. }
        ));
        let mine = workspaces(&store, &id).await?;
        let seen: Vec<(&str, &str)> = mine
            .iter()
            .map(|m| (m.tenant.as_str(), m.user.as_str()))
            .collect();
        assert_eq!(
            seen,
            [("ws:crew", "user:bodil"), ("ws:lab", "user:bodil-l")]
        );

        // A request made with no verified address is not this issuer's to sign.
        store_join::ask(
            &store,
            store_join::Asking {
                id: "r3",
                code: "crew-code",
                handle: "carl",
                display: None,
                note: None,
                host: None,
                secret_digest: &joining::digest("my-secret"),
                email: None,
            },
            Utc::now(),
        )
        .await?;
        store_join::decide(&store, &crew, &admin, "r3", true, None, Utc::now()).await?;
        assert!(matches!(
            collect_join(&store, "r3", "my-secret", &issuer()).await?,
            Collected::Unverified
        ));
        Ok(())
    }
}
