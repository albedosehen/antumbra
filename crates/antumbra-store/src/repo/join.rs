//! Join codes and join requests ([`antumbra_core::join`]).
//!
//! Both tables are the owner's alone (no `PERMISSIONS`, so a member's session
//! is denied): a code is a way into a workspace and a request holds the digest
//! of the secret its token is collected with. Every function here is called on
//! the owner's connection, after the server has decided the caller may.
//!
//! The three steps that two callers could race on (spending a code's use,
//! deciding a request, collecting an approval) are each one statement that
//! checks and writes, so the second caller finds nothing to change.

use std::collections::BTreeMap;

use chrono::{DateTime, Duration, Utc};
use serde_json::{json, Value};

use surql::query::builder::Query;
use surql::query::crud::query_records;
use surql::types::operators::eq;

use antumbra_core::join::{
    self, Answer, JoinCode, JoinRequest, Standing, DENIAL_HOURS, PENDING_AT_MOST,
};
use antumbra_core::profile::Profile;
use antumbra_core::{AntumbraError, Result, TenantId, UserId};

use crate::error::map;
use crate::repo::{principal, profile};
use crate::store::Store;

const CODE: &str = crate::tables::JOIN_CODE;
const REQUEST: &str = crate::tables::JOIN_REQUEST;

/// What anyone asking with a code that will not do is told, whatever is wrong
/// with it: unknown, expired, spent and revoked are not told apart.
pub const NO_SUCH_CODE: &str =
    "that join code is not one this server takes: it is unknown, expired, used up or revoked";

/// What anyone asking after a request they cannot show the secret of is told.
pub const NO_SUCH_REQUEST: &str = "no such request, or that is not its secret";

fn vars<const N: usize>(pairs: [(&str, Value); N]) -> BTreeMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Keep a code an admin made.
pub async fn make_code(store: &Store, code: &JoinCode) -> Result<()> {
    store.put(CODE, &code.key, code).await
}

/// A workspace's codes, newest first.
pub async fn codes(store: &Store, tenant: &TenantId) -> Result<Vec<JoinCode>> {
    let query = Query::new()
        .select(None)
        .from_table(CODE)
        .map_err(map)?
        .where_(eq("tenant", tenant.as_str()));
    let mut all: Vec<JoinCode> = query_records(store.client(), &query).await.map_err(map)?;
    all.sort_by_key(|c| std::cmp::Reverse(c.created_at));
    Ok(all)
}

/// Revoke one of a workspace's codes; whether it was theirs to revoke. A
/// request already made with it stays, and can still be decided.
pub async fn revoke_code(store: &Store, tenant: &TenantId, id: &str) -> Result<bool> {
    let changed: Vec<Value> = store
        .query_rows(
            &format!("UPDATE {CODE} SET revoked = true WHERE id = type::record('{CODE}', $id) AND tenant = $tenant RETURN AFTER"),
            vars([("id", json!(id)), ("tenant", json!(tenant.as_str()))]),
        )
        .await?;
    Ok(!changed.is_empty())
}

/// A workspace's requests, newest first.
pub async fn requests(store: &Store, tenant: &TenantId) -> Result<Vec<JoinRequest>> {
    let query = Query::new()
        .select(None)
        .from_table(REQUEST)
        .map_err(map)?
        .where_(eq("tenant", tenant.as_str()));
    let mut all: Vec<JoinRequest> = query_records(store.client(), &query).await.map_err(map)?;
    all.sort_by_key(|r| std::cmp::Reverse(r.created_at));
    Ok(all)
}

/// What a person sends to ask: the code, who they would be, and the digest of
/// the secret they will collect with.
pub struct Asking<'a> {
    /// The id the request will have, made by the caller.
    pub id: &'a str,
    pub code: &'a str,
    pub handle: &'a str,
    pub display: Option<String>,
    pub note: Option<String>,
    pub host: Option<String>,
    pub secret_digest: &'a str,
}

/// Ask to join the workspace a code is for.
///
/// Asking again with the same secret for a handle that already waits answers
/// with the request there is, so a retry is not a second request.
pub async fn ask(store: &Store, asking: Asking<'_>, now: DateTime<Utc>) -> Result<JoinRequest> {
    let code_id = join::digest(asking.code);
    let Some(code) = store.get_row::<JoinCode>(CODE, &code_id).await? else {
        return Err(AntumbraError::rejected(NO_SUCH_CODE));
    };
    if !code.admits(now) {
        return Err(AntumbraError::rejected(NO_SUCH_CODE));
    }
    let handle = join::handle_of(asking.handle)?;
    let user = join::user_of(&handle);
    let taken =
        || AntumbraError::rejected(format!("the handle `{handle}` is taken; choose another"));

    let theirs = requests(store, &code.tenant).await?;
    for held in theirs.iter().filter(|r| r.handle == handle) {
        let live = held.waits(now) || (held.standing == Standing::Approved && !held.lapsed(now));
        if live && held.secret == asking.secret_digest {
            return Ok(held.clone());
        }
        if live {
            return Err(taken());
        }
        let just_denied = held.standing == Standing::Denied
            && held
                .decided_at
                .is_some_and(|at| at + Duration::hours(DENIAL_HOURS) > now);
        if just_denied {
            return Err(AntumbraError::rejected(format!(
                "a request for `{handle}` was refused; it may be asked for again a day after"
            )));
        }
    }
    // A user id names one user in one workspace on a server.
    if !principal::workspaces_of(store, &user).await?.is_empty() {
        return Err(taken());
    }
    if theirs.iter().filter(|r| r.waits(now)).count() >= PENDING_AT_MOST {
        return Err(AntumbraError::rejected(
            "this workspace has as many requests waiting as it holds; ask its admins to decide some",
        ));
    }
    // One use of the code, spent only if one is left: two people racing on a
    // last use cannot both spend it.
    let spent: Vec<Value> = store
        .query_rows(
            &format!(
                "UPDATE type::record('{CODE}', $id) SET uses_left -= 1 \
                 WHERE uses_left > 0 AND revoked != true RETURN AFTER"
            ),
            vars([("id", json!(code_id))]),
        )
        .await?;
    if spent.is_empty() {
        return Err(AntumbraError::rejected(NO_SUCH_CODE));
    }
    let admitted = !code.needs_approval;
    let request = JoinRequest {
        key: asking.id.to_string(),
        tenant: code.tenant.clone(),
        handle,
        display: asking.display,
        note: asking.note,
        host: asking.host,
        secret: asking.secret_digest.to_string(),
        standing: if admitted {
            Standing::Approved
        } else {
            Standing::Pending
        },
        created_at: now,
        expires_at: code.expires_at,
        // A code that asks no one was its maker's decision when they made it.
        decided_at: admitted.then_some(now),
        decided_by: admitted.then(|| code.created_by.clone()),
        reason: None,
    };
    store.put(REQUEST, &request.key, &request).await?;
    Ok(request)
}

/// Decide a request, as `by`. The first decision stands: a second is refused
/// and told what the first was.
pub async fn decide(
    store: &Store,
    tenant: &TenantId,
    by: &UserId,
    id: &str,
    approve: bool,
    reason: Option<String>,
    now: DateTime<Utc>,
) -> Result<JoinRequest> {
    let unknown = || AntumbraError::rejected("no such request in this workspace");
    let Some(held) = store.get_row::<JoinRequest>(REQUEST, id).await? else {
        return Err(unknown());
    };
    if &held.tenant != tenant {
        return Err(unknown());
    }
    if held.standing == Standing::Pending && !held.waits(now) {
        return Err(AntumbraError::rejected(
            "that request expired before anyone decided it; they can ask again with a new code",
        ));
    }
    let to = if approve {
        Standing::Approved
    } else {
        Standing::Denied
    };
    let decided: Vec<JoinRequest> = store
        .query_rows(
            &format!(
                "UPDATE type::record('{REQUEST}', $id) \
                 SET standing = $to, decided_at = $now, decided_by = $by, reason = $reason \
                 WHERE standing = 'pending' RETURN AFTER"
            ),
            vars([
                ("id", json!(id)),
                ("to", json!(to.as_str())),
                ("now", json!(now.to_rfc3339())),
                ("by", json!(by.as_str())),
                ("reason", json!(reason)),
            ]),
        )
        .await?;
    match decided.into_iter().next() {
        Some(request) => Ok(request),
        None => Err(AntumbraError::rejected(format!(
            "that request was already {}{}",
            held.standing.as_str(),
            held.decided_by
                .map(|who| format!(" by {}", who.as_str()))
                .unwrap_or_default()
        ))),
    }
}

/// How a request stands, for the person who can show its secret. An approval
/// is collected here, once: the person becomes a member of the workspace,
/// with the name they asked under, and the caller mints their token.
pub async fn collect(
    store: &Store,
    id: &str,
    secret_digest: &str,
    now: DateTime<Utc>,
) -> Result<Answer> {
    let held = store.get_row::<JoinRequest>(REQUEST, id).await?;
    let Some(held) = held.filter(|r| r.secret == secret_digest) else {
        return Err(AntumbraError::rejected(NO_SUCH_REQUEST));
    };
    match held.standing {
        Standing::Pending if held.waits(now) => return Ok(Answer::Pending),
        Standing::Pending => return Ok(Answer::Expired),
        Standing::Denied => {
            return Ok(Answer::Denied {
                reason: held.reason,
            })
        }
        Standing::Collected => return Ok(Answer::AlreadyCollected),
        Standing::Approved if held.lapsed(now) => return Ok(Answer::Expired),
        Standing::Approved => {}
    }
    let taken: Vec<Value> = store
        .query_rows(
            &format!(
                "UPDATE type::record('{REQUEST}', $id) SET standing = 'collected', collected_at = $now \
                 WHERE standing = 'approved' RETURN AFTER"
            ),
            vars([("id", json!(id)), ("now", json!(now.to_rfc3339()))]),
        )
        .await?;
    if taken.is_empty() {
        return Ok(Answer::AlreadyCollected);
    }
    let user = held.user();
    principal::provision(store, &held.tenant, &user).await?;
    if held.display.is_some() {
        profile::put(
            store,
            &Profile {
                tenant_id: held.tenant.clone(),
                user: user.clone(),
                display: held.display.clone(),
                avatar: None,
                updated_at: now,
            },
        )
        .await?;
    }
    Ok(Answer::Admitted {
        tenant: held.tenant,
        user,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mem_store;
    use antumbra_core::membership::Role;

    fn code(tenant: &str, secret: &str, uses: u32, needs_approval: bool) -> JoinCode {
        let now = Utc::now();
        JoinCode {
            key: join::digest(secret),
            tenant: TenantId::new(tenant),
            label: None,
            created_by: UserId::new("user:ada"),
            created_at: now,
            expires_at: now + Duration::days(14),
            uses_left: uses,
            needs_approval,
            revoked: false,
        }
    }

    fn asking<'a>(id: &'a str, code: &'a str, handle: &'a str, secret: &'a str) -> Asking<'a> {
        Asking {
            id,
            code,
            handle,
            display: Some("Bo Lindqvist".to_string()),
            note: None,
            host: Some("laptop".to_string()),
            secret_digest: secret,
        }
    }

    fn refused<T>(r: Result<T>) -> String {
        r.err().map(|e| e.to_string()).unwrap_or_default()
    }

    /// Asked, approved, collected once: the person is a member under their
    /// handle, with the name they asked under, and the token is not given twice.
    #[tokio::test]
    async fn a_request_is_asked_decided_and_collected_once() -> Result<()> {
        let s = mem_store().await?;
        let t = TenantId::new("ws:crew");
        let ada = UserId::new("user:ada");
        principal::provision(&s, &t, &ada).await?;
        make_code(&s, &code("ws:crew", "the-code", 1, true)).await?;
        let (now, secret) = (Utc::now(), join::digest("my-secret"));

        let asked = ask(&s, asking("r1", " the-code ", "Bodil", &secret), now).await?;
        assert_eq!(
            (asked.handle.as_str(), asked.standing),
            ("bodil", Standing::Pending)
        );
        assert_eq!(collect(&s, "r1", &secret, now).await?, Answer::Pending);
        // Not theirs to read without the secret, and said as if it were not there.
        assert!(
            refused(collect(&s, "r1", &join::digest("guess"), now).await)
                .contains("no such request")
        );
        assert!(refused(collect(&s, "nope", &secret, now).await).contains("no such request"));

        decide(&s, &t, &ada, "r1", true, None, now).await?;
        // The second decision is told what the first was.
        assert!(refused(decide(&s, &t, &ada, "r1", false, None, now).await)
            .contains("already approved by user:ada"));
        let bodil = UserId::new("user:bodil");
        assert_eq!(
            collect(&s, "r1", &secret, now).await?,
            Answer::Admitted {
                tenant: t.clone(),
                user: bodil.clone()
            }
        );
        assert_eq!(
            collect(&s, "r1", &secret, now).await?,
            Answer::AlreadyCollected
        );
        assert_eq!(
            principal::role_of(&s, &t, &bodil, false).await?,
            Some(Role::Member)
        );
        assert_eq!(
            profile::get(&s, &t, &bodil)
                .await?
                .and_then(|p| p.display)
                .as_deref(),
            Some("Bo Lindqvist")
        );
        Ok(())
    }

    /// A code that will not do gets one answer whatever is wrong with it, a
    /// use is spent once, and a code for one workspace opens no other.
    #[tokio::test]
    async fn a_code_admits_as_many_as_it_says_and_fails_one_way() -> Result<()> {
        let s = mem_store().await?;
        let now = Utc::now();
        make_code(&s, &code("ws:crew", "one-use", 1, true)).await?;
        let expired = JoinCode {
            expires_at: now - Duration::hours(1),
            ..code("ws:crew", "old", 5, true)
        };
        make_code(&s, &expired).await?;
        make_code(&s, &code("ws:crew", "to-revoke", 5, true)).await?;
        assert!(!revoke_code(&s, &TenantId::new("ws:other"), &join::digest("to-revoke")).await?);
        assert!(revoke_code(&s, &TenantId::new("ws:crew"), &join::digest("to-revoke")).await?);

        ask(&s, asking("r1", "one-use", "bodil", "s1"), now).await?;
        for (what, c) in [
            ("spent", "one-use"),
            ("expired", "old"),
            ("revoked", "to-revoke"),
            ("unknown", "never"),
        ] {
            let why = refused(ask(&s, asking("rx", c, "cyrus", "s2"), now).await);
            assert_eq!(why, NO_SUCH_CODE, "{what}");
        }
        assert_eq!(requests(&s, &TenantId::new("ws:crew")).await?.len(), 1);
        assert!(requests(&s, &TenantId::new("ws:other")).await?.is_empty());
        Ok(())
    }

    /// A handle is one person's: asking again with the same secret is the
    /// same request, anyone else is told it is taken, a member's is taken,
    /// and a refused one waits a day.
    #[tokio::test]
    async fn a_handle_is_asked_for_by_one_person_at_a_time() -> Result<()> {
        let s = mem_store().await?;
        let t = TenantId::new("ws:crew");
        let ada = UserId::new("user:ada");
        principal::provision(&s, &t, &ada).await?;
        make_code(&s, &code("ws:crew", "many", 9, true)).await?;
        let now = Utc::now();

        let first = ask(&s, asking("r1", "many", "bodil", "s1"), now).await?;
        let again = ask(&s, asking("r2", "many", "bodil", "s1"), now).await?;
        assert_eq!(again.key, first.key, "a retry is the same request");
        assert!(
            refused(ask(&s, asking("r3", "many", "bodil", "other"), now).await).contains("taken")
        );
        assert!(refused(ask(&s, asking("r4", "many", "ada", "s4"), now).await).contains("taken"));
        assert!(
            refused(ask(&s, asking("r5", "many", "default", "s5"), now).await)
                .contains("not a handle anyone may take")
        );

        decide(&s, &t, &ada, "r1", false, Some("not yet".to_string()), now).await?;
        assert_eq!(
            collect(&s, "r1", "s1", now).await?,
            Answer::Denied {
                reason: Some("not yet".to_string())
            }
        );
        assert!(
            refused(ask(&s, asking("r6", "many", "bodil", "s1"), now).await)
                .contains("a day after")
        );
        let later = now + Duration::hours(DENIAL_HOURS + 1);
        assert_eq!(
            ask(&s, asking("r7", "many", "bodil", "s1"), later)
                .await?
                .key,
            "r7"
        );
        Ok(())
    }

    /// A request nobody decides expires, an approval nobody collects lapses,
    /// and a code that asks no one admits whoever holds it.
    #[tokio::test]
    async fn requests_expire_and_an_invitation_asks_no_one() -> Result<()> {
        let s = mem_store().await?;
        let t = TenantId::new("ws:crew");
        let ada = UserId::new("user:ada");
        principal::provision(&s, &t, &ada).await?;
        make_code(&s, &code("ws:crew", "ask", 3, true)).await?;
        make_code(&s, &code("ws:crew", "invite", 1, false)).await?;
        let now = Utc::now();

        ask(&s, asking("r1", "ask", "bodil", "s1"), now).await?;
        let after = now + Duration::days(15);
        assert_eq!(collect(&s, "r1", "s1", after).await?, Answer::Expired);
        assert!(refused(decide(&s, &t, &ada, "r1", true, None, after).await).contains("expired"));

        ask(&s, asking("r2", "ask", "cyrus", "s2"), now).await?;
        decide(&s, &t, &ada, "r2", true, None, now).await?;
        let late = now + Duration::days(join::COLLECT_DAYS + 1);
        assert_eq!(collect(&s, "r2", "s2", late).await?, Answer::Expired);

        let invited = ask(&s, asking("r3", "invite", "dee", "s3"), now).await?;
        assert_eq!(invited.standing, Standing::Approved);
        assert_eq!(invited.decided_by, Some(ada));
        assert!(matches!(
            collect(&s, "r3", "s3", now).await?,
            Answer::Admitted { .. }
        ));
        // Another workspace's admin decides nothing here.
        assert!(refused(
            decide(
                &s,
                &TenantId::new("ws:other"),
                &UserId::new("user:x"),
                "r2",
                false,
                None,
                now
            )
            .await
        )
        .contains("no such request"));
        Ok(())
    }
}
