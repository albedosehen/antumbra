//! Members' profiles: the name each goes by and their image.
//!
//! One row per (tenant, user). The engine enforces who writes it (the table
//! carries a device row's permissions): a session reads its workspace's
//! profiles and writes only its own user's, so no member can rename another or
//! put a picture beside another's name.

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use surql::query::builder::Query;
use surql::query::crud::query_records;
use surql::types::operators::eq;

use antumbra_core::profile::{Avatar, Profile};
use antumbra_core::{Result, TenantId, UserId};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const PROFILE: &str = crate::tables::MEMBER_PROFILE;

#[derive(Serialize, Deserialize)]
struct ProfileRow {
    tenant_id: String,
    user: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    display: Option<String>,
    /// The image's media type, and its bytes as base64: a row is JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    avatar_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    avatar: Option<String>,
    updated_at: String,
}

impl ProfileRow {
    fn from_domain(p: &Profile) -> Self {
        let encode = |a: &Avatar| base64::engine::general_purpose::STANDARD.encode(&a.bytes);
        ProfileRow {
            tenant_id: p.tenant_id.as_str().to_string(),
            user: p.user.as_str().to_string(),
            display: p.display.clone(),
            avatar_type: p.avatar.as_ref().map(|a| a.media_type.clone()),
            avatar: p.avatar.as_ref().map(encode),
            updated_at: p.updated_at.to_rfc3339(),
        }
    }

    fn into_domain(self) -> Result<Profile> {
        // An image that does not decode is no image: the name still stands.
        let avatar = match (self.avatar_type, self.avatar) {
            (Some(media_type), Some(encoded)) => base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .ok()
                .map(|bytes| Avatar { media_type, bytes }),
            _ => None,
        };
        Ok(Profile {
            tenant_id: TenantId::new(self.tenant_id),
            user: UserId::new(self.user),
            display: self.display,
            avatar,
            updated_at: parse_dt(&self.updated_at)?,
        })
    }
}

/// A record key for (tenant, user): each part's length, then the part with
/// what a key may not hold made `_`. The lengths keep two pairs that differ
/// only in where their punctuation falls from sharing a key.
fn key(tenant: &TenantId, user: &UserId) -> String {
    let plain = |s: &str| -> String {
        s.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    };
    format!(
        "{}_{}_{}_{}",
        tenant.as_str().len(),
        plain(tenant.as_str()),
        user.as_str().len(),
        plain(user.as_str())
    )
}

/// Write a member's profile, whole: what it held before is replaced.
pub async fn put(store: &Store, profile: &Profile) -> Result<()> {
    store
        .put(
            PROFILE,
            &key(&profile.tenant_id, &profile.user),
            &ProfileRow::from_domain(profile),
        )
        .await
}

/// One member's profile, `None` when they have written none.
pub async fn get(store: &Store, tenant: &TenantId, user: &UserId) -> Result<Option<Profile>> {
    Ok(list(store, tenant)
        .await?
        .into_iter()
        .find(|p| &p.user == user))
}

/// Every profile in a workspace, by user.
pub async fn list(store: &Store, tenant: &TenantId) -> Result<Vec<Profile>> {
    let query = Query::new()
        .select(None)
        .from_table(PROFILE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()));
    let rows: Vec<ProfileRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut found = rows
        .into_iter()
        .map(ProfileRow::into_domain)
        .collect::<Result<Vec<_>>>()?;
    found.sort_by(|a, b| a.user.as_str().cmp(b.user.as_str()));
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mem_store;

    fn profile(tenant: &str, user: &str, display: Option<&str>, image: Option<&[u8]>) -> Profile {
        Profile {
            tenant_id: TenantId::new(tenant),
            user: UserId::new(user),
            display: display.map(str::to_string),
            avatar: image.map(|bytes| Avatar {
                media_type: "image/png".to_string(),
                bytes: bytes.to_vec(),
            }),
            updated_at: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn a_profile_is_read_back_as_written_and_rewritten_in_place() -> Result<()> {
        let s = mem_store().await?;
        let (tenant, user) = (TenantId::new("ws:t"), UserId::new("user:dana"));
        assert_eq!(get(&s, &tenant, &user).await?, None);

        let image: Vec<u8> = (0..=255).collect();
        put(
            &s,
            &profile("ws:t", "user:dana", Some("Dana"), Some(&image)),
        )
        .await?;
        let held = get(&s, &tenant, &user).await?;
        assert_eq!(
            held.as_ref().and_then(|p| p.display.as_deref()),
            Some("Dana")
        );
        assert_eq!(
            held.and_then(|p| p.avatar).map(|a| a.bytes),
            Some(image),
            "every byte of the image comes back"
        );

        // Written again without an image: one row, and the image is gone.
        put(&s, &profile("ws:t", "user:dana", Some("Dana W"), None)).await?;
        let all = list(&s, &tenant).await?;
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].display.as_deref(), Some("Dana W"));
        assert_eq!(all[0].avatar, None);
        Ok(())
    }

    #[tokio::test]
    async fn a_workspace_lists_its_own_members_and_no_other() -> Result<()> {
        let s = mem_store().await?;
        put(&s, &profile("ws:t", "user:b", Some("B"), None)).await?;
        put(&s, &profile("ws:t", "user:a", Some("A"), None)).await?;
        put(&s, &profile("ws:other", "user:c", Some("C"), None)).await?;
        let names: Vec<String> = list(&s, &TenantId::new("ws:t"))
            .await?
            .into_iter()
            .filter_map(|p| p.display)
            .collect();
        assert_eq!(names, ["A", "B"]);
        Ok(())
    }

    #[test]
    fn two_pairs_do_not_share_a_key() -> anyhow::Result<()> {
        let k = |t: &str, u: &str| key(&TenantId::new(t), &UserId::new(u));
        assert_ne!(k("ws:a", "b:c"), k("ws:a:b", "c"));
        assert_ne!(k("ws:a_b", "user:c"), k("ws:a", "b_user:c"));
        assert_eq!(k("ws:t", "user:dana"), k("ws:t", "user:dana"));
        Ok(())
    }
}
