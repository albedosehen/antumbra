//! Which workspace a repository's memories live in. A webhook delivery carries
//! no Antumbra identity, so the operator says up front: one workspace for
//! everything the App sees, or an explicit slug-to-tenant map. A repository
//! the map does not name is acknowledged and ignored, never guessed into a
//! workspace.

use std::collections::HashMap;

use antumbra_core::{normalize_repo, TenantId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepoMap {
    /// Every repository lands in this one workspace.
    All(TenantId),
    /// Repository slug (normalized) to workspace.
    Explicit(HashMap<String, TenantId>),
}

#[derive(Debug, thiserror::Error)]
pub enum RepoMapError {
    #[error("the repository map is not a JSON object of slug -> tenant: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("the repository map is empty")]
    Empty,
    #[error("the repository map entry {0:?} is not a slug (host/org/name -> \"ws:...\")")]
    BadEntry(String),
}

impl RepoMap {
    pub fn all(tenant: impl Into<TenantId>) -> Self {
        Self::All(tenant.into())
    }

    /// Parse `{"github.com/acme/orders": "ws:acme", ...}`. Slugs are
    /// normalized (case, trailing `.git`), so the map matches however the
    /// repository was spelled.
    pub fn from_json(text: &str) -> Result<Self, RepoMapError> {
        let raw: HashMap<String, String> = serde_json::from_str(text)?;
        if raw.is_empty() {
            return Err(RepoMapError::Empty);
        }
        let mut map = HashMap::with_capacity(raw.len());
        for (slug, tenant) in raw {
            let key = normalize_repo(&slug);
            if !key.contains('/') || tenant.trim().is_empty() {
                return Err(RepoMapError::BadEntry(slug));
            }
            map.insert(key, TenantId::new(tenant.trim()));
        }
        Ok(Self::Explicit(map))
    }

    /// The workspace for a repository slug, if it has one.
    pub fn tenant_for(&self, slug: &str) -> Option<&TenantId> {
        match self {
            Self::All(tenant) => Some(tenant),
            Self::Explicit(map) => map.get(&normalize_repo(slug)),
        }
    }

    /// How many repositories are mapped (`None` when every repository is).
    pub fn mapped(&self) -> Option<usize> {
        match self {
            Self::All(_) => None,
            Self::Explicit(map) => Some(map.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_map_normalizes_slugs_and_ignores_the_rest() {
        let map = RepoMap::from_json(
            r#"{"GitHub.com/Acme/Orders.git": "ws:acme", "github.com/acme/billing": " ws:billing "}"#,
        )
        .unwrap();
        assert_eq!(map.mapped(), Some(2));
        assert_eq!(
            map.tenant_for("github.com/acme/orders").map(|t| t.as_str()),
            Some("ws:acme")
        );
        assert_eq!(
            map.tenant_for("GITHUB.COM/ACME/BILLING")
                .map(|t| t.as_str()),
            Some("ws:billing")
        );
        assert_eq!(map.tenant_for("github.com/acme/other"), None);
    }

    #[test]
    fn all_sends_every_repository_to_one_workspace() {
        let map = RepoMap::all("ws:everything");
        assert_eq!(map.mapped(), None);
        assert_eq!(
            map.tenant_for("github.com/anyone/anything")
                .map(|t| t.as_str()),
            Some("ws:everything")
        );
    }

    #[test]
    fn bad_maps_are_refused_with_the_reason() {
        assert!(matches!(RepoMap::from_json("{}"), Err(RepoMapError::Empty)));
        assert!(matches!(
            RepoMap::from_json("[]"),
            Err(RepoMapError::Parse(_))
        ));
        assert!(matches!(
            RepoMap::from_json(r#"{"orders": "ws:acme"}"#),
            Err(RepoMapError::BadEntry(s)) if s == "orders"
        ));
        assert!(matches!(
            RepoMap::from_json(r#"{"github.com/acme/orders": " "}"#),
            Err(RepoMapError::BadEntry(_))
        ));
    }
}
