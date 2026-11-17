//! What a test needs to stand a store up: an embedded store with the schema
//! applied, the members of a workspace provisioned, and a session signed in as
//! one of them. Thirty test modules opened the store and signed in by hand, and
//! every per-crate sweep of their `unwrap`s would have rewritten the same
//! fixtures; they are written once here, behind the `testing` feature for the
//! crates whose tests drive a store (the MCP server's, first of all).

use antumbra_core::{Result, TenantId, UserId};

use crate::repo::principal;
use crate::schema::EMBED_DIM;
use crate::Store;

/// An embedded in-memory store with the schema applied, at the index
/// dimension the binaries use. Nothing persists past the test.
pub async fn mem_store() -> Result<Store> {
    Store::connect_memory(EMBED_DIM).await
}

/// [`mem_store`] at a dimension of the test's choosing: a test that writes its
/// vectors by hand wants four axes, not three hundred and eighty-four.
pub async fn mem_store_of(dim: usize) -> Result<Store> {
    Store::connect_memory(dim).await
}

/// Provision `users` as members of `tenant`, in the owner view: the principals
/// the record access signs in against.
pub async fn members(store: &Store, tenant: &TenantId, users: &[&UserId]) -> Result<()> {
    for user in users {
        principal::provision(store, tenant, user).await?;
    }
    Ok(())
}

/// A fresh store with `user` provisioned as a member of `tenant` and the
/// session signed in as them, so the engine's permissions apply to every read
/// and write that follows.
pub async fn signed_in(tenant: &str, user: &str) -> Result<Store> {
    let store = mem_store().await?;
    let (tenant, user) = (TenantId::new(tenant), UserId::new(user));
    principal::provision(&store, &tenant, &user).await?;
    store.signin(&tenant, &user).await?;
    Ok(store)
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use antumbra_core::{Memory, MemoryNetwork};

    use super::*;
    use crate::repo::memory;

    #[tokio::test]
    async fn a_mem_store_has_the_schema_and_takes_a_memory() -> Result<()> {
        let store = mem_store().await?;
        let m = Memory::new(
            "memory:one",
            "ws:t",
            MemoryNetwork::World,
            "a memory",
            0.8,
            Utc::now(),
        )
        .with_embedding(vec![1.0; EMBED_DIM]);
        memory::upsert(&store, &m).await?;
        let read = memory::get(&store, &TenantId::new("ws:t"), &m.id).await?;
        if read.map(|r| r.content) != Some("a memory".into()) {
            return Err(antumbra_core::AntumbraError::other(
                "the memory did not read back",
            ));
        }
        Ok(())
    }

    #[tokio::test]
    async fn signed_in_binds_the_session_to_the_member() -> Result<()> {
        let store = signed_in("ws:t", "user:dana").await?;
        if store.session_user() != Some(UserId::new("user:dana")) {
            return Err(antumbra_core::AntumbraError::other(
                "the session is not the member's",
            ));
        }
        // The owner view is one invalidate away, as the tests expect.
        store.invalidate().await?;
        if store.session_user().is_some() {
            return Err(antumbra_core::AntumbraError::other(
                "the owner view still names a user",
            ));
        }
        Ok(())
    }

    #[tokio::test]
    async fn members_are_provisioned_so_each_can_sign_in() -> Result<()> {
        let store = mem_store_of(4).await?;
        let tenant = TenantId::new("ws:org");
        let (lily, oslo) = (UserId::new("user:lily"), UserId::new("user:oslo"));
        members(&store, &tenant, &[&lily, &oslo]).await?;
        for user in [&lily, &oslo] {
            store.signin(&tenant, user).await?;
            store.invalidate().await?;
        }
        Ok(())
    }
}
