//! Listing memories: what a workspace holds, without a query.
//!
//! Its own router, joined to the others in `engine.rs`, because `server.rs`
//! is at the size rule.

use super::*;

#[tool_router(router = listing_router, vis = "pub(super)")]
impl McpServer {
    /// List this tenant's memories (optionally one network, or one machine's),
    /// whole or a page at a time.
    #[tool(
        description = "List your workspace's memories, optionally filtered to one network (world/bank/opinion) or to those written from one machine (`host`). With `limit`, a page: the most recently updated first, `offset` skipping those already seen, and `more` saying whether another page follows. Without it, every memory in no set order."
    )]
    pub(super) async fn list_memories(
        &self,
        Parameters(p): Parameters<ListParams>,
    ) -> Result<Json<MemoriesOut>, ErrorData> {
        let network = p.network.as_deref().map(parse_network);
        let host = host_filter(p.host.as_deref());
        let (mems, more) = match p.limit {
            Some(limit) => {
                let limit = limit.clamp(1, MAX_PAGE);
                // One past the page says whether another follows.
                let mut page = memory::recent(
                    &self.store,
                    &self.tenant,
                    network,
                    host.as_deref(),
                    limit + 1,
                    p.offset.unwrap_or(0),
                )
                .await
                .map_err(err)?;
                let more = page.len() > limit as usize;
                page.truncate(limit as usize);
                (page, more)
            }
            None => {
                let all = match network {
                    Some(net) => memory::list_by_network(&self.store, &self.tenant, net).await,
                    None => memory::list(&self.store, &self.tenant).await,
                };
                // Every row is read either way, so the machine is filtered here.
                let all = all.map_err(err)?.into_iter();
                let all: Vec<Memory> = match &host {
                    Some(h) => all.filter(|m| written_from(m, h)).collect(),
                    None => all.collect(),
                };
                (all, false)
            }
        };
        Ok(Json(MemoriesOut {
            memories: mems.iter().map(MemoryView::from).collect(),
            // list_memories has no query, so there is nothing to floor.
            nothing_cleared_the_floor: false,
            more,
        }))
    }
}

/// The most a page may hold, whatever `limit` asks for.
const MAX_PAGE: u32 = 200;

/// A caller's `host` filter as host names are compared (trimmed, lowercased),
/// or `None` when it names nothing.
pub(super) fn host_filter(raw: Option<&str>) -> Option<String> {
    raw.map(|h| h.trim().to_lowercase())
        .filter(|h| !h.is_empty())
}

/// Whether `m` was written from the machine `host` names, compared as
/// [`host_filter`] normalizes it. A memory with no stamp was written from no
/// machine in particular.
pub(super) fn written_from(m: &Memory, host: &str) -> bool {
    m.author_host
        .as_deref()
        .is_some_and(|a| a.trim().to_lowercase() == host)
}
