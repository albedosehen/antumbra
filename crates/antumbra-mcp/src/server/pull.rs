//! Pulling an adapter this server does not hold from the server that does.
//!
//! An expert's row reaches every server on its database; its weights stay on
//! the machine that trained it. A server told where that machine's Antumbra
//! is ([`ExpertSource`]) pulls the adapter the first time it routes to the
//! expert, checks it against the sha256 on the expert's card, keeps it under
//! its own adapter directory and serves it from then on without the source.
//!
//! The store is the authority on what the expert is. The source is trusted
//! for availability only: whatever it answers is kept only if it is the
//! bytes the row names.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use antumbra_core::adapter::{self, Seal};
use antumbra_core::ports::Serve;
use antumbra_core::{Expert, ExpertId, TenantId, UserId};
use antumbra_store::repo::expert as store_expert;

use super::McpServer;

/// How long a token signed for one pull is good for.
const TOKEN_TTL: Duration = Duration::from_secs(60);

/// How long one pull may take, from connecting to the last byte.
const PATIENCE: Duration = Duration::from_secs(120);

/// How long connecting to the source may take: a source that is down costs a
/// request seconds, not the whole of [`PATIENCE`].
const CONNECT: Duration = Duration::from_secs(5);

/// How long an expert whose pull failed is left before it is tried again, so
/// a source that is down is not asked on every answer.
const RETRY_AFTER: Duration = Duration::from_secs(30);

/// The route an adapter is pulled from, on the server that holds it.
pub(crate) const ADAPTER_PATH: &str = "/expert/adapter";

/// The server adapters are pulled from, and what this one keeps of them.
pub struct ExpertSource {
    /// The source's address, without a trailing slash.
    url: String,
    /// The HS256 secret both servers verify with: a pull is signed with it,
    /// for the user being answered.
    secret: Vec<u8>,
    /// This server's adapter directory.
    dir: String,
    /// One pull at a time: two requests for one expert are one download.
    pulling: tokio::sync::Mutex<()>,
    /// The file each pulled expert is registered under.
    held: std::sync::Mutex<HashMap<ExpertId, String>>,
    /// When each expert's pull last failed.
    failed: std::sync::Mutex<HashMap<ExpertId, Instant>>,
}

impl ExpertSource {
    /// Pull from the Antumbra at `url`, signing with `secret`, into `dir`.
    pub fn new(url: &str, secret: &[u8], dir: &str) -> Self {
        Self {
            url: url.trim_end_matches('/').to_string(),
            secret: secret.to_vec(),
            dir: dir.to_string(),
            pulling: tokio::sync::Mutex::new(()),
            held: std::sync::Mutex::new(HashMap::new()),
            failed: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Whether `expert` is registered under `path` already.
    fn holds(&self, expert: &ExpertId, path: &str) -> bool {
        // The map is ids to paths, and a poisoned lock holds the same map.
        self.held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(expert)
            .is_some_and(|held| held == path)
    }

    /// Whether a pull of `expert` failed within [`RETRY_AFTER`].
    fn failed_lately(&self, expert: &ExpertId) -> bool {
        self.failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(expert)
            .is_some_and(|at| at.elapsed() < RETRY_AFTER)
    }

    /// Make `expert`, placed on another machine, servable here: registered
    /// with `serve` under a file that is the adapter its row names now.
    /// Whether it is.
    ///
    /// `false` is an expert with no seal on its card, which nothing pulled
    /// could be checked against. A file already here under the name the row
    /// determines is checked and registered without asking the source. An
    /// expert retrained since it was pulled has another sha256 on its row:
    /// that adapter is pulled, and the copy before it removed once the new
    /// one is registered. A pull that fails is an error once, and `false`
    /// without another attempt for [`RETRY_AFTER`]; what was registered
    /// before it stays registered.
    pub async fn hold(
        &self,
        serve: &dyn Serve,
        expert: &Expert,
        tenant: &TenantId,
        user: &UserId,
    ) -> anyhow::Result<bool> {
        let Some(seal) = Seal::of_card(&expert.capability_card) else {
            return Ok(false);
        };
        let path = format!(
            "{}/{}",
            self.dir,
            adapter::file_name(&expert.id, &seal.sha256)
        );
        if self.holds(&expert.id, &path) {
            return Ok(true);
        }
        if self.failed_lately(&expert.id) {
            return Ok(false);
        }
        let _one = self.pulling.lock().await;
        // Pulled by the request that held the lock before this one.
        if self.holds(&expert.id, &path) {
            return Ok(true);
        }
        let token =
            crate::auth::mint_hs256(&self.secret, tenant.as_str(), user.as_str(), TOKEN_TTL)?;
        let (url, id, want, to) = (
            self.url.clone(),
            expert.id.clone(),
            seal.clone(),
            path.clone(),
        );
        // A download and a hash: off the runtime's workers.
        let pulled =
            tokio::task::spawn_blocking(move || fetch(&url, &token, &id, &want, Path::new(&to)))
                .await?;
        let mut failed = self
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(e) = pulled {
            failed.insert(expert.id.clone(), Instant::now());
            return Err(e);
        }
        failed.remove(&expert.id);
        drop(failed);
        serve.register_expert(&expert.id, &path);
        self.held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(expert.id.clone(), path);
        // Only now that requests resolve to the new file: the copies an
        // earlier training of this expert left here.
        for old in adapter::others(Path::new(&self.dir), &expert.id, &seal.sha256) {
            if let Err(e) = std::fs::remove_file(&old) {
                eprintln!(
                    "antumbra-mcp: an earlier adapter of {} was left in place: {e}",
                    expert.id.as_str()
                );
            }
        }
        Ok(true)
    }
}

/// Have the adapter `want` describes at `to`: the file already there when it
/// is that adapter, else the source's answer when that is.
fn fetch(url: &str, token: &str, expert: &ExpertId, want: &Seal, to: &Path) -> anyhow::Result<()> {
    if to.exists() && Seal::of_file(to)? == *want {
        return Ok(());
    }
    // As `antumbra_http::agent`, with a bound on connecting: a refusal's status
    // is read here, not turned into an error.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(PATIENCE))
        .timeout_connect(Some(CONNECT))
        .http_status_as_error(false)
        .build()
        .into();
    let mut response = agent
        .get(format!("{url}{ADAPTER_PATH}"))
        .query("id", expert.as_str())
        .header("authorization", &format!("Bearer {token}"))
        .call()
        .map_err(|e| anyhow::anyhow!("the expert source did not answer: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("the expert source answered {}", status.as_u16());
    }
    // One byte past the size on the row is already not the adapter.
    let bytes = response
        .body_mut()
        .with_config()
        .limit(want.bytes.saturating_add(1))
        .read_to_vec()
        .map_err(|e| anyhow::anyhow!("the expert source's answer could not be read: {e}"))?;
    let got = Seal::of_bytes(&bytes);
    if got != *want {
        anyhow::bail!(
            "the expert source answered {} bytes with sha256 {}, and the expert's row names {} \
             bytes with sha256 {}; nothing was kept",
            got.bytes,
            got.sha256,
            want.bytes,
            want.sha256
        );
    }
    if let Some(dir) = to.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // Written beside its name and moved onto it, so a file under the name is
    // always a whole adapter.
    let partial = to.with_extension("part");
    std::fs::write(&partial, &bytes)?;
    std::fs::rename(&partial, to)?;
    Ok(())
}

impl McpServer {
    /// Pull `expert`'s adapter when it is placed on another machine and this
    /// server has a source to pull from. Without a source, or when the pull
    /// fails, the expert stays one this server cannot serve, as it was.
    pub(super) async fn hold(&self, serve: &dyn Serve, expert: &Expert) {
        let Some(source) = &self.source else {
            return;
        };
        if expert.is_placed_on(&self.host) {
            return;
        }
        if let Err(e) = source.hold(serve, expert, &self.tenant, &self.user).await {
            eprintln!(
                "antumbra-mcp: {} was not pulled from the expert source: {e:#}",
                expert.id.as_str()
            );
        }
    }

    /// [`Self::hold`] for an expert known by its id, as a route names one.
    pub(super) async fn hold_routed(&self, serve: &dyn Serve, id: &ExpertId) {
        if self.source.is_none() || (serve.can_serve(id) && !self.pulled(id)) {
            return;
        }
        match store_expert::get(&self.store, id).await {
            Ok(Some(expert)) => self.hold(serve, &expert).await,
            Ok(None) => {}
            Err(e) => eprintln!(
                "antumbra-mcp: {} could not be read to pull its adapter: {e}",
                id.as_str()
            ),
        }
    }

    /// Whether `id` is served here from a pulled copy, whose row may since
    /// name another adapter.
    fn pulled(&self, id: &ExpertId) -> bool {
        self.source.as_ref().is_some_and(|source| {
            source
                .held
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains_key(id)
        })
    }

    /// The expert `id` and its adapter's seal, when this caller may be
    /// served it and this machine holds the file: a shared expert, or the
    /// caller's own standing one.
    pub(crate) async fn expert_to_ship(
        &self,
        id: &ExpertId,
    ) -> antumbra_core::Result<Option<(Expert, Seal)>> {
        let Some(expert) = store_expert::get(&self.store, id).await? else {
            return Ok(None);
        };
        let theirs = expert.owner.is_none()
            || antumbra_core::behavior::is_standing_for(&expert, &self.tenant, &self.user);
        if !theirs || !expert.is_placed_on(&self.host) {
            return Ok(None);
        }
        Ok(Seal::of_card(&expert.capability_card).map(|seal| (expert, seal)))
    }
}
