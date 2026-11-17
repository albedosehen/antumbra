//! Getting an adapter this server does not hold, and keeping a second copy
//! of the ones it trains.
//!
//! An expert's row reaches every server on its database, and every store
//! that syncs with it; its weights stay on the machine that trained it. A
//! server told where that machine's Antumbra
//! is ([`ExpertSource`]) pulls the adapter the first time it routes to the
//! expert, checks it against the sha256 on the expert's card, keeps it under
//! its own adapter directory and serves it from then on without the source.
//!
//! A server that keeps a copal archive has a second place to look: the
//! training server archives each adapter it admits ([`archive_adapter`]),
//! and a server whose source does not answer, or that has none, reads the
//! adapter back from there. The server that trained an expert restores its
//! own file the same way when it is gone.
//!
//! The store is the authority on what the expert is. The source and the
//! archive are trusted for availability only: whatever either answers is
//! kept only if it is the bytes the row names.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use antumbra_copal::CopalArchive;
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

/// The server adapters are pulled from, and the secret a pull is signed with.
struct Remote {
    /// The source's address, without a trailing slash.
    url: String,
    /// The HS256 secret both servers verify with: a pull is signed with it,
    /// for the user being answered.
    secret: Vec<u8>,
}

/// Where this server gets the adapters it does not hold, and what it keeps
/// of them: the server that trains them, a copal archive, or both.
pub struct ExpertSource {
    /// `None` for a server with an archive and no source.
    remote: Option<Remote>,
    /// The archive adapters are read back from, when this server keeps one.
    archive: Option<Arc<CopalArchive>>,
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
    fn over(remote: Option<Remote>, archive: Option<Arc<CopalArchive>>, dir: &str) -> Self {
        Self {
            remote,
            archive,
            dir: dir.to_string(),
            pulling: tokio::sync::Mutex::new(()),
            held: std::sync::Mutex::new(HashMap::new()),
            failed: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Pull from the Antumbra at `url`, signing with `secret`, into `dir`.
    pub fn new(url: &str, secret: &[u8], dir: &str) -> Self {
        let remote = Remote {
            url: url.trim_end_matches('/').to_string(),
            secret: secret.to_vec(),
        };
        Self::over(Some(remote), None, dir)
    }

    /// Read adapters back from `archive` alone, into `dir`: a server with no
    /// source to pull from.
    pub fn archive_only(archive: Arc<CopalArchive>, dir: &str) -> Self {
        Self::over(None, Some(archive), dir)
    }

    /// Fall back to `archive` when the source does not give an adapter.
    #[must_use]
    pub fn with_archive(mut self, archive: Arc<CopalArchive>) -> Self {
        self.archive = Some(archive);
        self
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

    /// Note how a pull of `expert` went, for [`Self::failed_lately`].
    fn note<T>(&self, expert: &ExpertId, got: &anyhow::Result<T>) {
        let mut failed = self
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match got {
            Ok(_) => failed.remove(expert),
            Err(_) => failed.insert(expert.clone(), Instant::now()),
        };
    }

    /// Make `expert`, placed on another machine, servable here: registered
    /// with `serve` under a file that is the adapter its row names now.
    /// Whether it is.
    ///
    /// `false` is an expert with no seal on its card, which nothing pulled
    /// could be checked against, and one there is nowhere to get: no source,
    /// and no copy in an archive this server keeps. A file already here
    /// under the name the row determines is checked and registered without
    /// asking anyone. Otherwise the source is asked, and then the archive
    /// when the source does not give it. An expert retrained since it was
    /// pulled has another sha256 on its row: that adapter is pulled, and the
    /// copy before it removed once the new one is registered. A pull that
    /// fails is an error once, and `false` without another attempt for
    /// [`RETRY_AFTER`]; what was registered before it stays registered.
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
        let had = self.have(expert, &seal, &path, tenant, user).await;
        self.note(&expert.id, &had);
        if !had? {
            return Ok(false);
        }
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

    /// Have the adapter `seal` describes at `path`: the file already there
    /// when it is that adapter, else what the source or the archive gives.
    /// `false` when it is not there and there is nowhere to get it.
    async fn have(
        &self,
        expert: &Expert,
        seal: &Seal,
        path: &str,
        tenant: &TenantId,
        user: &UserId,
    ) -> anyhow::Result<bool> {
        let (to, want) = (PathBuf::from(path), seal.clone());
        // A hash and a download: off the runtime's workers.
        if tokio::task::spawn_blocking(move || present(&to, &want)).await?? {
            return Ok(true);
        }
        let Some(bytes) = self.obtain(expert, seal, tenant, user).await? else {
            return Ok(false);
        };
        let to = PathBuf::from(path);
        tokio::task::spawn_blocking(move || keep(&bytes, &to)).await??;
        Ok(true)
    }

    /// The adapter's bytes, checked against `seal`: the source's when there
    /// is one and it gives them, else the archive's when the expert's card
    /// names a copy there. `None` when neither is there to ask.
    async fn obtain(
        &self,
        expert: &Expert,
        seal: &Seal,
        tenant: &TenantId,
        user: &UserId,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let archived = self
            .archive
            .as_ref()
            .zip(adapter::archived_in(&expert.capability_card));
        let mut refused = Vec::new();
        if let Some(remote) = &self.remote {
            let token =
                crate::auth::mint_hs256(&remote.secret, tenant.as_str(), user.as_str(), TOKEN_TTL)?;
            let (url, id, want) = (remote.url.clone(), expert.id.clone(), seal.clone());
            match tokio::task::spawn_blocking(move || pull(&url, &token, &id, &want)).await? {
                Ok(bytes) => return Ok(Some(bytes)),
                Err(e) => refused.push(format!("{e:#}")),
            }
        }
        if let Some((archive, file)) = archived {
            match from_archive(archive, tenant, file, seal).await {
                Ok(bytes) => return Ok(Some(bytes)),
                Err(e) => refused.push(format!("{e:#}")),
            }
        }
        if refused.is_empty() {
            return Ok(None);
        }
        Err(anyhow::anyhow!(refused.join("; ")))
    }

    /// Restore the adapter of an expert placed on this machine, when the file
    /// its row names is gone and the archive holds a copy. Whether it was.
    ///
    /// The file is written only at the name the expert's id and sha256
    /// determine, as the row of an expert admitted here names it; a row that
    /// names anything else is left alone.
    pub async fn restore(
        &self,
        serve: &dyn Serve,
        expert: &Expert,
        tenant: &TenantId,
    ) -> anyhow::Result<bool> {
        let Some(archive) = &self.archive else {
            return Ok(false);
        };
        let path = PathBuf::from(&expert.artifact_uri);
        if path.exists() {
            return Ok(false);
        }
        let card = &expert.capability_card;
        let (Some(seal), Some(file)) = (Seal::of_card(card), adapter::archived_in(card)) else {
            return Ok(false);
        };
        let named = path.file_name().and_then(|name| name.to_str())
            == Some(adapter::file_name(&expert.id, &seal.sha256).as_str());
        if !named || self.failed_lately(&expert.id) {
            return Ok(false);
        }
        let _one = self.pulling.lock().await;
        // Restored by the request that held the lock before this one.
        if path.exists() {
            return Ok(true);
        }
        let got = from_archive(archive, tenant, file, &seal).await;
        self.note(&expert.id, &got);
        let bytes = got?;
        tokio::task::spawn_blocking(move || keep(&bytes, &path)).await??;
        serve.register_expert(&expert.id, &expert.artifact_uri);
        Ok(true)
    }
}

/// Whether the file at `to` is already the adapter `want` describes.
fn present(to: &Path, want: &Seal) -> anyhow::Result<bool> {
    Ok(to.exists() && Seal::of_file(to)? == *want)
}

/// `bytes` as the adapter `want` describes, or what `whose` answer was instead.
fn checked(bytes: Vec<u8>, want: &Seal, whose: &str) -> anyhow::Result<Vec<u8>> {
    let got = Seal::of_bytes(&bytes);
    if got != *want {
        anyhow::bail!(
            "{whose} answered {} bytes with sha256 {}, and the expert's row names {} bytes with \
             sha256 {}; nothing was kept",
            got.bytes,
            got.sha256,
            want.bytes,
            want.sha256
        );
    }
    Ok(bytes)
}

/// The source's answer for `expert`, when it is the adapter `want` describes.
fn pull(url: &str, token: &str, expert: &ExpertId, want: &Seal) -> anyhow::Result<Vec<u8>> {
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
    checked(bytes, want, "the expert source")
}

/// The archive's copy at `file`, when it is the adapter `want` describes.
async fn from_archive(
    archive: &CopalArchive,
    tenant: &TenantId,
    file: &str,
    want: &Seal,
) -> anyhow::Result<Vec<u8>> {
    // One byte past the size on the row is already not the adapter.
    let bytes = archive
        .read_adapter(tenant.as_str(), file, want.bytes.saturating_add(1))
        .await
        .map_err(|e| anyhow::anyhow!("the archive did not give the adapter: {e}"))?;
    checked(bytes, want, "the archive")
}

/// Write `bytes` at `to`: beside its name first and then moved onto it, so a
/// file under the name is always a whole adapter.
fn keep(bytes: &[u8], to: &Path) -> anyhow::Result<()> {
    if let Some(dir) = to.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let partial = to.with_extension("part");
    std::fs::write(&partial, bytes)?;
    std::fs::rename(&partial, to)?;
    Ok(())
}

/// Archive the adapter at `adapter_uri`, which a training has just had
/// admitted with `seal`: the copal file it is now in.
///
/// `None` when the upload fails, which is said and costs the expert nothing:
/// the archive is a second copy, and the expert is minted and served without
/// it.
// Called by the keeper, which is built only with the model features.
#[cfg_attr(not(feature = "models"), allow(dead_code))]
pub(crate) async fn archive_adapter(
    archive: &CopalArchive,
    tenant: &TenantId,
    expert: &ExpertId,
    seal: &Seal,
    adapter_uri: &str,
) -> Option<String> {
    let path = PathBuf::from(adapter_uri);
    let archived = async {
        let bytes = tokio::task::spawn_blocking(move || std::fs::read(&path)).await??;
        let stored = archive
            .put_adapter(tenant.as_str(), expert, &seal.sha256, bytes)
            .await?;
        anyhow::Ok(stored.file_id)
    }
    .await;
    match archived {
        Ok(file) => Some(file),
        Err(e) => {
            eprintln!(
                "antumbra-mcp: the adapter of {} was not archived, and is served without a \
                 second copy: {e:#}",
                expert.as_str()
            );
            None
        }
    }
}

impl McpServer {
    /// Bring `expert`'s adapter here when this server has somewhere to get
    /// it: pulled when the expert is placed on another machine, restored
    /// from the archive when it is this machine's own and its file is gone.
    /// With nowhere to get it, or when that fails, the expert stays one this
    /// server cannot serve, as it was.
    pub(super) async fn hold(&self, serve: &dyn Serve, expert: &Expert) {
        let Some(source) = &self.source else {
            return;
        };
        let brought = if expert.is_placed_on(&self.host) {
            source.restore(serve, expert, &self.tenant).await
        } else {
            source.hold(serve, expert, &self.tenant, &self.user).await
        };
        if let Err(e) = brought {
            eprintln!(
                "antumbra-mcp: the adapter of {} was not brought here: {e:#}",
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
