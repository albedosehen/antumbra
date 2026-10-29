//! Where the console reads and acts: the Antumbra server, with the same
//! address and token the hooks use, or (for `--demo`) a seeded store in
//! memory. The console never opens a database of its own.
//!
//! What the server sends is the token's workspace, and the substrate besides
//! when the token is an operator's ([`antumbra_core::console`]); only an
//! operator's token may act.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use antumbra_core::console::{Action, Snapshot};
use antumbra_core::ports::Embedder;
use antumbra_core::testing::FixedEmbedder;
use antumbra_core::{ExpertId, LearnedRouter};
use antumbra_store::repo::console;
use antumbra_store::{Store, EMBED_DIM};

use crate::settings::Found;

/// The most a snapshot may weigh. It holds every memory the workspace has,
/// without their vectors; `ureq` reads at most 10 MiB unless told otherwise.
const MAX_SNAPSHOT: u64 = 512 * 1024 * 1024;

/// How long one request to the server may take, a whole snapshot included.
const TIMEOUT: Duration = Duration::from_secs(60);

/// The workspace the demo seeds and reads as.
fn demo_tenant() -> antumbra_core::TenantId {
    antumbra_core::TenantId::new(crate::DEMO_TENANT)
}

#[derive(Clone)]
pub enum Backend {
    Server(Arc<Server>),
    Demo(Arc<Store>),
}

impl Backend {
    /// The demo, over a store seeded in memory.
    pub fn demo(store: Store) -> Self {
        Backend::Demo(Arc::new(store))
    }

    /// What the console shows now, or `None` when it is what the server sent
    /// last time.
    pub async fn snapshot(&self) -> Result<Option<Snapshot>> {
        match self {
            Backend::Server(server) => {
                let server = server.clone();
                tokio::task::spawn_blocking(move || server.snapshot()).await?
            }
            Backend::Demo(store) => Ok(Some(console::everything(store, &demo_tenant()).await?)),
        }
    }

    /// How often the console asks again: the demo's store is in this process,
    /// and a server's snapshot crosses the network.
    pub fn reload_every_ms(&self) -> f64 {
        match self {
            Backend::Server(_) => 5000.0,
            Backend::Demo(_) => 2000.0,
        }
    }

    /// Carry out an operator's action. `false` when its target is gone.
    pub async fn act(&self, action: Action) -> Result<bool> {
        match self {
            Backend::Server(server) => {
                let server = server.clone();
                tokio::task::spawn_blocking(move || server.act(&action)).await?
            }
            Backend::Demo(store) => Ok(console::apply(store, &demo_tenant(), &action).await?),
        }
    }

    /// Route `task` through the gate, most likely expert first. The server
    /// embeds it with the model its population was built with and applies its
    /// boundaries; the demo embeds with its stand-in over `router`, which is
    /// only coherent over the population it seeded.
    pub async fn route(
        &self,
        task: &str,
        router: Option<&LearnedRouter>,
    ) -> Result<Vec<(ExpertId, f32)>> {
        match self {
            Backend::Server(server) => {
                let server = server.clone();
                let task = task.to_string();
                tokio::task::spawn_blocking(move || server.route(&task)).await?
            }
            Backend::Demo(_) => {
                let vec = FixedEmbedder::new(EMBED_DIM).embed(task).await?;
                Ok(router.map(|r| r.route(&vec)).unwrap_or_default())
            }
        }
    }

    /// Where the console is reading from, for its connect panel.
    pub fn describe(&self) -> String {
        match self {
            Backend::Server(server) => server.url.clone(),
            Backend::Demo(_) => "a demo store in memory; nothing persists".into(),
        }
    }

    /// Where the server's address and the token were found, for the connect
    /// panel; `None` for the demo.
    pub fn found_from(&self) -> Option<(String, String)> {
        match self {
            Backend::Server(server) => {
                Some((server.surface_from.clone(), server.token_from.clone()))
            }
            Backend::Demo(_) => None,
        }
    }

    /// Why the first read failed, with where the server was found, and what
    /// to do when nothing named one.
    pub fn unreachable(&self) -> String {
        match self.found_from() {
            Some((from, _)) if from == crate::settings::DEFAULT_FROM => format!(
                "could not read the console from {} ({from}); run `antumbra setup`, or pass --surface",
                self.describe()
            ),
            Some((from, _)) => {
                format!("could not read the console from {} (from {from})", self.describe())
            }
            None => format!("could not read the console from {}", self.describe()),
        }
    }

    /// Whether `ask` embeds with the population's own model.
    pub fn routes_for_real(&self) -> bool {
        matches!(self, Backend::Server(_))
    }
}

/// The Antumbra server the console reads from.
pub struct Server {
    url: String,
    token: String,
    /// Where the address and the token were found ([`crate::settings`]).
    surface_from: String,
    token_from: String,
    agent: ureq::Agent,
    /// The tag of the last snapshot read, so an unchanged one is not sent again.
    seen: Mutex<Option<String>>,
}

impl Server {
    pub fn new(surface: Found, token: Found) -> Self {
        Self {
            url: surface.value.trim_end_matches('/').to_string(),
            token: token.value,
            surface_from: surface.from,
            token_from: token.from,
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(TIMEOUT))
                .http_status_as_error(false)
                .build()
                .into(),
            seen: Mutex::new(None),
        }
    }

    fn bearer(&self) -> String {
        format!("Bearer {}", self.token)
    }

    fn snapshot(&self) -> Result<Option<Snapshot>> {
        let mut req = self
            .agent
            .get(&format!("{}/console/snapshot", self.url))
            .header("authorization", &self.bearer());
        let seen = self.seen.lock().map(|s| s.clone()).unwrap_or_default();
        if let Some(tag) = &seen {
            req = req.header("if-none-match", tag);
        }
        let mut res = req
            .call()
            .with_context(|| format!("could not reach {}", self.url))?;
        match res.status().as_u16() {
            304 => return Ok(None),
            404 => bail!(
                "the server at {} has no console: it runs an Antumbra older than 0.2.2",
                self.url
            ),
            status => self.check(status)?,
        }
        let tag = res
            .headers()
            .get("etag")
            .and_then(|t| t.to_str().ok())
            .map(str::to_string);
        let snapshot = res
            .body_mut()
            .with_config()
            .limit(MAX_SNAPSHOT)
            .read_json()
            .with_context(|| format!("{} sent a snapshot this console cannot read", self.url))?;
        if let Ok(mut seen) = self.seen.lock() {
            *seen = tag;
        }
        Ok(Some(snapshot))
    }

    fn act(&self, action: &Action) -> Result<bool> {
        let res = self
            .agent
            .post(&format!("{}/console/action", self.url))
            .header("authorization", &self.bearer())
            .send_json(action)
            .with_context(|| format!("could not reach {}", self.url))?;
        match res.status().as_u16() {
            404 => Ok(false),
            403 => bail!(
                "this token is not an operator's; the server's operator mints one with \
                 `antumbra-mcp --mint-token --operator`"
            ),
            status => self.check(status).map(|()| true),
        }
    }

    fn route(&self, task: &str) -> Result<Vec<(ExpertId, f32)>> {
        let mut res = self
            .agent
            .post(&format!("{}/mcp/call", self.url))
            .header("authorization", &self.bearer())
            .send_json(json!({ "tool": "route", "arguments": { "task": task, "top_k": 5 } }))
            .with_context(|| format!("could not reach {}", self.url))?;
        self.check(res.status().as_u16())?;
        let out: Value = res.body_mut().read_json()?;
        Ok(out["routes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|hit| {
                Some((
                    ExpertId::new(hit["expert_id"].as_str()?),
                    hit["probability"].as_f64()? as f32,
                ))
            })
            .collect())
    }

    fn check(&self, status: u16) -> Result<()> {
        match status {
            200..=299 => Ok(()),
            401 => bail!(
                "the server at {} refused the token from {}; it may have expired, or be for another server",
                self.url,
                self.token_from
            ),
            _ => bail!("the server at {} answered {status}", self.url),
        }
    }
}
