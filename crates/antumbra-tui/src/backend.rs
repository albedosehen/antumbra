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
use serde_json::json;

use antumbra_core::console::{Action, Snapshot};
use antumbra_core::ports::Embedder;
use antumbra_core::testing::FixedEmbedder;
use antumbra_core::{ExpertId, LearnedRouter};
use antumbra_store::repo::console;
use antumbra_store::{Store, EMBED_DIM};

use antumbra_http::{SurfaceClient, SurfaceError};

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

/// What routing a task came to: the experts it would go to, most likely
/// first, and, when there is none, why.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Routed {
    pub routes: Vec<(ExpertId, f32)>,
    /// Why no expert is routed to it, in the server's words: a failure
    /// boundary covers the task, there is no learned router yet, the experts
    /// that cover it are resting, or it is outside what the population covers.
    pub reason: Option<String>,
    /// What the server read beside the routes.
    pub near: Near,
}

/// What a task is near, of the user's own: their standing experts and the
/// behaviors they have taught. A standing expert is never routed to, so this
/// is the half of an answer a workspace of standing experts has.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Near {
    /// The standing experts, the closest to the task first. `None` from a
    /// server that says nothing of them (one from before it did), which is
    /// not the same as a user who has none.
    pub standing: Option<Vec<StandingHit>>,
    /// The behaviors nearest the task, closest first.
    pub taught: Vec<TaughtHit>,
}

/// One standing expert, as the server reads it for a task.
#[derive(Debug, Clone, PartialEq)]
pub struct StandingHit {
    pub expert: ExpertId,
    pub scope: String,
    /// How close the task is to what it was taught.
    pub similarity: f32,
    /// Whether the task is in its scope: everywhere, or the repository the
    /// console was started in.
    pub in_scope: bool,
    /// Whether it answers the task with no expert routed to it.
    pub answers: bool,
}

/// One behavior near a task.
#[derive(Debug, Clone, PartialEq)]
pub struct TaughtHit {
    pub rule: String,
    pub scope: String,
    /// `proposed`, `accepted` or `trained`.
    pub status: String,
    pub similarity: f32,
    /// Whether its scope's expert learned it at its last training; `None`
    /// before it has been through one.
    pub learned: Option<bool>,
}

/// The least a task's likeness to what a standing expert was taught may be
/// for the expert to answer it alone: the server's own floor, which the demo
/// applies as the server does.
const ANSWERS_FROM: f32 = 0.3;

impl Routed {
    /// Read the server's `route` answer: its `routes`, and its `reason` when
    /// it gives one.
    pub fn read(out: &serde_json::Value) -> Self {
        let routes = out["routes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|hit| {
                Some((
                    ExpertId::new(hit["expert_id"].as_str()?),
                    hit["probability"].as_f64()? as f32,
                ))
            })
            .collect();
        let reason = out["reason"]
            .as_str()
            .filter(|reason| !reason.is_empty())
            .map(str::to_string);
        let standing = out["standing"].as_array().map(|hits| {
            hits.iter()
                .filter_map(|hit| {
                    Some(StandingHit {
                        expert: ExpertId::new(hit["expert_id"].as_str()?),
                        scope: hit["scope"].as_str()?.to_string(),
                        similarity: hit["similarity"].as_f64()? as f32,
                        in_scope: hit["in_scope"].as_bool()?,
                        answers: hit["answers"].as_bool()?,
                    })
                })
                .collect()
        });
        let taught = out["taught"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|hit| {
                Some(TaughtHit {
                    rule: hit["rule"].as_str()?.to_string(),
                    scope: hit["scope"].as_str()?.to_string(),
                    status: hit["status"].as_str()?.to_string(),
                    similarity: hit["similarity"].as_f64()? as f32,
                    learned: hit["learned"].as_bool(),
                })
            })
            .collect();
        Routed {
            routes,
            reason,
            near: Near { standing, taught },
        }
    }
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
        repo: Option<&str>,
        router: Option<&LearnedRouter>,
    ) -> Result<Routed> {
        match self {
            Backend::Server(server) => {
                let server = server.clone();
                let (task, repo) = (task.to_string(), repo.map(str::to_string));
                tokio::task::spawn_blocking(move || server.route(&task, repo.as_deref())).await?
            }
            Backend::Demo(store) => {
                let vec = FixedEmbedder::new(EMBED_DIM).embed(task).await?;
                let routes = router.map(|r| r.route(&vec)).unwrap_or_default();
                // The demo says why in the server's words, for the two cases
                // it has.
                let reason = match (routes.is_empty(), router) {
                    (false, _) => None,
                    (true, None) => Some("no learned router yet".to_string()),
                    (true, Some(_)) => Some("outside what the population covers".to_string()),
                };
                // And reads its standing experts as the server reads a
                // user's: by the task's likeness to what each was taught.
                let mut standing: Vec<StandingHit> = antumbra_store::repo::expert::list(store)
                    .await?
                    .iter()
                    .filter_map(|e| {
                        let scope = e.standing_scope()?.to_string();
                        let similarity = e.capability_similarity(&vec).unwrap_or(0.0);
                        let in_scope = scope == antumbra_core::behavior::EVERYWHERE
                            || Some(scope.as_str()) == repo;
                        Some(StandingHit {
                            expert: e.id.clone(),
                            answers: in_scope && similarity >= ANSWERS_FROM,
                            scope,
                            similarity,
                            in_scope,
                        })
                    })
                    .collect();
                standing.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
                Ok(Routed {
                    routes,
                    reason,
                    near: Near {
                        standing: Some(standing),
                        taught: Vec::new(),
                    },
                })
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
    /// Tool calls, through the client every door shares.
    surface: SurfaceClient,
    /// The tag of the last snapshot read, so an unchanged one is not sent again.
    seen: Mutex<Option<String>>,
}

impl Server {
    pub fn new(surface: Found, token: Found) -> Self {
        let url = surface.value.trim_end_matches('/').to_string();
        let agent = antumbra_http::agent(TIMEOUT);
        Self {
            // From the machine `ANTUMBRA_HOST_ID` names when it names one, so
            // the server stamps what the console writes with it.
            surface: SurfaceClient::with_agent(&url, Some(&token.value), agent.clone())
                .from_machine(antumbra_http::host_id()),
            url,
            token: token.value,
            surface_from: surface.from,
            token_from: token.from,
            agent,
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

    fn route(&self, task: &str, repo: Option<&str>) -> Result<Routed> {
        let out = match self
            .surface
            .call("route", json!({ "task": task, "top_k": 5, "repo": repo }))
        {
            Ok(out) => out,
            // The shared refusal, with where this console found the token.
            Err(SurfaceError::TokenRefused { .. }) => bail!(
                "the server at {} refused the token from {}; it may have expired, or be for another server",
                self.url,
                self.token_from
            ),
            Err(e) => return Err(e.into()),
        };
        Ok(Routed::read(&out))
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
