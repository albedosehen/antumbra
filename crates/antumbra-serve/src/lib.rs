//! # antumbra-serve (hardware-adaptive serving)
//!
//! S-LoRA-style serving: one shared base resident on a single device plus a
//! library of frozen LoRA adapters hot-swapped per request. [`CandleServe`]
//! pins a single adapter for the life of the server (the one-shot `ask` path);
//! [`MultiAdapterServe`] keeps the multi-GB base hot and swaps only the LoRA
//! factors when the routed expert changes, so a long-running server (the MCP
//! engine, a future `serve` daemon) answers across the whole population without
//! reloading the base each call.

use async_trait::async_trait;

use antumbra_core::ports::Serve;

mod probe;

// The schedule is plain logic over a trait: tested in every build, used by
// the engine a `models` build has.
#[cfg(any(feature = "models", test))]
mod engine;
pub use probe::GenerateVerifyProbe;

#[cfg(feature = "models")]
mod embed;
#[cfg(feature = "models")]
pub use embed::BertEmbedder;

#[cfg(feature = "models")]
pub mod decision_probe;

#[cfg(feature = "models")]
pub mod pair_encoder;

#[cfg(feature = "models")]
mod serve_candle;
#[cfg(feature = "models")]
pub use serve_candle::CandleServe;

#[cfg(feature = "models")]
pub mod behave;
#[cfg(all(test, feature = "models"))]
mod gpu_blend;

/// Re-exported so the server can set the generation seed its trainings shift
/// their seeded streams by.
#[cfg(feature = "models")]
pub use antumbra_train::seed::seed_generation;
/// Re-exported so callers can construct [`MultiAdapterServe`] without depending
/// on `antumbra-train` directly.
#[cfg(feature = "models")]
pub use antumbra_train::{RaftConfig, ServeQuant};

// --- the real resident multi-adapter engine (models build) ----------------

/// Multi-adapter server over a shared base. Loads the base once (with neutral
/// zero LoRA factors), then [`load_blend`](antumbra_train::models::QwenCausalLm::load_blend)s
/// the request's experts into the resident model: an O(adapter) swap, not an
/// O(base) reload. The base is built at [`BLEND_SLOTS`] times the trained rank,
/// alpha raised in step, so one expert serves exactly as at its trained rank
/// and a blend of up to that many (a routed expert with the user's standing
/// ones) needs no rebuild. A `register`ed map resolves the gate's `ExpertId`s
/// to adapter files; the resident blend is tracked by those files, so repeated
/// requests for the same one skip the swap and an expert registered again
/// under another file is loaded from it.
#[cfg(feature = "models")]
pub struct MultiAdapterServe {
    base_model: String,
    config: antumbra_train::RaftConfig,
    // `RwLock` so the engine can hot-register a freshly-minted expert (a standing
    // expert the keeper just trained) through a shared `&self`, instead of
    // snapshotting at build.
    registry: std::sync::RwLock<std::collections::HashMap<antumbra_core::ExpertId, String>>,
    /// Where requests go: the thread that owns the resident model, started
    /// by the first of them.
    worker: std::sync::Mutex<Option<std::sync::mpsc::Sender<engine::Job>>>,
    /// Whether the base's files are on this machine and loaded, and the
    /// download and the load that make them so ([`Serve::prepare`]).
    fetch: std::sync::Arc<std::sync::Mutex<Fetch>>,
    /// Set by the serving thread once the base is resident, however it came
    /// to be loaded: ahead of the first answer, or by one.
    base_loaded: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Where the base is, as [`Serve::prepare`] found it.
#[cfg(feature = "models")]
#[derive(Debug, Clone, PartialEq, Eq)]
enum Fetch {
    /// Not asked yet.
    Unknown,
    /// In the Hugging Face cache.
    Here,
    /// Being downloaded on a thread of its own.
    Fetching,
    /// Being loaded by the serving thread, started to load it before any
    /// answer asks it to.
    Loading,
    /// The download or the load failed, for this reason; the next ask starts
    /// again.
    Failed(String),
}

/// How many adapters one request may blend: a routed expert and the user's
/// standing experts for everywhere and for the repository.
#[cfg(feature = "models")]
pub const BLEND_SLOTS: usize = 3;

/// The resident model's configuration: [`BLEND_SLOTS`] times the trained rank,
/// with alpha raised in step so `alpha / rank`, the scale every expert was
/// trained at, holds.
#[cfg(feature = "models")]
fn blend_config(trained: &antumbra_train::RaftConfig) -> antumbra_train::RaftConfig {
    antumbra_train::RaftConfig {
        lora_rank: trained.lora_rank * BLEND_SLOTS,
        lora_alpha: trained.lora_alpha * BLEND_SLOTS as f64,
        ..trained.clone()
    }
}

/// Blends of experts the resident model keeps at once, so that requests
/// answered together may each read its own.
#[cfg(feature = "models")]
const KEPT_BLENDS: usize = 8;

/// Requests that may wait for the engine at once. One past it is refused
/// with "busy" where it would have queued: at a few seconds an answer,
/// thirty-two answered together, the last of this many waits about a
/// minute, which is as long as a caller's patience is likely to be.
#[cfg(feature = "models")]
const WAITING_AT_MOST: usize = 256;

/// Held by a call to [`MultiAdapterServe::act`] for as long as it waits:
/// when the call ends, by an answer or by its future being dropped, the
/// flag is set and the engine lets go of the request.
#[cfg(feature = "models")]
#[derive(Default)]
struct Waiting(std::sync::Arc<std::sync::atomic::AtomicBool>);

#[cfg(feature = "models")]
impl Drop for Waiting {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The resident model as the schedule drives it: loaded when it is first
/// asked for, and the requests it is answering side by side.
#[cfg(feature = "models")]
struct Resident {
    base_model: String,
    config: antumbra_train::RaftConfig,
    model: Option<antumbra_train::models::QwenCausalLm>,
    together: antumbra_train::models::Together,
    /// What the model's own factors are set to, where it answers one
    /// request at a time: a request of the same blend skips the swap.
    loaded: Option<engine::BlendKey>,
    runtime: tokio::runtime::Runtime,
    seed: u64,
    /// Raised once [`Resident::model`] has loaded the base, for
    /// [`Serve::prepare`] to read from another thread.
    base_loaded: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "models")]
impl Resident {
    /// The base, loaded with neutral factors the first time: the trainer's
    /// loader, so the model is byte-identical to what training produced.
    fn model(&mut self) -> antumbra_core::Result<&mut antumbra_train::models::QwenCausalLm> {
        if self.model.is_none() {
            let loader = antumbra_train::CandleModelLoader::new(blend_config(&self.config));
            let model = self.runtime.block_on(antumbra_train::ModelLoader::load(
                &loader,
                &self.base_model,
                None,
            ))?;
            self.model = Some(model);
            self.loaded = None;
            self.base_loaded
                .store(true, std::sync::atomic::Ordering::Release);
        }
        match self.model.as_mut() {
            Some(model) => Ok(model),
            None => Err(antumbra_core::AntumbraError::other(
                "the base model is not resident after loading",
            )),
        }
    }
}

#[cfg(feature = "models")]
impl engine::Model for Resident {
    fn keep(&mut self, blend: u16, specs: &[(String, f32)]) -> antumbra_core::Result<bool> {
        // Keeping a blend sets the model's own factors to it on the way.
        self.loaded = None;
        self.model()?.keep_blend(blend, specs)
    }

    fn admit(&mut self, prompt: &str, blend: u16) -> antumbra_core::Result<Option<usize>> {
        self.seed = self.seed.wrapping_add(1);
        let seed = self.seed;
        let mut together = std::mem::take(&mut self.together);
        let place = self
            .model()
            .and_then(|model| model.admit(&mut together, prompt, blend, seed));
        self.together = together;
        place
    }

    fn advance(&mut self) -> antumbra_core::Result<Vec<(usize, String)>> {
        let mut together = std::mem::take(&mut self.together);
        let ended = self
            .model()
            .and_then(|model| model.advance(&mut together))
            .map(|ended| {
                ended
                    .into_iter()
                    .filter_map(|place| together.take(place).map(|answer| (place, answer)))
                    .collect()
            });
        self.together = together;
        ended
    }

    fn leave(&mut self, place: usize) {
        self.together.leave(place);
    }

    fn reset(&mut self) {
        self.together = antumbra_train::models::Together::default();
    }

    fn alone(&mut self, prompt: &str, specs: &[(String, f32)]) -> antumbra_core::Result<String> {
        use antumbra_train::CausalLm;
        let key = engine::blend_key(specs);
        // Swap only when the blend differs from the one that is loaded.
        if self.loaded.as_ref() != Some(&key) {
            // A failed load can leave the factors half-written: forget what
            // was loaded before trying.
            self.loaded = None;
            self.model()?.load_blend(specs)?;
            self.loaded = Some(key);
        }
        let handle = self.runtime.handle().clone();
        let model = self.model()?;
        let outputs = handle.block_on(model.generate(prompt, 1))?;
        Ok(outputs.into_iter().next().unwrap_or_default())
    }
}

/// The thread that owns the resident model: it takes requests as they come,
/// and gives the schedule a turn for as long as any waits or is being
/// answered. It ends when the engine that feeds it is dropped.
#[cfg(feature = "models")]
fn serve_requests(
    base_model: String,
    config: antumbra_train::RaftConfig,
    jobs: std::sync::mpsc::Receiver<engine::Job>,
    base_loaded: std::sync::Arc<std::sync::atomic::AtomicBool>,
    warm: Option<std::sync::Arc<std::sync::Mutex<Fetch>>>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread().build() {
        Ok(runtime) => runtime,
        Err(err) => {
            // No runtime, no model: every request is told why.
            for job in jobs {
                (job.reply)(Err(antumbra_core::AntumbraError::other(format!(
                    "the serving thread could not start a runtime: {err}"
                ))));
            }
            return;
        }
    };
    let mut resident = Resident {
        base_model,
        config,
        model: None,
        together: antumbra_train::models::Together::default(),
        loaded: None,
        runtime,
        seed: 0,
        base_loaded,
    };
    // Started to load the base before any answer asks for it: requests that
    // arrive meanwhile wait in the channel, and a load that fails is said to
    // the next ask, whose answer then loads it itself.
    if let Some(fetch) = warm {
        let base = resident.base_model.clone();
        eprintln!("antumbra-serve: loading the base model {base} ahead of the first answer");
        let failed = match resident.model() {
            Ok(_) => {
                eprintln!("antumbra-serve: the base model {base} is loaded");
                None
            }
            Err(e) => {
                eprintln!("antumbra-serve: loading the base model {base} failed: {e}");
                Some(Fetch::Failed(format!("the base model did not load: {e}")))
            }
        };
        if let Some(failed) = failed {
            *fetch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = failed;
        }
    }
    let mut schedule = engine::Schedule::new(KEPT_BLENDS, WAITING_AT_MOST);
    loop {
        if schedule.idle() {
            match jobs.recv() {
                Ok(job) => schedule.push(job),
                Err(_) => return,
            }
        }
        while let Ok(job) = jobs.try_recv() {
            schedule.push(job);
        }
        schedule.turn(&mut resident);
    }
}

#[cfg(feature = "models")]
impl MultiAdapterServe {
    /// A server over `base_model` with no adapters registered yet. Register the
    /// population's experts with [`register`](Self::register) / [`with_adapter`](Self::with_adapter).
    pub fn new(base_model: impl Into<String>, config: antumbra_train::RaftConfig) -> Self {
        Self {
            base_model: base_model.into(),
            config,
            registry: std::sync::RwLock::new(std::collections::HashMap::new()),
            worker: std::sync::Mutex::new(None),
            fetch: std::sync::Arc::new(std::sync::Mutex::new(Fetch::Unknown)),
            base_loaded: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start the thread that owns the resident model, loading the base on it
    /// before it takes any request when `warm`.
    fn spawn_worker(
        &self,
        warm: bool,
    ) -> antumbra_core::Result<std::sync::mpsc::Sender<engine::Job>> {
        let (jobs, taken) = std::sync::mpsc::channel();
        let (base_model, config) = (self.base_model.clone(), self.config.clone());
        let base_loaded = self.base_loaded.clone();
        let warm = warm.then(|| self.fetch.clone());
        std::thread::Builder::new()
            .name("antumbra-serve".into())
            .spawn(move || serve_requests(base_model, config, taken, base_loaded, warm))
            .map_err(|e| {
                antumbra_core::AntumbraError::other(format!(
                    "the serving thread did not start: {e}"
                ))
            })?;
        Ok(jobs)
    }

    /// Map an expert to its adapter file (its `artifact_uri`). Routes naming this
    /// expert hot-swap that file onto the resident base.
    ///
    /// The registry is a map of names to paths, and a poisoned lock holds the
    /// same map, so every lock on it here is taken as it is.
    pub fn register(&mut self, expert: antumbra_core::ExpertId, adapter_path: impl Into<String>) {
        self.registry
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(expert, adapter_path.into());
    }

    /// Builder form of [`register`](Self::register) for fluent construction from
    /// a population listing.
    #[must_use]
    pub fn with_adapter(
        mut self,
        expert: antumbra_core::ExpertId,
        adapter_path: impl Into<String>,
    ) -> Self {
        self.register(expert, adapter_path);
        self
    }

    /// How many adapters are registered.
    pub fn len(&self) -> usize {
        self.registry
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Whether no adapter is registered.
    pub fn is_empty(&self) -> bool {
        self.registry
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }

    /// Each of `blend`'s experts as the adapter file registered for it now,
    /// with its weight.
    fn resolve(
        &self,
        blend: &[(antumbra_core::ExpertId, f32)],
    ) -> antumbra_core::Result<Vec<(String, f32)>> {
        let registry = self
            .registry
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        blend
            .iter()
            .map(|(e, w)| {
                let path = registry.get(e).cloned().ok_or_else(|| {
                    antumbra_core::AntumbraError::other(format!(
                        "MultiAdapterServe: no adapter registered for {e} (register the \
                         population before serving)"
                    ))
                })?;
                Ok((path, *w))
            })
            .collect()
    }
}

#[cfg(feature = "models")]
#[async_trait]
impl Serve for MultiAdapterServe {
    /// Servable iff the expert's adapter was registered (this engine snapshots the
    /// population at build time), so a route to an unregistered expert escalates.
    fn can_serve(&self, expert: &antumbra_core::ExpertId) -> bool {
        self.registry
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(expert)
    }

    /// Hot-register a freshly-minted expert's adapter, so a route to it serves
    /// without a restart (a standing expert the keeper just trained). An
    /// expert registered again under another file is served from that file by
    /// the next request to take the engine.
    fn register_expert(&self, expert: &antumbra_core::ExpertId, adapter_uri: &str) {
        self.registry
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(expert.clone(), adapter_uri.to_string());
    }

    fn answers_from_its_own_thread(&self) -> bool {
        true
    }

    /// Where the base is read and in what form, from the configuration alone:
    /// asked before the first answer, it says what that answer will be served
    /// on without loading anything.
    fn serving(&self) -> Option<antumbra_core::ports::Serving> {
        let quant = self.config.serve_quant;
        let device = antumbra_train::device::serving_device(quant.is_some());
        let kernels = (device == "cpu").then(|| match quant {
            Some(_) => antumbra_train::device::cpu_kernels().to_string(),
            // A dense base on a CPU is multiplied by candle's kernels.
            None => "none".to_string(),
        });
        Some(antumbra_core::ports::Serving {
            device: device.to_string(),
            base: quant.map_or("dense", ServeQuant::name).to_string(),
            kernels,
        })
    }

    /// Ready once the base is loaded. Asked the first time, the cache is
    /// looked at without the network; where the base's files are not there,
    /// they are downloaded on a thread of their own. Once they are here, the
    /// serving thread is started to load the base before any answer asks it
    /// to. Every ask until then says which of the two is under way, and a
    /// failed download or load is said once and started again by the next
    /// ask. Where a serving thread is already running, started by an answer
    /// served some other way, that answer loads the base as it always did.
    fn prepare(&self) -> antumbra_core::ports::Readiness {
        use antumbra_core::ports::Readiness;
        let base = &self.base_model;
        let downloading = || {
            Readiness::NotYet(format!(
                "the base model ({base}) is downloading to this server"
            ))
        };
        let loading =
            || Readiness::NotYet(format!("the base model ({base}) is loading on this server"));
        if self.base_loaded.load(std::sync::atomic::Ordering::Acquire) {
            return Readiness::Ready;
        }
        let mut state = self
            .fetch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &*state {
            Fetch::Fetching => return downloading(),
            Fetch::Loading => return loading(),
            Fetch::Failed(why) => {
                let said = Readiness::Failed(why.clone());
                *state = Fetch::Unknown;
                return said;
            }
            Fetch::Here => {}
            Fetch::Unknown => {
                if antumbra_train::models::base_is_cached(base) {
                    *state = Fetch::Here;
                }
            }
        }
        if *state == Fetch::Here {
            let mut worker = self
                .worker
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if worker.is_some() {
                return Readiness::Ready;
            }
            return match self.spawn_worker(true) {
                Ok(jobs) => {
                    *worker = Some(jobs);
                    *state = Fetch::Loading;
                    loading()
                }
                Err(e) => Readiness::Failed(e.to_string()),
            };
        }
        *state = Fetch::Fetching;
        let (base, fetch) = (self.base_model.clone(), self.fetch.clone());
        let started = std::thread::Builder::new()
            .name("antumbra-fetch".into())
            .spawn(move || {
                eprintln!("antumbra-serve: fetching the base model {base}, about 3 GB, for the first answer through an expert");
                let done = match antumbra_train::models::fetch_base(&base) {
                    Ok(()) => {
                        eprintln!("antumbra-serve: the base model {base} is fetched");
                        Fetch::Here
                    }
                    Err(e) => {
                        eprintln!("antumbra-serve: fetching the base model {base} failed: {e}");
                        Fetch::Failed(e.to_string())
                    }
                };
                *fetch.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = done;
            });
        match started {
            Ok(_) => downloading(),
            Err(e) => {
                *state = Fetch::Unknown;
                Readiness::Failed(format!("the download thread did not start: {e}"))
            }
        }
    }

    async fn act(
        &self,
        req: antumbra_core::ports::ActRequest,
    ) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        use antumbra_core::ports::{ActOutput, StepOutput};
        use antumbra_core::AntumbraError;

        let blend = req.blend();
        if blend.is_empty() {
            return Err(AntumbraError::other(
                "MultiAdapterServe: no adapter selected (the gate escalates out-of-scope tasks; \
                 base-only generation is the probe's job, not this engine's)",
            ));
        }
        if blend.len() > BLEND_SLOTS {
            return Err(AntumbraError::other(format!(
                "MultiAdapterServe: {} adapters asked for; a blend holds at most {BLEND_SLOTS}",
                blend.len()
            )));
        }
        // Resolved before the request is handed on: an unknown expert is a
        // config error, not a generation failure, and must not pay a model
        // load. The files named are the ones registered now.
        let specs = self.resolve(&blend)?;

        // The model is one thread's: candle generation is synchronous and
        // device-bound, and requests that wait together are answered
        // together, a pass over the weights for all of them. The request
        // goes to that thread and its answer comes back here.
        let (answer, answered) = tokio::sync::oneshot::channel();
        // Set when this call ends, however it ends: a caller that hung up
        // or timed out drops it with the future, and the engine stops
        // decoding for a request no one waits for.
        let waiting = Waiting::default();
        let gone = waiting.0.clone();
        let job = engine::Job {
            gone: Box::new(move || gone.load(std::sync::atomic::Ordering::Relaxed)),
            owner: req.owner.unwrap_or_default(),
            prompt: req.prompt,
            specs,
            reply: Box::new(move |result| {
                // A caller that stopped waiting has no use for it.
                let _ = answer.send(result);
            }),
        };
        {
            // A poisoned lock holds a sender or nothing, never half of one.
            let mut worker = self
                .worker
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let job = match worker.as_ref() {
                Some(jobs) => jobs.send(job).err().map(|unsent| unsent.0),
                None => Some(job),
            };
            if let Some(job) = job {
                // No thread yet, or the one there has gone: start one.
                let jobs = self.spawn_worker(false)?;
                if jobs.send(job).is_err() {
                    return Err(AntumbraError::other(
                        "the serving thread ended before it took a request",
                    ));
                }
                *worker = Some(jobs);
            }
        }
        let final_output = answered
            .await
            .map_err(|_| AntumbraError::other("the serving thread ended before it answered"))??;

        Ok(ActOutput {
            steps: vec![StepOutput {
                step_idx: 0,
                content: final_output.clone(),
            }],
            final_output,
        })
    }
}

// --- non-models stub for default build ----

/// Multi-adapter server seam (non-models build). The real engine needs candle;
/// without `--features models` it reports `Unimplemented`.
#[cfg(not(feature = "models"))]
#[derive(Debug, Clone)]
pub struct MultiAdapterServe {
    /// The shared, code-capable base: a Hugging Face model id.
    pub base_model_uri: String,
    /// Directory of frozen LoRA adapters to hot-swap.
    pub adapter_dir: String,
}

#[cfg(not(feature = "models"))]
impl MultiAdapterServe {
    pub fn new(base_model_uri: impl Into<String>, adapter_dir: impl Into<String>) -> Self {
        Self {
            base_model_uri: base_model_uri.into(),
            adapter_dir: adapter_dir.into(),
        }
    }
}

#[cfg(not(feature = "models"))]
#[async_trait]
impl Serve for MultiAdapterServe {
    async fn act(
        &self,
        _req: antumbra_core::ports::ActRequest,
    ) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        Err(antumbra_core::AntumbraError::Unimplemented(
            "multi-adapter serving requires antumbra-serve built with --features models",
        ))
    }
}

#[cfg(all(test, not(feature = "models")))]
mod tests {
    use super::*;
    use antumbra_core::ports::ActRequest;
    use antumbra_core::{AntumbraError, Result};

    #[tokio::test]
    async fn serve_seam_reports_unimplemented() -> Result<()> {
        let serve = MultiAdapterServe::new("Qwen/Qwen2.5-Coder-1.5B-Instruct", "/adapters");
        let req = ActRequest {
            task_id: "t:1".into(),
            prompt: "hello".into(),
            adapters: vec![],
            weights: Vec::new(),
            owner: None,
        };
        let Err(err) = serve.act(req).await else {
            return Err(AntumbraError::other("the seam served the request"));
        };
        assert!(matches!(err, AntumbraError::Unimplemented(_)));
        Ok(())
    }
}

#[cfg(all(test, feature = "models"))]
mod tests {
    use super::*;
    use antumbra_core::ports::ActRequest;
    use antumbra_core::{AntumbraError, ExpertId, Result};

    /// The message `serve` refuses `req` with.
    async fn refused(serve: &MultiAdapterServe, req: ActRequest) -> Result<String> {
        let Err(e) = serve.act(req).await else {
            return Err(AntumbraError::other("the request was served"));
        };
        Ok(e.to_string())
    }

    // The resolution path is device-free: an empty request and an unregistered
    // expert both fail *before* any base load, so these run without a GPU.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn empty_request_errors_before_load() -> Result<()> {
        let serve = MultiAdapterServe::new("Qwen/Qwen2.5-Coder-1.5B", Default::default());
        let req = ActRequest {
            task_id: "t".into(),
            prompt: "hi".into(),
            adapters: vec![],
            weights: Vec::new(),
            owner: None,
        };
        let msg = refused(&serve, req).await?;
        assert!(msg.contains("no adapter selected"), "{msg}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unregistered_expert_errors_before_load() -> Result<()> {
        let serve = MultiAdapterServe::new("Qwen/Qwen2.5-Coder-1.5B", Default::default());
        let req = ActRequest {
            task_id: "t".into(),
            prompt: "hi".into(),
            adapters: vec![ExpertId::new("expert:ghost")],
            weights: Vec::new(),
            owner: None,
        };
        let msg = refused(&serve, req).await?;
        assert!(msg.contains("no adapter registered"), "{msg}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_blend_beyond_the_slots_errors_before_load() -> Result<()> {
        let mut serve = MultiAdapterServe::new("Qwen/Qwen2.5-Coder-1.5B", Default::default());
        let ids: Vec<ExpertId> = (0..=BLEND_SLOTS)
            .map(|i| ExpertId::new(format!("expert:{i}")))
            .collect();
        for id in &ids {
            serve.register(id.clone(), "adapters/never-read.safetensors");
        }
        let msg = refused(&serve, ActRequest::new("t", "hi", ids)).await?;
        assert!(msg.contains("at most"), "{msg}");
        Ok(())
    }

    #[test]
    fn the_resident_rank_holds_every_slot_at_the_trained_scale() -> Result<()> {
        let trained = RaftConfig::default();
        let resident = blend_config(&trained);
        assert_eq!(resident.lora_rank, trained.lora_rank * BLEND_SLOTS);
        assert!((resident.lora_scale() - trained.lora_scale()).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn registry_tracks_adapters() -> Result<()> {
        let serve = MultiAdapterServe::new("base", Default::default())
            .with_adapter(ExpertId::new("expert:a"), "adapters/a.safetensors")
            .with_adapter(ExpertId::new("expert:b"), "adapters/b.safetensors");
        assert_eq!(serve.len(), 2);
        assert!(!serve.is_empty());
        Ok(())
    }

    /// The regression: a retrained expert keeps its id, and the resident
    /// blend was known by id, so its new adapter was never loaded.
    #[test]
    fn an_expert_registered_under_another_file_is_another_blend() -> antumbra_core::Result<()> {
        let expert = ExpertId::new("expert:a");
        let serve = MultiAdapterServe::new("base", Default::default())
            .with_adapter(expert.clone(), "adapters/a.1.safetensors");
        let blend = [(expert.clone(), 1.0f32)];
        let before = engine::blend_key(&serve.resolve(&blend)?);
        assert_eq!(
            before,
            engine::blend_key(&serve.resolve(&blend)?),
            "the same file at the same weight is the blend already resident"
        );

        serve.register_expert(&expert, "adapters/a.2.safetensors");
        let after = engine::blend_key(&serve.resolve(&blend)?);

        assert_ne!(before, after);
        assert_eq!(
            after,
            vec![("adapters/a.2.safetensors".to_string(), 1.0f32.to_bits())]
        );
        Ok(())
    }

    #[test]
    fn a_blend_at_another_weight_is_another_blend() {
        let at = |weight: f32| engine::blend_key(&[("adapters/a.safetensors".to_string(), weight)]);
        assert_eq!(at(0.5), at(0.5));
        assert_ne!(at(0.5), at(1.0));
    }
}
