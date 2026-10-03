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
mod consolidate;
#[cfg(feature = "models")]
pub use consolidate::{consolidate_compartment, Consolidation, ConsolidationOutcome};

/// Re-exported so callers can construct [`MultiAdapterServe`] without depending
/// on `antumbra-train` directly.
#[cfg(feature = "models")]
pub use antumbra_train::RaftConfig;

/// Re-exported so the consolidation gate can be tuned without a direct
/// `antumbra-train` dependency (the MCP server's autonomous trigger uses it).
#[cfg(feature = "models")]
pub use antumbra_train::consolidate::{ConsolidationPolicy, GateReport};

// --- the real resident multi-adapter engine (models build) ----------------

/// Multi-adapter server over a shared base. Loads the base once (with neutral
/// zero LoRA factors), then [`load_adapter`](antumbra_train::models::QwenCausalLm::load_adapter)s
/// the routed expert's weights into the resident model: an O(adapter) swap, not
/// an O(base) reload. A `register`ed map resolves the gate's `ExpertId`s to
/// adapter files; the currently-resident adapter is tracked so repeated routes
/// to the same expert skip the swap.
#[cfg(feature = "models")]
pub struct MultiAdapterServe {
    base_model: String,
    config: antumbra_train::RaftConfig,
    // `RwLock` so the engine can hot-register a freshly-minted expert (autonomous
    // consolidation) through a shared `&self`, instead of snapshotting at build.
    registry: std::sync::RwLock<std::collections::HashMap<antumbra_core::ExpertId, String>>,
    state: tokio::sync::Mutex<Resident>,
}

/// The resident base and which adapter's factors currently sit on it.
#[cfg(feature = "models")]
#[derive(Default)]
struct Resident {
    model: Option<antumbra_train::models::QwenCausalLm>,
    current: Option<antumbra_core::ExpertId>,
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
            state: tokio::sync::Mutex::new(Resident::default()),
        }
    }

    /// Map an expert to its adapter file (its `artifact_uri`). Routes naming this
    /// expert hot-swap that file onto the resident base.
    pub fn register(&mut self, expert: antumbra_core::ExpertId, adapter_path: impl Into<String>) {
        self.registry
            .get_mut()
            .expect("registry lock")
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
        self.registry.read().expect("registry lock").len()
    }

    /// Whether no adapter is registered.
    pub fn is_empty(&self) -> bool {
        self.registry.read().expect("registry lock").is_empty()
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
            .expect("registry lock")
            .contains_key(expert)
    }

    /// Hot-register a freshly-minted expert's adapter, so a route to it serves
    /// without a restart (closes the autonomous consolidation loop end to end).
    fn register_expert(&self, expert: &antumbra_core::ExpertId, adapter_uri: &str) {
        self.registry
            .write()
            .expect("registry lock")
            .insert(expert.clone(), adapter_uri.to_string());
    }

    async fn act(
        &self,
        req: antumbra_core::ports::ActRequest,
    ) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        use antumbra_core::ports::{ActOutput, StepOutput};
        use antumbra_core::AntumbraError;
        use antumbra_train::CausalLm;

        // The gate selects experts in rank order; v0 serves the top one. A true
        // latent blend of several adapters (routing-as-retrieval) needs `compose_adapters`,
        // which changes the LoRA rank and so a differently-shaped base. That is
        // the `ask --with` path, not an in-place swap, and is deferred here.
        let target = req.adapters.first().cloned().ok_or_else(|| {
            AntumbraError::other(
                "MultiAdapterServe: no adapter selected (the gate escalates out-of-scope tasks; \
                 base-only generation is the probe's job, not this engine's)",
            )
        })?;
        // Resolve before touching the device: an unknown expert is a config
        // error, not a generation failure, and must not pay a model load.
        let path = self
            .registry
            .read()
            .expect("registry lock")
            .get(&target)
            .cloned()
            .ok_or_else(|| {
                AntumbraError::other(format!(
                    "MultiAdapterServe: no adapter registered for {target} (register the population \
                     before serving)"
                ))
            })?;

        // candle generation is synchronous and device-bound. Run it under
        // `block_in_place` so the runtime spawns a replacement worker and other
        // tasks are not starved while a request is served (a v1 serving-throughput concern). Requires
        // a multi-threaded runtime.
        let final_output = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut st = self.state.lock().await;
                if st.model.is_none() {
                    // Load the base with neutral (zero) LoRA factors; adapters are
                    // swapped in below. Reuse the trainer's loader so the model is
                    // byte-identical to what training produced.
                    let loader = antumbra_train::CandleModelLoader::new(self.config.clone());
                    let model =
                        antumbra_train::ModelLoader::load(&loader, &self.base_model, None).await?;
                    st.model = Some(model);
                    st.current = None;
                }
                // Hot-swap only when the routed expert differs from the resident
                // one; repeated routes to the same expert reuse the loaded factors.
                if st.current.as_ref() != Some(&target) {
                    st.model
                        .as_mut()
                        .expect("base loaded above")
                        .load_adapter(&path)?;
                    st.current = Some(target.clone());
                }
                let outputs = st
                    .model
                    .as_mut()
                    .expect("base loaded above")
                    .generate(&req.prompt, 1)
                    .await?;
                antumbra_core::Result::Ok(outputs.into_iter().next().unwrap_or_default())
            })
        })?;

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
    /// GGUF / safetensors path of the shared, code-capable base.
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
    use antumbra_core::AntumbraError;

    #[tokio::test]
    async fn serve_seam_reports_unimplemented() {
        let serve = MultiAdapterServe::new("mem://base.gguf", "/adapters");
        let req = ActRequest {
            task_id: "t:1".into(),
            prompt: "hello".into(),
            adapters: vec![],
        };
        let err = serve.act(req).await.unwrap_err();
        assert!(matches!(err, AntumbraError::Unimplemented(_)));
    }
}

#[cfg(all(test, feature = "models"))]
mod tests {
    use super::*;
    use antumbra_core::ports::ActRequest;
    use antumbra_core::ExpertId;

    // The resolution path is device-free: an empty request and an unregistered
    // expert both fail *before* any base load, so these run without a GPU.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn empty_request_errors_before_load() {
        let serve = MultiAdapterServe::new("Qwen/Qwen2.5-Coder-1.5B", Default::default());
        let req = ActRequest {
            task_id: "t".into(),
            prompt: "hi".into(),
            adapters: vec![],
        };
        let msg = serve.act(req).await.unwrap_err().to_string();
        assert!(msg.contains("no adapter selected"), "{msg}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unregistered_expert_errors_before_load() {
        let serve = MultiAdapterServe::new("Qwen/Qwen2.5-Coder-1.5B", Default::default());
        let req = ActRequest {
            task_id: "t".into(),
            prompt: "hi".into(),
            adapters: vec![ExpertId::new("expert:ghost")],
        };
        let msg = serve.act(req).await.unwrap_err().to_string();
        assert!(msg.contains("no adapter registered"), "{msg}");
    }

    #[test]
    fn registry_tracks_adapters() {
        let serve = MultiAdapterServe::new("base", Default::default())
            .with_adapter(ExpertId::new("expert:a"), "adapters/a.safetensors")
            .with_adapter(ExpertId::new("expert:b"), "adapters/b.safetensors");
        assert_eq!(serve.len(), 2);
        assert!(!serve.is_empty());
    }
}
