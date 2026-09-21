//! Everything about the server that is not one of its tools: how it is built,
//! the triggers it runs after a write, the routing and reranking the tools
//! lean on, the one-shot `call_tool` the REST shim dispatches through, and the
//! `ServerHandler` impl that hands the router to rmcp.
//!
//! A child module of `server`, so these are the same inherent methods on the
//! same `McpServer` and they see the helpers next door; they moved here only
//! because `server.rs` had grown past the size rule. The `#[tool_router]`
//! block stays there, because that macro builds the router from the block it
//! is applied to.

use super::*;

impl McpServer {
    /// `serve` is the engine the `answer` tool drives (a real `MultiAdapterServe`
    /// under `--features models`, a fake in tests, or `None` for a route-only
    /// surface where `answer` reports serving is not configured).
    pub fn new(
        store: Store,
        embedder: Arc<dyn Embedder>,
        tenant: TenantId,
        user: UserId,
        host: String,
        default_compartment: CompartmentId,
        serve: Option<Arc<dyn antumbra_core::ports::Serve>>,
    ) -> Self {
        Self {
            store,
            consolidation_store: None,
            embedder,
            tenant,
            user,
            host,
            default_compartment,
            auto_propose: None,
            auto_consolidate: None,
            consolidating: consolidation::SharedConsolidation::default(),
            serve,
            registry: None,
            reranker: None,
            reranker_cache: Arc::new(tokio::sync::Mutex::new(RerankCache::new())),
            copal: None,
            profile: None,
            session: None,
        }
    }

    /// Make a copal file service the document of record for ingested documents:
    /// the original content is archived there before any chunk is stored, and
    /// every chunk carries the copal file id + digest as provenance. Off by
    /// default; without it, ingest keeps only the chunks (the v0 behavior).
    #[must_use]
    pub fn with_copal_archive(mut self, archive: Arc<antumbra_copal::CopalArchive>) -> Self {
        self.copal = Some(archive);
        self
    }

    /// Narrow this session to a tool profile: only its tools are listed and
    /// callable. Off by default (every tool).
    pub fn with_tool_profile(mut self, profile: Arc<crate::profile::ToolProfile>) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Keep the record session alive across long-lived use: the keeper re-signs
    /// the connection in before the store's session duration runs out.
    pub fn with_session_keeper(mut self, keeper: Arc<crate::session::SessionKeeper>) -> Self {
        self.session = Some(keeper);
        self
    }

    /// Refresh the record session when it is about to expire, before a tool
    /// runs on it. A no-op without a keeper or while the session is fresh.
    pub(super) async fn keep_session(&self) -> Result<(), ErrorData> {
        if let Some(keeper) = &self.session {
            keeper.refresh_if_stale().await.map_err(|e| {
                ErrorData::internal_error(format!("session refresh failed: {e}"), None)
            })?;
        }
        Ok(())
    }

    /// Every tool this server has, by name, profile or not: what `--tools`
    /// is validated against.
    pub fn all_tool_names() -> Vec<String> {
        Self::tool_router()
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect()
    }

    /// The tools this session advertises: everything, or the profile's subset.
    pub fn advertised_tools(&self) -> Vec<rmcp::model::Tool> {
        let all = Self::tool_router().list_all();
        match &self.profile {
            Some(profile) => profile.filter(all, |t| t.name.as_ref()),
            None => all,
        }
    }

    /// Refuse a call to a tool outside the profile with an error that names
    /// what is advertised, rather than running it or reporting it unknown.
    pub(super) fn check_profile(&self, name: &str) -> Result<(), ErrorData> {
        match &self.profile {
            Some(profile) if !profile.allows(name) => Err(ErrorData::invalid_params(
                format!(
                    "tool `{name}` is not in this server's tool profile; advertised: {}",
                    profile.names().collect::<Vec<_>>().join(", ")
                ),
                None,
            )),
            _ => Ok(()),
        }
    }

    /// Enable the cross-encoder precision stage (P-2): after hybrid recall, the
    /// wide RRF candidate pool is re-scored over (query, content) by `reranker`
    /// and reordered before truncating to the caller's `top_k`. Off by default;
    /// a reranker fault degrades to the RRF order, never failing recall.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn antumbra_core::ports::Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Register this session's peer into `registry` on initialize, so live
    /// shared-memory changes (R-2) are pushed to it over its SSE stream.
    #[must_use]
    pub fn with_registry(mut self, registry: crate::notify::PeerRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Enable the autonomous propose trigger: once the unorganized inbox reaches
    /// `threshold` memories, a write auto-clusters it into `Origin::Proposed`
    /// compartments (reversible, the user curates). Off by default.
    #[must_use]
    pub fn with_auto_propose(mut self, threshold: usize) -> Self {
        self.auto_propose = Some(AutoProposeConfig {
            threshold,
            min_size: 3,
            similarity_threshold: 0.6,
        });
        self
    }

    /// Enable the autonomous consolidation trigger: when a reinforced memory's
    /// compartment clears the consolidation gate, it graduates into a private
    /// expert on the GPU in the background (the "during sleep" trigger). Off by
    /// default; needs the server built with `--features models` + a GPU to train.
    #[must_use]
    pub fn with_auto_consolidate(mut self) -> Self {
        self.auto_consolidate = Some(AutoConsolidateConfig {
            min_recurrence: 2,
            min_confidence: 0.5,
            // Lighter than the manual `consolidate-compartment` (40): an autonomous
            // trigger fires repeatedly and supersedes its expert each time, so a
            // short capture per reinforce is the right trade.
            rounds: 8,
            samples: 4,
            max_new_tokens: 32,
            lr: 3e-4,
            replay_ratio: 0.5,
        });
        self
    }

    /// Set the OWNER connection the autonomous consolidation runs on. The HTTP
    /// server passes its stable root connection here, because the per-request
    /// scoped `store` it builds each `McpServer` with is not safe to use from a
    /// detached background task (its signin is rebound per request). Unset, the
    /// trigger uses `store` (correct for stdio / the embedded owner connection).
    #[must_use]
    pub fn with_consolidation_store(mut self, store: Store) -> Self {
        self.consolidation_store = Some(store);
        self
    }

    /// Share the consolidation state across every per-identity server. The HTTP
    /// server builds a fresh `McpServer` per request, so per-instance state never
    /// coalesces; passing one shared handle makes concurrent reinforces of the
    /// same compartment collapse into a single train (and stops them racing on the
    /// same model-weight download). Unset, each server keeps its own (correct for
    /// stdio).
    #[must_use]
    pub fn with_consolidating(mut self, shared: consolidation::SharedConsolidation) -> Self {
        self.consolidating = shared;
        self
    }

    /// Leave this compartment's genesis run for the machine that can do it, and
    /// say whether that happened (ADR-0017 A2). `true` means the caller should
    /// not train: the work is recorded, waiting on another of the user's nodes.
    ///
    /// The request is a row rather than a log line because a node that only said
    /// "this belongs on the rig" would have failed in a way indistinguishable,
    /// from outside, from a compartment that never cleared the gate.
    ///
    /// Every failure here falls back to training locally. A fabric that cannot
    /// be read, or a request that cannot be written, is a reason to do the work
    /// badly rather than a reason to lose it.
    #[cfg_attr(not(feature = "models"), allow(dead_code))]
    pub(super) async fn escalate_genesis(&self, store: &Store, comp: &CompartmentId) -> bool {
        let genesis =
            match antumbra_store::repo::device::genesis_for_user(store, &self.tenant, &self.user)
                .await
            {
                Ok(found) => found,
                Err(e) => {
                    eprintln!("[auto-consolidate] could not read the fabric, training here: {e}");
                    None
                }
            };
        let antumbra_core::GenesisPlacement::On(there) = antumbra_core::genesis_placement(
            &self.host,
            crate::hardware::role().can_train(),
            genesis.as_ref(),
        ) else {
            return false;
        };
        let asking = antumbra_core::GenesisRequest::new(
            self.tenant.clone(),
            self.user.clone(),
            comp.clone(),
            self.host.clone(),
            there.clone(),
            chrono::Utc::now(),
        );
        match antumbra_store::repo::genesis::ask(store, &asking).await {
            Ok(open) => {
                eprintln!(
                    "[auto-consolidate] {} : left for {there}, which can train it (waiting since {})",
                    comp.as_str(),
                    open.created_at.to_rfc3339()
                );
                true
            }
            Err(e) => {
                eprintln!(
                    "[auto-consolidate] {} : could not leave the run for {there}, training here: {e}",
                    comp.as_str()
                );
                false
            }
        }
    }

    /// Take the genesis run that has waited longest, if this user's fabric left
    /// one here (ADR-0017 A2, the taking half). `None` when the queue is empty,
    /// which is the ordinary case and the only one in a fabric of one node.
    ///
    /// Claiming is a write, so two trainers in one fabric do not both take the
    /// same run -- and a claim that cannot be recorded is not taken at all,
    /// because a run this node believed it owned while the row still read
    /// pending is exactly how the same compartment gets trained twice.
    #[cfg_attr(not(feature = "models"), allow(dead_code))]
    pub(super) async fn claim_genesis_request(
        &self,
        store: &Store,
    ) -> Option<antumbra_core::GenesisRequest> {
        let waiting = match antumbra_store::repo::genesis::list_open_for_user(
            store,
            &self.tenant,
            &self.user,
        )
        .await
        {
            Ok(waiting) => waiting,
            Err(e) => {
                eprintln!("[auto-consolidate] could not read the genesis queue: {e}");
                return None;
            }
        };
        // Oldest first, and skip what another node already has in hand.
        let next = waiting
            .into_iter()
            .find(|r| r.status == antumbra_core::GenesisStatus::Pending)?;
        let now = chrono::Utc::now();
        if let Err(e) = antumbra_store::repo::genesis::set_status(
            store,
            &next,
            antumbra_core::GenesisStatus::Claimed,
            now,
        )
        .await
        {
            eprintln!("[auto-consolidate] could not claim a waiting genesis run: {e}");
            return None;
        }
        eprintln!(
            "[auto-consolidate] {} : taking the run {} left here",
            next.compartment.as_str(),
            next.from_host
        );
        Some(next.with_status(antumbra_core::GenesisStatus::Claimed, now))
    }

    /// Autonomous consolidation: if the just-written/reinforced `mem` belongs to
    /// a compartment, graduate that compartment into a private expert in the
    /// background (one train per compartment at a time; a burst coalesces). The
    /// shared `consolidate_compartment` gathers and scores, so it returns cheaply
    /// when nothing in the compartment clears the gate. A no-op unless built with
    /// `--features models` and enabled via `with_auto_consolidate`.
    pub(super) async fn maybe_consolidate(&self, _mem: &Memory) {
        #[cfg(feature = "models")]
        {
            let Some(acfg) = self.auto_consolidate.clone() else {
                return;
            };
            let Some(comp) = _mem.compartment.clone() else {
                return;
            };
            // The default/inbox compartment (`comp:{tenant}:{user}:default`) is an
            // unorganized grab-bag that can hold thousands of unrelated memories;
            // it is neither a coherent skill to graduate into one expert nor cheap
            // to gather on every write. Auto-consolidate only deliberately-created
            // compartments (`comp:<id>`); the inbox is organized first, via
            // `propose_compartments`, then those compartments consolidate.
            if comp.as_str().ends_with(":default") {
                return;
            }
            // Run the gather + provision + mint as OWNER on a stable connection,
            // not the per-request scoped `store` (which a detached task cannot
            // rely on); falls back to `store` for stdio / the embedded owner.
            let store = self
                .consolidation_store
                .clone()
                .unwrap_or_else(|| self.store.clone());
            // Where this run belongs (ADR-0017 A2). A node that cannot train
            // does not grind the model on a CPU while the user's GPU box sits
            // idle; it leaves the work where that machine will find it.
            if self.escalate_genesis(&store, &comp).await {
                return;
            }
            // This node is training. Clear what it was *asked* to do before what
            // it happened to be handed: a request has been waiting on another of
            // the user's machines, and this write has not waited at all. The
            // memory's own compartment is not lost -- the next write reaches it,
            // and by then the queue is shorter.
            let (comp, request) = match self.claim_genesis_request(&store).await {
                Some(waiting) => (waiting.compartment.clone(), Some(waiting)),
                None => (comp, None),
            };
            let key = comp.as_str().to_string();
            // Already consolidating this compartment: the write is remembered, and
            // the run in flight is followed by another (a memory that arrives
            // during a train would otherwise never be looked at again).
            if !self.consolidating.lock().await.begin(&key) {
                return;
            }
            let embedder = self.embedder.clone();
            let tenant = self.tenant.clone();
            // The private expert is owned by the session user who reinforced the
            // memory, not the memory's original author: in a shared compartment,
            // owning it by the author would mint an expert the reinforcing user
            // could not serve (owner-scoped), and attribute their training to
            // someone else.
            let user = self.user.clone();
            let state = self.consolidating.clone();
            let serve = self.serve.clone();
            let policy = antumbra_serve::ConsolidationPolicy {
                min_recurrence: acfg.min_recurrence,
                min_confidence: acfg.min_confidence,
                ..Default::default()
            };
            let cfg = antumbra_serve::RaftConfig {
                samples_per_task: acfg.samples,
                rounds: acfg.rounds,
                max_new_tokens: acfg.max_new_tokens,
                learning_rate: acfg.lr,
                replay_ratio: acfg.replay_ratio,
                ..Default::default()
            };
            // A train is minutes of synchronous compute inside an `async fn`. On a
            // runtime worker it starves whatever queues behind it, the store's
            // connection driver included, and every tool call hangs until it ends.
            consolidation::spawn_heavy(async move {
                use antumbra_serve::Consolidation;
                loop {
                    let result = antumbra_serve::consolidate_compartment(
                        &store,
                        embedder.as_ref(),
                        &tenant,
                        &user,
                        &comp,
                        &policy,
                        &cfg,
                    )
                    .await;
                    // Only a run that loaded the model cools down.
                    let trained = matches!(
                        result,
                        Ok(Consolidation::Minted(_) | Consolidation::DidNotLearn { .. })
                    );
                    match result {
                        Ok(Consolidation::Minted(o)) => {
                            // Close the loop: hot-register the minted expert so the
                            // `answer` tool can serve it now, with no server restart.
                            let servable = match &serve {
                                Some(serve) => {
                                    serve.register_expert(&o.expert, &o.adapter_uri);
                                    "now servable"
                                }
                                None => "not servable: this server has no serving engine",
                            };
                            state.lock().await.forget_report(&key);
                            eprintln!(
                                "[auto-consolidate] {} : {} graduated -> {} (internalized {:.2}, {servable})",
                                comp.as_str(),
                                o.graduated,
                                o.expert.as_str(),
                                o.fitness
                            );
                        }
                        // The trigger fires on every write, so an unchanged
                        // "nothing graduates" is said once, not per write.
                        Ok(Consolidation::HeldBack(report)) => {
                            let summary = report.summary();
                            if state.lock().await.changed(&key, &summary) {
                                eprintln!("[auto-consolidate] {} : {summary}", comp.as_str());
                            }
                        }
                        Ok(Consolidation::DidNotLearn { graduated, fitness }) => eprintln!(
                            "[auto-consolidate] {} : {graduated} graduated but the capture did not learn (fitness {fitness:.2}); no expert minted",
                            comp.as_str()
                        ),
                        Err(e) => eprintln!("[auto-consolidate] {} failed: {e}", comp.as_str()),
                    }
                    // Debounce: after an actual train, hold the compartment's slot a
                    // little longer so a burst of reinforces collapses into one train
                    // instead of retraining the same expert on every reinforce. No
                    // cooldown when nothing trained, so a later graduate is not delayed.
                    if trained {
                        tokio::time::sleep(std::time::Duration::from_secs(45)).await;
                    }
                    if !state.lock().await.finish(&key) {
                        break;
                    }
                }
                // The trainer has looked, whatever it concluded. A request is
                // closed by being serviced, not by producing an expert: if the
                // compartment held nothing back today and is reinforced again
                // tomorrow, the asking node raises a fresh request, which is
                // the loop working rather than a request that never closes.
                if let Some(request) = request {
                    if let Err(e) = antumbra_store::repo::genesis::set_status(
                        &store,
                        &request,
                        antumbra_core::GenesisStatus::Done,
                        chrono::Utc::now(),
                    )
                    .await
                    {
                        eprintln!(
                            "[auto-consolidate] ran {} for {} but could not close the request: {e}",
                            request.compartment.as_str(),
                            request.from_host
                        );
                    }
                }
            });
        }
    }

    /// The *unorganized* memory pool: the inbox (default compartment) plus
    /// anything uncompartmented. Deliberately-filed compartments are left alone;
    /// the antumbra proposes structure only over what the user has not organized.
    pub(super) async fn inbox_pool(&self) -> antumbra_core::Result<Vec<Memory>> {
        Ok(memory::list(&self.store, &self.tenant)
            .await?
            .into_iter()
            .filter(|m| {
                m.compartment.is_none() || m.compartment.as_ref() == Some(&self.default_compartment)
            })
            .collect())
    }

    /// Persist one proposal as an `Origin::Proposed` compartment you own and move
    /// its members in. Reversible: deleting the compartment undoes it. Returns the
    /// new compartment id.
    pub(super) async fn apply_proposal(
        &self,
        prop: &antumbra_core::ProposedCompartment,
    ) -> antumbra_core::Result<String> {
        let id = next_id("comp");
        let c = Compartment::new(
            id.clone(),
            self.tenant.clone(),
            self.user.clone(),
            prop.label.clone(),
            Utc::now(),
        )
        .proposed();
        compartment::create(&self.store, &c).await?;
        for mid in &prop.members {
            if let Some(mut m) = memory::get(&self.store, &self.tenant, mid).await? {
                m.compartment = Some(CompartmentId::new(id.clone()));
                m.updated_at = Utc::now();
                memory::upsert(&self.store, &m).await?;
            }
        }
        Ok(id)
    }

    /// The autonomous propose trigger: once the inbox reaches the configured
    /// threshold, cluster it and auto-create the proposals (so the inbox shrinks
    /// below the threshold and the trigger quiets until it grows again). Returns
    /// the created compartment ids; empty when disabled or below threshold.
    pub(super) async fn maybe_auto_propose(&self) -> antumbra_core::Result<Vec<String>> {
        let Some(cfg) = self.auto_propose.clone() else {
            return Ok(Vec::new());
        };
        let pool = self.inbox_pool().await?;
        if pool.len() < cfg.threshold {
            return Ok(Vec::new());
        }
        let cluster_cfg = ClusterConfig {
            similarity_threshold: cfg.similarity_threshold,
            min_size: cfg.min_size,
            ..ClusterConfig::default()
        };
        let proposals = antumbra_core::propose_compartments(&pool, &cluster_cfg);
        let mut created = Vec::with_capacity(proposals.len());
        for prop in &proposals {
            created.push(self.apply_proposal(prop).await?);
        }
        Ok(created)
    }

    /// Rank the experts covering an embedded task: shared experts via the learned
    /// router, plus the user's own private experts by centroid. Top-`k`, best
    /// first. The expert ACL already scopes `expert::list` to shared + own-private.
    ///
    /// Boundary inhibition at the counterfactual boundary of competence: if this task falls inside a known failure
    /// scope, escalate (return no routes) rather than route confidently; the
    /// same gate the CLI's learned-route path applies, so the two front doors
    /// agree. Only *actionable* boundaries inhibit (the relative C/C' margin), so
    /// this is a no-op until a verified correction has scoped one.
    pub(super) async fn ranked_routes(
        &self,
        v: &[f32],
        k: usize,
    ) -> antumbra_core::Result<Vec<RouteHit>> {
        let inhibition = boundary::list(&self.store)
            .await?
            .iter()
            .map(|b| b.inhibition_for(v, INHIBITION_RADIUS))
            .fold(0.0f32, f32::max);
        if inhibition > BOUNDARY_ESCALATE_THRESHOLD {
            return Ok(Vec::new());
        }
        let mut routes: Vec<RouteHit> = Vec::new();
        if let Some(router) = router::load(&self.store).await? {
            if router.covers(v) {
                for (id, probability) in router.route(v) {
                    routes.push(RouteHit {
                        expert_id: id.as_str().to_string(),
                        probability,
                        private: false,
                    });
                }
            }
        }
        for e in expert::list(&self.store).await? {
            if e.owner.as_ref() == Some(&self.user) {
                if let Some(sim) = e.capability_similarity(v) {
                    if sim >= PRIVATE_ROUTE_FLOOR {
                        routes.push(RouteHit {
                            expert_id: e.id.as_str().to_string(),
                            probability: sim,
                            private: true,
                        });
                    }
                }
            }
        }
        routes.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        routes.truncate(k);
        Ok(routes)
    }
}

impl McpServer {
    /// Dispatch a tool by name with raw JSON `arguments`, returning its result as
    /// JSON. This is the same set of tools `#[tool_router]` exposes over JSON-RPC,
    /// reached directly so a one-shot caller (the REST `/mcp/call` shim, P-1b) can
    /// invoke one without an MCP session/handshake. The bound `(tenant, user)` and
    /// the engine ACL apply exactly as they do over `/mcp` -- this is a transport,
    /// not a second authority.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ErrorData> {
        self.check_profile(name)?;
        self.keep_session().await?;
        // Deserialize `arguments` into the tool's params, call it, and serialize
        // the result -- one arm per tool, mirroring the `#[tool]` methods.
        macro_rules! dispatch {
            ($params:ty, $method:ident) => {{
                let p: $params = serde_json::from_value(arguments)
                    .map_err(|e| ErrorData::invalid_params(format!("bad arguments: {e}"), None))?;
                let Json(out) = self.$method(Parameters(p)).await?;
                serde_json::to_value(out)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))
            }};
        }
        match name {
            "store_memory" => dispatch!(StoreParams, store_memory),
            "recall_memories" => dispatch!(RecallParams, recall_memories),
            "ingest_document" => dispatch!(IngestDocumentParams, ingest_document),
            "recall_documents" => dispatch!(RecallDocumentsParams, recall_documents),
            "reinforce_memory" => dispatch!(IdParams, reinforce_memory),
            "forget_memory" => dispatch!(IdParams, forget_memory),
            "list_memories" => dispatch!(ListParams, list_memories),
            "relate_memories" => dispatch!(RelateParams, relate_memories),
            "get_neighbors" => dispatch!(NeighborsParams, get_neighbors),
            "route" => dispatch!(RouteParams, route),
            "answer" => dispatch!(AnswerParams, answer),
            "create_compartment" => dispatch!(CreateCompartmentParams, create_compartment),
            "propose_compartments" => dispatch!(ProposeCompartmentsParams, propose_compartments),
            "list_compartments" => {
                let Json(out) = self.list_compartments().await?;
                serde_json::to_value(out)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))
            }
            "population" => {
                let Json(out) = self.population().await?;
                serde_json::to_value(out)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))
            }
            "workspace_stats" => {
                let Json(out) = self.workspace_stats().await?;
                serde_json::to_value(out)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))
            }
            "share_compartment" => dispatch!(ShareParams, share_compartment),
            "revoke_compartment" => dispatch!(RevokeParams, revoke_compartment),
            other => Err(ErrorData::invalid_params(
                format!("unknown tool: {other}"),
                None,
            )),
        }
    }
}

#[tool_handler]
impl ServerHandler for McpServer {
    /// `tools/list` under the session's profile. Written out (the macro
    /// generates it only when absent) so the profile is enforced where the
    /// agent first sees the tools: the advertisement.
    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, ErrorData> {
        let supports_cache_hints = context
            .protocol_version()
            .is_some_and(|version| version >= rmcp::model::ProtocolVersion::V_2026_07_28);
        Ok(rmcp::model::ListToolsResult {
            result_type: Some(rmcp::model::ResultType::COMPLETE),
            tools: self.advertised_tools(),
            meta: None,
            next_cursor: None,
            ttl_ms: supports_cache_hints.then_some(0),
            cache_scope: supports_cache_hints.then_some(rmcp::model::CacheScope::Public),
        })
    }

    /// JSON-RPC `tools/call` under the session's profile: a tool outside it is
    /// refused before the router sees the request.
    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, ErrorData> {
        self.check_profile(&request.name)?;
        self.keep_session().await?;
        let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        Self::tool_router().call(tcc).await
    }

    /// On initialize, record this session's peer under its (tenant, user) identity
    /// so the live-propagation watcher can push shared-memory changes to it (R-2).
    /// A no-op when no registry is wired (stdio / route-only / tests).
    async fn on_initialized(
        &self,
        context: rmcp::service::NotificationContext<rmcp::service::RoleServer>,
    ) {
        // Bind THIS session's connection to its identity. rmcp builds one server
        // (and, on a remote, one DB connection) per session, so the binding must
        // happen here -- the HTTP layer's signin runs on a different handle and a
        // cloned remote connection does not share it. Signing in as the record
        // scopes the engine ACL for every tool call in this session (R-6).
        if let Err(e) = self.store.signin(&self.tenant, &self.user).await {
            eprintln!(
                "antumbra-mcp: session signin failed for {}/{}: {e}",
                self.tenant.as_str(),
                self.user.as_str()
            );
        } else if let Some(keeper) = &self.session {
            keeper.mark_signed_in().await;
        }
        if let Some(registry) = &self.registry {
            let identity = crate::auth::Identity {
                tenant: self.tenant.as_str().to_string(),
                user: self.user.as_str().to_string(),
            };
            registry.register(identity, context.peer.clone()).await;
        }
    }
}
