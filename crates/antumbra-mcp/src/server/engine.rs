//! Everything about the server that is not one of its tools: how it is built,
//! the triggers it runs after a write, the routing and reranking the tools
//! lean on, the one-shot `call_tool` the REST shim dispatches through, and the
//! `ServerHandler` impl that hands the router to rmcp.
//!
//! A child module of `server`, so these are the same inherent methods on the
//! same `McpServer` and they see the helpers next door; they moved here only
//! because `server.rs` had grown past the size rule. The `#[tool_router]`
//! blocks stay in `server.rs` and `provenance.rs`, because that macro builds a
//! router from the block it is applied to; `tool_router` here joins them.

use super::*;
use tracing::Instrument as _;

/// The learned router to route with. One the store holds but cannot decode,
/// as after a rollback to a server older than the router's format, degrades
/// to none, as one never trained does: the task escalates to the agent rather
/// than failing the call (ADR-0024 Validation 7). Anything else the store
/// reports still fails it.
fn readable(
    loaded: antumbra_core::Result<Option<antumbra_core::LearnedRouter>>,
) -> antumbra_core::Result<Option<antumbra_core::LearnedRouter>> {
    match loaded {
        Err(antumbra_core::AntumbraError::Serde(e)) => {
            eprintln!("antumbra-mcp: the learned router is unreadable, escalating instead: {e}");
            Ok(None)
        }
        other => other,
    }
}

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
            embedder,
            tenant,
            user,
            host,
            default_compartment,
            auto_propose: None,
            serve,
            registry: None,
            reranker: None,
            reranker_cache: Arc::new(tokio::sync::Mutex::new(RerankCache::new())),
            copal: None,
            profile: None,
            session: None,
            decider: None,
        }
    }

    /// Give recall a relevance floor (ADR-0024 D-2, closing ADR-0023 B-2).
    ///
    /// Off by default, and off is not a degraded mode: without a decider recall
    /// returns every row it found, which is what it did before this existed.
    ///
    /// Reached only from tests until a head exists to pass it. The seam is built
    /// first on purpose (the port, the fake, this setter, the floor and its
    /// tests), so the trained head lands as the one changed piece rather than as
    /// a change to the recall path at the same time.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn with_decider(mut self, decider: Arc<dyn antumbra_core::ports::TypedDecider>) -> Self {
        self.decider = Some(decider);
        self
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
    /// Every tool: the memory tools in `server.rs`, the provenance tools in
    /// `provenance.rs`, and the compartment and document tools in theirs, each
    /// block building its own router.
    pub(super) fn tool_router() -> rmcp::handler::server::router::tool::ToolRouter<Self> {
        Self::memory_router()
            + Self::provenance_router()
            + Self::compartment_router()
            + Self::document_router()
            + Self::listing_router()
            + Self::handoff_router()
            + Self::device_router()
            + Self::behaviour_router()
            + Self::answer_router()
            + Self::depgraph_router()
    }

    pub fn all_tool_names() -> Vec<String> {
        Self::tool_router()
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect()
    }

    /// Whether `name` is one of this server's tools, profile or not: what
    /// decides whether a call's span may carry the name it asked for. Read off
    /// the router once.
    fn is_tool(name: &str) -> bool {
        static NAMES: std::sync::OnceLock<std::collections::HashSet<String>> =
            std::sync::OnceLock::new();
        NAMES
            .get_or_init(|| Self::all_tool_names().into_iter().collect())
            .contains(name)
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
    /// first. The expert ACL already scopes `expert::list` to shared + own-private,
    /// and only experts the gate may route to (the active ones) are candidates.
    ///
    /// Boundary inhibition at the counterfactual boundary of competence: if this task falls inside a known failure
    /// scope, escalate (return no routes) rather than route confidently; the
    /// same gate the CLI's learned-route path applies, so the two front doors
    /// agree. Only *actionable* boundaries inhibit (the relative C/C' margin), so
    /// this is a no-op until a verified correction has scoped one.
    /// The ranked routes for a task, and when there are none, why: what a
    /// caller needs to decide between waiting for an expert and going to the
    /// generalist.
    pub(super) async fn routed(
        &self,
        v: &[f32],
        k: usize,
    ) -> antumbra_core::Result<(Vec<RouteHit>, Option<&'static str>)> {
        let inhibition = boundary::list(&self.store)
            .await?
            .iter()
            .map(|b| b.inhibition_for(v, INHIBITION_RADIUS))
            .fold(0.0f32, f32::max);
        if inhibition > BOUNDARY_ESCALATE_THRESHOLD {
            return Ok((Vec::new(), Some(UNCOVERED_INHIBITED)));
        }
        let mut routes: Vec<RouteHit> = Vec::new();
        let router = readable(lifecycle::load_router(&self.store).await)?;
        let uncovered = match &router {
            None => UNCOVERED_NO_ROUTER,
            Some(_) => UNCOVERED_OUT_OF_DISTRIBUTION,
        };
        if let Some(router) = router {
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
        for e in lifecycle::routable(&self.store).await? {
            // A standing expert is composed into the answers of its scope
            // (`answer`), never routed: its centroid would otherwise carry one
            // repository's behaviours into another's tasks.
            if e.owner.as_ref() == Some(&self.user) && e.standing_scope().is_none() {
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
        if !routes.is_empty() {
            return Ok((routes, None));
        }
        // The router as trained, before the experts it may not route to were
        // masked out: when it covers the task, the experts for it are resting.
        let resting = uncovered == UNCOVERED_OUT_OF_DISTRIBUTION
            && readable(antumbra_store::repo::router::load(&self.store).await)?
                .is_some_and(|trained| trained.covers(v));
        Ok((
            routes,
            Some(if resting {
                UNCOVERED_RESTING
            } else {
                uncovered
            }),
        ))
    }
}

/// Why a task found no expert.
const UNCOVERED_INHIBITED: &str =
    "a failure boundary covers this task: the population has failed at it before";
const UNCOVERED_NO_ROUTER: &str =
    "no learned router yet, and none of your private experts is close to this task";
const UNCOVERED_RESTING: &str =
    "the experts that cover this task are dormant or archived, and none of your private experts is close to it";
const UNCOVERED_OUT_OF_DISTRIBUTION: &str =
    "outside what the shared population covers, and none of your private experts is close to it";

impl McpServer {
    /// Dispatch a tool by name with raw JSON `arguments`, returning its result as
    /// JSON. This is the same set of tools `#[tool_router]` exposes over JSON-RPC,
    /// reached directly so a one-shot caller (the REST `/mcp/call` shim, P-1b) can
    /// invoke one without an MCP session/handshake. The bound `(tenant, user)` and
    /// the engine ACL apply exactly as they do over `/mcp` -- this is a transport,
    /// not a second authority.
    ///
    /// Traced as one tool call ([`crate::telemetry::tool_span`]), inside the
    /// span of the request that made it.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ErrorData> {
        let span =
            crate::telemetry::tool_span(name, Self::is_tool(name), self.tenant.as_str(), None);
        let result = self
            .dispatch(name, arguments)
            .instrument(span.clone())
            .await;
        crate::telemetry::record_outcome(
            &span,
            result.as_ref().err().map(crate::telemetry::error_kind),
        );
        result
    }

    /// [`Self::call_tool`]'s dispatch, inside its span.
    async fn dispatch(
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
            "penalize_memory" => dispatch!(IdParams, penalize_memory),
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
            "record_dependency" => dispatch!(RecordDependencyParams, record_dependency),
            "blast_radius" => dispatch!(BlastRadiusParams, blast_radius),
            "list_dependencies" => dispatch!(ListDependenciesParams, list_dependencies),
            "leave_handoff" => dispatch!(LeaveHandoffParams, leave_handoff),
            "handoffs" => dispatch!(HandoffsParams, handoffs),
            "complete_handoff" => dispatch!(CompleteHandoffParams, complete_handoff),
            "register_device" => dispatch!(device::RegisterDeviceParams, register_device),
            "devices" => {
                let Json(out) = self.devices().await?;
                serde_json::to_value(out)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))
            }
            "record_behaviour" => dispatch!(behaviour::RecordBehaviourParams, record_behaviour),
            "list_behaviours" => dispatch!(behaviour::ListBehavioursParams, list_behaviours),
            "accept_behaviour" => dispatch!(behaviour::BehaviourIdParams, accept_behaviour),
            "retire_behaviour" => dispatch!(behaviour::BehaviourIdParams, retire_behaviour),
            "list_documents" => {
                let Json(out) = self.list_documents().await?;
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
            "record_merges" => dispatch!(super::provenance::RecordMergesParams, record_merges),
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
    ///
    /// Traced as one tool call ([`crate::telemetry::tool_span`]). rmcp runs it
    /// on the session's task, away from the HTTP request that carried it, so
    /// the span joins that request's trace through the request parts rmcp
    /// hands over; over stdio there is no request, and the span stands alone.
    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, ErrorData> {
        let parent = context
            .extensions
            .get::<axum::http::request::Parts>()
            .and_then(|parts| parts.extensions.get::<crate::telemetry::RequestTrace>());
        let span = crate::telemetry::tool_span(
            &request.name,
            Self::is_tool(&request.name),
            self.tenant.as_str(),
            parent,
        );
        let result = async {
            self.check_profile(&request.name)?;
            self.keep_session().await?;
            let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
            Self::tool_router().call(tcc).await
        }
        .instrument(span.clone())
        .await;
        let failure = match &result {
            Err(e) => Some(crate::telemetry::error_kind(e)),
            // A tool that reports its failure in the result rather than as an
            // error is still a failed call.
            Ok(rmcp::model::CallToolResponse::Complete(done)) if done.is_error == Some(true) => {
                Some("tool_error")
            }
            Ok(_) => None,
        };
        crate::telemetry::record_outcome(&span, failure);
        result
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

#[cfg(test)]
mod tests {
    use super::readable;
    use antumbra_core::AntumbraError;

    #[test]
    fn an_unreadable_router_routes_as_none_and_a_store_fault_still_fails() {
        let undecodable = serde_json::from_str::<antumbra_core::LearnedRouter>("{}")
            .expect_err("a router needs its fields");
        assert!(matches!(readable(Err(undecodable.into())), Ok(None)));
        assert!(matches!(
            readable(Err(AntumbraError::Store("connection reset".into()))),
            Err(AntumbraError::Store(_))
        ));
        assert!(matches!(readable(Ok(None)), Ok(None)));
    }
}
