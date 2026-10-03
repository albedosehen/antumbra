//! The evidence graph (ADR-0019 section 2): dependency claims recorded as
//! memories with evidence, and blast radius as a weighted walk over them.
//!
//! Its own router, joined to the others in `engine.rs`, because `server.rs`
//! is at the size rule. The model and the walk live in
//! `antumbra_core::depgraph`; this is the surface over them.

use super::*;

use antumbra_core::depgraph::{self, Claim, Direction, Edge, Hop, Source};

/// The most services one answer lists, strongest first.
const MAX_REACHED: usize = 100;
/// The deepest walk a caller may ask for.
const MAX_DEPTH: u32 = 6;
/// The most pairs one listing returns, and how many it returns by default.
const MAX_PAIRS: u32 = 200;
const DEFAULT_PAIRS: u32 = 100;
/// The most service names one listing returns.
const MAX_SERVICES: usize = 400;

/// One pair as a caller reads it, with each edge's current weight.
fn hop_view(hop: Hop, now: chrono::DateTime<Utc>) -> HopView {
    HopView {
        from: hop.from,
        to: hop.to,
        weight: hop.weight,
        evidence: hop
            .edges
            .iter()
            .map(|e| EdgeView {
                source: e.source.as_str().to_string(),
                detail: e.detail.clone(),
                confidence: e.confidence,
                weight: e.weight(now),
                reinforcement: e.reinforcement,
                last_seen: e.last_seen.to_rfc3339(),
                memory_id: e.memory_id.clone(),
                anchor: e.anchor.as_ref().map(ProvenanceView::from),
            })
            .collect(),
    }
}

impl McpServer {
    /// Every dependency edge in the workspace, found by the store's evidence
    /// filter rather than by reading every memory.
    async fn dependency_edges(&self) -> Result<Vec<Edge>, ErrorData> {
        let memories =
            memory::with_any_evidence(&self.store, &self.tenant, &depgraph::source_entries())
                .await
                .map_err(err)?;
        Ok(memories.iter().filter_map(Edge::of).collect())
    }
}

#[tool_router(router = depgraph_router, vis = "pub(super)")]
impl McpServer {
    /// Record that one service depends on another, with its evidence.
    #[tool(
        description = "Record that service `from` depends on service `to` (repository slugs, host/org/name), by `source`: declared (a manifest or config names it), observed (telemetry saw the calls), learned (they change or deploy together) or claimed (read from the code, believed little until corroborated). Recording the same edge and source again reinforces it rather than duplicating it."
    )]
    pub(super) async fn record_dependency(
        &self,
        Parameters(p): Parameters<RecordDependencyParams>,
    ) -> Result<Json<RecordedDependencyOut>, ErrorData> {
        let source = Source::parse(&p.source).ok_or_else(|| {
            ErrorData::invalid_params(
                format!(
                    "unknown source `{}`: one of declared, observed, learned, claimed",
                    p.source
                ),
                None,
            )
        })?;
        if p.from.trim().is_empty() || p.to.trim().is_empty() {
            return Err(ErrorData::invalid_params(
                "a dependency needs both `from` and `to`",
                None,
            ));
        }
        let claim = Claim::new(&p.from, &p.to, source, p.detail.as_deref().unwrap_or(""));
        if claim.from == claim.to {
            return Err(ErrorData::invalid_params(
                "a service does not depend on itself",
                None,
            ));
        }
        let anchor = p.provenance.map(anchor_from).transpose()?;
        let recorded = crate::dependencies::record(
            &self.store,
            self.embedder.as_ref(),
            &self.tenant,
            (&self.user, &self.host),
            &claim,
            anchor.as_ref(),
            Utc::now(),
        )
        .await
        .map_err(err)?;
        Ok(Json(RecordedDependencyOut {
            id: recorded.id.as_str().to_string(),
            created: recorded.created,
            confidence: recorded.confidence,
            reinforcement: recorded.reinforcement,
        }))
    }

    /// The graph as a list of pairs, for a person looking it over.
    #[tool(
        description = "The workspace's dependency graph as a list: each pair of services (`from` depends on `to`) with its combined current weight and every edge's evidence, strongest first, weak pairs included; at most `limit` pairs (default 100, at most 200), with how many there are in all and the services named. Pass `service` for only the pairs it is on either end of."
    )]
    pub(super) async fn list_dependencies(
        &self,
        Parameters(p): Parameters<ListDependenciesParams>,
    ) -> Result<Json<DependenciesOut>, ErrorData> {
        let edges = self.dependency_edges().await?;
        let now = Utc::now();
        let service = p
            .service
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let hops = depgraph::hops(&edges, service, now);
        let total = hops.len() as u32;
        let services: std::collections::BTreeSet<String> = hops
            .iter()
            .flat_map(|h| [h.from.clone(), h.to.clone()])
            .collect();
        let limit = p.limit.unwrap_or(DEFAULT_PAIRS).clamp(1, MAX_PAIRS) as usize;
        Ok(Json(DependenciesOut {
            pairs: hops
                .into_iter()
                .take(limit)
                .map(|hop| hop_view(hop, now))
                .collect(),
            total,
            services: services.into_iter().take(MAX_SERVICES).collect(),
        }))
    }

    /// What depends on a service, or what it depends on, with the evidence.
    #[tool(
        description = "Blast radius: the services that depend on `service` (direction `dependents`, the default: what breaks if it does) or that it depends on (`dependencies`), up to `depth` hops (default 3, at most 6), each by its strongest path with every edge's evidence, source and current weight. Edges fade when nobody records them again; a pair's sources combine as independent evidence, and pairs under `min_weight` (default 0.4) are not walked, so a claim alone does not count until something corroborates it."
    )]
    pub(super) async fn blast_radius(
        &self,
        Parameters(p): Parameters<BlastRadiusParams>,
    ) -> Result<Json<BlastRadiusOut>, ErrorData> {
        let direction = match p.direction.as_deref().map(str::trim) {
            None | Some("") | Some("dependents") => Direction::Dependents,
            Some("dependencies") => Direction::Dependencies,
            Some(other) => {
                return Err(ErrorData::invalid_params(
                    format!("unknown direction `{other}`: dependents or dependencies"),
                    None,
                ))
            }
        };
        let depth = p.depth.unwrap_or(3).clamp(1, MAX_DEPTH);
        let min_weight = p.min_weight.unwrap_or(0.4).clamp(0.0, 1.0);
        let edges = self.dependency_edges().await?;
        let now = Utc::now();
        let reached = depgraph::blast_radius(&edges, &p.service, direction, depth, min_weight, now)
            .into_iter()
            .take(MAX_REACHED)
            .map(|r| ReachedView {
                service: r.service,
                depth: r.depth,
                weight: r.weight,
                path: r.path.into_iter().map(|hop| hop_view(hop, now)).collect(),
            })
            .collect();
        Ok(Json(BlastRadiusOut {
            service: antumbra_core::provenance::normalize_repo(&p.service),
            direction: match direction {
                Direction::Dependents => "dependents",
                Direction::Dependencies => "dependencies",
            }
            .to_string(),
            edges: edges.len() as u32,
            reached,
        }))
    }
}
