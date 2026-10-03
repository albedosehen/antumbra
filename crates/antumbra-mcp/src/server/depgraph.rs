//! The evidence graph (ADR-0019 section 2): dependency claims recorded as
//! memories with evidence, and blast radius as a weighted walk over them.
//!
//! Its own router, joined to the others in `engine.rs`, because `server.rs`
//! is at the size rule. The model and the walk live in
//! `antumbra_core::depgraph`; this is the surface over them.

use super::*;

use antumbra_core::depgraph::{self, Claim, Direction, Edge, Source};

/// The most services one answer lists, strongest first.
const MAX_REACHED: usize = 100;
/// The deepest walk a caller may ask for.
const MAX_DEPTH: u32 = 6;

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
        let evidence = claim.evidence(anchor.as_ref());
        let content = claim.content();
        let id = MemoryId::new(claim.memory_id(&self.tenant));
        let now = Utc::now();

        if let Some(mut m) = memory::get(&self.store, &self.tenant, &id)
            .await
            .map_err(err)?
        {
            // Seen again: the latest evidence replaces the old, and the edge is
            // reinforced, which is what keeps it from fading.
            if m.content != content {
                m.embedding = Some(self.embedder.embed(&content).await.map_err(err)?);
                m.content = content;
            }
            m.evidence = evidence;
            memory::upsert(&self.store, &m).await.map_err(err)?;
            let reinforced = memory::reinforce(&self.store, &self.tenant, &id, now)
                .await
                .map_err(err)?
                .ok_or_else(|| ErrorData::internal_error("the edge vanished", None))?;
            return Ok(Json(RecordedDependencyOut {
                id: id.as_str().to_string(),
                created: false,
                confidence: reinforced.confidence,
                reinforcement: reinforced.reinforcement,
            }));
        }

        let embedding = self.embedder.embed(&content).await.map_err(err)?;
        let m = Memory::new(
            id.clone(),
            self.tenant.clone(),
            MemoryNetwork::World,
            content,
            source.confidence(),
            now,
        )
        .with_embedding(embedding)
        // The tenant's shared pool, not the recording user's default
        // compartment: a dependency is the workspace's knowledge, and every
        // member's blast radius should read it.
        .by(self.user.clone(), self.host.clone())
        .with_evidence(evidence);
        memory::upsert(&self.store, &m).await.map_err(err)?;
        if memory::get(&self.store, &self.tenant, &id)
            .await
            .map_err(err)?
            .is_none()
        {
            return Err(ErrorData::internal_error(
                "the dependency did not land in the workspace's shared pool",
                None,
            ));
        }
        Ok(Json(RecordedDependencyOut {
            id: id.as_str().to_string(),
            created: true,
            confidence: source.confidence(),
            reinforcement: 0,
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
        let memories =
            memory::with_any_evidence(&self.store, &self.tenant, &depgraph::source_entries())
                .await
                .map_err(err)?;
        let edges: Vec<Edge> = memories.iter().filter_map(Edge::of).collect();
        let now = Utc::now();
        let reached = depgraph::blast_radius(&edges, &p.service, direction, depth, min_weight, now)
            .into_iter()
            .take(MAX_REACHED)
            .map(|r| ReachedView {
                service: r.service,
                depth: r.depth,
                weight: r.weight,
                path: r
                    .path
                    .into_iter()
                    .map(|hop| HopView {
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
                    })
                    .collect(),
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
