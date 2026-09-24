//! The compartment tools: create, propose, list, share and revoke the
//! private latent-spaces a user keeps memories in.
//!
//! Their own router, joined to the memory and provenance tools' in
//! `engine.rs`, because `server.rs` is past the size rule.

use super::*;

#[tool_router(router = compartment_router, vis = "pub(super)")]
impl McpServer {
    /// Create a private compartment (latent-space) owned by you.
    #[tool(
        description = "Create a new compartment (a private latent-space of memory) you own. Store into it via store_memory's compartment arg. Returns its id."
    )]
    pub(super) async fn create_compartment(
        &self,
        Parameters(p): Parameters<CreateCompartmentParams>,
    ) -> Result<Json<CompartmentView>, ErrorData> {
        let id = next_id("comp");
        let c = Compartment::new(
            id.clone(),
            self.tenant.clone(),
            self.user.clone(),
            p.name.clone(),
            Utc::now(),
        );
        compartment::create(&self.store, &c).await.map_err(err)?;
        Ok(Json(CompartmentView {
            id,
            name: p.name,
            origin: Origin::User.as_str().to_string(),
        }))
    }

    /// Propose compartments by clustering your uncompartmented memories.
    #[tool(
        description = "Have the antumbra propose compartments by clustering your uncompartmented memories into competence-coherent regions. Returns proposals (label, member ids, cohesion). With apply=true it also creates each as a proposed compartment you own and moves its members in (reversible by deleting the compartment)."
    )]
    pub(super) async fn propose_compartments(
        &self,
        Parameters(p): Parameters<ProposeCompartmentsParams>,
    ) -> Result<Json<ProposalsOut>, ErrorData> {
        let candidates = self.inbox_pool().await.map_err(err)?;
        let cfg = ClusterConfig {
            similarity_threshold: p.similarity_threshold,
            min_size: p.min_size,
            ..ClusterConfig::default()
        };
        // Fully-qualified: the crate fn shares this tool's name.
        let proposals = antumbra_core::propose_compartments(&candidates, &cfg);
        let mut views = Vec::with_capacity(proposals.len());
        for prop in proposals {
            let compartment_id = if p.apply {
                Some(self.apply_proposal(&prop).await.map_err(err)?)
            } else {
                None
            };
            views.push(ProposalView {
                label: prop.label,
                members: prop
                    .members
                    .iter()
                    .map(|m| m.as_str().to_string())
                    .collect(),
                cohesion: prop.cohesion,
                compartment_id,
            });
        }
        Ok(Json(ProposalsOut { proposals: views }))
    }

    /// List the compartments you own.
    #[tool(description = "List the compartments you own (including any the antumbra proposed).")]
    pub(super) async fn list_compartments(&self) -> Result<Json<CompartmentsOut>, ErrorData> {
        let comps = compartment::list_owned(&self.store, &self.tenant, &self.user)
            .await
            .map_err(err)?;
        Ok(Json(CompartmentsOut {
            compartments: comps
                .iter()
                .map(|c| CompartmentView {
                    id: c.id.as_str().to_string(),
                    name: c.name.clone(),
                    origin: c.origin.as_str().to_string(),
                })
                .collect(),
        }))
    }

    /// Share a compartment you own with another user.
    #[tool(
        description = "Share one of your compartments with another user: reference (they can recall it) or link (they can also connect to it)."
    )]
    pub(super) async fn share_compartment(
        &self,
        Parameters(p): Parameters<ShareParams>,
    ) -> Result<Json<ShareOut>, ErrorData> {
        let g = Grant::new(
            self.tenant.clone(),
            CompartmentId::new(p.compartment_id),
            UserId::new(p.grantee),
            parse_capability(&p.capability),
            self.user.clone(),
            Utc::now(),
        );
        compartment::grant(&self.store, &g).await.map_err(err)?;
        Ok(Json(ShareOut { shared: true }))
    }

    /// Revoke a user's access to one of your compartments.
    #[tool(
        description = "Revoke a user's access to one of your compartments (takes effect immediately)."
    )]
    pub(super) async fn revoke_compartment(
        &self,
        Parameters(p): Parameters<RevokeParams>,
    ) -> Result<Json<RevokeOut>, ErrorData> {
        compartment::revoke(
            &self.store,
            &self.tenant,
            &CompartmentId::new(p.compartment_id),
            &UserId::new(p.grantee),
            Utc::now(),
        )
        .await
        .map_err(err)?;
        Ok(Json(RevokeOut { revoked: true }))
    }
}
