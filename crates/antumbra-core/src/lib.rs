//! # antumbra-core
//!
//! The storage-agnostic domain for typed entities, their
//! lifecycle state machines, and the port traits that seam off serving and
//! training so the rest of the system is exercisable without a GPU.
//!
//! Metaphor map in case you need it:
//! - **umbra** = frozen experts ([`expert`])
//! - **penumbra** = shadows-in-training ([`shadow`])
//! - **antumbra** = the counterfactual boundary, the ([`boundary`])

pub mod boundary;
pub mod calibrate;
pub mod compartment;
pub mod contribution;
pub mod device;
pub mod document;
pub mod error;
pub mod evaluation;
pub mod expert;
pub mod generational;
pub mod genesis;
pub mod hive;
pub mod ids;
pub mod keyed;
pub mod lifecycle;
pub mod memory;
pub mod orchestration;
pub mod penumbra;
pub mod platt;
pub mod ports;
pub mod provenance;
pub mod recipe;
pub mod reward;
pub mod router;
pub mod shadow;
pub mod slice;
pub mod vector;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use error::{AntumbraError, Result};

// Re-export the load-bearing types at the crate root for ergonomic downstream use.
pub use boundary::{governing_feature_from_pair, BoundaryFinding, FailureBoundary, Grain};
pub use compartment::{Capability, Compartment, Grant, Origin};
pub use contribution::{BaselineRecord, ContributionRecord};
pub use device::{
    backend_can_train, genesis_placement, role_for, this_host, DeviceProfile, DeviceRole,
    GenesisPlacement, GENESIS_MIN_VRAM_MIB,
};
pub use document::{chunk_text, DocumentChunk};
pub use evaluation::{EvalStatus, EvaluationRun, SubjectKind};
pub use expert::{cosine_similarity, Expert};
pub use generational::{GenerationHead, LoopCommand, LoopControl, LoopState};
pub use genesis::{GenesisRequest, GenesisStatus};
pub use hive::{is_open, Hive, HiveMembership, HiveOffer, OfferStatus, OfferedKind};
pub use ids::{
    BoundaryId, CompartmentId, DocumentChunkId, ExpertId, Generation, MemoryId, RunId, ShadowId,
    TenantId, UserId,
};
pub use lifecycle::{current_status, ExpertStatus, ExpertTransition, TransitionCause};
pub use memory::{EdgeType, Memory, MemoryEdge, MemoryNetwork, MemoryStatus};
pub use orchestration::{ComposeStrategy, OrchestrationRun, OrchestrationStatus};
pub use penumbra::{propose_compartments, ClusterConfig, ProposedCompartment};
pub use provenance::{
    demote_out_of_scope, mark_orphaned, normalize_repo, orphan_of, reanchor, repo_slug_from_remote,
    scope_from_str, scope_of, scope_of_evidence, BranchOrphan, GitContext, GitProvenance, Scope,
    GIT_EVIDENCE_PREFIX, ORPHAN_EVIDENCE_PREFIX,
};
pub use recipe::{RecipeRecord, TrainingRecipe};
pub use reward::{fold_step, RewardSignal, RewardSource};
pub use router::{LearnedRouter, RouterExpert};
pub use shadow::{Shadow, ShadowStatus};
pub use vector::truncate_renormalize;
