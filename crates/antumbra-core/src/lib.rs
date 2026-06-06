//! # antumbra-core
//!
//! The storage-agnostic domain heart of Antumbra: typed entities, their
//! lifecycle state machines, and the port traits that seam off serving and
//! training so the rest of the system is exercisable without a GPU.
//!
//! Metaphor map (a cast shadow has three regions):
//! - **umbra** = frozen experts ([`expert`], ADR-0001)
//! - **penumbra** = shadows-in-training ([`shadow`], ADR-0002)
//! - **antumbra** = the counterfactual boundary, the keystone ([`boundary`], ADR-0004)

pub mod boundary;
pub mod compartment;
pub mod error;
pub mod evaluation;
pub mod expert;
pub mod generational;
pub mod ids;
pub mod memory;
pub mod orchestration;
pub mod penumbra;
pub mod ports;
pub mod reward;
pub mod router;
pub mod shadow;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use error::{AntumbraError, Result};

// Re-export the load-bearing types at the crate root for ergonomic downstream use.
pub use boundary::{FailureBoundary, Grain};
pub use compartment::{Capability, Compartment, Grant, Origin};
pub use evaluation::{EvalStatus, EvaluationRun, SubjectKind};
pub use expert::{cosine_similarity, Expert};
pub use generational::{GenerationHead, LoopState};
pub use ids::{
    BoundaryId, CompartmentId, ExpertId, Generation, MemoryId, RunId, ShadowId, TenantId, UserId,
};
pub use memory::{EdgeType, Memory, MemoryEdge, MemoryNetwork, MemoryStatus};
pub use orchestration::{ComposeStrategy, OrchestrationRun, OrchestrationStatus};
pub use penumbra::{propose_compartments, ClusterConfig, ProposedCompartment};
pub use reward::{fold_step, RewardSignal, RewardSource};
pub use router::{LearnedRouter, RouterExpert};
pub use shadow::{Shadow, ShadowStatus};
