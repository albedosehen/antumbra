//! The name of every table the schema defines, written once. The repositories,
//! the store sync and the control server's wire contract all name tables; a
//! literal in each let a renamed table silently stop replicating or drift out
//! of the contract, where a constant fails to compile. [`ALL`] is what the
//! schema is held to: every table it defines is here, and nothing else is.

pub const ACCOUNT: &str = "account";
pub const BEHAVIOR_TUNING: &str = "behavior_tuning";
pub const COMPARTMENT: &str = "compartment";
pub const CONTRIBUTION: &str = "contribution";
pub const DEVICE_PROFILE: &str = "device_profile";
pub const DOCUMENT_CHUNK: &str = "document_chunk";
/// A member's name and image.
pub const MEMBER_PROFILE: &str = "member_profile";
pub const EMBEDDER_CONFIG: &str = "embedder_config";
pub const EVALUATION_RUN: &str = "evaluation_run";
pub const EXPERT: &str = "expert";
pub const EXPERT_TRANSITION: &str = "expert_transition";
pub const FAILURE_BOUNDARY: &str = "failure_boundary";
pub const GENERATION_HEAD: &str = "generation_head";
pub const GENESIS_REQUEST: &str = "genesis_request";
pub const GRANT: &str = "grant";
pub const GROW: &str = "grow";
pub const HIVE: &str = "hive";
pub const HIVE_MEMBERSHIP: &str = "hive_membership";
pub const HIVE_OFFER: &str = "hive_offer";
pub const INVITE_CODE: &str = "invite_code";
pub const LEARNED_ROUTER: &str = "learned_router";
pub const LOOP_CONTROL: &str = "loop_control";
pub const MAGIC_LINK_USE: &str = "magic_link_use";
pub const MANIFEST_SET: &str = "manifest_set";
pub const MEMORY: &str = "memory";
pub const MEMORY_CHUNK: &str = "memory_chunk";
pub const MEMORY_EDGE: &str = "memory_edge";
pub const MEMORY_POLICY: &str = "memory_policy";
pub const MEMORY_USE: &str = "memory_use";
pub const ORCHESTRATION_RUN: &str = "orchestration_run";
pub const POPULATION_BASELINE: &str = "population_baseline";
pub const PRINCIPAL: &str = "principal";
pub const RECIPE: &str = "recipe";
pub const REGION_CENSUS: &str = "region_census";
pub const RELIC: &str = "relic";
pub const REVOKED_TOKEN: &str = "revoked_token";
pub const REWARD_SIGNAL: &str = "reward_signal";
pub const ROUTING_OUTCOME: &str = "routing_outcome";
pub const SHADOW: &str = "shadow";
pub const VERIFIER: &str = "verifier";
pub const VERIFIER_MEASUREMENT: &str = "verifier_measurement";
pub const VERIFIER_TRANSITION: &str = "verifier_transition";

/// Every table the schema defines, in the schema's order.
pub const ALL: &[&str] = &[
    ACCOUNT,
    BEHAVIOR_TUNING,
    COMPARTMENT,
    CONTRIBUTION,
    DEVICE_PROFILE,
    DOCUMENT_CHUNK,
    EMBEDDER_CONFIG,
    EVALUATION_RUN,
    EXPERT,
    EXPERT_TRANSITION,
    FAILURE_BOUNDARY,
    GENERATION_HEAD,
    GENESIS_REQUEST,
    GRANT,
    GROW,
    HIVE,
    HIVE_MEMBERSHIP,
    HIVE_OFFER,
    INVITE_CODE,
    LEARNED_ROUTER,
    LOOP_CONTROL,
    MAGIC_LINK_USE,
    MANIFEST_SET,
    MEMBER_PROFILE,
    MEMORY,
    MEMORY_CHUNK,
    MEMORY_EDGE,
    MEMORY_POLICY,
    MEMORY_USE,
    ORCHESTRATION_RUN,
    POPULATION_BASELINE,
    PRINCIPAL,
    RECIPE,
    REGION_CENSUS,
    RELIC,
    REVOKED_TOKEN,
    REWARD_SIGNAL,
    ROUTING_OUTCOME,
    SHADOW,
    VERIFIER,
    VERIFIER_MEASUREMENT,
    VERIFIER_TRANSITION,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The schema defines exactly these tables, so a table added to one and
    /// not the other is caught here, by name.
    #[test]
    fn every_table_the_schema_defines_is_named_here_and_nothing_else() {
        let mut defined: Vec<String> = crate::schema::tables(crate::EMBED_DIM as u32)
            .iter()
            .map(|t| t.name.clone())
            .collect();
        defined.sort();
        let mut named: Vec<String> = ALL.iter().map(|n| (*n).to_owned()).collect();
        named.sort();
        assert_eq!(named, defined);
        assert_eq!(ALL.len(), 42, "every name is distinct");
    }
}
