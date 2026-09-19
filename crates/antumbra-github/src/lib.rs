//! The native GitHub integration (ADR-0019): the events that make a memory's
//! git anchor stale (a merge, a branch deletion) originate in the hosting
//! platform, so the platform tells Antumbra directly instead of a session hook
//! happening to notice later.
//!
//! This crate is the pure half. [`verify`] checks a delivery's HMAC signature;
//! [`parse`] turns a delivery into the one [`Event`] shape Antumbra acts on;
//! the handlers ([`reanchor_merged`], [`orphan_branch`],
//! [`pull_request_memory`]) take memories in and return the changed ones, with
//! no store in sight, so every rule is a unit test. The MCP server's webhook
//! route does the I/O: verify, parse, load the workspace's memories, apply,
//! write back.
//!
//! Which workspace a repository's memories live in is the [`RepoMap`]: one
//! workspace for everything the App sees, or an explicit slug-to-tenant map.

mod event;
mod handlers;
mod repos;
mod signature;

pub use event::{
    parse, Actor, DeleteEvent, Event, EventError, GitRef, PullRequest, PullRequestEvent, Repository,
};
pub use handlers::{
    orphan_branch, pull_request_memory, reanchor_merged, Merge, SYSTEM_HOST, SYSTEM_USER,
};
pub use repos::{RepoMap, RepoMapError};
pub use signature::{
    sign, verify, SignatureError, DELIVERY_HEADER, EVENT_HEADER, SIGNATURE_HEADER,
};
