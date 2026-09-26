//! The native GitHub integration (ADR-0019): the events that make a memory's
//! git anchor stale (a merge, a branch deletion) originate in the hosting
//! platform, so the platform tells Antumbra directly instead of a session hook
//! happening to notice later.
//!
//! This crate is the pure half plus the API client. [`verify`] checks a
//! delivery's HMAC signature; [`parse`] turns a delivery into the one
//! [`Event`] shape Antumbra acts on; the handlers ([`reanchor_merged`],
//! [`orphan_branch`], [`pull_request_memory`]) take memories in and return the
//! changed ones, with no store in sight, so every rule is a unit test.
//! [`GithubApi`] reads what a handler needs from GitHub as the App
//! ([`AppCredentials`]): a merged pull request's changed files and their
//! contents at the merge commit, or a newly installed repository's tree. The
//! MCP server's webhook route does the I/O: verify, parse, load the
//! workspace's memories, apply, write back, ingest.
//!
//! Which workspace a repository's memories live in is the [`RepoMap`]: one
//! workspace for everything the App sees, or an explicit slug-to-tenant map.

mod api;
mod app;
mod docs;
mod event;
mod handlers;
mod knowledge;
mod repos;
mod signature;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use api::{
    ApiError, ChangedFile, GithubApi, GithubTransport, HttpResponse, InstallationToken, Tree,
    DEFAULT_API_URL,
};
pub use app::{AppCredentials, AppError, APP_JWT_TTL_SECS};
pub use docs::{is_knowledge_document, MAX_DOCUMENT_BYTES};
pub use event::{
    parse, Actor, DeleteEvent, Event, EventError, GitRef, Installation, InstallationEvent,
    PullRequest, PullRequestEvent, RepoRef, Repository,
};
pub use handlers::{
    orphan_branch, pull_request_memory, reanchor_merged, Merge, SYSTEM_HOST, SYSTEM_USER,
};
pub use knowledge::{AtPath, CheckOutput, KnowledgeDiff, Named};
pub use repos::{RepoMap, RepoMapError};
pub use signature::{
    sign, verify, SignatureError, DELIVERY_HEADER, EVENT_HEADER, SIGNATURE_HEADER,
};
