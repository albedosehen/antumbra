//! `GET /expert/adapter?id=<expert id>`: an expert's adapter, for another of
//! its user's servers to serve it from.
//!
//! An expert is trained on one machine and its weights stay on that machine's
//! disk, while its row reaches every server on the database. This route is
//! how one of those servers gets the weights (see `server::pull`).
//!
//! It answers under the bearer token every other route takes, and only for an
//! expert the caller may be served: a shared one, or the caller's own
//! standing expert. Whatever else is asked for is the same 404 as an expert
//! that does not exist, so the route says nothing about what it will not
//! give.
//!
//! It sends only what was admitted. The expert's card holds the sha256 and
//! size of the adapter it was admitted with, the row may name only the file
//! that sha256 determines, and the bytes are hashed before they are sent.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::rejection::ExtensionRejection;
use axum::extract::{ConnectInfo, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use serde::Deserialize;

use antumbra_core::adapter::{self, Seal};
use antumbra_core::limit::{self, Rate};
use antumbra_core::{Expert, ExpertId};

use super::relics::{client_of, problem, too_many};
use super::HttpState;
use crate::auth::Identity;
use crate::server::pull::ADAPTER_PATH;

/// The header that carries the sha256 of the adapter sent.
pub(crate) const SHA256_HEADER: &str = "x-antumbra-sha256";

/// One client's pulls: a server pulls an expert once and keeps it, so a few
/// at once is every expert a user has, and a steady stream is not a server.
pub(super) const PULLS_PER_CLIENT: Rate = Rate {
    burst: 8,
    per_minute: 8,
};

/// Everyone's pulls together. Each one reads a whole adapter into memory.
const PULLS_EVERYONE: Rate = Rate {
    burst: 16,
    per_minute: 32,
};

/// What bounds the route here.
pub(super) struct Guards {
    pulls: limit::Limiter,
}

impl Guards {
    pub(super) fn new() -> Self {
        Self {
            pulls: limit::Limiter::new(PULLS_PER_CLIENT, PULLS_EVERYONE),
        }
    }
}

pub(super) fn routes() -> Router<Arc<HttpState>> {
    Router::new().route(ADAPTER_PATH, get(adapter_file))
}

#[derive(Deserialize)]
struct AdapterQuery {
    /// In the query, not the path: an expert's id holds colons and, for a
    /// repository's scope, slashes.
    id: String,
}

fn no_adapter() -> Response {
    problem(
        StatusCode::NOT_FOUND,
        "no adapter of that expert that this server can give you",
    )
}

async fn adapter_file(
    State(state): State<Arc<HttpState>>,
    Query(query): Query<AdapterQuery>,
    headers: HeaderMap,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
) -> Response {
    let header_val = headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());
    let Ok(identity) = state.verifier.verify_header(header_val) else {
        return problem(StatusCode::UNAUTHORIZED, "the token was refused");
    };
    if !state.experts.pulls.allow(&client_of(peer)) {
        return too_many();
    }
    let id = ExpertId::new(query.id);
    let (expert, seal) = match shippable(&state, &identity, &id).await {
        Ok(Some(found)) => found,
        Ok(None) => return no_adapter(),
        Err(response) => return response,
    };
    // The row may name only the file its own id and sha256 determine: a row
    // is never a way to read another file off this machine.
    let path = std::path::PathBuf::from(&expert.artifact_uri);
    let named = path.file_name().and_then(|name| name.to_str())
        == Some(adapter::file_name(&expert.id, &seal.sha256).as_str());
    if !named {
        return no_adapter();
    }
    // A whole file read and hashed: off the runtime's workers.
    let read = tokio::task::spawn_blocking(move || {
        let bytes = std::fs::read(&path)?;
        let got = Seal::of_bytes(&bytes);
        std::io::Result::Ok((bytes, got))
    })
    .await;
    let bytes = match read {
        Ok(Ok((bytes, got))) if got == seal => bytes,
        Ok(Ok(_)) => {
            eprintln!(
                "antumbra-mcp: the adapter of {} is not the one its card seals; it was not sent",
                expert.id.as_str()
            );
            return no_adapter();
        }
        Ok(Err(e)) => {
            eprintln!(
                "antumbra-mcp: the adapter of {} could not be read: {e}",
                expert.id.as_str()
            );
            return no_adapter();
        }
        Err(e) => {
            eprintln!("antumbra-mcp: reading an adapter failed: {e}");
            return problem(
                StatusCode::INTERNAL_SERVER_ERROR,
                "could not read the adapter",
            );
        }
    };
    (
        [
            (header::CONTENT_TYPE, "application/octet-stream".to_string()),
            // Weights, never a page this origin runs, and not for a cache.
            (header::CACHE_CONTROL, "no-store".to_string()),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
            (
                header::HeaderName::from_static(SHA256_HEADER),
                seal.sha256.clone(),
            ),
        ],
        bytes,
    )
        .into_response()
}

/// The expert and its seal as `identity` may be given them, read on that
/// identity's own connection.
async fn shippable(
    state: &HttpState,
    identity: &Identity,
    id: &ExpertId,
) -> Result<Option<(Expert, Seal)>, Response> {
    let failed = |what: &str, e: &dyn std::fmt::Display| {
        eprintln!(
            "antumbra-mcp: adapter pull {what} failed for {}/{}: {e}",
            identity.tenant, identity.user
        );
        problem(
            StatusCode::INTERNAL_SERVER_ERROR,
            "could not open your workspace",
        )
    };
    let mcp = state
        .rest_server_for(identity)
        .await
        .map_err(|e| failed("init", &e))?;
    let _guard = state
        .bind(identity)
        .await
        .map_err(|e| failed("signin", &e))?;
    mcp.expert_to_ship(id).await.map_err(|e| failed("read", &e))
}
