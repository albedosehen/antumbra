//! Knowledge-document ingest: the one path every door into the store takes,
//! so the MCP `ingest_document` tool, the CLI `ingest`, and the GitHub
//! integration write the same shape and keep the same promises.
//!
//! - **Document of record first.** With a copal archive configured, the
//!   original content is uploaded before any chunk is stored, and a failure
//!   fails the whole ingest: a configured archive that silently dropped
//!   originals would be worse than none.
//! - **Replace in place.** A title names one document within a workspace.
//!   Re-ingesting it replaces its chunks rather than accumulating copies, so
//!   the corpus never carries two generations of the same document.
//! - **Anchored.** A git anchor, when known, is folded into every chunk's
//!   `source`, so a recalled chunk names the commit it describes.

use chrono::Utc;

use antumbra_copal::{ArchivedDocument, CopalArchive};
use antumbra_core::document::{DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP};
use antumbra_core::ports::Embedder;
use antumbra_core::{chunk_text, DocumentChunk, DocumentChunkId, GitProvenance, TenantId};
use antumbra_store::repo::document;
use antumbra_store::Store;

/// A knowledge document to ingest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    /// The document's identity within the workspace: one title, one document.
    pub title: String,
    /// Where it came from (a path, a URL, the command that printed it).
    pub source: Option<String>,
    pub content: String,
    /// The git anchor folded into every chunk's `source`, when known.
    pub provenance: Option<GitProvenance>,
}

/// What one ingest produced: the chunk count, and the copal document of
/// record when an archive was configured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ingested {
    pub chunks: u32,
    pub archived: Option<ArchivedDocument>,
}

/// Ingest `doc` under `tenant`: archive the original when an `archive` is
/// configured (upload first, fail closed), chunk and embed the content, then
/// replace the title's previous chunks with the new ones.
///
/// Every chunk is embedded before the previous generation goes, so a failed
/// embed leaves the old document readable; the moment between the delete and
/// the insert is the one window a recall can miss the title.
pub async fn ingest_text(
    store: &Store,
    embedder: &dyn Embedder,
    tenant: &TenantId,
    archive: Option<&CopalArchive>,
    doc: &Document,
) -> anyhow::Result<Ingested> {
    let archived = match archive {
        Some(archive) => Some(
            archive
                .archive_document(
                    tenant.as_str(),
                    &doc.title,
                    doc.source.as_deref(),
                    &doc.content,
                )
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "copal document-of-record upload failed, nothing was ingested: {e}"
                    )
                })?,
        ),
        None => None,
    };
    let source = source_of(doc);
    let now = Utc::now();
    let mut chunks = Vec::new();
    for (ordinal, content) in chunk_text(&doc.content, DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP)
        .into_iter()
        .enumerate()
    {
        let embedding = embedder.embed(&content).await?;
        let mut chunk = DocumentChunk {
            id: DocumentChunkId::new(chunk_id(tenant, &doc.title, ordinal)),
            tenant: tenant.clone(),
            title: doc.title.clone(),
            source: source.clone(),
            ordinal: ordinal as u32,
            content,
            embedding: Some(embedding),
            created_at: now,
            copal_file: None,
            copal_digest: None,
        };
        if let Some(d) = &archived {
            chunk = chunk.with_copal(d.file_id.clone(), d.digest.clone());
        }
        chunks.push(chunk);
    }
    // Ids repeat per (tenant, title, ordinal), so the common case overwrites
    // in place; the delete is what drops the tail when the document shrank.
    document::delete_title(store, tenant, &doc.title).await?;
    document::insert_chunks(store, &chunks).await?;
    Ok(Ingested {
        chunks: chunks.len() as u32,
        archived,
    })
}

/// The `source` every chunk carries: the caller's source and the git anchor,
/// joined so a recalled chunk names both where the document came from and the
/// commit it describes.
pub fn source_of(doc: &Document) -> Option<String> {
    match (&doc.source, &doc.provenance) {
        (Some(s), Some(p)) => Some(format!("{s} @ {p}")),
        (Some(s), None) => Some(s.clone()),
        (None, Some(p)) => Some(p.to_string()),
        (None, None) => None,
    }
}

/// A stable chunk id: the same (tenant, title, ordinal) always names the same
/// record, which is what makes re-ingest an in-place replacement.
pub fn chunk_id(tenant: &TenantId, title: &str, ordinal: usize) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tenant.as_str().hash(&mut hasher);
    title.hash(&mut hasher);
    ordinal.hash(&mut hasher);
    format!("docchunk:{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_copal::{CopalCredential, CopalTenancy, CopalTransport};
    use antumbra_core::testing::FixedEmbedder;
    use antumbra_store::EMBED_DIM;
    use serde_json::{json, Value};
    use std::sync::Arc;

    /// A copal that answers the create + upload calls, or fails them, offline.
    struct FakeCopal {
        up: bool,
    }

    impl CopalTransport for FakeCopal {
        fn post_json(
            &self,
            _url: &str,
            _credential: &CopalCredential,
            _body: &Value,
        ) -> antumbra_core::Result<Value> {
            if self.up {
                Ok(json!({ "id": "file:01J" }))
            } else {
                Err(antumbra_core::AntumbraError::other("copal is down"))
            }
        }
        fn put_bytes(
            &self,
            _url: &str,
            _credential: &CopalCredential,
            _content_type: &str,
            _body: &[u8],
        ) -> antumbra_core::Result<Value> {
            Ok(json!({ "digest": "sha256:abc" }))
        }
    }

    fn archive(up: bool) -> CopalArchive {
        CopalArchive::with_transport(
            "127.0.0.1:9010",
            CopalTenancy::PerWorkspace,
            Arc::new(FakeCopal { up }),
        )
    }

    fn doc(title: &str, content: &str, provenance: Option<GitProvenance>) -> Document {
        Document {
            title: title.into(),
            source: Some("$ deno task routes".into()),
            content: content.into(),
            provenance,
        }
    }

    /// The chunks land under the tenant with the anchor in their source, and a
    /// second ingest of the same title replaces them, even when the document
    /// shrank from several chunks to one.
    #[tokio::test]
    async fn reingest_replaces_the_title_in_place() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("ws:t");
        let embedder = FixedEmbedder::new(EMBED_DIM);
        let anchor = GitProvenance::new("github.com/o/r", "b697da7").on_branch("main");
        let long = "GET /orders\n".repeat(400); // several chunks' worth
        let first = ingest_text(
            &store,
            &embedder,
            &tenant,
            None,
            &doc("routes", &long, Some(anchor.clone())),
        )
        .await
        .unwrap();
        assert!(first.chunks > 1, "{} chunks", first.chunks);
        assert!(first.archived.is_none());
        let again = ingest_text(
            &store,
            &embedder,
            &tenant,
            None,
            &doc("routes", "GET /health\nDELETE /orders/{id}\n", Some(anchor)),
        )
        .await
        .unwrap();
        assert_eq!(again.chunks, 1);
        assert_eq!(
            document::list_titles(&store, &tenant).await.unwrap(),
            vec!["routes".to_string()]
        );
        let hits = document::recall(
            &store,
            &tenant,
            &embedder.embed("orders").await.unwrap(),
            50,
        )
        .await
        .unwrap();
        assert_eq!(hits.len(), 1, "the old generation's tail is gone");
        let source = hits[0].source.as_deref().unwrap();
        assert!(
            source.starts_with("$ deno task routes @ git:github.com/o/r@b697da7#main"),
            "{source}"
        );
        assert!(
            hits[0].content.contains("DELETE /orders"),
            "the newer content won"
        );
    }

    /// With an archive configured the original lands in copal first and every
    /// chunk names it.
    #[tokio::test]
    async fn archives_to_copal_and_stamps_every_chunk() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("ws:t");
        let embedder = FixedEmbedder::new(EMBED_DIM);
        let out = ingest_text(
            &store,
            &embedder,
            &tenant,
            Some(&archive(true)),
            &doc("routes", "GET /health\n", None),
        )
        .await
        .unwrap();
        let archived = out.archived.expect("archived");
        assert_eq!(
            (archived.file_id.as_str(), archived.digest.as_str()),
            ("file:01J", "sha256:abc")
        );
        let hits = document::recall(&store, &tenant, &embedder.embed("health").await.unwrap(), 5)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].copal_file.as_deref(), Some("file:01J"));
        assert_eq!(hits[0].copal_digest.as_deref(), Some("sha256:abc"));
        assert_eq!(hits[0].source.as_deref(), Some("$ deno task routes"));
    }

    /// A configured archive that is unreachable fails the ingest and stores
    /// nothing: no chunks whose original was silently dropped.
    #[tokio::test]
    async fn fails_closed_when_copal_is_down() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("ws:t");
        let embedder = FixedEmbedder::new(EMBED_DIM);
        let err = ingest_text(
            &store,
            &embedder,
            &tenant,
            Some(&archive(false)),
            &doc("routes", "GET /health\n", None),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("nothing was ingested"), "{err}");
        assert!(document::list_titles(&store, &tenant)
            .await
            .unwrap()
            .is_empty());
    }

    #[test]
    fn source_joins_the_anchor_and_ids_are_stable() {
        let anchor = GitProvenance::new("github.com/o/r", "b697da7").at_path("README.md");
        let both = Document {
            title: "t".into(),
            source: Some("https://x".into()),
            content: String::new(),
            provenance: Some(anchor.clone()),
        };
        assert_eq!(
            source_of(&both).as_deref(),
            Some("https://x @ git:github.com/o/r@b697da7:README.md")
        );
        let anchor_only = Document {
            source: None,
            ..both.clone()
        };
        assert_eq!(
            source_of(&anchor_only).as_deref(),
            Some("git:github.com/o/r@b697da7:README.md")
        );
        let neither = Document {
            source: None,
            provenance: None,
            ..both
        };
        assert_eq!(source_of(&neither), None);
        let t = TenantId::new("ws:t");
        assert_eq!(chunk_id(&t, "routes", 0), chunk_id(&t, "routes", 0));
        assert_ne!(chunk_id(&t, "routes", 0), chunk_id(&t, "routes", 1));
        assert_ne!(
            chunk_id(&t, "routes", 0),
            chunk_id(&TenantId::new("ws:u"), "routes", 0)
        );
    }
}
