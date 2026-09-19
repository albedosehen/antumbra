//! Knowledge-document ingest: the one path every door into the store takes,
//! so the MCP `ingest_document` tool, the CLI `ingest`, and the GitHub
//! integration write the same shape and keep the same promises.
//!
//! - **Document of record first.** With a copal archive configured, the
//!   original content is uploaded before any chunk is stored, and a failure
//!   fails the whole ingest: a configured archive that silently dropped
//!   originals would be worse than none.
//! - **Replace in place.** A title names one document within a workspace and a
//!   compartment. Re-ingesting it replaces its chunks rather than accumulating
//!   copies, so the corpus never carries two generations of the same document.
//! - **Private when placed.** A document ingested into a compartment is readable
//!   by that compartment's owner and grantees only, and its identity (chunk ids,
//!   the archived original) is its own: two members may each keep a private
//!   document under one title without ever touching the other's.
//! - **Anchored.** A git anchor, when known, is folded into every chunk's
//!   `source`, so a recalled chunk names the commit it describes.

use chrono::Utc;

use antumbra_copal::{ArchivedDocument, CopalArchive};
use antumbra_core::document::{DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP};
use antumbra_core::ports::Embedder;
use antumbra_core::{
    chunk_text, CompartmentId, DocumentChunk, DocumentChunkId, GitProvenance, TenantId,
};
use antumbra_store::repo::document;
use antumbra_store::Store;

/// A knowledge document to ingest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    /// The document's identity within its workspace and compartment: one title,
    /// one document.
    pub title: String,
    /// Where it came from (a path, a URL, the command that printed it).
    pub source: Option<String>,
    pub content: String,
    /// The git anchor folded into every chunk's `source`, when known.
    pub provenance: Option<GitProvenance>,
    /// The compartment to place it in, which decides who may read it. `None` =
    /// the tenant's shared pool, readable by every member.
    pub compartment: Option<CompartmentId>,
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
                    &archive_title(doc),
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
            id: DocumentChunkId::new(chunk_id(
                tenant,
                doc.compartment.as_ref(),
                &doc.title,
                ordinal,
            )),
            tenant: tenant.clone(),
            title: doc.title.clone(),
            source: source.clone(),
            ordinal: ordinal as u32,
            content,
            embedding: Some(embedding),
            created_at: now,
            copal_file: None,
            copal_digest: None,
            compartment: doc.compartment.clone(),
        };
        if let Some(d) = &archived {
            chunk = chunk.with_copal(d.file_id.clone(), d.digest.clone());
        }
        chunks.push(chunk);
    }
    // Ids repeat per (tenant, compartment, title, ordinal), so the common case
    // overwrites in place; the delete is what drops the tail when the document
    // shrank.
    document::delete_title(store, tenant, &doc.title, doc.compartment.as_ref()).await?;
    document::insert_chunks(store, &chunks).await?;
    // Under a record session the engine refuses a write into a compartment the
    // session may not write to by persisting nothing, without an error. Look
    // before reporting chunks that are not there.
    if !chunks.is_empty()
        && !document::title_exists(store, tenant, &doc.title, doc.compartment.as_ref()).await?
    {
        anyhow::bail!(
            "the document did not land: '{}' is not a compartment this session can write to",
            doc.compartment
                .as_ref()
                .map(|c| c.as_str())
                .unwrap_or("(the shared pool)")
        );
    }
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

/// The title the archive files the original under. The archive's identity is
/// (workspace, title), so a document in a compartment carries the compartment in
/// it: otherwise two members' private documents of one title would revision a
/// single archived file, and "the original" of one would be the other's text. A
/// shared-pool document keeps its bare title, so files archived before documents
/// had compartments keep revisioning where they are.
pub fn archive_title(doc: &Document) -> String {
    match &doc.compartment {
        Some(compartment) => format!("{}/{}", compartment.as_str(), doc.title),
        None => doc.title.clone(),
    }
}

/// A stable chunk id: the same (tenant, compartment, title, ordinal) always names
/// the same record, which is what makes re-ingest an in-place replacement. The
/// compartment is hashed only when there is one, so a shared-pool document keeps
/// the ids it had before documents had compartments.
pub fn chunk_id(
    tenant: &TenantId,
    compartment: Option<&CompartmentId>,
    title: &str,
    ordinal: usize,
) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tenant.as_str().hash(&mut hasher);
    if let Some(compartment) = compartment {
        compartment.as_str().hash(&mut hasher);
    }
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
        /// Every create body, so a test can see what identity was archived.
        created: std::sync::Mutex<Vec<Value>>,
    }

    impl FakeCopal {
        /// A poisoned lock means another test thread already panicked; the
        /// recorded bodies are still the truth, so read through it.
        fn record(&self, body: &Value) {
            match self.created.lock() {
                Ok(mut created) => created.push(body.clone()),
                Err(poisoned) => poisoned.into_inner().push(body.clone()),
            }
        }

        /// The titles of the documents it was asked to archive, in order.
        fn archived_titles(&self) -> Vec<Option<String>> {
            let created = match self.created.lock() {
                Ok(created) => created,
                Err(poisoned) => poisoned.into_inner(),
            };
            created
                .iter()
                .map(|body| body["metadata"]["title"].as_str().map(str::to_string))
                .collect()
        }
    }

    impl CopalTransport for FakeCopal {
        fn post_json(
            &self,
            _url: &str,
            _credential: &CopalCredential,
            body: &Value,
        ) -> antumbra_core::Result<Value> {
            self.record(body);
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
        recording_archive(up).0
    }

    fn recording_archive(up: bool) -> (CopalArchive, Arc<FakeCopal>) {
        let copal = Arc::new(FakeCopal {
            up,
            created: std::sync::Mutex::new(Vec::new()),
        });
        let archive = CopalArchive::with_transport(
            "127.0.0.1:9010",
            CopalTenancy::PerWorkspace,
            copal.clone(),
        );
        (archive, copal)
    }

    fn doc(title: &str, content: &str, provenance: Option<GitProvenance>) -> Document {
        Document {
            title: title.into(),
            source: Some("$ deno task routes".into()),
            content: content.into(),
            provenance,
            compartment: None,
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
            compartment: None,
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
        assert_eq!(
            chunk_id(&t, None, "routes", 0),
            chunk_id(&t, None, "routes", 0)
        );
        assert_ne!(
            chunk_id(&t, None, "routes", 0),
            chunk_id(&t, None, "routes", 1)
        );
        assert_ne!(
            chunk_id(&t, None, "routes", 0),
            chunk_id(&TenantId::new("ws:u"), None, "routes", 0)
        );
    }

    /// Every document ingested before documents had compartments is a shared-pool
    /// document, and its next re-ingest must land on the same records and the
    /// same archived file. If either identity moved, that re-ingest would leave
    /// the old generation behind as a duplicate.
    #[test]
    fn a_shared_pool_document_keeps_the_identity_it_always_had() {
        use std::hash::{Hash, Hasher};
        let t = TenantId::new("ws:t");
        // The id formula as it was before the compartment joined it.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        t.as_str().hash(&mut hasher);
        "routes".hash(&mut hasher);
        3usize.hash(&mut hasher);
        let before = format!("docchunk:{:x}", hasher.finish());
        assert_eq!(chunk_id(&t, None, "routes", 3), before);
        assert_eq!(archive_title(&doc("routes", "", None)), "routes");
    }

    /// Two members each keep a private document under one title. Neither ingest
    /// reaches the other's chunks, and each has its own archived original.
    #[tokio::test]
    async fn private_documents_of_one_title_never_touch_each_other() -> anyhow::Result<()> {
        let store = Store::connect_memory(EMBED_DIM).await?;
        let tenant = TenantId::new("ws:t");
        let embedder = FixedEmbedder::new(EMBED_DIM);
        let (archive, copal) = recording_archive(true);
        let placed = |compartment: &str, content: &str| Document {
            compartment: Some(CompartmentId::new(compartment)),
            ..doc("notes", content, None)
        };
        let lily = CompartmentId::new("comp:lily");
        let oslo = CompartmentId::new("comp:oslo");
        assert_ne!(
            chunk_id(&tenant, Some(&lily), "notes", 0),
            chunk_id(&tenant, Some(&oslo), "notes", 0)
        );
        assert_ne!(
            chunk_id(&tenant, Some(&lily), "notes", 0),
            chunk_id(&tenant, None, "notes", 0),
            "a private document is not the shared pool's document of that title"
        );

        for (compartment, content) in [
            ("comp:lily", "lily's salary review"),
            ("comp:oslo", "oslo's launch plan"),
        ] {
            ingest_text(
                &store,
                &embedder,
                &tenant,
                Some(&archive),
                &placed(compartment, content),
            )
            .await?;
        }
        // Lily revises hers. Oslo's is still there, untouched.
        ingest_text(
            &store,
            &embedder,
            &tenant,
            Some(&archive),
            &placed("comp:lily", "lily's revised salary review"),
        )
        .await?;

        let hits = document::recall(&store, &tenant, &embedder.embed("plan").await?, 10).await?;
        let mut seen: Vec<(Option<CompartmentId>, String)> = hits
            .iter()
            .map(|c| (c.compartment.clone(), c.content.clone()))
            .collect();
        seen.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(
            seen,
            vec![
                (Some(lily), "lily's revised salary review".to_string()),
                (Some(oslo), "oslo's launch plan".to_string()),
            ]
        );

        // Two archived files, not one revisioned three times: the archive's
        // identity is the title it is given, and that carries the compartment.
        let expected: Vec<Option<String>> =
            ["comp:lily/notes", "comp:oslo/notes", "comp:lily/notes"]
                .iter()
                .map(|title| Some(title.to_string()))
                .collect();
        assert_eq!(copal.archived_titles(), expected);
        Ok(())
    }
}
