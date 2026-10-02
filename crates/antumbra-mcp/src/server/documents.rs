//! The document list: which knowledge documents a workspace holds, without
//! reading them.
//!
//! Its own router, joined to the others in `engine.rs`, because `server.rs`
//! (where ingest and recall live) is past the size rule.

use super::*;

#[tool_router(router = document_router, vis = "pub(super)")]
impl McpServer {
    /// Every knowledge document you can see, by title.
    #[tool(
        description = "List your ingested knowledge documents by title, with how many chunks each was cut into and whether its file of record is archived. Lists rather than searches; use recall_documents to search their text."
    )]
    pub(super) async fn list_documents(&self) -> Result<Json<DocumentsOut>, ErrorData> {
        let documents = document::summaries(&self.store, &self.tenant)
            .await
            .map_err(err)?;
        Ok(Json(DocumentsOut {
            documents: documents
                .into_iter()
                .map(|d| DocumentView {
                    title: d.title,
                    chunks: d.chunks,
                    archived: d.archived,
                })
                .collect(),
        }))
    }
}
