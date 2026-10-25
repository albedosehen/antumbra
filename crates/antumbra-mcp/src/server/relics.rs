//! Relics: the pages a user keeps, published by an agent and found and viewed
//! in the dashboard (`/relics`, `/relic/<slug>`).
//!
//! Its own router, joined to the others in `engine.rs`. A relic's bytes live in
//! copal, one file per version; its row here says who owns it, which
//! compartment decides who else may see it, and whether it is public; its text
//! is ingested as the document `relic:<slug>`, in the same compartment, so
//! recall finds it by what it says. Nothing a person types to reach a relic
//! expires: a public relic has a permanent address on copal's render origin,
//! and a private one is shown through a render link minted for each view.

use super::*;

use antumbra_core::relic::{self as core_relic, Visibility};
use antumbra_core::{Relic, RelicVersion};
use antumbra_store::repo::relic as store_relic;

/// The most HTML one relic version may hold.
const MAX_HTML_BYTES: usize = 2 * 1024 * 1024;
/// The most text of a relic indexed for search; the rest is still kept and
/// shown, only not searched.
const MAX_INDEXED_CHARS: usize = 64_000;
/// How long a private relic's render link lasts: one view's worth.
const VIEW_TTL_SECS: u32 = 300;
/// The most relics one listing returns.
const MAX_LISTED: usize = 100;

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct PublishRelicParams {
    /// What the gallery calls it.
    pub(super) title: String,
    /// The page: a complete HTML document. Inline its images, styles and
    /// scripts; it is shown in a sandbox with no network access of its own.
    pub(super) html: String,
    /// The address, `/relic/<slug>`: lowercase letters, digits and hyphens.
    /// Omit it to derive one from the title. Naming a relic you already have
    /// publishes a new version of it at the same address.
    #[serde(default)]
    pub(super) slug: Option<String>,
    /// The compartment that decides who besides you may see it while private.
    /// Omit it for your session's own space. A relic stays in the compartment
    /// it was first published in.
    #[serde(default)]
    pub(super) compartment: Option<String>,
    /// `private` (the default for a new relic) or `public` (anyone who can
    /// reach the server, no sign-in). Omit it to keep a relic's current one.
    #[serde(default)]
    pub(super) visibility: Option<String>,
    /// Where it came from: the task, the repository, the conversation.
    #[serde(default)]
    pub(super) source: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct PublishedRelicOut {
    pub(super) slug: String,
    pub(super) title: String,
    /// The version this call published: 1 for a new relic.
    pub(super) version: u32,
    pub(super) visibility: String,
    /// The relic's address on this server's dashboard.
    pub(super) path: String,
    /// The page's permanent address on copal's render origin, when public.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) public_url: Option<String>,
    /// Anything that did not go as planned without failing the publish.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) note: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ListRelicsParams {
    /// Find relics by what they say (and by title). Omit to list them all,
    /// most recently updated first.
    #[serde(default)]
    pub(super) query: Option<String>,
    /// `all` (default), `mine` (relics you published), or `shared` (others'
    /// relics in compartments granted to you).
    #[serde(default)]
    pub(super) scope: Option<String>,
    /// At most this many (default and ceiling 100).
    #[serde(default)]
    pub(super) limit: Option<u32>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RelicCard {
    pub(super) slug: String,
    pub(super) title: String,
    /// Whether you published it.
    pub(super) yours: bool,
    pub(super) owner: String,
    pub(super) visibility: String,
    pub(super) version: u32,
    pub(super) updated_at: String,
    pub(super) path: String,
    /// A public relic's permanent address, which a gallery can preview without
    /// asking for a link. Absent for a private relic: ask `get_relic`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) preview_url: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RelicsOut {
    pub(super) relics: Vec<RelicCard>,
    /// Why the list is empty, when it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) note: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RelicSlugParams {
    /// The relic's address, as in `/relic/<slug>`.
    pub(super) slug: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RelicVersionView {
    pub(super) number: u32,
    pub(super) created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) author_host: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RelicOut {
    pub(super) slug: String,
    pub(super) title: String,
    pub(super) yours: bool,
    pub(super) owner: String,
    pub(super) compartment: String,
    pub(super) visibility: String,
    pub(super) version: u32,
    /// Oldest first.
    pub(super) versions: Vec<RelicVersionView>,
    pub(super) created_at: String,
    pub(super) updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) source: Option<String>,
    pub(super) path: String,
    /// Where the current version is displayed: its permanent address when
    /// public, else a render link made for this view.
    pub(super) view_url: String,
    /// When `view_url` stops working; absent for a public relic's address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) expires_at: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RelicVisibilityParams {
    pub(super) slug: String,
    /// `private` or `public`.
    pub(super) visibility: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RelicVisibilityOut {
    pub(super) slug: String,
    pub(super) visibility: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) public_url: Option<String>,
}

fn invalid(message: impl Into<String>) -> ErrorData {
    ErrorData::invalid_params(message.into(), None)
}

fn path_of(slug: &str) -> String {
    format!("/relic/{slug}")
}

/// What the first `n` characters of `text` are, cut on a character boundary.
fn first_chars(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

impl McpServer {
    /// The archive relics are kept in, refused unless it can display them.
    fn relic_archive(&self) -> Result<&antumbra_copal::CopalArchive, ErrorData> {
        match self.copal.as_deref() {
            Some(archive) if archive.render_origin().is_some() => Ok(archive),
            _ => Err(invalid(
                "relics need a copal archive that can display them: this server has no \
                 --copal-addr or no --copal-render-origin",
            )),
        }
    }

    /// The relic `slug` as this session may see it, or the error every
    /// missing and unseen relic gets alike.
    async fn seen_relic(&self, slug: &str) -> Result<Relic, ErrorData> {
        let slug = slug.trim();
        let found = if core_relic::is_valid_slug(slug) {
            store_relic::get(&self.store, &self.tenant, slug)
                .await
                .map_err(err)?
        } else {
            None
        };
        found.ok_or_else(|| invalid(format!("no relic `{slug}` that you can see")))
    }

    /// Where the current version of `relic` is displayed, and until when.
    async fn view_of(&self, relic: &Relic) -> Result<(String, Option<String>), ErrorData> {
        let archive = self.relic_archive()?;
        let current = relic
            .current()
            .ok_or_else(|| err("the relic has no versions"))?;
        if relic.visibility == Visibility::Public {
            if let Some(url) = archive.public_url(&current.copal_file) {
                return Ok((url, None));
            }
        }
        let link = archive
            .render_url(relic.tenant.as_str(), &current.copal_file, VIEW_TTL_SECS)
            .await
            .map_err(err)?;
        Ok((link.url, link.expires_at))
    }

    fn card(&self, relic: &Relic) -> RelicCard {
        let preview_url = (relic.visibility == Visibility::Public)
            .then(|| {
                relic.current().and_then(|v| {
                    self.copal
                        .as_deref()
                        .and_then(|a| a.public_url(&v.copal_file))
                })
            })
            .flatten();
        RelicCard {
            slug: relic.slug.clone(),
            title: relic.title.clone(),
            yours: relic.owner == self.user,
            owner: relic.owner.as_str().to_string(),
            visibility: relic.visibility.as_str().to_string(),
            version: relic.current().map_or(0, |v| v.number),
            updated_at: relic.updated_at.to_rfc3339(),
            path: path_of(&relic.slug),
            preview_url,
        }
    }
}

#[tool_router(router = relic_router, vis = "pub(super)")]
impl McpServer {
    /// Keep an HTML page as a relic the user can find and view later.
    #[tool(
        description = "Publish an HTML page as a relic: kept in copal, listed in the dashboard's relics gallery, viewed at /relic/<slug>, and found by recall_documents. Private by default (you, and whoever its compartment is shared with); public means anyone who can reach the server. Naming a slug you already have publishes a new version at the same address. Returns the slug, version, and address."
    )]
    pub(super) async fn publish_relic(
        &self,
        Parameters(p): Parameters<PublishRelicParams>,
    ) -> Result<Json<PublishedRelicOut>, ErrorData> {
        let archive = self.relic_archive()?;
        let title = p.title.trim().to_string();
        if title.is_empty() {
            return Err(invalid("a relic needs a title"));
        }
        if p.html.trim().is_empty() {
            return Err(invalid("a relic needs a page: html is empty"));
        }
        if p.html.len() > MAX_HTML_BYTES {
            return Err(invalid(format!(
                "the page is {} bytes; a relic holds at most {MAX_HTML_BYTES}",
                p.html.len()
            )));
        }
        let asked: Option<Visibility> = p
            .visibility
            .as_deref()
            .map(str::parse)
            .transpose()
            .map_err(|e: antumbra_core::AntumbraError| invalid(e.to_string()))?;

        // The address: one named, or one drawn from the title. A named slug
        // that belongs to someone else is refused; a drawn one steps aside.
        let named = p.slug.as_deref().map(str::trim).filter(|s| !s.is_empty());
        let (slug, existing) = match named {
            Some(slug) => {
                if !core_relic::is_valid_slug(slug) {
                    return Err(invalid(format!(
                        "`{slug}` cannot be a relic's address: use lowercase letters, digits \
                         and hyphens, at most {} characters",
                        core_relic::MAX_SLUG_LEN
                    )));
                }
                let existing = store_relic::get(&self.store, &self.tenant, slug)
                    .await
                    .map_err(err)?;
                if existing.as_ref().is_some_and(|r| r.owner != self.user) {
                    return Err(invalid(format!(
                        "the relic `{slug}` is someone else's; choose another slug"
                    )));
                }
                (slug.to_string(), existing)
            }
            None => {
                let base = core_relic::slugify(&title);
                let existing = store_relic::get(&self.store, &self.tenant, &base)
                    .await
                    .map_err(err)?;
                match existing {
                    Some(r) if r.owner == self.user => (base, Some(r)),
                    Some(_) => (format!("{base}-{:06x}", random_suffix()), None),
                    None => (base, None),
                }
            }
        };

        // A relic stays where it was first placed; a new one goes where it is
        // asked to, or into this session's own space. Asked before anything is
        // uploaded, so a refusal leaves nothing behind in copal.
        let compartment = match &existing {
            Some(r) => {
                if let Some(asked) = p.compartment.as_deref().filter(|c| !c.trim().is_empty()) {
                    if asked != r.compartment.as_str() {
                        return Err(invalid(format!(
                            "the relic `{slug}` lives in compartment '{}'; a relic stays in \
                             the compartment it was first published in",
                            r.compartment.as_str()
                        )));
                    }
                }
                r.compartment.clone()
            }
            None => p
                .compartment
                .as_deref()
                .map(str::trim)
                .filter(|c| !c.is_empty())
                .map(CompartmentId::new)
                .unwrap_or_else(|| self.default_compartment.clone()),
        };
        let writable = compartment::can_write(&self.store, &self.tenant, &self.user, &compartment)
            .await
            .map_err(err)?;
        if !writable {
            return Err(invalid(format!(
                "cannot publish into compartment '{}': it is not one you can write to",
                compartment.as_str()
            )));
        }

        let visibility = asked
            .or(existing.as_ref().map(|r| r.visibility))
            .unwrap_or(Visibility::Private);
        let number = existing.as_ref().map_or(1, Relic::next_version);
        let stored = archive
            .put_relic(
                self.tenant.as_str(),
                &slug,
                number,
                &title,
                &p.html,
                visibility == Visibility::Public,
            )
            .await
            .map_err(|e| {
                err(format!(
                    "copal refused the page, nothing was published: {e}"
                ))
            })?;

        let now = Utc::now();
        let version = RelicVersion {
            number,
            copal_file: stored.file_id.clone(),
            copal_digest: stored.digest.clone(),
            created_at: now,
            author_host: Some(self.device()),
        };
        let previous = existing.as_ref().and_then(|r| r.current().cloned());
        let relic = match existing {
            Some(mut r) => {
                r.title = title.clone();
                r.visibility = visibility;
                r.versions.push(version);
                if p.source.is_some() {
                    r.source = p.source.clone();
                }
                r.updated_at = now;
                r
            }
            None => Relic {
                slug: slug.clone(),
                tenant: self.tenant.clone(),
                owner: self.user.clone(),
                compartment: compartment.clone(),
                title: title.clone(),
                visibility,
                versions: vec![version],
                source: p.source.clone(),
                created_at: now,
                updated_at: now,
            },
        };
        if !store_relic::put(&self.store, &relic).await.map_err(err)? {
            return Err(invalid(format!(
                "the relic `{slug}` did not land: the slug is held by someone else; choose \
                 another"
            )));
        }

        let mut notes = Vec::new();
        // Only the current version is ever public: the one before it closes.
        if let Some(previous) = previous {
            if let Err(e) = archive
                .set_access(self.tenant.as_str(), &previous.copal_file, false)
                .await
            {
                notes.push(format!(
                    "version {} may still be public at its old address: {e}",
                    previous.number
                ));
            }
        }
        let text = core_relic::html_to_text(&p.html);
        let doc = antumbra_ingest::Document {
            title: core_relic::document_title(&slug),
            source: p.source.clone(),
            content: format!("{title}\n\n{}", first_chars(&text, MAX_INDEXED_CHARS)),
            provenance: None,
            compartment: Some(compartment),
        };
        if let Err(e) = antumbra_ingest::ingest_recorded(
            &self.store,
            self.embedder.as_ref(),
            &self.tenant,
            &doc,
            Some(stored.clone()),
        )
        .await
        {
            notes.push(format!(
                "published, but its text was not indexed, so search will not find it yet: {e:#}"
            ));
        }

        Ok(Json(PublishedRelicOut {
            path: path_of(&slug),
            public_url: (visibility == Visibility::Public)
                .then(|| archive.public_url(&stored.file_id))
                .flatten(),
            slug,
            title,
            version: number,
            visibility: visibility.as_str().to_string(),
            note: (!notes.is_empty()).then(|| notes.join(" ")),
        }))
    }

    /// The relics you can see, newest first or by a search.
    #[tool(
        description = "List the relics you can see: your own and those in compartments shared with you, most recently updated first, or the ones matching a query by what they say. Each card has the slug, title, owner, visibility, version, and, for a public relic, a preview address."
    )]
    pub(super) async fn list_relics(
        &self,
        Parameters(p): Parameters<ListRelicsParams>,
    ) -> Result<Json<RelicsOut>, ErrorData> {
        let scope = p.scope.as_deref().map(str::trim).unwrap_or("all");
        if !matches!(scope, "all" | "mine" | "shared") {
            return Err(invalid(format!(
                "unknown scope `{scope}` (use all | mine | shared)"
            )));
        }
        let limit = p
            .limit
            .map_or(MAX_LISTED, |l| (l as usize).clamp(1, MAX_LISTED));
        let mut relics: Vec<Relic> = store_relic::list(&self.store, &self.tenant)
            .await
            .map_err(err)?
            .into_iter()
            .filter(|r| match scope {
                "mine" => r.owner == self.user,
                "shared" => r.owner != self.user,
                _ => true,
            })
            .collect();
        let total = relics.len();

        let query = p.query.as_deref().map(str::trim).filter(|q| !q.is_empty());
        if let Some(query) = query {
            // By meaning first (the indexed text, through recall), then any
            // other relic whose title holds the words.
            let vector = self.embedder.embed(query).await.map_err(err)?;
            let hits = document::recall_hybrid(
                &self.store,
                &self.tenant,
                query,
                &vector,
                (limit * 5).max(50),
            )
            .await
            .map_err(err)?;
            let mut ranked: Vec<String> = Vec::new();
            for hit in hits {
                if let Some(slug) = core_relic::slug_of_document(&hit.title) {
                    if !ranked.iter().any(|s| s == slug) {
                        ranked.push(slug.to_string());
                    }
                }
            }
            let needle = query.to_lowercase();
            for r in &relics {
                if r.title.to_lowercase().contains(&needle) && !ranked.contains(&r.slug) {
                    ranked.push(r.slug.clone());
                }
            }
            relics = ranked
                .iter()
                .filter_map(|slug| relics.iter().find(|r| &r.slug == slug).cloned())
                .collect();
        }
        relics.truncate(limit);
        let note = relics.is_empty().then(|| {
            if total == 0 {
                "no relics yet: publish_relic keeps an HTML page as one".to_string()
            } else {
                format!("none of your {total} relics match")
            }
        });
        Ok(Json(RelicsOut {
            relics: relics.iter().map(|r| self.card(r)).collect(),
            note,
        }))
    }

    /// One relic, with an address that displays it now.
    #[tool(
        description = "Get one relic by slug: its title, owner, compartment, visibility, versions, and view_url, where the current version is displayed (its permanent address when public, otherwise a render link made for this view that lasts a few minutes)."
    )]
    pub(super) async fn get_relic(
        &self,
        Parameters(p): Parameters<RelicSlugParams>,
    ) -> Result<Json<RelicOut>, ErrorData> {
        let relic = self.seen_relic(&p.slug).await?;
        let (view_url, expires_at) = self.view_of(&relic).await?;
        Ok(Json(RelicOut {
            yours: relic.owner == self.user,
            owner: relic.owner.as_str().to_string(),
            compartment: relic.compartment.as_str().to_string(),
            visibility: relic.visibility.as_str().to_string(),
            version: relic.current().map_or(0, |v| v.number),
            versions: relic
                .versions
                .iter()
                .map(|v| RelicVersionView {
                    number: v.number,
                    created_at: v.created_at.to_rfc3339(),
                    author_host: v.author_host.clone(),
                })
                .collect(),
            created_at: relic.created_at.to_rfc3339(),
            updated_at: relic.updated_at.to_rfc3339(),
            source: relic.source.clone(),
            path: path_of(&relic.slug),
            view_url,
            expires_at,
            title: relic.title,
            slug: relic.slug,
        }))
    }

    /// Make one of your relics public or private.
    #[tool(
        description = "Make one of your relics public (anyone who can reach the server, no sign-in) or private (you and its compartment's audience). Only the relic's owner can change it."
    )]
    pub(super) async fn set_relic_visibility(
        &self,
        Parameters(p): Parameters<RelicVisibilityParams>,
    ) -> Result<Json<RelicVisibilityOut>, ErrorData> {
        let archive = self.relic_archive()?;
        let visibility: Visibility = p
            .visibility
            .parse()
            .map_err(|e: antumbra_core::AntumbraError| invalid(e.to_string()))?;
        let mut relic = self.seen_relic(&p.slug).await?;
        if relic.owner != self.user {
            return Err(invalid(format!(
                "the relic `{}` is someone else's; only its owner can change who sees it",
                relic.slug
            )));
        }
        let current = relic
            .current()
            .cloned()
            .ok_or_else(|| err("the relic has no versions"))?;
        let public = visibility == Visibility::Public;
        // Copal first: it is what actually serves the page. If the row then
        // refuses, the file goes back to how it was.
        archive
            .set_access(self.tenant.as_str(), &current.copal_file, public)
            .await
            .map_err(err)?;
        let was = relic.visibility;
        relic.visibility = visibility;
        relic.updated_at = Utc::now();
        if !store_relic::put(&self.store, &relic).await.map_err(err)? {
            let _ = archive
                .set_access(
                    self.tenant.as_str(),
                    &current.copal_file,
                    was == Visibility::Public,
                )
                .await;
            return Err(err("the change did not land; nothing changed"));
        }
        Ok(Json(RelicVisibilityOut {
            public_url: public
                .then(|| archive.public_url(&current.copal_file))
                .flatten(),
            slug: relic.slug,
            visibility: visibility.as_str().to_string(),
        }))
    }
}

/// 24 bits for a drawn slug's suffix when its base is taken.
fn random_suffix() -> u32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or_default();
    (nanos ^ std::process::id().rotate_left(16)) & 0x00ff_ffff
}
