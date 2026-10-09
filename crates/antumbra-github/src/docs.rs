//! Which files in a repository are knowledge documents worth ingesting: the
//! prose a developer reads (READMEs, docs, ADRs, changelogs) and the API
//! descriptions a service publishes (OpenAPI, AsyncAPI). Source code is not
//! ingested: what a service exposes comes from running its own lister, never
//! from reading its source.

/// Files larger than this are skipped: a generated changelog or a vendored
/// spec that big is noise in recall.
pub const MAX_DOCUMENT_BYTES: usize = 512 * 1024;

const SKIPPED_DIRS: &[&str] = &[
    "node_modules",
    "vendor",
    "target",
    "dist",
    "build",
    "third_party",
    ".git",
];
const PROSE_EXTENSIONS: &[&str] = &["md", "mdx", "markdown", "rst", "adoc", "asciidoc", "txt"];
const SPEC_STEMS: &[&str] = &["openapi", "swagger", "asyncapi"];
const SPEC_EXTENSIONS: &[&str] = &["yaml", "yml", "json"];
const BARE_NAMES: &[&str] = &[
    "readme",
    "contributing",
    "changelog",
    "architecture",
    "codeowners",
];
// Both spellings: these match other people's file names, LICENCE as well as LICENSE.
const LEGAL_PREFIXES: &[&str] = &["licence", "license", "notice", "copying", "patents"];

/// Whether `path` (repository-relative, `/`-separated) is a knowledge
/// document.
pub fn is_knowledge_document(path: &str) -> bool {
    let lower = path.trim_matches('/').to_ascii_lowercase();
    if lower.is_empty()
        || lower
            .split('/')
            .any(|segment| SKIPPED_DIRS.contains(&segment))
    {
        return false;
    }
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    if LEGAL_PREFIXES.iter().any(|p| name.starts_with(p)) {
        return false;
    }
    let (stem, ext) = match name.rsplit_once('.') {
        Some((stem, ext)) => (stem, ext),
        None => (name, ""),
    };
    if PROSE_EXTENSIONS.contains(&ext) {
        return true;
    }
    if SPEC_EXTENSIONS.contains(&ext) && SPEC_STEMS.iter().any(|s| stem.starts_with(s)) {
        return true;
    }
    ext.is_empty() && BARE_NAMES.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_specs_and_well_known_files_are_documents() {
        for yes in [
            "README.md",
            "docs/adr/0018-provenance.md",
            "CHANGELOG",
            "Docs/Guide.MDX",
            "api/openapi.yaml",
            "spec/swagger-v2.json",
            "events/asyncapi.yml",
            "notes/plan.txt",
            "/README.rst",
        ] {
            assert!(is_knowledge_document(yes), "{yes}");
        }
    }

    #[test]
    fn code_vendored_trees_legal_files_and_data_are_not() {
        for no in [
            "src/main.rs",
            "package.json",
            "node_modules/left-pad/README.md",
            "vendor/spec/openapi.yaml",
            "target/doc/index.md",
            "LICENSE",
            "LICENSE.md",
            "NOTICE.txt",
            "config/settings.yaml",
            "data/orders.json",
            "",
            "readme.py",
        ] {
            assert!(!is_knowledge_document(no), "{no}");
        }
    }
}
