//! `antumbra claude dependencies`: which of your repositories depend on which,
//! read from their manifests and recorded in the workspace's dependency graph
//! as `declared` edges.
//!
//! A repository depends on another when one of its manifests names a package
//! the other publishes, in the same ecosystem. Only the repositories read
//! together are matched, so a dependency on a package none of them publishes
//! is not an edge here. Recording an edge again reinforces it, which is what
//! keeps it from fading: run this as often as the manifests change.

use antumbra_core::manifest::{self, Manifest};
use antumbra_core::GitProvenance;
use serde_json::{json, Value};

use super::conventions::Call;

/// One working tree as it was read: where (its slug, commit and branch) and
/// the manifests it ships.
pub struct Tree {
    pub anchor: GitProvenance,
    pub manifests: Vec<Manifest>,
}

impl Tree {
    /// Read the manifests among `files` (paths from the tree's root) with
    /// `read`, which returns a file's text. A file that cannot be read or does
    /// not parse is passed over.
    pub fn read(
        anchor: GitProvenance,
        files: &[String],
        read: impl Fn(&str) -> Option<String>,
    ) -> Self {
        let manifests = files
            .iter()
            .filter(|path| manifest::is_manifest(path))
            .filter_map(|path| manifest::read(path, &read(path)?))
            .collect();
        Self { anchor, manifests }
    }
}

/// Each repository once, by slug, the first tree read for it: the same
/// repository checked out twice, or under the project and under `--repos`,
/// is one repository.
fn distinct(trees: &[Tree]) -> Vec<(String, &Tree)> {
    let mut out: Vec<(String, &Tree)> = Vec::new();
    for tree in trees {
        let repo = antumbra_core::provenance::normalize_repo(&tree.anchor.repo);
        if out.iter().all(|(seen, _)| *seen != repo) {
            out.push((repo, tree));
        }
    }
    out
}

/// The `record_dependency` arguments for each edge among `trees`, each with
/// the manifest that declares it, at the commit it was read, as provenance.
pub fn edges(trees: &[Tree]) -> Vec<Value> {
    let distinct = distinct(trees);
    let repos: Vec<(String, Vec<Manifest>)> = distinct
        .iter()
        .map(|(repo, tree)| (repo.clone(), tree.manifests.clone()))
        .collect();
    manifest::declared(&repos)
        .into_iter()
        .map(|(claim, path)| {
            let mut arguments = json!({
                "from": claim.from,
                "to": claim.to,
                "source": claim.source.as_str(),
                "detail": claim.detail,
            });
            let read = distinct.iter().find(|(repo, _)| *repo == claim.from);
            if let Some(anchor) = read.map(|(_, tree)| &tree.anchor) {
                arguments["provenance"] = json!({
                    "repo": claim.from,
                    "commit": anchor.commit,
                    "branch": anchor.branch,
                    "path": path,
                });
            }
            arguments
        })
        .collect()
}

/// Record each edge among `trees` and say what was done, one line each. With
/// `dry_run` nothing is written.
pub fn record(call: Call<'_>, trees: &[Tree], dry_run: bool) -> anyhow::Result<Vec<String>> {
    let mut said: Vec<String> = distinct(trees)
        .iter()
        .map(|(repo, t)| format!("read     {repo} ({} manifest(s))", t.manifests.len()))
        .collect();
    for arguments in edges(trees) {
        let edge = format!(
            "{} -> {} ({})",
            arguments["from"].as_str().unwrap_or_default(),
            arguments["to"].as_str().unwrap_or_default(),
            arguments["detail"].as_str().unwrap_or_default()
        );
        if dry_run {
            said.push(format!("found    {edge}"));
            continue;
        }
        let answer = call("record_dependency", arguments)?;
        said.push(if answer["created"].as_bool().unwrap_or(false) {
            format!("recorded {edge}")
        } else {
            format!(
                "seen     {edge}, reinforced to {}",
                answer["reinforcement"].as_u64().unwrap_or_default()
            )
        });
    }
    Ok(said)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn tree(repo: &str, files: &[(&str, &str)]) -> Tree {
        let anchor = GitProvenance::new(repo, "0123456789abcdef").on_branch("main");
        let paths: Vec<String> = files.iter().map(|(p, _)| p.to_string()).collect();
        Tree::read(anchor, &paths, |path| {
            files
                .iter()
                .find(|(p, _)| *p == path)
                .map(|(_, text)| text.to_string())
        })
    }

    fn fleet() -> Vec<Tree> {
        vec![
            tree(
                "github.com/Acme/Web",
                &[
                    (
                        "package.json",
                        r#"{"name":"@acme/web","dependencies":{"@acme/orders-client":"2"}}"#,
                    ),
                    ("README.md", "not a manifest"),
                    (
                        "node_modules/@acme/orders-client/package.json",
                        r#"{"name":"@acme/orders-client"}"#,
                    ),
                ],
            ),
            tree(
                "github.com/acme/orders",
                &[(
                    "clients/js/package.json",
                    r#"{"name":"@acme/orders-client"}"#,
                )],
            ),
        ]
    }

    /// The edge carries the manifest that declares it, at the commit the
    /// depending repository was read; installed copies are not read as the
    /// repository's own, and a slug is one repository whatever its case.
    #[test]
    fn an_edge_names_its_manifest_at_the_commit_it_was_read() {
        let trees = fleet();
        assert_eq!(trees[0].manifests.len(), 1, "node_modules is not read");
        let mut twice = fleet();
        twice.extend(fleet());
        assert_eq!(
            edges(&twice),
            edges(&trees),
            "a tree read twice counts once"
        );
        assert_eq!(
            edges(&trees),
            [json!({
                "from": "github.com/acme/web",
                "to": "github.com/acme/orders",
                "source": "declared",
                "detail": "package.json names @acme/orders-client",
                "provenance": {
                    "repo": "github.com/acme/web",
                    "commit": "0123456789abcdef",
                    "branch": "main",
                    "path": "package.json",
                },
            })]
        );
    }

    #[test]
    fn records_each_edge_and_says_whether_it_was_new() {
        let asked = RefCell::new(Vec::new());
        let seen_before = RefCell::new(false);
        let surface = |tool: &str, arguments: Value| {
            asked.borrow_mut().push((tool.to_string(), arguments));
            let again = seen_before.replace(true);
            Ok(json!({ "created": !again, "reinforcement": u32::from(again) }))
        };
        let trees = fleet();
        let mut twice = fleet();
        twice.extend(fleet());
        let first = record(&surface, &twice, false).unwrap();
        let second = record(&surface, &trees, false).unwrap();
        assert_eq!(
            first,
            [
                "read     github.com/acme/web (1 manifest(s))",
                "read     github.com/acme/orders (1 manifest(s))",
                "recorded github.com/acme/web -> github.com/acme/orders (package.json names @acme/orders-client)",
            ]
        );
        assert_eq!(
            second[2],
            "seen     github.com/acme/web -> github.com/acme/orders (package.json names @acme/orders-client), reinforced to 1"
        );
        assert!(asked.borrow().iter().all(|(t, _)| t == "record_dependency"));

        asked.borrow_mut().clear();
        let dry = record(&surface, &trees, true).unwrap();
        assert!(dry[2].starts_with("found    "), "{dry:?}");
        assert!(
            asked.borrow().is_empty(),
            "a dry run asks the surface nothing"
        );
    }
}
