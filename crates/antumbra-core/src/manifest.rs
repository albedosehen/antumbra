//! What a repository's manifests declare (ADR-0019's declared source): the
//! package names it publishes and the ones it depends on.
//!
//! A manifest is read, never run, and only for names: which packages a
//! repository puts out, and which it pulls in. Versions, features and
//! scripts are not the graph's business. Matching one repository's
//! dependencies to another's publications ([`declared`]) is what turns two
//! manifests into an edge, and the edge carries the file and the name as its
//! evidence.
//!
//! The TOML here is read line by line for the few shapes the graph needs
//! (`[package] name`, dependency tables and their keys, `[project]` name and
//! its `dependencies` array), which keeps the core free of a parser
//! dependency. Something it does not recognize yields no name rather than a
//! wrong one: a missed edge is a gap the other sources can fill, a false one
//! is a claim with forged evidence.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::depgraph::{Claim, Source};
use crate::provenance::{normalize_repo, GitProvenance};

/// Which package namespace a name belongs to. Two ecosystems can share a name
/// without being the same package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Ecosystem {
    Npm,
    Cargo,
    Go,
    Python,
}

/// What one manifest file declares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub ecosystem: Ecosystem,
    /// The file, relative to the repository root.
    pub path: String,
    pub publishes: Vec<String>,
    pub depends: Vec<String>,
    /// The repository the manifest says it comes from (`repository` in
    /// Cargo.toml or package.json, a project URL in pyproject.toml, a forge
    /// module path in go.mod), as a `host/org/name` slug, when it says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home: Option<String>,
}

/// What a parser reads from one file: publications, dependencies, home.
type Parsed = (Vec<String>, Vec<String>, Option<String>);

/// Directories whose manifests are not the project's own: installed or
/// vendored packages, and test fixtures, which would otherwise publish names
/// the repository does not.
const NOT_THE_PROJECTS: [&str; 6] = [
    "node_modules",
    "vendor",
    "third_party",
    "testdata",
    "fixtures",
    "__fixtures__",
];

/// Whether `path` (relative to the repository root, `/`-separated) is a
/// manifest this reads and one the project itself ships, for a caller
/// choosing what to fetch.
pub fn is_manifest(path: &str) -> bool {
    let mut parts = path.rsplit('/');
    let name = parts.next().unwrap_or(path);
    let readable = matches!(
        name,
        "package.json" | "Cargo.toml" | "go.mod" | "pyproject.toml"
    ) || (name.starts_with("requirements") && name.ends_with(".txt"));
    readable && !parts.any(|dir| NOT_THE_PROJECTS.contains(&dir))
}

/// Read one manifest by its file name. `None` for a file this does not read,
/// or one that does not parse.
pub fn read(path: &str, text: &str) -> Option<Manifest> {
    let name = path.rsplit('/').next().unwrap_or(path);
    let (ecosystem, (publishes, depends, home)) = match name {
        "package.json" => (Ecosystem::Npm, npm(text)?),
        "Cargo.toml" => (Ecosystem::Cargo, cargo(text)),
        "go.mod" => (Ecosystem::Go, go(text)),
        "pyproject.toml" => (Ecosystem::Python, pyproject(text)),
        n if n.starts_with("requirements") && n.ends_with(".txt") => {
            (Ecosystem::Python, (Vec::new(), requirements(text), None))
        }
        _ => return None,
    };
    let tidy = |names: Vec<String>, ecosystem: Ecosystem| -> Vec<String> {
        let set: BTreeSet<String> = names
            .into_iter()
            .map(|n| canonical(ecosystem, &n))
            .filter(|n| !n.is_empty())
            .collect();
        set.into_iter().collect()
    };
    Some(Manifest {
        ecosystem,
        path: path.trim_start_matches("./").to_string(),
        publishes: tidy(publishes, ecosystem),
        depends: tidy(depends, ecosystem),
        home,
    })
}

/// The forges whose paths are `host/org/name`, for reading a repository out
/// of a Go module path.
const FORGES: [&str; 4] = ["github.com", "gitlab.com", "bitbucket.org", "codeberg.org"];

/// A declared repository URL as a `host/org/name` slug: a git or web URL
/// (`git+https://github.com/acme/orders.git`, `git@github.com:acme/orders`,
/// a page under the repository), or npm's `github:acme/orders` and bare
/// `acme/orders` shorthands.
fn repository_slug(url: &str) -> Option<String> {
    let url = url.trim();
    let url = url.strip_prefix("git+").unwrap_or(url);
    let url = match url.strip_prefix("github:") {
        Some(rest) => format!("github.com/{rest}"),
        None if !url.contains(':') && url.matches('/').count() == 1 => {
            format!("github.com/{url}")
        }
        None => url.to_string(),
    };
    let slug = crate::provenance::repo_slug_from_remote(&url)?;
    let parts: Vec<&str> = slug.split('/').filter(|p| !p.is_empty()).collect();
    (parts.len() >= 3 && parts[0].contains('.')).then(|| parts[..3].join("/"))
}

/// A name as its ecosystem compares them. Python folds case and treats `-`,
/// `_` and `.` alike (PEP 503); Cargo treats `-` and `_` alike; npm and Go
/// compare as written, lowercased for npm and Go.
fn canonical(ecosystem: Ecosystem, name: &str) -> String {
    let name = name.trim().trim_matches('"').trim_matches('\'');
    match ecosystem {
        Ecosystem::Python => {
            let mut out = String::new();
            let mut last_sep = false;
            for c in name.to_lowercase().chars() {
                if matches!(c, '-' | '_' | '.') {
                    if !last_sep {
                        out.push('-');
                    }
                    last_sep = true;
                } else {
                    out.push(c);
                    last_sep = false;
                }
            }
            out
        }
        Ecosystem::Cargo => name.to_lowercase().replace('_', "-"),
        Ecosystem::Npm | Ecosystem::Go => name.to_lowercase(),
    }
}

fn npm(text: &str) -> Option<Parsed> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let publishes = v
        .get("name")
        .and_then(|n| n.as_str())
        .map(|n| vec![n.to_string()])
        .unwrap_or_default();
    let mut depends = Vec::new();
    for table in [
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
    ] {
        if let Some(obj) = v.get(table).and_then(|t| t.as_object()) {
            depends.extend(obj.keys().cloned());
        }
    }
    let home = v
        .get("repository")
        .and_then(|r| r.as_str().or_else(|| r.get("url")?.as_str()))
        .and_then(repository_slug);
    Some((publishes, depends, home))
}

/// A TOML line without its comment, trimmed. A `#` inside a quoted string is
/// kept, which is all a manifest line needs.
fn uncomment(line: &str) -> &str {
    let mut quoted: Option<char> = None;
    for (i, c) in line.char_indices() {
        match (quoted, c) {
            (None, '"' | '\'') => quoted = Some(c),
            (Some(q), c) if c == q => quoted = None,
            (None, '#') => return line[..i].trim(),
            _ => {}
        }
    }
    line.trim()
}

/// The section a `[header]` line names, without its brackets.
fn header(line: &str) -> Option<&str> {
    let inner = line.strip_prefix('[')?.strip_suffix(']')?;
    Some(inner.trim_matches(|c| c == '[' || c == ']').trim())
}

/// The key of a `key = value` line: the first part of a dotted bare key
/// (`serde.workspace = true` is `serde`), or a quoted key whole
/// (`"acme.orders" = "1"` is `acme.orders`).
fn key(line: &str) -> Option<&str> {
    let (k, _) = line.split_once('=')?;
    let k = k.trim();
    let k = match k.strip_prefix('"') {
        Some(rest) => rest.split('"').next()?,
        None => k.split('.').next()?.trim(),
    };
    (!k.is_empty()).then_some(k)
}

/// A quoted string value of `key = "value"`.
fn string_value<'a>(line: &'a str, want: &str) -> Option<&'a str> {
    let (k, v) = line.split_once('=')?;
    if k.trim() != want {
        return None;
    }
    Some(v.trim().trim_matches('"').trim_matches('\''))
}

const CARGO_TABLES: [&str; 3] = ["dependencies", "dev-dependencies", "build-dependencies"];

/// `[dependencies]`, `[dev-dependencies]`, `[workspace.dependencies]`,
/// `[target.'cfg(unix)'.dependencies]` and the like: a table whose keys are
/// dependencies.
fn is_cargo_dependency_table(section: &str) -> bool {
    let last = section
        .rsplit('.')
        .next()
        .unwrap_or(section)
        .trim_matches('"');
    CARGO_TABLES.contains(&last)
}

/// The dependency a `[dependencies.serde]` header names, if it is one.
fn cargo_dependency_header(section: &str) -> Option<&str> {
    let parts: Vec<&str> = section.split('.').collect();
    let n = parts.len();
    (n >= 2 && CARGO_TABLES.contains(&parts[n - 2])).then(|| parts[n - 1].trim_matches('"'))
}

/// The real name in a renamed dependency, `alias = { package = "real", ... }`.
/// Only a `package` key inside the inline table counts, so a dependency that
/// merely has "package" in its name is not mistaken for a rename.
fn cargo_renamed(value: &str) -> Option<&str> {
    let inner = value.trim().strip_prefix('{')?;
    inner.split(',').find_map(|field| {
        let (k, v) = field.split_once('=')?;
        (k.trim() == "package").then(|| v.trim().trim_end_matches('}').trim().trim_matches('"'))
    })
}

fn cargo(text: &str) -> Parsed {
    let mut section = String::new();
    let mut publishes = Vec::new();
    let mut depends = Vec::new();
    let mut home = None;
    for raw in text.lines() {
        let line = uncomment(raw);
        if line.is_empty() {
            continue;
        }
        if let Some(h) = header(line) {
            section = h.to_string();
            // `[dependencies.serde]` names a dependency in its header.
            if let Some(name) = cargo_dependency_header(&section) {
                depends.push(name.to_string());
            }
            continue;
        }
        if section == "package" || section == "workspace.package" {
            if let Some(url) = string_value(line, "repository") {
                home = repository_slug(url);
            }
        }
        if section == "package" {
            if let Some(name) = string_value(line, "name") {
                publishes.push(name.to_string());
            }
        } else if is_cargo_dependency_table(&section) {
            if let Some(k) = key(line) {
                let renamed = line.split_once('=').and_then(|(_, v)| cargo_renamed(v));
                depends.push(renamed.unwrap_or(k).to_string());
            }
        }
    }
    (publishes, depends, home)
}

fn go(text: &str) -> Parsed {
    let mut publishes = Vec::new();
    let mut depends = Vec::new();
    let mut in_require = false;
    for raw in text.lines() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(module) = line.strip_prefix("module ") {
            publishes.push(module.trim().trim_matches('"').to_string());
        } else if line.starts_with("require (") || line == "require(" {
            in_require = true;
        } else if in_require && line == ")" {
            in_require = false;
        } else if in_require {
            if let Some(path) = line.split_whitespace().next() {
                depends.push(path.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("require ") {
            if let Some(path) = rest.split_whitespace().next() {
                depends.push(path.to_string());
            }
        }
    }
    // A module path on a forge names its repository; a vanity path does not.
    let home = publishes.first().and_then(|module: &String| {
        let parts: Vec<&str> = module.split('/').collect();
        (parts.len() >= 3 && FORGES.contains(&parts[0])).then(|| parts[..3].join("/"))
    });
    (publishes, depends, home)
}

/// The distribution name at the head of a PEP 508 requirement:
/// `requests[socks]>=2; python_version < "3.12"` is `requests`.
fn requirement_name(spec: &str) -> Option<String> {
    let spec = spec
        .trim()
        .trim_matches(',')
        .trim()
        .trim_matches('"')
        .trim_matches('\'');
    if spec.is_empty() || spec.starts_with('-') || spec.contains("://") {
        return None;
    }
    let name: String = spec
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .collect();
    (!name.is_empty()).then_some(name)
}

fn requirements(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter_map(requirement_name)
        .collect()
}

/// The `[project.urls]` keys that name the source repository.
const SOURCE_URL_KEYS: [&str; 4] = ["repository", "source", "source code", "code"];

fn pyproject(text: &str) -> Parsed {
    let mut section = String::new();
    let mut publishes = Vec::new();
    let mut depends = Vec::new();
    let mut home = None;
    let mut in_array = false;
    for raw in text.lines() {
        let line = uncomment(raw);
        if line.is_empty() {
            continue;
        }
        if in_array {
            for item in line.split(',') {
                let item = item.trim().trim_end_matches(']').trim();
                if let Some(name) = requirement_name(item) {
                    depends.push(name);
                }
            }
            if line.contains(']') {
                in_array = false;
            }
            continue;
        }
        if let Some(h) = header(line) {
            section = h.to_string();
            continue;
        }
        match section.as_str() {
            "project" => {
                if let Some(name) = string_value(line, "name") {
                    publishes.push(name.to_string());
                } else if let Some(rest) = line
                    .strip_prefix("dependencies")
                    .and_then(|r| r.trim_start().strip_prefix('='))
                {
                    let rest = rest.trim();
                    let inner = rest.trim_start_matches('[');
                    for item in inner.split(',') {
                        let item = item.trim().trim_end_matches(']').trim();
                        if let Some(name) = requirement_name(item) {
                            depends.push(name);
                        }
                    }
                    in_array = rest.starts_with('[') && !rest.contains(']');
                }
            }
            "project.urls" => {
                let named = key(line).map(str::to_lowercase);
                if named.is_some_and(|k| SOURCE_URL_KEYS.contains(&k.as_str())) {
                    let url = line
                        .split_once('=')
                        .map(|(_, v)| v.trim().trim_matches('"'));
                    home = home.or_else(|| repository_slug(url?));
                }
            }
            "tool.poetry" => {
                if let Some(name) = string_value(line, "name") {
                    publishes.push(name.to_string());
                } else if let Some(url) = string_value(line, "repository") {
                    home = repository_slug(url);
                }
            }
            s if s.starts_with("tool.poetry") && s.ends_with("dependencies") => {
                if let Some(k) = key(line) {
                    if k != "python" {
                        depends.push(k.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    (publishes, depends, home)
}

/// Whether a manifest's packages are published from `repo`, the repository it
/// was read in. One that says it comes from a repository of another name is a
/// fork or a copy, and the names it carries are its upstream's; one that names
/// a repository of the same name under another owner is the project after a
/// move, and stands.
fn published_from(m: &Manifest, repo: &str) -> bool {
    let name = |slug: &str| slug.rsplit('/').next().unwrap_or(slug).to_ascii_lowercase();
    m.home
        .as_deref()
        .is_none_or(|home| name(home) == name(repo))
}

/// Declared claims between repositories: `from` depends on `to` wherever a
/// manifest of `from` depends on a name a manifest of `to` publishes, in the
/// same ecosystem. A repository's dependency on its own packages is not an
/// edge, and a fork does not publish its upstream's names
/// (`published_from`). Each claim's detail names the file and the package, and the claims
/// come back in a stable order with the file each was read from.
pub fn declared(repos: &[(String, Vec<Manifest>)]) -> Vec<(Claim, String)> {
    // Go names a dependency by module path, and a module's path can run past
    // the module root (`/v2`, a subpackage), so it matches by prefix too.
    let mut publishers: BTreeMap<(Ecosystem, String), BTreeSet<String>> = BTreeMap::new();
    for (repo, manifests) in repos {
        for m in manifests.iter().filter(|m| published_from(m, repo)) {
            for name in &m.publishes {
                publishers
                    .entry((m.ecosystem, name.clone()))
                    .or_default()
                    .insert(repo.clone());
            }
        }
    }
    let go_owner = |dep: &str| -> Option<&BTreeSet<String>> {
        publishers
            .iter()
            .filter(|((e, name), _)| {
                *e == Ecosystem::Go && (dep == name || dep.starts_with(&format!("{name}/")))
            })
            .max_by_key(|((_, name), _)| name.len())
            .map(|(_, repos)| repos)
    };
    let mut out: BTreeMap<(String, String), (Claim, String)> = BTreeMap::new();
    for (repo, manifests) in repos {
        for m in manifests {
            for dep in &m.depends {
                let owners = match m.ecosystem {
                    Ecosystem::Go => go_owner(dep),
                    e => publishers.get(&(e, dep.clone())),
                };
                for owner in owners.into_iter().flatten() {
                    if owner == repo {
                        continue;
                    }
                    let claim = Claim::new(
                        repo,
                        owner,
                        Source::Declared,
                        &format!("{} names {dep}", m.path),
                    );
                    out.entry((claim.from.clone(), claim.to.clone()))
                        .or_insert((claim, m.path.clone()));
                }
            }
        }
    }
    out.into_values().collect()
}

/// A repository's manifests as they were read at one commit: what is kept per
/// repository so the edges into and out of any one of them can be worked out
/// again when it changes, without reading every other repository again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestSet {
    /// The repository slug.
    pub repo: String,
    /// The commit the manifests were read at.
    pub commit: String,
    pub branch: Option<String>,
    pub manifests: Vec<Manifest>,
}

/// What reading one repository's manifests again changes in the declared
/// edges.
#[derive(Debug, Default, PartialEq)]
pub struct DeclaredSync {
    /// Every declared edge into or out of the repository, each with the file
    /// that declares it at the commit it was read, to record or reinforce.
    pub record: Vec<(Claim, GitProvenance)>,
    /// Declared edges into or out of it that no manifest declares any more.
    pub retract: Vec<Claim>,
}

/// The declared edges into and out of `repo` among every repository whose
/// manifests are known (`sets`), against the declared edges already recorded
/// (`existing`, as `(from, to)`). An edge whose other end has no set is never
/// retracted: nothing read here can say it is gone, and it may have come from
/// a scan of a repository this side cannot see.
pub fn sync(sets: &[ManifestSet], repo: &str, existing: &[(String, String)]) -> DeclaredSync {
    let repo = normalize_repo(repo);
    let repos: Vec<(String, Vec<Manifest>)> = sets
        .iter()
        .map(|s| (normalize_repo(&s.repo), s.manifests.clone()))
        .collect();
    let known: BTreeSet<&str> = repos.iter().map(|(r, _)| r.as_str()).collect();
    let touching: Vec<(Claim, String)> = declared(&repos)
        .into_iter()
        .filter(|(c, _)| c.from == repo || c.to == repo)
        .collect();
    let still: BTreeSet<(&str, &str)> = touching
        .iter()
        .map(|(c, _)| (c.from.as_str(), c.to.as_str()))
        .collect();
    let retract = existing
        .iter()
        .map(|(from, to)| (normalize_repo(from), normalize_repo(to)))
        .filter(|(from, to)| {
            (*from == repo || *to == repo)
                && known.contains(from.as_str())
                && known.contains(to.as_str())
                && !still.contains(&(from.as_str(), to.as_str()))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|(from, to)| Claim::new(&from, &to, Source::Declared, ""))
        .collect();
    let record = touching
        .iter()
        .filter_map(|(claim, path)| {
            let set = sets
                .iter()
                .find(|s| normalize_repo(&s.repo) == claim.from)?;
            let mut anchor =
                GitProvenance::new(claim.from.clone(), set.commit.clone()).at_path(path.clone());
            if let Some(branch) = &set.branch {
                anchor = anchor.on_branch(branch.clone());
            }
            Some((claim.clone(), anchor))
        })
        .collect();
    DeclaredSync { record, retract }
}

#[cfg(test)]
mod tests;
