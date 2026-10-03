use super::*;

fn names(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}

#[test]
fn which_files_are_manifests() {
    for yes in [
        "package.json",
        "svc/Cargo.toml",
        "go.mod",
        "pyproject.toml",
        "requirements-dev.txt",
    ] {
        assert!(is_manifest(yes), "{yes}");
    }
    for no in [
        "package-lock.json",
        "Cargo.lock",
        "go.sum",
        "README.md",
        "setup.cfg",
        "web/node_modules/react/package.json",
        "tests/fixtures/Cargo.toml",
        "internal/testdata/go.mod",
    ] {
        assert!(!is_manifest(no), "{no}");
    }
}

#[test]
fn package_json_publishes_its_name_and_depends_on_every_table() {
    let m = read(
        "web/package.json",
        r#"{
            "name": "@acme/web",
            "repository": { "type": "git", "url": "git+https://github.com/Acme/web.git" },
            "dependencies": { "@acme/orders-client": "^2.1.0", "react": "18" },
            "devDependencies": { "vitest": "1" },
            "peerDependencies": { "@acme/ui": "*" }
        }"#,
    )
    .unwrap();
    assert_eq!(m.ecosystem, Ecosystem::Npm);
    assert_eq!(m.path, "web/package.json");
    assert_eq!(names(&m.publishes), ["@acme/web"]);
    assert_eq!(
        names(&m.depends),
        ["@acme/orders-client", "@acme/ui", "react", "vitest"]
    );
    assert_eq!(m.home.as_deref(), Some("github.com/acme/web"));
    for (repository, home) in [
        (r#""github:acme/web""#, Some("github.com/acme/web")),
        (r#""acme/web""#, Some("github.com/acme/web")),
        (
            r#""https://gitlab.com/acme/web/-/tree/main""#,
            Some("gitlab.com/acme/web"),
        ),
        (r#""not a url""#, None),
    ] {
        let m = read(
            "package.json",
            &format!(r#"{{"repository": {repository}}}"#),
        )
        .unwrap();
        assert_eq!(m.home.as_deref(), home, "{repository}");
    }
    assert_eq!(read("package.json", "{ not json"), None);
}

#[test]
fn cargo_reads_the_package_and_every_kind_of_dependency_table() {
    let m = read(
        "Cargo.toml",
        r#"
[package]
name = "orders_service" # comment
version = "0.1.0"
repository = "https://github.com/acme/orders-service"

[dependencies]
serde = { version = "1", features = ["derive"] }
acme-client.workspace = true
my-package-tools = "0.3"
http = { package = "acme-http", version = "2" }

[dependencies.tokio]
version = "1"
features = ["full"]

[dev-dependencies]
proptest = "1"

[target.'cfg(unix)'.dependencies]
nix = "0.29"

[features]
default = ["std"]
"#,
    )
    .unwrap();
    assert_eq!(m.ecosystem, Ecosystem::Cargo);
    assert_eq!(
        names(&m.publishes),
        ["orders-service"],
        "`_` and `-` are one name in Cargo"
    );
    assert_eq!(
        names(&m.depends),
        ["acme-client", "acme-http", "my-package-tools", "nix", "proptest", "serde", "tokio"],
        "a rename depends on the real name; a key merely containing `package` is not a rename; features are not dependencies"
    );
    assert_eq!(m.home.as_deref(), Some("github.com/acme/orders-service"));
    let inherited = read(
        "Cargo.toml",
        "[workspace.package]\nrepository = \"https://github.com/acme/mono\"\n\n[package]\nname = \"x\"\nrepository.workspace = true\n",
    )
    .unwrap();
    assert_eq!(inherited.home.as_deref(), Some("github.com/acme/mono"));
}

#[test]
fn go_mod_publishes_its_module_and_depends_on_requires() {
    let m = read(
        "go.mod",
        "module github.com/acme/checkout\n\ngo 1.22\n\nrequire github.com/acme/orders v1.4.0 // indirect\n\nrequire (\n\tgithub.com/acme/pay/v2 v2.0.1\n\tgolang.org/x/sync v0.7.0\n)\n",
    )
    .unwrap();
    assert_eq!(names(&m.publishes), ["github.com/acme/checkout"]);
    assert_eq!(m.home.as_deref(), Some("github.com/acme/checkout"));
    let vanity = read("go.mod", "module golang.org/x/sync\n").unwrap();
    assert_eq!(vanity.home, None, "a vanity path names no repository");
    assert_eq!(
        names(&m.depends),
        [
            "github.com/acme/orders",
            "github.com/acme/pay/v2",
            "golang.org/x/sync"
        ]
    );
}

#[test]
fn python_reads_pep_621_poetry_and_requirements() {
    let pep = read(
        "pyproject.toml",
        r#"
[project]
name = "Acme_Billing"
dependencies = [
    "acme-orders-client>=2",  # pinned below
    "requests[socks] >= 2.31; python_version < '3.13'",
]

[project.optional-dependencies]
dev = ["pytest"]

[project.urls]
Homepage = "https://acme.example"
"Source Code" = "https://github.com/acme/billing"
"#,
    )
    .unwrap();
    assert_eq!(
        names(&pep.publishes),
        ["acme-billing"],
        "PEP 503 normalization"
    );
    assert_eq!(names(&pep.depends), ["acme-orders-client", "requests"]);
    assert_eq!(pep.home.as_deref(), Some("github.com/acme/billing"));

    let inline = read(
        "pyproject.toml",
        "[project]\nname = \"x\"\ndependencies = [\"a\", \"b>=1\"]\n",
    )
    .unwrap();
    assert_eq!(names(&inline.depends), ["a", "b"]);

    let poetry = read(
        "pyproject.toml",
        "[tool.poetry]\nname = \"acme-search\"\n\n[tool.poetry.dependencies]\npython = \"^3.11\"\n\"acme.orders\" = \"^1\"\nhttpx = \"*\"\n",
    )
    .unwrap();
    assert_eq!(names(&poetry.publishes), ["acme-search"]);
    assert_eq!(
        names(&poetry.depends),
        ["acme-orders", "httpx"],
        "python itself is not a dependency"
    );

    let reqs = read(
        "requirements.txt",
        "# pinned\nAcme.Orders==1.2\n-r base.txt\ngit+https://example.com/x.git\nflask>=3 ; python_version>'3.8'\n",
    )
    .unwrap();
    assert!(reqs.publishes.is_empty());
    assert_eq!(names(&reqs.depends), ["acme-orders", "flask"]);
}

fn repo(name: &str, manifests: Vec<Manifest>) -> (String, Vec<Manifest>) {
    (name.to_string(), manifests)
}

/// Two manifests make an edge only when one depends on a name the other
/// publishes in the same ecosystem; a repository's own packages are not an
/// edge; Go matches a module path by prefix.
#[test]
fn declared_edges_join_dependencies_to_publications() {
    let web = read(
        "package.json",
        r#"{"name":"@acme/web","dependencies":{"@acme/orders-client":"2","left-pad":"1"}}"#,
    )
    .unwrap();
    let orders_npm = read(
        "clients/js/package.json",
        r#"{"name":"@acme/orders-client"}"#,
    )
    .unwrap();
    let orders_go = read("go.mod", "module github.com/acme/orders\n").unwrap();
    let checkout = read(
        "go.mod",
        "module github.com/acme/checkout\nrequire (\n\tgithub.com/acme/orders/api v1.0.0\n\tgithub.com/acme/checkout/internal v0.0.0\n)\n",
    )
    .unwrap();
    // A Python package of the same name is a different package.
    let unrelated = read("requirements.txt", "acme-orders-client==1\n").unwrap();

    let claims = declared(&[
        repo("github.com/acme/web", vec![web]),
        repo("github.com/acme/orders", vec![orders_npm, orders_go]),
        repo("github.com/acme/checkout", vec![checkout]),
        repo("github.com/acme/tools", vec![unrelated]),
    ]);
    let edges: Vec<(&str, &str, &str, &str)> = claims
        .iter()
        .map(|(c, file)| {
            (
                c.from.as_str(),
                c.to.as_str(),
                c.detail.as_str(),
                file.as_str(),
            )
        })
        .collect();
    assert_eq!(
        edges,
        [
            (
                "github.com/acme/checkout",
                "github.com/acme/orders",
                "go.mod names github.com/acme/orders/api",
                "go.mod"
            ),
            (
                "github.com/acme/web",
                "github.com/acme/orders",
                "package.json names @acme/orders-client",
                "package.json"
            ),
        ]
    );
    assert!(claims.iter().all(|(c, _)| c.source == Source::Declared));
}

/// A fork carries its upstream's name but does not publish it: a dependency
/// on that name is a dependency on the upstream. A project that moved to
/// another owner, keeping its name, still publishes what it publishes.
#[test]
fn a_fork_does_not_publish_its_upstreams_names() {
    let fork = read(
        "Cargo.toml",
        "[package]\nname = \"winit\"\nrepository = \"https://github.com/rust-windowing/winit\"\n",
    )
    .unwrap();
    let moved = read(
        "Cargo.toml",
        "[package]\nname = \"acme-orders\"\nrepository = \"https://github.com/acme/orders\"\n",
    )
    .unwrap();
    let engine = read(
        "Cargo.toml",
        "[package]\nname = \"engine\"\n\n[dependencies]\nwinit = \"0.30\"\nacme-orders = \"1\"\n",
    )
    .unwrap();
    let claims = declared(&[
        repo("github.com/me/winit-fork", vec![fork]),
        repo("github.com/me/orders", vec![moved]),
        repo("github.com/me/engine", vec![engine]),
    ]);
    let edges: Vec<(&str, &str)> = claims
        .iter()
        .map(|(c, _)| (c.from.as_str(), c.to.as_str()))
        .collect();
    assert_eq!(edges, [("github.com/me/engine", "github.com/me/orders")]);
}
