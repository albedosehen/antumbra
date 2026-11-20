//! What an organization's settings stop of what setup wires into Claude Code.
//!
//! Setup adds hooks to the user's own settings and registers a remote MCP
//! server, `antumbra`, for the user. An organization can stop either from its
//! managed settings, and Claude Code then says nothing: the hooks are not
//! run, or the server is gone from `/mcp`, and a check that reads back only
//! what setup wrote calls everything in place. So the check reads what the
//! managed settings say as well, and both are judged here, over plain data,
//! by the rules the vendor's pages give
//! (https://code.claude.com/docs/en/managed-mcp and the settings reference,
//! read on 2026-11-20):
//!
//! - **Hooks.** `allowManagedHooksOnly` runs only the organization's.
//!   `strictPluginOnlyCustomization`, when it is `true` or names `hooks`,
//!   stops the ones in a user's settings. `disableAllHooks` stops them from
//!   whichever file it is in.
//! - **The server.** A `managed-mcp.json` fixes the set of servers, so one it
//!   does not name is not loaded. `strictPluginOnlyCustomization`, when it is
//!   `true` or names `mcp`, leaves out the ones a user registered. A
//!   `deniedMcpServers` entry in any file blocks a server it matches. And
//!   once any file sets `allowedMcpServers`, a remote server loads only when
//!   a `serverUrl` entry matches its address, or, in a list with no
//!   `serverUrl` entry, a `serverName` entry names it. The lists of every
//!   file are one list, unless the organization sets
//!   `allowManagedMcpServersOnly`, which keeps the allowlist to its own.
//!
//! Only files are read, as in [`super::managed`]: a policy delivered another
//! way (the vendor's console, an MDM profile, the registry) is not seen.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// The key that keeps kinds of customization to plugins and managed settings.
const PLUGINS_ONLY: &str = "strictPluginOnlyCustomization";

/// One settings file with a say: one of the organization's, or one of the
/// user's.
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub path: PathBuf,
    pub settings: Value,
    /// Whether it is the organization's. Some keys count only from there.
    pub managed: bool,
}

impl Source {
    /// One of the organization's files.
    pub fn managed(path: PathBuf, settings: Value) -> Self {
        Source {
            path,
            settings,
            managed: true,
        }
    }

    /// One of the user's: their settings file, or one a launch passes.
    pub fn user(path: PathBuf, settings: Value) -> Self {
        Source {
            path,
            settings,
            managed: false,
        }
    }
}

/// The organization's value for `key`, with the file it is in. Its files are
/// merged in order, so the last to hold the key is the one that stands.
fn managed<'a>(sources: &'a [Source], key: &str) -> Option<(&'a Source, &'a Value)> {
    sources
        .iter()
        .filter(|source| source.managed)
        .filter_map(|source| source.settings.get(key).map(|value| (source, value)))
        .next_back()
}

/// Whether a `strictPluginOnlyCustomization` value locks `surface`.
fn locks(value: &Value, surface: &str) -> bool {
    match value {
        Value::Bool(all) => *all,
        Value::Array(surfaces) => surfaces.iter().any(|named| named == surface),
        _ => false,
    }
}

/// Whether the hooks in a user's settings run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hooks {
    Run,
    /// Stopped by `key`, in the file at `by`, which is the organization's
    /// when `managed`.
    Stopped {
        by: PathBuf,
        key: &'static str,
        managed: bool,
    },
}

/// Whether hooks in the user's own settings run, under `sources`.
pub fn hooks(sources: &[Source]) -> Hooks {
    let stopped = |source: &Source, key: &'static str| Hooks::Stopped {
        by: source.path.clone(),
        key,
        managed: source.managed,
    };
    for key in ["disableAllHooks", "allowManagedHooksOnly"] {
        if let Some((source, Value::Bool(true))) = managed(sources, key) {
            return stopped(source, key);
        }
    }
    if let Some((source, value)) = managed(sources, PLUGINS_ONLY) {
        if locks(value, "hooks") {
            return stopped(source, PLUGINS_ONLY);
        }
    }
    sources
        .iter()
        .filter(|source| !source.managed)
        .find(|source| source.settings.get("disableAllHooks") == Some(&Value::Bool(true)))
        .map_or(Hooks::Run, |source| stopped(source, "disableAllHooks"))
}

/// Why a server is kept out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Why {
    /// A `managed-mcp.json` that does not name it fixes the set of servers.
    Fixed,
    /// `strictPluginOnlyCustomization` covers MCP servers.
    PluginsOnly,
    /// A `deniedMcpServers` entry matches it.
    Denied,
    /// `allowedMcpServers` is set, and nothing in it admits the server.
    /// `entry` is one that would, in the form the list goes by, and `open`
    /// says whether a list in the user's own settings adds to it.
    Unlisted { entry: String, open: bool },
}

/// Whether a remote MCP server the user registered loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Server {
    Loads,
    /// Kept out by the file at `by`, which is the organization's when
    /// `managed`.
    Blocked {
        by: PathBuf,
        managed: bool,
        why: Why,
    },
}

/// Whether the remote server registered as `name` at `url` loads, under
/// `sources` and the `managed-mcp.json` in `fixed`, where there is one.
pub fn server(sources: &[Source], fixed: Option<&Source>, name: &str, url: &str) -> Server {
    let blocked = |source: &Source, why: Why| Server::Blocked {
        by: source.path.clone(),
        managed: source.managed,
        why,
    };
    // A fixed set: its own servers load, and no other.
    if let Some(fixed) = fixed {
        if let Some(servers) = fixed.settings.get("mcpServers").and_then(Value::as_object) {
            return if servers.contains_key(name) {
                Server::Loads
            } else {
                blocked(fixed, Why::Fixed)
            };
        }
    }
    if let Some((source, value)) = managed(sources, PLUGINS_ONLY) {
        if locks(value, "mcp") {
            return blocked(source, Why::PluginsOnly);
        }
    }
    // The denylist of every file, which nothing overrides.
    for source in sources {
        let denied = list(source, "deniedMcpServers")
            .into_iter()
            .flatten()
            .any(|entry| names(entry, name) || addresses(entry, url));
        if denied {
            return blocked(source, Why::Denied);
        }
    }
    // The allowlist: the organization's, and the user's unless the
    // organization keeps it to its own. Set nowhere, everything loads; set
    // and empty, nothing does.
    let own_only = matches!(
        managed(sources, "allowManagedMcpServersOnly"),
        Some((_, Value::Bool(true)))
    );
    let lists: Vec<(&Source, &Vec<Value>)> = sources
        .iter()
        .filter(|source| source.managed || !own_only)
        .filter_map(|source| list(source, "allowedMcpServers").map(|list| (source, list)))
        .collect();
    let Some(&(first, _)) = lists.first() else {
        return Server::Loads;
    };
    let entries = || lists.iter().flat_map(|(_, list)| list.iter());
    // Once a list holds an address, a remote server is admitted by address
    // alone: a name no longer does it.
    let by_address = entries().any(|entry| entry.get("serverUrl").is_some());
    let admitted = if by_address {
        entries().any(|entry| addresses(entry, url))
    } else {
        entries().any(|entry| names(entry, name))
    };
    if admitted {
        return Server::Loads;
    }
    // The organization's list is the one to name, where it has one.
    let source = lists
        .iter()
        .map(|(source, _)| *source)
        .find(|source| source.managed)
        .unwrap_or(first);
    let entry = if by_address {
        json!({ "serverUrl": url })
    } else {
        json!({ "serverName": name })
    };
    blocked(
        source,
        Why::Unlisted {
            entry: entry.to_string(),
            open: !own_only,
        },
    )
}

/// The list a file holds under `key`, when it holds one.
fn list<'a>(source: &'a Source, key: &str) -> Option<&'a Vec<Value>> {
    source.settings.get(key).and_then(Value::as_array)
}

/// Whether `entry` is a `serverName` entry for `name`. A name matches whole.
fn names(entry: &Value, name: &str) -> bool {
    entry.get("serverName").and_then(Value::as_str) == Some(name)
}

/// Whether `entry` is a `serverUrl` entry whose pattern matches `url`.
fn addresses(entry: &Value, url: &str) -> bool {
    entry
        .get("serverUrl")
        .and_then(Value::as_str)
        .is_some_and(|pattern| url_matches(pattern, url))
}

/// The parts of a URL, or of a pattern for one, that matching goes by.
struct Parts<'a> {
    scheme: &'a str,
    host: &'a str,
    port: Option<&'a str>,
    path: &'a str,
}

impl<'a> Parts<'a> {
    fn of(text: &'a str) -> Option<Self> {
        let (scheme, rest) = text.trim().split_once("://")?;
        let cut = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let (authority, path) = rest.split_at(cut);
        let authority = authority.rsplit('@').next().unwrap_or(authority);
        // A bracketed IPv6 address holds colons of its own.
        let (host, port) = match authority.find(']') {
            Some(end) => {
                let (host, after) = authority.split_at(end + 1);
                (host, after.strip_prefix(':'))
            }
            None => match authority.rsplit_once(':') {
                Some((host, port)) => (host, Some(port)),
                None => (authority, None),
            },
        };
        Some(Parts {
            scheme,
            // A fully qualified name's closing dot is not part of the name.
            host: host.trim_end_matches('.'),
            port: port.filter(|port| !port.is_empty()),
            path,
        })
    }
}

/// The port a scheme goes to when a URL names none.
fn default_port(scheme: &str) -> Option<&'static str> {
    match scheme.to_ascii_lowercase().as_str() {
        "https" => Some("443"),
        "http" => Some("80"),
        _ => None,
    }
}

/// Whether `pattern` matches all of `text`, a `*` standing for any run of
/// characters, an empty one included.
fn glob(pattern: &str, text: &str) -> bool {
    let (pattern, text) = (pattern.as_bytes(), text.as_bytes());
    let (mut p, mut t) = (0, 0);
    // The last `*` passed, and how much of the text it has taken so far.
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        if pattern.get(p) == Some(&b'*') {
            star = Some((p, t));
            p += 1;
        } else if pattern.get(p) == Some(&text[t]) {
            p += 1;
            t += 1;
        } else if let Some((at, taken)) = star {
            star = Some((at, taken + 1));
            p = at + 1;
            t = taken + 1;
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|rest| *rest == b'*')
}

/// Whether `pattern`, a `serverUrl` entry, matches `url`, a server's address.
///
/// As the vendor's page has it: `*` stands for anything, a whole scheme
/// included. A host matches in any case. A pattern with no path matches every
/// path, and a path matches in its own case. A pattern that names no port
/// matches the scheme's own port alone when its host is written out, and
/// every port when its host holds a `*`.
pub fn url_matches(pattern: &str, url: &str) -> bool {
    let (Some(want), Some(have)) = (Parts::of(pattern), Parts::of(url)) else {
        return pattern.trim() == url.trim();
    };
    let port = have.port.or(default_port(have.scheme));
    let any_case =
        |want: &str, have: &str| glob(&want.to_ascii_lowercase(), &have.to_ascii_lowercase());
    any_case(want.scheme, have.scheme)
        && any_case(want.host, have.host)
        && match want.port {
            Some(named) => port.is_some_and(|port| glob(named, port)),
            None => want.host.contains('*') || port == default_port(have.scheme),
        }
        && (want.path.is_empty()
            || glob(
                want.path,
                if have.path.is_empty() { "/" } else { have.path },
            ))
}

/// The `managed-mcp.json` in `dir`, the managed files' directory, when it
/// holds a JSON object.
pub fn fixed(dir: &Path) -> Option<Source> {
    let path = dir.join("managed-mcp.json");
    let settings: Value = serde_json::from_str(&std::fs::read_to_string(&path).ok()?).ok()?;
    settings
        .is_object()
        .then(|| Source::managed(path, settings))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAME: &str = "antumbra";

    fn theirs(settings: Value) -> Source {
        Source::managed(PathBuf::from("managed-settings.json"), settings)
    }

    fn mine(settings: Value) -> Source {
        Source::user(PathBuf::from("settings.json"), settings)
    }

    /// The rows of the vendor's table of what common patterns allow.
    #[test]
    fn an_address_pattern_matches_as_the_vendors_table_says() {
        let rows: [(&str, &[&str], &[&str]); 8] = [
            (
                "https://mcp.example.com/*",
                &[
                    "https://mcp.example.com/api",
                    "https://mcp.example.com:443/mcp",
                    "https://mcp.example.com",
                ],
                &[
                    "https://mcp.example.com:8443/api",
                    "http://mcp.example.com/api",
                    "https://other.example.com/api",
                ],
            ),
            // No path: every path, and still the scheme's own port.
            (
                "https://mcp.example.com",
                &["https://mcp.example.com/api/v2", "https://mcp.example.com"],
                &["https://mcp.example.com:8443/api"],
            ),
            (
                "https://mcp.example.com:8443/*",
                &["https://mcp.example.com:8443/api"],
                &["https://mcp.example.com/api"],
            ),
            (
                "https://mcp.example.com:*/*",
                &[
                    "https://mcp.example.com/api",
                    "https://mcp.example.com:8443/api",
                ],
                &["http://mcp.example.com:8443/api"],
            ),
            // A `*` in the host: every port.
            (
                "https://*.example.com/*",
                &[
                    "https://api.internal.example.com/mcp",
                    "https://staging.example.com:8443/api",
                ],
                &["https://example.org/mcp", "https://other.com/mcp"],
            ),
            (
                "http://localhost:*/*",
                &["http://localhost:8081/mcp", "http://localhost/mcp"],
                &["https://localhost:8081/mcp", "http://127.0.0.1:8081/mcp"],
            ),
            // Any scheme, each on its own port.
            (
                "*://mcp.example.com/*",
                &["https://mcp.example.com/api", "http://mcp.example.com/api"],
                &["https://mcp.example.com:8443/api"],
            ),
            // A host in any case, with or without its closing dot; a path in
            // its own.
            (
                "https://Mcp.Example.com./Api/*",
                &["https://mcp.example.com/Api/v1"],
                &["https://mcp.example.com/api/v1"],
            ),
        ];
        for (pattern, matched, unmatched) in rows {
            for url in matched {
                assert!(url_matches(pattern, url), "{pattern} should match {url}");
            }
            for url in unmatched {
                assert!(
                    !url_matches(pattern, url),
                    "{pattern} should not match {url}"
                );
            }
        }
        // The address written out is its own pattern, port and all.
        for url in [
            "http://127.0.0.1:8081/mcp",
            "https://memory.example/mcp",
            "http://[::1]:8081/mcp",
        ] {
            assert!(url_matches(url, url), "{url}");
        }
        assert!(!url_matches(
            "http://[::1]:8081/mcp",
            "http://[::1]:8082/mcp"
        ));
    }

    #[test]
    fn with_no_list_anywhere_the_server_loads() {
        let url = "https://memory.example/mcp";
        assert_eq!(server(&[], None, NAME, url), Server::Loads);
        let sources = [theirs(json!({ "env": {} })), mine(json!({ "hooks": {} }))];
        assert_eq!(server(&sources, None, NAME, url), Server::Loads);
        // A denylist that is empty, or names something else, blocks nothing.
        let sources = [theirs(json!({
            "deniedMcpServers": [{ "serverName": "other" }, { "serverUrl": "https://bad.example/*" }]
        }))];
        assert_eq!(server(&sources, None, NAME, url), Server::Loads);
    }

    /// The vendor's worked examples, for a remote server.
    #[test]
    fn an_allowlist_admits_by_address_once_it_holds_one_and_by_name_until_then() {
        let url = "https://memory.example/mcp";
        let unlisted = |entry: Value, open: bool| Server::Blocked {
            by: PathBuf::from("managed-settings.json"),
            managed: true,
            why: Why::Unlisted {
                entry: entry.to_string(),
                open,
            },
        };
        // Names only: its name admits it, another's does not.
        let by_name = |names: &[&str]| {
            [theirs(json!({
                "allowedMcpServers": names.iter().map(|n| json!({ "serverName": n })).collect::<Vec<_>>()
            }))]
        };
        assert_eq!(
            server(&by_name(&["github", NAME]), None, NAME, url),
            Server::Loads
        );
        assert_eq!(
            server(&by_name(&["github"]), None, NAME, url),
            unlisted(json!({ "serverName": NAME }), true)
        );
        // Commands and names: a remote server still goes by its name.
        let mixed = [theirs(json!({ "allowedMcpServers": [
            { "serverName": NAME },
            { "serverCommand": ["npx", "-y", "approved-package"] }
        ]}))];
        assert_eq!(server(&mixed, None, NAME, url), Server::Loads);
        // An address in the list: the name no longer admits it.
        let by_address = |patterns: &[&str]| {
            let mut entries: Vec<Value> =
                patterns.iter().map(|p| json!({ "serverUrl": p })).collect();
            entries.push(json!({ "serverName": NAME }));
            [theirs(json!({ "allowedMcpServers": entries }))]
        };
        assert_eq!(
            server(
                &by_address(&["https://*.internal.example.com/*"]),
                None,
                NAME,
                url
            ),
            unlisted(json!({ "serverUrl": url }), true)
        );
        assert_eq!(
            server(&by_address(&["https://memory.example/*"]), None, NAME, url),
            Server::Loads
        );
        // Set and empty: nothing is allowed.
        let none = [theirs(json!({ "allowedMcpServers": [] }))];
        assert_eq!(
            server(&none, None, NAME, url),
            unlisted(json!({ "serverName": NAME }), true)
        );
    }

    #[test]
    fn the_users_own_list_adds_to_the_organizations_unless_it_keeps_the_list_to_its_own() {
        let url = "http://127.0.0.1:8081/mcp";
        let soft = json!({ "allowedMcpServers": [{ "serverName": "github" }] });
        let own = mine(json!({ "allowedMcpServers": [{ "serverName": NAME }] }));
        assert_eq!(
            server(&[theirs(soft.clone()), own.clone()], None, NAME, url),
            Server::Loads
        );
        let mut locked = soft;
        locked["allowManagedMcpServersOnly"] = json!(true);
        assert_eq!(
            server(&[theirs(locked.clone()), own.clone()], None, NAME, url),
            Server::Blocked {
                by: PathBuf::from("managed-settings.json"),
                managed: true,
                why: Why::Unlisted {
                    entry: json!({ "serverName": NAME }).to_string(),
                    open: false
                },
            }
        );
        // The lock in a user's file is not the organization's, and does nothing.
        let mut own_lock = own.clone();
        own_lock.settings["allowManagedMcpServersOnly"] = json!(true);
        assert_eq!(
            server(
                &[
                    theirs(json!({ "allowedMcpServers": [{ "serverName": "github" }] })),
                    own_lock
                ],
                None,
                NAME,
                url
            ),
            Server::Loads
        );
        // A list of the user's alone is a list all the same, and theirs to change.
        let only_mine = [mine(
            json!({ "allowedMcpServers": [{ "serverName": "github" }] }),
        )];
        assert_eq!(
            server(&only_mine, None, NAME, url),
            Server::Blocked {
                by: PathBuf::from("settings.json"),
                managed: false,
                why: Why::Unlisted {
                    entry: json!({ "serverName": NAME }).to_string(),
                    open: true
                },
            }
        );
    }

    #[test]
    fn a_denylist_entry_in_any_file_blocks_over_any_allowlist() {
        let url = "https://staging.example.com/mcp";
        let allowed = json!({ "allowedMcpServers": [{ "serverUrl": "https://*.example.com/*" }] });
        assert_eq!(
            server(&[theirs(allowed.clone())], None, NAME, url),
            Server::Loads
        );
        let mut denied = allowed.clone();
        denied["deniedMcpServers"] = json!([{ "serverUrl": "https://staging.example.com/*" }]);
        let blocked = Server::Blocked {
            by: PathBuf::from("managed-settings.json"),
            managed: true,
            why: Why::Denied,
        };
        assert_eq!(server(&[theirs(denied.clone())], None, NAME, url), blocked);
        // The vendor's own example: that entry is for port 443 alone.
        assert_eq!(
            server(
                &[theirs(denied)],
                None,
                NAME,
                "https://staging.example.com:8443/mcp"
            ),
            Server::Loads
        );
        // By name, and from the user's own file, with the lock on.
        let own = mine(json!({ "deniedMcpServers": [{ "serverName": NAME }] }));
        let mut locked = allowed;
        locked["allowManagedMcpServersOnly"] = json!(true);
        assert_eq!(
            server(&[theirs(locked), own], None, NAME, url),
            Server::Blocked {
                by: PathBuf::from("settings.json"),
                managed: false,
                why: Why::Denied,
            }
        );
        // A command entry is for a server that is started, never a remote one.
        let command = [theirs(json!({
            "deniedMcpServers": [{ "serverCommand": ["antumbra", "mcp", "bridge"] }]
        }))];
        assert_eq!(server(&command, None, NAME, url), Server::Loads);
    }

    #[test]
    fn a_fixed_set_or_a_plugin_only_lock_leaves_a_users_server_out() {
        let url = "https://memory.example/mcp";
        let file = |settings: Value| Source::managed(PathBuf::from("managed-mcp.json"), settings);
        let fixed =
            file(json!({ "mcpServers": { "github": { "type": "http", "url": "https://x" } } }));
        assert_eq!(
            server(&[], Some(&fixed), NAME, url),
            Server::Blocked {
                by: PathBuf::from("managed-mcp.json"),
                managed: true,
                why: Why::Fixed,
            }
        );
        // An empty set is no MCP at all.
        let empty = file(json!({ "mcpServers": {} }));
        assert!(matches!(
            server(&[], Some(&empty), NAME, url),
            Server::Blocked {
                why: Why::Fixed,
                ..
            }
        ));
        // The organization's own entry for it is the one that loads.
        let provided = file(json!({ "mcpServers": { NAME: { "type": "http", "url": url } } }));
        assert_eq!(server(&[], Some(&provided), NAME, url), Server::Loads);
        // A file that is not a set of servers fixes nothing.
        let other = file(json!({ "note": "nothing" }));
        assert_eq!(server(&[], Some(&other), NAME, url), Server::Loads);

        for (lock, locked) in [
            (json!(true), true),
            (json!(["skills", "mcp"]), true),
            (json!(["skills", "hooks"]), false),
            (json!(false), false),
        ] {
            let sources = [theirs(json!({ "strictPluginOnlyCustomization": lock }))];
            assert_eq!(
                matches!(
                    server(&sources, None, NAME, url),
                    Server::Blocked {
                        why: Why::PluginsOnly,
                        ..
                    }
                ),
                locked,
                "{lock}"
            );
        }
        // In a user's file the key is not read.
        let own = [mine(json!({ "strictPluginOnlyCustomization": true }))];
        assert_eq!(server(&own, None, NAME, url), Server::Loads);
    }

    #[test]
    fn the_users_hooks_run_unless_a_file_says_otherwise() {
        assert_eq!(hooks(&[]), Hooks::Run);
        assert_eq!(
            hooks(&[
                theirs(json!({ "allowManagedHooksOnly": false })),
                mine(json!({ "hooks": {} }))
            ]),
            Hooks::Run
        );
        let stopped = |key: &'static str, managed: bool| Hooks::Stopped {
            by: PathBuf::from(if managed {
                "managed-settings.json"
            } else {
                "settings.json"
            }),
            key,
            managed,
        };
        assert_eq!(
            hooks(&[theirs(json!({ "allowManagedHooksOnly": true }))]),
            stopped("allowManagedHooksOnly", true)
        );
        assert_eq!(
            hooks(&[theirs(json!({ "disableAllHooks": true }))]),
            stopped("disableAllHooks", true)
        );
        for (lock, locked) in [
            (json!(true), true),
            (json!(["hooks"]), true),
            (json!(["skills", "mcp"]), false),
        ] {
            assert_eq!(
                hooks(&[theirs(json!({ "strictPluginOnlyCustomization": lock }))]),
                if locked {
                    stopped("strictPluginOnlyCustomization", true)
                } else {
                    Hooks::Run
                },
                "{lock}"
            );
        }
        // The user's own switch stops their own hooks.
        assert_eq!(
            hooks(&[theirs(json!({})), mine(json!({ "disableAllHooks": true }))]),
            stopped("disableAllHooks", false)
        );
        // The organization's key in a user's file is not read.
        assert_eq!(
            hooks(&[mine(json!({ "allowManagedHooksOnly": true }))]),
            Hooks::Run
        );
    }

    #[test]
    fn a_later_file_of_the_organizations_stands_over_an_earlier_one() {
        let first = Source::managed(
            PathBuf::from("managed-settings.json"),
            json!({ "allowManagedHooksOnly": true, "allowManagedMcpServersOnly": true,
                    "allowedMcpServers": [{ "serverName": "github" }] }),
        );
        let later = Source::managed(
            PathBuf::from("managed-settings.d/20-relax.json"),
            json!({ "allowManagedHooksOnly": false, "allowManagedMcpServersOnly": false }),
        );
        let own = mine(json!({ "allowedMcpServers": [{ "serverName": NAME }] }));
        assert_eq!(hooks(&[first.clone(), later.clone()]), Hooks::Run);
        assert_eq!(
            server(
                &[first.clone(), later, own.clone()],
                None,
                NAME,
                "https://m.example/mcp"
            ),
            Server::Loads
        );
        assert!(matches!(
            server(&[first, own], None, NAME, "https://m.example/mcp"),
            Server::Blocked { .. }
        ));
    }
}
