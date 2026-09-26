//! Sovereign mode for Claude Code (ADR-0021): what the agent loses when its
//! telemetry is turned off, and what is done about each loss.
//!
//! `DISABLE_TELEMETRY` and its siblings also turn off feature-flag fetching, and
//! a list of features that have nothing to do with telemetry is gated on those
//! flags. Nothing announces it. This module decides whether a session is in that
//! state and why ([`detect`]), holds the list of what goes with it ([`rules`]),
//! and judges each entry against the machine it runs on ([`examine`]).
//!
//! Everything here is a function over plain data. The file system and the
//! process environment are read once, at the edge ([`Inputs::gather`]), so the
//! judgment itself is table-tested with no disk and no environment.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub mod apply;
pub mod auto_mode;
pub mod bridge;
pub mod brief;
pub mod conventions;
pub mod hooks;
pub mod mcp_lint;
pub mod mcp_stdio;
pub mod reanchor;
pub mod run;
pub mod skills;
pub mod version;

/// The Claude Code release the rules below were verified against, the day, and
/// the page that says so. The gated list changes between releases, so a rule is
/// only as good as its last check; the report says when the installed version
/// differs.
pub const VERIFIED_AGAINST: &str = "2.1.283";
pub const VERIFIED_ON: &str = "2026-09-26";
pub const SOURCE: &str =
    "https://code.claude.com/docs/en/env-vars#features-that-need-feature-flag-fetching";

/// Where a variable's value was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    /// The process environment.
    Environment,
    /// The `env` block of one of the agent's settings files.
    Settings(PathBuf),
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Origin::Environment => write!(f, "the environment"),
            Origin::Settings(path) => write!(f, "{}", path.display()),
        }
    }
}

/// One variable as the agent will see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable {
    pub value: String,
    pub origin: Origin,
}

/// The operating system, as far as the rules care: the PowerShell tool is a
/// Windows concern (the host's dominant shell decides, ADR-0021 principle 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Windows,
    Other,
}

impl Os {
    pub fn current() -> Self {
        if cfg!(windows) {
            Os::Windows
        } else {
            Os::Other
        }
    }
}

/// Everything the judgment needs, gathered once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inputs {
    pub os: Os,
    /// Settings `env` blocks and the environment, merged: the first source to
    /// name a variable keeps it, in the order user, project, project-local,
    /// environment.
    pub variables: BTreeMap<String, Variable>,
    /// `permissions.defaultMode` from the USER's settings. The agent ignores
    /// `auto` there anywhere else.
    pub user_default_mode: Option<String>,
    /// Whether an `AGENTS.md` here is unread, bridged, or not in question.
    pub instructions: bridge::Instructions,
    /// The installed agent's version, when it could be asked.
    pub installed_version: Option<String>,
}

/// Why feature-flag fetching is off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trigger {
    pub variable: String,
    pub origin: Origin,
}

/// Set to any non-empty value, `0` and `false` included, these turn the flags
/// off. The vendor's table is explicit that they differ from its other switches.
const PRESENCE_TRIGGERS: [&str; 2] = [
    "DISABLE_TELEMETRY",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
];

/// Ordinary booleans: `0` and `false` leave the flags on.
const BOOLEAN_TRIGGERS: [&str; 2] = ["DO_NOT_TRACK", "DISABLE_GROWTHBOOK"];

/// A third-party provider skips the flag fetch, unless the host platform has
/// declared that it manages the provider.
const PROVIDER_SWITCHES: [&str; 4] = [
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
];
const PROVIDER_MANAGED_BY_HOST: &str = "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST";

fn is_true(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Every reason the session is in sovereign mode; empty means it is not.
pub fn detect(variables: &BTreeMap<String, Variable>) -> Vec<Trigger> {
    let named = |name: &str, counts: &dyn Fn(&str) -> bool| {
        variables
            .get(name)
            .filter(|v| counts(&v.value))
            .map(|v| Trigger {
                variable: name.to_string(),
                origin: v.origin.clone(),
            })
    };
    let host_managed = variables
        .get(PROVIDER_MANAGED_BY_HOST)
        .is_some_and(|v| !v.value.is_empty());
    let presence = PRESENCE_TRIGGERS
        .iter()
        .filter_map(|name| named(name, &|value| !value.is_empty()));
    let boolean = BOOLEAN_TRIGGERS
        .iter()
        .filter_map(|name| named(name, &is_true));
    let provider = PROVIDER_SWITCHES
        .iter()
        .filter(|_| !host_managed)
        .filter_map(|name| named(name, &is_true));
    presence.chain(boolean).chain(provider).collect()
}

/// What is done about a loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Antumbra supplies it.
    Restored,
    /// A local setting brings it back. `required` settings fail the doctor.
    Setting { required: bool },
    /// It lives on the vendor's side and stays lost.
    AcceptedLoss,
}

/// A variable in the agent's `env` block, and the value that settles a rule.
/// These are the only names `antumbra claude apply` will ever write, and they
/// come from the matrix below, never from an argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvSetting {
    pub name: &'static str,
    pub value: &'static str,
}

/// One feature that goes when the flags go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rule {
    pub id: &'static str,
    pub lost: &'static str,
    pub class: Class,
    /// What Antumbra does, or why nothing is done.
    pub response: &'static str,
    /// How an agent is told it cannot use this, in a list of such things. `None`
    /// when an agent never reaches for it, or when its standing decides what is
    /// said (see [`brief`]).
    pub unavailable: Option<&'static str>,
    /// The `env` entry that brings it back, when one does. `None` for a rule no
    /// environment variable settles, including the default permission mode:
    /// that lives under `permissions`, which nothing here writes.
    pub env: Option<EnvSetting>,
    /// The first release in which the loss is gone. `None` while it lasts.
    pub until: Option<&'static str>,
}

/// The matrix of ADR-0021, in the order the report prints it.
pub fn rules() -> Vec<Rule> {
    use Class::{AcceptedLoss, Restored, Setting};
    vec![
        Rule {
            id: "agents-md",
            lost: "AGENTS.md is no longer read as project instructions",
            class: Restored,
            response: "`antumbra claude bridge` writes an untracked CLAUDE.local.md that imports it, so it is read natively again",
            unavailable: None,
            env: None,
            until: Some("2.1.281"),
        },
        Rule {
            id: "mcp-schemas",
            lost: "MCP tools whose input schema the API rejects are no longer excluded, so one bad tool fails every request it is sent with",
            class: Restored,
            response: "`antumbra claude mcp-lint`, run outside the agent, names each such tool and prints the deny rule that keeps it out of the request",
            unavailable: None,
            env: None,
            until: None,
        },
        Rule {
            id: "mcp-root-combinators",
            lost: "MCP tools whose input schema has anyOf, oneOf or allOf at its root are skipped instead of rewritten, and only a debug log says so (not on the vendor's list; measured)",
            class: AcceptedLoss,
            response: "`antumbra claude mcp-lint` names them; only the server can fix it, by flattening the schema",
            unavailable: None,
            env: None,
            until: None,
        },
        Rule {
            id: "powershell-tool",
            lost: "the PowerShell tool is off on Windows when Git Bash is installed",
            class: Setting { required: true },
            response: "set CLAUDE_CODE_USE_POWERSHELL_TOOL=1",
            unavailable: None,
            env: Some(EnvSetting { name: "CLAUDE_CODE_USE_POWERSHELL_TOOL", value: "1" }),
            until: None,
        },
        Rule {
            id: "mcp-protocol-probe",
            lost: "claude.ai connector servers are not probed for MCP protocol 2026-07-28",
            class: Setting { required: false },
            response: "set MCP_PROTOCOL_NEGOTIATION=auto",
            unavailable: None,
            env: Some(EnvSetting { name: "MCP_PROTOCOL_NEGOTIATION", value: "auto" }),
            until: None,
        },
        Rule {
            id: "auto-mode-default",
            lost: "sessions no longer start in auto mode by default",
            class: Setting { required: false },
            response: "set permissions.defaultMode to \"auto\" in your own settings file (it is ignored in a project's)",
            unavailable: None,
            env: None,
            until: Some("2.1.283"),
        },
        Rule {
            id: "auto-mode-setup",
            lost: "/auto-mode-setup cannot draft autoMode.environment entries",
            class: Restored,
            response: "`antumbra claude auto-mode-env` drafts them from your working trees' remotes and Antumbra's memories, reads no transcript, and prints the block without writing it",
            unavailable: Some("/auto-mode-setup (`antumbra claude auto-mode-env` drafts the same entries, from outside the agent)"),
            env: None,
            until: None,
        },
        Rule {
            id: "skill-doctor",
            lost: "/skill-doctor cannot report unused skills, nor the /plugin Stats tab show its report",
            class: Restored,
            response: "`antumbra claude skill-used`, run by two hooks, counts each use as a reinforcement of one volatile memory per skill, and `antumbra claude skills` reports the ones never or no longer used",
            unavailable: Some("/skill-doctor (`antumbra claude skills` reports the same, from outside the agent)"),
            env: None,
            until: None,
        },
        Rule {
            id: "remote-control",
            lost: "Remote Control, and messaging sessions on other machines",
            class: Restored,
            response: "in part and later: an asynchronous handoff compartment, no live control",
            unavailable: Some("Remote Control and messaging sessions on other machines"),
            env: None,
            until: None,
        },
        Rule {
            id: "account-sync",
            lost: "skills and plugins enabled on the hosted account no longer sync",
            class: AcceptedLoss,
            response: "off is the sovereign default",
            unavailable: None,
            env: None,
            until: None,
        },
        Rule {
            id: "vscode-starting-mode",
            lost: "the VS Code extension ignores every settings file for its starting permission mode",
            class: AcceptedLoss,
            response: "out of Antumbra's reach",
            unavailable: None,
            env: None,
            until: Some("2.1.283"),
        },
        Rule {
            id: "advisor",
            lost: "the advisor tool",
            class: AcceptedLoss,
            response: "it sends the whole conversation to a stronger model on the vendor's infrastructure, the opposite of `route` and `answer`",
            unavailable: Some("the advisor tool"),
            env: None,
            until: None,
        },
        Rule {
            id: "artifact-comments",
            lost: "reading and replying to comments on hosted artifacts",
            class: AcceptedLoss,
            response: "read them in the browser; ADR-0020 makes a comment a memory",
            unavailable: Some("comments on hosted artifacts (the user reads them in the browser)"),
            env: None,
            until: None,
        },
        Rule {
            id: "drafted-feedback",
            lost: "vendor-bound drafted feedback",
            class: AcceptedLoss,
            response: "friction is kept locally as `bank` memories by the capture hook",
            unavailable: Some("vendor-bound drafted feedback (keep friction as a `bank` memory instead)"),
            env: None,
            until: None,
        },
        Rule {
            id: "pasted-text",
            lost: "a large paste reaches Claude as typed text: what sits behind a [Pasted text #N] placeholder is not marked as pasted",
            class: AcceptedLoss,
            response: "out of Antumbra's reach",
            unavailable: None,
            env: None,
            until: None,
        },
        Rule {
            id: "import",
            lost: "`claude import`",
            class: AcceptedLoss,
            response: "a one-time migration from other agents; nothing to compensate",
            unavailable: None,
            env: None,
            until: None,
        },
    ]
}

/// Where one rule stands on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Standing {
    /// Nothing to do: the setting is in place, or the rule does not apply here.
    Fine(String),
    /// A setting is missing. `required` ones fail the doctor.
    Missing { required: bool, fix: String },
    /// Antumbra's answer is not built yet, or needs the user's attention now.
    Attention(String),
    /// Lost, and staying lost.
    Lost,
}

/// One line of the report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub rule: Rule,
    pub standing: Standing,
}

/// The whole judgment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub triggers: Vec<Trigger>,
    pub findings: Vec<Finding>,
    /// The installed version, when it differs from [`VERIFIED_AGAINST`].
    pub unverified_version: Option<String>,
}

impl Report {
    /// How many required settings are missing: the doctor's exit status.
    pub fn required_missing(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| matches!(f.standing, Standing::Missing { required: true, .. }))
            .count()
    }
}

fn is_set_to(variables: &BTreeMap<String, Variable>, name: &str, want: &str) -> bool {
    variables
        .get(name)
        .is_some_and(|v| v.value.trim().eq_ignore_ascii_case(want))
}

fn standing_of(rule: &Rule, inputs: &Inputs) -> Standing {
    match (rule.id, rule.class) {
        ("powershell-tool", Class::Setting { required }) => match inputs.os {
            Os::Other => Standing::Fine("not Windows: the host's shell is not PowerShell".into()),
            Os::Windows if is_set_to(&inputs.variables, "CLAUDE_CODE_USE_POWERSHELL_TOOL", "1") => {
                Standing::Fine("CLAUDE_CODE_USE_POWERSHELL_TOOL=1".into())
            }
            Os::Windows => Standing::Missing {
                required,
                fix: "add \"CLAUDE_CODE_USE_POWERSHELL_TOOL\": \"1\" to the env block of your settings file".into(),
            },
        },
        ("mcp-protocol-probe", Class::Setting { required }) => {
            if is_set_to(&inputs.variables, "MCP_PROTOCOL_NEGOTIATION", "auto") {
                Standing::Fine("MCP_PROTOCOL_NEGOTIATION=auto".into())
            } else {
                Standing::Missing {
                    required,
                    fix: "add \"MCP_PROTOCOL_NEGOTIATION\": \"auto\" to the env block, if you use claude.ai connectors".into(),
                }
            }
        }
        ("auto-mode-default", Class::Setting { required }) => {
            if inputs.user_default_mode.as_deref() == Some("auto") {
                Standing::Fine("permissions.defaultMode is \"auto\"".into())
            } else {
                Standing::Missing {
                    required,
                    fix: "set permissions.defaultMode to \"auto\" in your own settings file, if you want auto mode".into(),
                }
            }
        }
        ("remote-control", _) if version::remote_control_spared(inputs) => Standing::Fine(
            "available: only a telemetry switch turned the flags off, which it survives unless your organization requires Trusted Devices".into(),
        ),
        ("agents-md", _) => match &inputs.instructions {
            bridge::Instructions::Unread(path) => Standing::Attention(format!(
                "{} is not being read. Run `antumbra claude bridge` here: it writes an untracked CLAUDE.local.md that imports it",
                path.display()
            )),
            bridge::Instructions::Bridged(path) => {
                Standing::Fine(format!("imported natively by {}", path.display()))
            }
            bridge::Instructions::NotApplicable => {
                Standing::Fine("no AGENTS.md here depends on it".into())
            }
        },
        (_, Class::Restored) => Standing::Attention(rule.response.to_string()),
        (_, Class::AcceptedLoss) => Standing::Lost,
        (_, Class::Setting { required }) => Standing::Missing {
            required,
            fix: rule.response.to_string(),
        },
    }
}

/// Judge every rule against this machine. Outside sovereign mode nothing is
/// lost, so there is nothing to find.
pub fn examine(inputs: &Inputs) -> Report {
    let triggers = detect(&inputs.variables);
    let findings = if triggers.is_empty() {
        Vec::new()
    } else {
        rules()
            .into_iter()
            .filter(|rule| {
                rule.until
                    .is_none_or(|ended| !version::at_least(version::judged_for(inputs), ended))
            })
            .map(|rule| Finding {
                standing: standing_of(&rule, inputs),
                rule,
            })
            .collect()
    };
    let unverified_version = inputs
        .installed_version
        .clone()
        .filter(|installed| installed != VERIFIED_AGAINST);
    Report {
        triggers,
        findings,
        unverified_version,
    }
}

/// What one settings file contributes: its `env` block and its default mode.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Settings {
    pub env: BTreeMap<String, String>,
    pub default_mode: Option<String>,
}

/// Read what matters out of a settings file's text. A file that does not parse
/// contributes nothing: the agent itself would have refused it.
pub fn parse_settings(text: &str) -> Settings {
    let Ok(json) = serde_json::from_str::<serde_json::Value>(text) else {
        return Settings::default();
    };
    let env = json
        .get("env")
        .and_then(serde_json::Value::as_object)
        .map(|block| {
            block
                .iter()
                .filter_map(|(name, value)| value.as_str().map(|v| (name.clone(), v.to_string())))
                .collect()
        })
        .unwrap_or_default();
    let default_mode = json
        .pointer("/permissions/defaultMode")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Settings { env, default_mode }
}

/// Merge the sources in the order given: the first to name a variable keeps it.
pub fn merge(sources: Vec<(Origin, BTreeMap<String, String>)>) -> BTreeMap<String, Variable> {
    sources
        .into_iter()
        .fold(BTreeMap::new(), |mut merged, (origin, values)| {
            for (name, value) in values {
                merged.entry(name).or_insert(Variable {
                    value,
                    origin: origin.clone(),
                });
            }
            merged
        })
}

impl Inputs {
    /// Read the environment, the settings files and the working directory, and
    /// ask the installed agent for its version.
    pub fn gather(home: Option<&Path>, project: &Path) -> Self {
        Inputs {
            installed_version: installed_version(),
            ..Self::gather_without_version(home, project)
        }
    }

    /// As [`Inputs::gather`], without asking the agent for its version. Asking
    /// starts the agent's runtime, which a session-start hook has no time for.
    pub fn gather_without_version(home: Option<&Path>, project: &Path) -> Self {
        let read = |path: PathBuf| {
            std::fs::read_to_string(&path)
                .ok()
                .map(|text| (path, parse_settings(&text)))
        };
        let user = home.and_then(|home| read(home.join(".claude").join("settings.json")));
        let project_files = [
            project.join(".claude").join("settings.json"),
            project.join(".claude").join("settings.local.json"),
        ]
        .into_iter()
        .filter_map(read);

        let user_default_mode = user.as_ref().and_then(|(_, s)| s.default_mode.clone());
        // Settings files first. The agent copies its settings' `env` block into
        // the environment, so a variable configured in a file shows up in both;
        // the file is where the user goes to change it, so the file gets the
        // credit. Only what no file names is attributed to the environment.
        let sources = user
            .into_iter()
            .chain(project_files)
            .map(|(path, settings)| (Origin::Settings(path), settings.env))
            .chain(std::iter::once((
                Origin::Environment,
                std::env::vars().collect(),
            )))
            .collect();
        Inputs {
            os: Os::current(),
            variables: merge(sources),
            user_default_mode,
            instructions: bridge::standing(&repository_root(project), project, &|path| {
                std::fs::read_to_string(path).ok()
            }),
            installed_version: None,
        }
    }
}

/// The top of the repository `project` sits in, or `project` itself outside one.
/// Nothing above it is ever read for instructions or written to.
pub fn repository_root(project: &Path) -> PathBuf {
    std::process::Command::new("git")
        .arg("-C")
        .arg(project)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
        .filter(|root| !root.as_os_str().is_empty())
        .unwrap_or_else(|| project.to_path_buf())
}

/// Ask the installed agent for its version. Best effort: no agent on the path,
/// or an answer in a shape this does not know, is simply "unknown".
fn installed_version() -> Option<String> {
    let output = if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/C", "claude", "--version"])
            .output()
    } else {
        std::process::Command::new("claude")
            .arg("--version")
            .output()
    }
    .ok()?;
    version_in(&String::from_utf8_lossy(&output.stdout))
}

/// The version number at the start of `claude --version` output, such as
/// `2.1.278 (Claude Code)`.
pub fn version_in(output: &str) -> Option<String> {
    output
        .split_whitespace()
        .next()
        .filter(|first| first.chars().next().is_some_and(|c| c.is_ascii_digit()))
        .map(str::to_string)
}

/// The report as text.
pub fn render(report: &Report) -> String {
    let mut out = Vec::new();
    if report.triggers.is_empty() {
        out.push(
            "Sovereign mode: off. Feature-flag fetching is on, so nothing below is lost."
                .to_string(),
        );
        return out.join("\n");
    }
    out.push("Sovereign mode: ON. Feature-flag fetching is off, because:".to_string());
    out.extend(
        report
            .triggers
            .iter()
            .map(|t| format!("  {} is set in {}", t.variable, t.origin)),
    );
    out.push(String::new());
    out.extend(report.findings.iter().map(|finding| {
        let (mark, detail) = match &finding.standing {
            Standing::Fine(why) => ("ok  ", why.clone()),
            Standing::Missing {
                required: true,
                fix,
            } => ("FAIL", fix.clone()),
            Standing::Missing {
                required: false,
                fix,
            } => ("hint", fix.clone()),
            Standing::Attention(what) => ("todo", what.clone()),
            Standing::Lost => ("lost", finding.rule.response.to_string()),
        };
        format!("[{mark}] {}\n       {detail}", finding.rule.lost)
    }));
    out.push(String::new());
    out.push(format!(
        "Verified against Claude Code {VERIFIED_AGAINST} on {VERIFIED_ON}: {SOURCE}"
    ));
    if let Some(installed) = &report.unverified_version {
        out.push(format!(
            "Installed is {installed}: the gated list changes between releases, so re-verify."
        ));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> BTreeMap<String, Variable> {
        pairs
            .iter()
            .map(|(name, value)| {
                (
                    name.to_string(),
                    Variable {
                        value: value.to_string(),
                        origin: Origin::Environment,
                    },
                )
            })
            .collect()
    }

    fn triggered(pairs: &[(&str, &str)]) -> Vec<String> {
        detect(&vars(pairs))
            .into_iter()
            .map(|t| t.variable)
            .collect()
    }

    fn inputs(os: Os, pairs: &[(&str, &str)]) -> Inputs {
        Inputs {
            os,
            variables: vars(pairs),
            user_default_mode: None,
            instructions: bridge::Instructions::NotApplicable,
            installed_version: Some(VERIFIED_AGAINST.to_string()),
        }
    }

    fn standing(report: &Report, id: &str) -> Option<Standing> {
        report
            .findings
            .iter()
            .find(|f| f.rule.id == id)
            .map(|f| f.standing.clone())
    }

    /// The vendor's table is explicit that these switches do not behave alike,
    /// and a detector that treats them alike is wrong in both directions.
    #[test]
    fn each_trigger_counts_exactly_the_values_the_vendor_says() {
        for (pairs, expected) in [
            // Presence: any non-empty value, `0` and `false` included.
            (vec![("DISABLE_TELEMETRY", "1")], vec!["DISABLE_TELEMETRY"]),
            (vec![("DISABLE_TELEMETRY", "0")], vec!["DISABLE_TELEMETRY"]),
            (
                vec![("DISABLE_TELEMETRY", "false")],
                vec!["DISABLE_TELEMETRY"],
            ),
            (vec![("DISABLE_TELEMETRY", "")], vec![]),
            (
                vec![("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "0")],
                vec!["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"],
            ),
            // Ordinary booleans: `0` and `false` leave the flags on.
            (vec![("DO_NOT_TRACK", "1")], vec!["DO_NOT_TRACK"]),
            (vec![("DO_NOT_TRACK", "0")], vec![]),
            (
                vec![("DISABLE_GROWTHBOOK", "true")],
                vec!["DISABLE_GROWTHBOOK"],
            ),
            (vec![("DISABLE_GROWTHBOOK", "false")], vec![]),
            // Error reporting alone has no such side effect.
            (vec![("DISABLE_ERROR_REPORTING", "1")], vec![]),
            // A third-party provider, unless the host manages it.
            (
                vec![("CLAUDE_CODE_USE_BEDROCK", "1")],
                vec!["CLAUDE_CODE_USE_BEDROCK"],
            ),
            (vec![("CLAUDE_CODE_USE_BEDROCK", "0")], vec![]),
            (
                vec![
                    ("CLAUDE_CODE_USE_VERTEX", "1"),
                    ("CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST", "1"),
                ],
                vec![],
            ),
            (vec![], vec![]),
        ] {
            assert_eq!(triggered(&pairs), expected, "{pairs:?}");
        }
    }

    #[test]
    fn outside_sovereign_mode_there_is_nothing_to_find() {
        let report = examine(&inputs(Os::Windows, &[("DISABLE_ERROR_REPORTING", "1")]));
        assert!(report.triggers.is_empty());
        assert!(report.findings.is_empty());
        assert_eq!(report.required_missing(), 0);
        assert!(render(&report).starts_with("Sovereign mode: off"));
    }

    /// The host's dominant shell decides: a missing PowerShell tool fails the
    /// doctor on Windows and is not a finding anywhere else.
    #[test]
    fn the_powershell_tool_is_required_on_windows_only() {
        let off = [("DISABLE_TELEMETRY", "1")];
        let on = [
            ("DISABLE_TELEMETRY", "1"),
            ("CLAUDE_CODE_USE_POWERSHELL_TOOL", "1"),
        ];
        let windows_without = examine(&inputs(Os::Windows, &off));
        assert_eq!(windows_without.required_missing(), 1);
        assert!(matches!(
            standing(&windows_without, "powershell-tool"),
            Some(Standing::Missing { required: true, .. })
        ));
        assert!(render(&windows_without).contains("[FAIL]"));

        assert_eq!(examine(&inputs(Os::Windows, &on)).required_missing(), 0);
        let elsewhere = examine(&inputs(Os::Other, &off));
        assert_eq!(elsewhere.required_missing(), 0);
        assert!(matches!(
            standing(&elsewhere, "powershell-tool"),
            Some(Standing::Fine(_))
        ));
    }

    #[test]
    fn advisory_settings_hint_and_never_fail() {
        // The default mode is a loss only before 2.1.283.
        let older = |i: Inputs| Inputs {
            installed_version: Some("2.1.278".into()),
            ..i
        };
        let bare = examine(&older(inputs(Os::Other, &[("DISABLE_TELEMETRY", "1")])));
        assert_eq!(bare.required_missing(), 0);
        for id in ["mcp-protocol-probe", "auto-mode-default"] {
            assert!(
                matches!(
                    standing(&bare, id),
                    Some(Standing::Missing {
                        required: false,
                        ..
                    })
                ),
                "{id}"
            );
        }
        let set = examine(&older(Inputs {
            user_default_mode: Some("auto".into()),
            ..inputs(
                Os::Other,
                &[
                    ("DISABLE_TELEMETRY", "1"),
                    ("MCP_PROTOCOL_NEGOTIATION", "auto"),
                ],
            )
        }));
        for id in ["mcp-protocol-probe", "auto-mode-default"] {
            assert!(
                matches!(standing(&set, id), Some(Standing::Fine(_))),
                "{id}"
            );
        }
    }

    #[test]
    fn an_unread_agents_md_is_called_out_by_path() -> Result<(), String> {
        // Read natively with telemetry off from 2.1.281; lost before it.
        let report = examine(&Inputs {
            instructions: bridge::Instructions::Unread(PathBuf::from("/work/repo/AGENTS.md")),
            installed_version: Some("2.1.278".into()),
            ..inputs(Os::Other, &[("DISABLE_TELEMETRY", "1")])
        });
        let Some(Standing::Attention(what)) = standing(&report, "agents-md") else {
            return Err(format!(
                "expected the file to be called out, got {:?}",
                standing(&report, "agents-md")
            ));
        };
        assert!(
            what.contains("AGENTS.md") && what.contains("not being read"),
            "{what}"
        );
        Ok(())
    }

    #[test]
    fn settings_contribute_their_env_block_and_default_mode() {
        let parsed = parse_settings(
            r#"{ "env": { "DISABLE_TELEMETRY": "1", "PORT": 8080 },
                 "permissions": { "defaultMode": "auto", "allow": ["Bash"] } }"#,
        );
        assert_eq!(
            parsed.env.get("DISABLE_TELEMETRY").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            parsed.env.get("PORT"),
            None,
            "a non-string value is not a variable"
        );
        assert_eq!(parsed.default_mode.as_deref(), Some("auto"));
        assert_eq!(parse_settings("not json"), Settings::default());
        assert_eq!(parse_settings("{}"), Settings::default());
    }

    #[test]
    fn the_first_source_to_name_a_variable_keeps_it() {
        let settings = PathBuf::from("/home/me/.claude/settings.json");
        let merged = merge(vec![
            (
                Origin::Environment,
                BTreeMap::from([("DISABLE_TELEMETRY".to_string(), "1".to_string())]),
            ),
            (
                Origin::Settings(settings.clone()),
                BTreeMap::from([
                    ("DISABLE_TELEMETRY".to_string(), "0".to_string()),
                    ("MCP_PROTOCOL_NEGOTIATION".to_string(), "auto".to_string()),
                ]),
            ),
        ]);
        assert_eq!(
            merged
                .get("DISABLE_TELEMETRY")
                .map(|v| (&v.origin, v.value.as_str())),
            Some((&Origin::Environment, "1"))
        );
        assert_eq!(
            merged.get("MCP_PROTOCOL_NEGOTIATION").map(|v| &v.origin),
            Some(&Origin::Settings(settings))
        );
    }

    /// The prose the doctor prints and the name `apply` writes must be the same
    /// name, or the doctor asks for one thing and the tool does another.
    #[test]
    fn what_the_doctor_asks_for_is_what_apply_would_write() -> anyhow::Result<()> {
        let report = examine(&inputs(Os::Windows, &[("DISABLE_TELEMETRY", "1")]));
        let mut settled = 0;
        for finding in &report.findings {
            let Some(env) = finding.rule.env else {
                continue;
            };
            settled += 1;
            let Standing::Missing { fix, .. } = &finding.standing else {
                anyhow::bail!("{} is not missing in a bare environment", finding.rule.id);
            };
            assert!(fix.contains(env.name), "{fix} does not name {}", env.name);
            assert!(fix.contains(env.value), "{fix} does not give {}", env.value);
        }
        assert_eq!(settled, 2, "the matrix has two env settings");
        // The default permission mode is asked for and deliberately not written.
        let mode = report
            .findings
            .iter()
            .find(|finding| finding.rule.id == "auto-mode-default");
        assert_eq!(mode.and_then(|finding| finding.rule.env), None);
        Ok(())
    }

    #[test]
    fn a_newer_installed_version_is_flagged_for_reverification() {
        assert_eq!(
            version_in("2.1.278 (Claude Code)\n").as_deref(),
            Some("2.1.278")
        );
        assert_eq!(version_in("command not found"), None);
        assert_eq!(version_in(""), None);

        let newer = examine(&Inputs {
            installed_version: Some("2.2.0".into()),
            ..inputs(Os::Other, &[("DISABLE_TELEMETRY", "1")])
        });
        assert_eq!(newer.unverified_version.as_deref(), Some("2.2.0"));
        assert!(render(&newer).contains("re-verify"));
        let same = examine(&inputs(Os::Other, &[("DISABLE_TELEMETRY", "1")]));
        assert_eq!(same.unverified_version, None);
    }

    #[test]
    fn every_rule_is_judged_and_the_report_names_why() {
        // On a release before any loss ended, every rule applies.
        let report = examine(&Inputs {
            installed_version: Some("2.1.278".into()),
            ..inputs(Os::Other, &[("DISABLE_TELEMETRY", "1")])
        });
        assert_eq!(report.findings.len(), rules().len());
        let text = render(&report);
        assert!(
            text.contains("DISABLE_TELEMETRY is set in the environment"),
            "{text}"
        );
        assert!(text.contains(VERIFIED_AGAINST) && text.contains(SOURCE));
        // Ids are what the conventions compartment will key on: unique.
        let mut ids: Vec<&str> = rules().iter().map(|r| r.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), rules().len());
    }
}
