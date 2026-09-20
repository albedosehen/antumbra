//! Drive the actual `antumbra claude ...` commands, with the machine they read
//! pinned down: an empty home, a project made for the test, and every variable
//! that decides sovereign mode removed or set on purpose.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Every variable the detector reads. A developer's own shell may have any of
/// them set, and the test must not depend on that.
const DECIDING: [&str; 9] = [
    "DISABLE_TELEMETRY",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "DO_NOT_TRACK",
    "DISABLE_GROWTHBOOK",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST",
];

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> std::io::Result<Self> {
        let dir = std::env::temp_dir().join(format!("antumbra-{name}-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("home"))?;
        std::fs::create_dir_all(dir.join("project"))?;
        Ok(Self(dir))
    }
    fn home(&self) -> PathBuf {
        self.0.join("home")
    }
    fn project(&self) -> PathBuf {
        self.0.join("project")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover directory under the temp dir fails no test.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn antumbra(home: &Path, set: &[(&str, &str)], args: &[&str]) -> std::io::Result<Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_antumbra"));
    for name in DECIDING {
        command.env_remove(name);
    }
    command
        .env("HOME", home)
        .env("USERPROFILE", home)
        .envs(set.iter().copied())
        .args(args)
        .output()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn brief_is_silent_outside_sovereign_mode_and_speaks_inside_it() -> anyhow::Result<()> {
    let scratch = Scratch::new("brief")?;
    std::fs::write(scratch.project().join("AGENTS.md"), "# Rules\n")?;
    let project = scratch.project();
    let Some(dir) = project.to_str() else {
        anyhow::bail!("the temp dir is not UTF-8");
    };
    let args = ["claude", "brief", "--dir", dir];

    let quiet = antumbra(&scratch.home(), &[], &args)?;
    assert!(quiet.status.success(), "{}", text(&quiet.stderr));
    assert_eq!(text(&quiet.stdout).trim(), "");

    // `0` counts: the vendor's table says any non-empty value turns the flags off.
    let loud = antumbra(&scratch.home(), &[("DISABLE_TELEMETRY", "0")], &args)?;
    assert!(loud.status.success(), "{}", text(&loud.stderr));
    let said = text(&loud.stdout);
    assert!(said.contains("Sovereign mode"), "{said}");
    assert!(said.contains("AGENTS.md"), "{said}");
    assert!(said.contains("Read it now"), "{said}");
    Ok(())
}

#[test]
fn remember_says_why_it_failed_and_never_shows_the_token() -> anyhow::Result<()> {
    let scratch = Scratch::new("remember")?;
    let token = "a-token-that-must-not-appear-anywhere";
    let out = antumbra(
        &scratch.home(),
        &[("ANTUMBRA_TOKEN", token)],
        &[
            "claude",
            "remember",
            "--surface",
            "http://127.0.0.1:1",
            "--dry-run",
        ],
    )?;
    assert!(!out.status.success());
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(stderr.contains("could not reach"), "{stderr}");
    assert!(!stdout.contains(token) && !stderr.contains(token));

    let help = antumbra(
        &scratch.home(),
        &[("ANTUMBRA_TOKEN", token)],
        &["claude", "remember", "--help"],
    )?;
    assert!(
        !text(&help.stdout).contains(token),
        "help must not print the token"
    );
    Ok(())
}

fn fixture(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
        .display()
        .to_string()
}

/// The fake server's command line: the PowerShell sibling on Windows, where
/// `bash` may be anything or nothing, and the POSIX one everywhere else.
fn fake_server() -> Vec<String> {
    if cfg!(windows) {
        vec![
            "pwsh".to_string(),
            "-NoProfile".to_string(),
            "-File".to_string(),
            fixture("fake_mcp_server.ps1"),
        ]
    } else {
        vec!["bash".to_string(), fixture("fake_mcp_server.sh")]
    }
}

fn assert_the_fixture_report(said: &str) {
    assert!(said.contains("6 tool(s) checked"), "{said}");
    assert!(said.contains("[FAIL] rootless"), "{said}");
    assert!(!said.contains("mcp__my_server__rootless"), "{said}");
    assert!(said.contains("[FAIL] breaks_names"), "{said}");
    assert!(said.contains("[FAIL] breaks_schema"), "{said}");
    assert!(said.contains("[note] dropped"), "{said}");
    assert!(said.contains("[note] dated"), "{said}");
    assert!(!said.contains("] sound"), "{said}");
    assert!(
        said.contains("\"mcp__my_server__breaks_names\", \"mcp__my_server__breaks_schema\""),
        "{said}"
    );
    assert!(!said.contains("mcp__my_server__dropped"), "{said}");
}

#[test]
fn mcp_lint_asks_a_live_server_and_fails_on_what_breaks_requests() -> anyhow::Result<()> {
    let scratch = Scratch::new("lint-live")?;
    let server = fake_server();
    let mut args = vec!["claude", "mcp-lint", "--server", "my server", "--"];
    args.extend(server.iter().map(String::as_str));
    let out = antumbra(&scratch.home(), &[], &args)?;
    assert!(!out.status.success(), "three tools fail");
    assert_the_fixture_report(&text(&out.stdout));
    assert!(
        text(&out.stderr).contains("3 tool(s)"),
        "{}",
        text(&out.stderr)
    );
    Ok(())
}

#[test]
fn mcp_lint_reads_a_saved_answer_and_passes_a_clean_one() -> anyhow::Result<()> {
    let scratch = Scratch::new("lint-file")?;
    let saved = fixture("fake_tools.json");
    let out = antumbra(
        &scratch.home(),
        &[],
        &[
            "claude",
            "mcp-lint",
            "--server",
            "my server",
            "--from",
            &saved,
        ],
    )?;
    assert!(!out.status.success());
    assert_the_fixture_report(&text(&out.stdout));

    let clean = scratch.project().join("clean.json");
    std::fs::write(
        &clean,
        r#"{"result":{"tools":[{"name":"ok","inputSchema":{"type":"object"}}]}}"#,
    )?;
    let Some(clean) = clean.to_str() else {
        anyhow::bail!("the temp dir is not UTF-8");
    };
    let out = antumbra(
        &scratch.home(),
        &[],
        &["claude", "mcp-lint", "--server", "s", "--from", clean],
    )?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains("nothing the API would refuse"));
    Ok(())
}

#[test]
fn mcp_lint_gives_up_on_a_server_that_never_answers() -> anyhow::Result<()> {
    let scratch = Scratch::new("lint-silent")?;
    // A process that reads its input and says nothing, on either platform.
    let silent: &[&str] = if cfg!(windows) {
        &[
            "pwsh",
            "-NoProfile",
            "-Command",
            "[Console]::In.ReadToEnd() | Out-Null",
        ]
    } else {
        &["cat"]
    };
    let mut args = vec![
        "claude",
        "mcp-lint",
        "--server",
        "s",
        "--timeout-secs",
        "2",
        "--",
    ];
    args.extend(silent);
    let started = std::time::Instant::now();
    let out = antumbra(&scratch.home(), &[], &args)?;
    assert!(!out.status.success());
    assert!(
        text(&out.stderr).contains("said nothing for 2 seconds"),
        "{}",
        text(&out.stderr)
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(30));
    Ok(())
}
