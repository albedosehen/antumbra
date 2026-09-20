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
