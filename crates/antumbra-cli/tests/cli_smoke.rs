//! Drive the actual `antumbra` binary through its no-GPU commands. The bin
//! modules (`main` dispatch, `commands`, `ops`) are exercised end to end against
//! an ephemeral `mem://` store -- each command runs its handler, which is what the
//! library-level e2e test does not reach. GPU/`models` commands are not run here.

use std::process::{Command, Output};

use anyhow::Context;

fn cli(args: &[&str]) -> anyhow::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_antumbra"))
        .args(args)
        .output()
        .context("spawn the antumbra binary")
}

fn ok(args: &[&str]) -> anyhow::Result<String> {
    let out = cli(args)?;
    assert!(
        out.status.success(),
        "`antumbra {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn schema_prints_ddl() -> anyhow::Result<()> {
    assert!(
        ok(&["schema"])?.contains("DEFINE"),
        "schema prints generated DDL"
    );
    Ok(())
}

#[test]
fn no_gpu_commands_run_end_to_end() -> anyhow::Result<()> {
    // Each command opens its own ephemeral store; we assert the handler runs
    // (exit 0), which records coverage of the dispatch + command code.
    ok(&["--url", "mem://", "migrate"])?;
    ok(&["--url", "mem://", "--fake-embedder", "seed"])?;
    ok(&["--url", "mem://", "status"])?;
    ok(&["--url", "mem://", "experts"])?;
    ok(&["--url", "mem://", "loop", "--generations", "1", "--demo"])?;
    ok(&[
        "--url",
        "mem://",
        "remember",
        "--fake-embedder",
        "--tenant",
        "ws:t",
        "--user",
        "user:u",
        "--compartment",
        "comp:c",
        "--content",
        "deno install left-pad",
    ])?;
    // Routing with no experts escalates rather than errors.
    ok(&[
        "--url",
        "mem://",
        "--fake-embedder",
        "route",
        "reverse a string",
    ])?;
    Ok(())
}

#[test]
fn unknown_command_is_an_error() -> anyhow::Result<()> {
    assert!(!cli(&["definitely-not-a-command"])?.status.success());
    Ok(())
}

/// Without `--features models` there is no built-in embedder, so a command that
/// embeds must stop and say what to pass rather than silently recall by byte
/// histogram. (Under `models` the built-in candle BERT would load instead.)
#[cfg(not(feature = "models"))]
#[test]
fn embedding_commands_refuse_to_run_without_an_embedder() -> anyhow::Result<()> {
    let out = cli(&["--url", "mem://", "route", "reverse a string"])?;
    assert!(
        !out.status.success(),
        "route ran with no embedder configured"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--embedder-url") && stderr.contains("--fake-embedder"),
        "the refusal names both ways out: {stderr}"
    );
    Ok(())
}

/// `loop` is the scripted demo trainer; it must not run as if it were training.
#[test]
fn loop_requires_the_demo_acknowledgement() -> anyhow::Result<()> {
    let out = cli(&["--url", "mem://", "loop", "--generations", "1"])?;
    assert!(!out.status.success(), "loop ran without --demo");
    assert!(String::from_utf8_lossy(&out.stderr).contains("--demo"));
    Ok(())
}

/// `ingest` takes a file or a command's output and stores it as a document.
#[test]
fn ingest_stores_a_document_from_a_file_and_from_a_command() -> anyhow::Result<()> {
    let path = std::env::temp_dir().join(format!("antumbra-ingest-{}.txt", std::process::id()));
    std::fs::write(&path, "GET /health\nPOST /orders\n")?;
    let out = ok(&[
        "--url",
        "mem://",
        "--fake-embedder",
        "ingest",
        "--tenant",
        "ws:t",
        "--user",
        "user:u",
        "--title",
        "routes",
        "--no-git",
        "--file",
        path.to_str().context("the temp path is not UTF-8")?,
    ])?;
    assert!(out.contains("ingested 1 chunk"), "{out}");
    std::fs::remove_file(&path).ok();
    let out = ok(&[
        "--url",
        "mem://",
        "--fake-embedder",
        "ingest",
        "--tenant",
        "ws:t",
        "--user",
        "user:u",
        "--title",
        "toolchain",
        "--",
        "cargo",
        "--version",
    ])?;
    assert!(out.contains("ingested 1 chunk"), "{out}");
    // Neither a file nor a command is a usage error, not a silent empty document.
    let out = cli(&[
        "--url",
        "mem://",
        "--fake-embedder",
        "ingest",
        "--tenant",
        "ws:t",
        "--user",
        "user:u",
        "--title",
        "x",
    ])?;
    assert!(!out.status.success());
    Ok(())
}

/// `git-facts --dry-run` derives facts from this repository's own history and
/// stores nothing (so it needs no embedder and no store).
#[test]
fn git_facts_dry_run_reads_this_repository() -> anyhow::Result<()> {
    let out = ok(&[
        "--url",
        "mem://",
        "git-facts",
        "--tenant",
        "ws:t",
        "--user",
        "user:u",
        "--compartment",
        "comp:c",
        "--days",
        "3650",
        "--dry-run",
    ])?;
    assert!(out.contains("dry run"), "{out}");
    Ok(())
}

/// Retiring demotes and reviving restores, through a store that
/// outlives each command. Nothing is deleted along the way.
#[test]
fn retire_demotes_and_revive_restores() -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join(format!("antumbra-retire-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).context("a temp dir")?;
    let url = format!("surrealkv://{}", dir.join("store.skv").display());
    let url = url.as_str();
    ok(&["--url", url, "migrate"])?;
    ok(&["--url", url, "--fake-embedder", "seed"])?;
    let status_of = |name: &str| -> anyhow::Result<String> {
        Ok(ok(&["--url", url, "experts"])?
            .lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or_default()
            .to_string())
    };
    assert_eq!(status_of("arith-specialist")?, "active");

    let said = ok(&["--url", url, "retire", "--expert", "arith-specialist"])?;
    assert!(said.contains("active -> dormant"), "{said}");
    assert_eq!(status_of("arith-specialist")?, "dormant");
    assert_eq!(status_of("string-specialist")?, "active");

    ok(&[
        "--url",
        url,
        "retire",
        "--expert",
        "arith-specialist",
        "--archive",
    ])?;
    assert_eq!(status_of("arith-specialist")?, "archived");
    let said = ok(&["--url", url, "revive", "--expert", "arith-specialist"])?;
    assert!(said.contains("archived -> active"), "{said}");
    assert_eq!(status_of("arith-specialist")?, "active");

    // Reviving an active expert is not a move, and says so.
    let refused = cli(&["--url", url, "revive", "--expert", "arith-specialist"])?;
    assert!(!refused.status.success());
    assert!(!cli(&["--url", url, "retire", "--expert", "nobody"])?
        .status
        .success());
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
