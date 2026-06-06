//! Drive the actual `antumbra` binary through its no-GPU commands. The bin
//! modules (`main` dispatch, `commands`, `ops`) are exercised end to end against
//! an ephemeral `mem://` store -- each command runs its handler, which is what the
//! library-level e2e test does not reach. GPU/`models` commands are not run here.

use std::process::{Command, Output};

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_antumbra"))
        .args(args)
        .output()
        .expect("spawn the antumbra binary")
}

fn ok(args: &[&str]) -> String {
    let out = cli(args);
    assert!(
        out.status.success(),
        "`antumbra {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn schema_prints_ddl() {
    assert!(
        ok(&["schema"]).contains("DEFINE"),
        "schema prints generated DDL"
    );
}

#[test]
fn no_gpu_commands_run_end_to_end() {
    // Each command opens its own ephemeral store; we assert the handler runs
    // (exit 0), which records coverage of the dispatch + command code.
    ok(&["--url", "mem://", "migrate"]);
    ok(&["--url", "mem://", "seed"]);
    ok(&["--url", "mem://", "status"]);
    ok(&["--url", "mem://", "experts"]);
    ok(&["--url", "mem://", "loop", "--generations", "1"]);
    ok(&[
        "--url",
        "mem://",
        "remember",
        "--tenant",
        "ws:t",
        "--user",
        "user:u",
        "--compartment",
        "comp:c",
        "--content",
        "deno install left-pad",
    ]);
    // Routing with no experts escalates rather than errors.
    ok(&["--url", "mem://", "route", "reverse a string"]);
}

#[test]
fn unknown_command_is_an_error() {
    assert!(!cli(&["definitely-not-a-command"]).status.success());
}
