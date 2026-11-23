//! `antumbra export`: your own memories, out of the server, in a file.
//!
//! Every memory that is yours on a server (those in a compartment you own,
//! and those you wrote into none) is read whole, a page at a time, and
//! written as JSON Lines: a first line that says what the file is, then a
//! line a memory, the newest first. Text, kind, confidence, whether it
//! changes over time, how often it was reinforced, its compartment, the
//! machine it was written on, when, and its git anchor.
//!
//! It is the server's own answer, so it is what recall would have drawn on
//! and nothing a tool call could not have read. What the server holds back
//! from sharing (the hooks' own records, anything that looks like a secret)
//! is in the file, since the file is the owner's: it is written readable by
//! its owner alone.
//!
//! What is not in it: the embeddings (a server makes its own), relations
//! between memories, documents, and experts.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Args;
use serde_json::{json, Value};

/// How many memories are read from the server at a time.
const PAGE: u64 = 100;

#[derive(Args, Debug)]
pub struct ExportArgs {
    /// The file to write. `-` writes to standard output. Left out:
    /// `antumbra-export-<date>.jsonl` in the current directory.
    #[arg(long, short)]
    pub out: Option<PathBuf>,
    /// Export what is yours in a workspace this machine joined, named by its
    /// id (`ws:...`) or its address, in place of this machine's own server.
    #[arg(long)]
    pub from: Option<String>,
    /// Replace the file when it is already there.
    #[arg(long)]
    pub force: bool,
}

/// One call of a tool on a server: its name and arguments, and its answer.
type Call<'a> = &'a dyn Fn(&str, Value) -> anyhow::Result<Value>;

/// Write every memory that is the caller's on the server `call` reaches to
/// `out`, after a first line made from `about`. How many were written.
fn export(call: Call<'_>, about: &Value, out: &mut dyn Write) -> anyhow::Result<u64> {
    let page = |offset: u64| {
        call(
            "bring_candidates",
            json!({
                "filter": { "state": "any", "include_held": true },
                "limit": PAGE,
                "offset": offset,
                "full": true,
            }),
        )
    };
    let first = page(0)?;
    let total = first.get("total").and_then(Value::as_u64).unwrap_or(0);
    let mut head = about.clone();
    head["antumbra_export"] = json!(1);
    head["memories"] = json!(total);
    writeln!(out, "{head}")?;
    let (mut written, mut answer) = (0u64, first);
    loop {
        let memories = answer
            .get("memories")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if memories.is_empty() {
            break;
        }
        for mut memory in memories {
            // What the page that chooses among them is told, and a file is not.
            if let Some(fields) = memory.as_object_mut() {
                fields.remove("truncated");
            }
            writeln!(out, "{memory}")?;
            written += 1;
        }
        if written >= total {
            break;
        }
        answer = page(written)?;
    }
    out.flush()?;
    Ok(written)
}

/// The file an export is written to when none is named.
fn default_file(today: &str) -> PathBuf {
    PathBuf::from(format!("antumbra-export-{today}.jsonl"))
}

/// Open `path` for an export: new, and its owner's alone to read.
fn create(path: &Path, force: bool) -> anyhow::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true);
    if force {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::AlreadyExists => anyhow::anyhow!(
            "{} is already there. Name another with --out, or replace it with --force",
            path.display()
        ),
        _ => anyhow::anyhow!("could not write {}: {e}", path.display()),
    })
}

pub fn run(args: ExportArgs) -> anyhow::Result<()> {
    let home = crate::machine::home().ok_or_else(|| anyhow::anyhow!("no home directory"))?;
    let host = antumbra_http::host_id();
    let patience = Duration::from_secs(120);
    let (url, workspace, token) = match &args.from {
        None => {
            let reach = crate::agents::Reach::read(&home, |name| std::env::var(name).ok())?;
            (reach.url.clone(), reach.workspace.clone(), reach.token()?)
        }
        Some(named) => {
            let joined = crate::bring::joined(&home.join(".antumbra"));
            let from = crate::bring::target(&joined, Some(named))?;
            let token = std::fs::read_to_string(&from.token_file)
                .map(|t| t.trim().to_string())
                .map_err(|e| {
                    anyhow::anyhow!(
                        "no token for {} in {}: {e}",
                        from.workspace,
                        from.token_file.display()
                    )
                })?;
            (from.url, Some(from.workspace), token)
        }
    };
    let client =
        antumbra_http::SurfaceClient::new(&url, Some(&token), patience).from_machine(host.clone());
    let call =
        |tool: &str, arguments: Value| client.call(tool, arguments).map_err(anyhow::Error::from);
    // Whose it is, as the token itself says: what a setup recorded may be
    // another workspace's, or nothing.
    let claims = antumbra_auth::unverified::claims(&token);
    let claim = |name: &str| {
        claims
            .as_ref()
            .and_then(|c| c.get(name))
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let about = json!({
        "exported_at": chrono::Utc::now().to_rfc3339(),
        "server": url,
        "workspace": claim("tenant").or(workspace),
        "user": claim("user"),
        "antumbra": env!("CARGO_PKG_VERSION"),
    });
    match args.out.as_deref() {
        Some(path) if path == Path::new("-") => {
            let stdout = std::io::stdout();
            let written = export(&call, &about, &mut stdout.lock())?;
            eprintln!("{written} memories written.");
        }
        named => {
            let path = named.map_or_else(
                || default_file(&chrono::Utc::now().format("%Y-%m-%d").to_string()),
                Path::to_path_buf,
            );
            let mut file = std::io::BufWriter::new(create(&path, args.force)?);
            let written = export(&call, &about, &mut file)?;
            println!(
                "{written} memories written to {}. The file is yours alone to read; it holds everything, what looks like a secret among it.",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A server holding `n` memories, answering a page at a time and noting
    /// what it was asked.
    fn server(
        n: usize,
    ) -> (
        impl Fn(&str, Value) -> anyhow::Result<Value>,
        RefCell<Vec<Value>>,
    ) {
        let asked = RefCell::new(Vec::new());
        let memories: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "id": format!("memory:{i}"), "content": format!("note {i}\nits second line"),
                    "truncated": false, "network": "bank", "confidence": 0.8, "volatile": false,
                    "reinforcement": i, "created_at": "2026-11-20T10:00:00+00:00",
                    "provenance": { "repo": "github.com/acme/billing", "commit": "3f9c2ab" },
                })
            })
            .collect();
        let asked_too = asked.clone();
        let call = move |tool: &str, arguments: Value| {
            anyhow::ensure!(tool == "bring_candidates", "asked for {tool}");
            asked_too.borrow_mut().push(arguments.clone());
            let (limit, offset) = (
                arguments["limit"].as_u64().unwrap_or(50) as usize,
                arguments["offset"].as_u64().unwrap_or(0) as usize,
            );
            let page: Vec<Value> = memories.iter().skip(offset).take(limit).cloned().collect();
            Ok(json!({ "total": memories.len(), "memories": page }))
        };
        (call, asked)
    }

    /// Every memory is written, whole, a line each after a first line that
    /// says what the file is; asked for a page at a time, in every state and
    /// with what is held back.
    #[test]
    fn every_memory_is_written_a_line_each() -> anyhow::Result<()> {
        let (call, _) = server(230);
        let mut out = Vec::new();
        let about = json!({ "server": "https://memory.example", "workspace": "ws:own" });
        assert_eq!(export(&call, &about, &mut out)?, 230);
        let text = String::from_utf8(out)?;
        let lines: Vec<Value> = text
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        assert_eq!(lines.len(), 231);
        assert_eq!(lines[0]["antumbra_export"], 1);
        assert_eq!(
            (lines[0]["memories"].clone(), lines[0]["workspace"].clone()),
            (json!(230), json!("ws:own"))
        );
        assert_eq!(lines[1]["id"], "memory:0");
        assert_eq!(lines[230]["id"], "memory:229");
        // Whole, on one line though it holds a line break, and with its anchor.
        assert_eq!(lines[1]["content"], "note 0\nits second line");
        assert_eq!(lines[1]["provenance"]["commit"], "3f9c2ab");
        assert!(lines[1].get("truncated").is_none());
        Ok(())
    }

    #[test]
    fn it_asks_for_everything_that_is_yours_a_page_at_a_time() -> anyhow::Result<()> {
        let asked = RefCell::new(Vec::new());
        let call = |_: &str, arguments: Value| {
            asked.borrow_mut().push(arguments.clone());
            let offset = arguments["offset"].as_u64().unwrap_or(0);
            let page: Vec<Value> = (offset..(offset + PAGE).min(230))
                .map(|i| json!({ "id": i }))
                .collect();
            Ok(json!({ "total": 230, "memories": page }))
        };
        export(&call, &json!({}), &mut Vec::new())?;
        let asked = asked.borrow();
        assert_eq!(
            asked
                .iter()
                .map(|a| a["offset"].as_u64().unwrap_or(9))
                .collect::<Vec<_>>(),
            [0, 100, 200]
        );
        for a in asked.iter() {
            assert_eq!(a["filter"], json!({ "state": "any", "include_held": true }));
            assert_eq!(
                (a["full"].clone(), a["limit"].clone()),
                (json!(true), json!(PAGE))
            );
        }
        Ok(())
    }

    /// An empty store is a file with its first line; and a server that says
    /// more than it gives does not keep the export asking for ever.
    #[test]
    fn an_empty_store_and_a_short_answer_both_end() -> anyhow::Result<()> {
        let (call, _) = server(0);
        let mut out = Vec::new();
        assert_eq!(export(&call, &json!({}), &mut out)?, 0);
        assert_eq!(String::from_utf8(out)?.lines().count(), 1);
        let short = |_: &str, arguments: Value| {
            let page = if arguments["offset"] == 0 {
                vec![json!({ "id": 1 })]
            } else {
                vec![]
            };
            Ok(json!({ "total": 5, "memories": page }))
        };
        assert_eq!(export(&short, &json!({}), &mut Vec::new())?, 1);
        Ok(())
    }

    /// The file is new unless told to replace one, and its owner's alone.
    #[test]
    fn the_file_is_new_and_its_owners_alone() -> anyhow::Result<()> {
        let dir = crate::agents::scratch::Scratch::new("export")?;
        let path = dir.path().join("mine.jsonl");
        drop(create(&path, false)?);
        let Err(there) = create(&path, false) else {
            anyhow::bail!("an export wrote over a file");
        };
        assert!(there.to_string().contains("--force"), "{there}");
        drop(create(&path, true)?);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path)?.permissions().mode() & 0o777,
                0o600
            );
        }
        assert_eq!(
            default_file("2026-11-22"),
            PathBuf::from("antumbra-export-2026-11-22.jsonl")
        );
        Ok(())
    }
}
