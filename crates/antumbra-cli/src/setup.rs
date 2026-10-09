//! `antumbra setup`: Antumbra connected to a coding agent in one command.
//!
//! Two ways to run it, and a check:
//!
//! - **`local`**: the store and the MCP server on this machine, in Docker, with
//!   embeddings from ollama. Setup checks what is installed, pulls the
//!   embedding model, fills in `docker/.env` with generated secrets, starts the
//!   stack, mints a token, and proves the server answers with it.
//! - **`hosted <url>`**: a workspace someone else runs. Setup takes the token
//!   (or the sign-in link that carries one), proves the server accepts it, and
//!   reads the workspace from it.
//! - **`check`**: every piece, each with what to do when it is not right.
//!
//! Both ways end the same: the hooks written to `~/.antumbra/hooks`, Claude
//! Code's settings wired to run them (see [`settings`]: added to, never
//! rewritten), and the MCP server registered with a headers helper that reads
//! the token from `~/.antumbra/token.txt`, so the token never sits in a
//! settings file. Running setup again keeps what is already in place.
//!
//! `--yes` answers every question with its default and `--dry-run` changes
//! nothing, so an agent can run it on someone's behalf (docs/agent-setup.md).

pub mod env_file;
pub mod settings;
pub mod token;

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine;
use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};

use crate::claude::hooks;

/// Where the store's vectors come from: ollama's default address.
const OLLAMA: &str = "http://127.0.0.1:11434";
/// The embedding model, 384 dimensions to match the store.
const MODEL: &str = "all-minilm";
const EMBED_DIM: usize = 384;
/// The MCP server's port in the local stack.
const MCP_PORT: u16 = 8081;

#[derive(Subcommand)]
pub enum SetupMode {
    /// Run the store and the MCP server on this machine, in Docker, and connect
    /// Claude Code to them. Run it from a clone of the repository.
    Local(LocalArgs),
    /// Connect Claude Code to a hosted workspace, with the token (or the
    /// sign-in link) you were given.
    Hosted(HostedArgs),
    /// Check every piece of the setup, and say what to do about anything that
    /// is not right. Exits non-zero when something is.
    Check(CheckArgs),
}

/// What both ways of running setup share.
#[derive(Args, Clone, Debug)]
pub struct Common {
    /// Answer every question with its default (for scripts and agents).
    #[arg(long, short = 'y')]
    pub yes: bool,
    /// Say what would change, and change nothing.
    #[arg(long)]
    pub dry_run: bool,
    /// Leave Claude Code alone: no hooks, no MCP server, no settings.
    #[arg(long)]
    pub no_claude: bool,
    /// Keep Claude Code's own file memory on. Setup otherwise turns it off when
    /// the settings do not say, so Antumbra is the one store.
    #[arg(long)]
    pub keep_auto_memory: bool,
    /// Where Claude Code keeps its settings (default: ~/.claude).
    #[arg(long, env = "CLAUDE_CONFIG_DIR")]
    pub claude_dir: Option<PathBuf>,
}

#[derive(Args, Clone, Debug)]
pub struct LocalArgs {
    /// The repository clone to run the stack from (default: the one you are
    /// in).
    #[arg(long)]
    pub repo: Option<PathBuf>,
    /// Where ollama answers on this machine.
    #[arg(long, default_value = OLLAMA)]
    pub ollama: String,
    /// The GPU build (Linux, or Windows through WSL2, with an NVIDIA card):
    /// embeddings in-process, answers from trained experts, consolidation on
    /// its own. ollama is not needed with it.
    #[arg(long)]
    pub gpu: bool,
    /// The workspace the token grants.
    #[arg(long, default_value = "ws:default")]
    pub workspace: String,
    /// The user the token names.
    #[arg(long, default_value = "user:default")]
    pub user: String,
    #[command(flatten)]
    pub common: Common,
}

#[derive(Args, Clone, Debug)]
pub struct HostedArgs {
    /// The workspace's address, as you were given it: https://...
    #[arg(value_name = "URL")]
    pub server: String,
    /// The token. Prefer --token-file, or ANTUMBRA_TOKEN, so it stays out of
    /// your shell history.
    #[arg(long, env = "ANTUMBRA_TOKEN", hide_env_values = true, conflicts_with_all = ["token_file", "signin_link"])]
    pub token: Option<String>,
    /// A file holding the token.
    #[arg(long, conflicts_with = "signin_link")]
    pub token_file: Option<PathBuf>,
    /// The sign-in link from your email; setup trades it for the token.
    #[arg(long)]
    pub signin_link: Option<String>,
    #[command(flatten)]
    pub common: Common,
}

#[derive(Args, Clone, Debug)]
pub struct CheckArgs {
    /// Where Claude Code keeps its settings (default: ~/.claude).
    #[arg(long, env = "CLAUDE_CONFIG_DIR")]
    pub claude_dir: Option<PathBuf>,
}

/// `local` with its defaults, for the interactive menu.
#[derive(Parser)]
struct LocalDefaults {
    #[command(flatten)]
    args: LocalArgs,
}

/// Prints the steps as they go: done, would be done, a warning, a failure.
struct Steps {
    dry_run: bool,
}

impl Steps {
    fn ok(&self, said: impl AsRef<str>) {
        println!("  \u{2713} {}", said.as_ref());
    }
    fn would(&self, said: impl AsRef<str>) {
        println!("  \u{00b7} would {}", said.as_ref());
    }
    fn warn(&self, said: impl AsRef<str>) {
        println!("  ! {}", said.as_ref());
    }
    fn info(&self, said: impl AsRef<str>) {
        println!("    {}", said.as_ref());
    }
}

pub fn run(mode: Option<SetupMode>) -> anyhow::Result<()> {
    match mode {
        Some(SetupMode::Local(args)) => local(&args),
        Some(SetupMode::Hosted(args)) => hosted(&args),
        Some(SetupMode::Check(args)) => check(&args),
        None => menu(),
    }
}

/// `antumbra setup` with no mode: ask, when someone is there to answer.
fn menu() -> anyhow::Result<()> {
    if !std::io::stdin().is_terminal() {
        anyhow::bail!(
            "pick one:\n  antumbra setup local           run Antumbra on this machine\n  antumbra setup hosted <url>    connect to a hosted workspace\n  antumbra setup check           check an existing setup"
        );
    }
    println!("Where should Antumbra run?\n");
    println!("  1  On this machine (needs Docker and ollama)");
    println!("  2  A hosted workspace (needs its address and your token)\n");
    let choice = prompt("Choose 1 or 2: ")?;
    match choice.trim() {
        "1" => local(&LocalDefaults::parse_from(["local"]).args),
        "2" => {
            let server = prompt("The workspace's address (https://...): ")?;
            hosted(&HostedArgs {
                server: server.trim().to_string(),
                token: None,
                token_file: None,
                signin_link: None,
                common: LocalDefaults::parse_from(["local"]).args.common,
            })
        }
        other => anyhow::bail!("{other:?} is not 1 or 2"),
    }
}

// ---------------------------------------------------------------- local

fn local(args: &LocalArgs) -> anyhow::Result<()> {
    let steps = Steps {
        dry_run: args.common.dry_run,
    };
    let home = home()?;
    let antumbra = home.join(".antumbra");
    println!("Setting up Antumbra on this machine.\n");

    let repo = find_repo(args.repo.as_deref())?;
    steps.ok(format!("repository at {}", repo.display()));
    let compose = repo.join("docker").join("docker-compose.yml");
    let mut compose_files = vec![compose.clone()];
    if args.gpu {
        compose_files.push(repo.join("docker").join("docker-compose.gpu.yml"));
    }

    // Docker, and compose.
    let version = output("docker", &["version", "--format", "{{.Server.Version}}"]).map_err(|_| {
        anyhow::anyhow!(
            "Docker is not installed, or not running.\n  Install Docker Desktop (macOS, Windows) or Docker Engine (Linux): https://docs.docker.com/get-docker/\n  Start it, then run this again."
        )
    })?;
    steps.ok(format!("Docker {} is running", version.trim()));
    output("docker", &["compose", "version", "--short"]).map_err(|_| {
        anyhow::anyhow!("Docker Compose v2 is missing (`docker compose version` failed). Update Docker, then run this again.")
    })?;

    // The hooks' own needs on macOS and Linux.
    if !cfg!(windows) && !args.common.no_claude {
        let missing: Vec<&str> = ["jq", "curl"]
            .into_iter()
            .filter(|p| find_program(p).is_none())
            .collect();
        if !missing.is_empty() {
            anyhow::bail!(
                "the hooks need {} on your PATH.\n  macOS: brew install {}\n  Debian, Ubuntu: sudo apt install {}\n  Then run this again.",
                missing.join(" and "),
                missing.join(" "),
                missing.join(" ")
            );
        }
        steps.ok("jq and curl are installed (the hooks use them)");
    }

    // Embeddings.
    if args.gpu {
        steps.ok("the GPU build embeds in-process; ollama is not needed");
    } else {
        ensure_ollama(&args.ollama, &steps)?;
    }

    // docker/.env.
    let data = antumbra.join("surrealdb");
    let env_path = repo.join("docker").join(".env");
    let example =
        std::fs::read_to_string(repo.join("docker").join(".env.example")).unwrap_or_default();
    let existing = std::fs::read_to_string(&env_path).ok();
    let wants = vec![
        env_file::Want {
            name: "SURREAL_PASS",
            value: secret(24, true),
        },
        env_file::Want {
            name: "ANTUMBRA_JWT_SECRET",
            value: secret(32, false),
        },
        env_file::Want {
            name: "ANTUMBRA_SURREAL_DATA",
            value: data.to_string_lossy().replace('\\', "/"),
        },
    ];
    let (env_text, wrote) = env_file::fill(existing.as_deref(), &example, &wants);
    if wrote.is_empty() {
        steps.ok("docker/.env already holds the stack's secrets");
    } else if steps.dry_run {
        steps.would(format!("write {} to docker/.env", wrote.join(", ")));
    } else {
        std::fs::create_dir_all(&data)?;
        write_private(&env_path, &env_text)?;
        steps.ok(format!("docker/.env: generated {}", wrote.join(", ")));
    }
    let data_dir = env_file::get(&env_text, "ANTUMBRA_SURREAL_DATA")
        .map(PathBuf::from)
        .unwrap_or(data);
    if !steps.dry_run {
        std::fs::create_dir_all(&data_dir)?;
        give_to_container(&data_dir, &steps)?;
    }

    // The stack.
    let mut up: Vec<String> = compose_args(&compose_files);
    up.extend(["up", "-d", "--build", "surrealdb", "antumbra-mcp"].map(String::from));
    if steps.dry_run {
        steps.would(format!("run docker {}", up.join(" ")));
    } else {
        println!("\n  Starting the store and the server. The first run builds the server image,\n  which takes several minutes; later runs rebuild only what the clone changed.\n");
        let status = Command::new("docker").args(&up).status()?;
        if !status.success() {
            anyhow::bail!("`docker {}` failed; its output is above", up.join(" "));
        }
        println!();
        steps.ok("the store and the server are up");
    }

    let bind = env_file::get(&env_text, "ANTUMBRA_MCP_BIND")
        .filter(|b| !b.is_empty() && *b != "0.0.0.0")
        .unwrap_or("127.0.0.1");
    let url = format!("http://{bind}:{MCP_PORT}");
    let token_path = antumbra.join("token.txt");

    if steps.dry_run {
        steps.would(format!(
            "wait for the server at {url}, mint a token into {}",
            token_path.display()
        ));
    } else {
        wait_for(&url)?;
        steps.ok(format!("the server answers at {url}"));
        let token = match std::fs::read_to_string(&token_path)
            .ok()
            .map(|t| t.trim().to_string())
            .filter(|t| token::looks_like_jwt(t) && verify(&url, t).is_ok())
        {
            Some(t) => {
                steps.ok(format!(
                    "kept the token in {}, which the server accepts",
                    token_path.display()
                ));
                t
            }
            None => {
                let t = mint(&compose_files, &args.workspace, &args.user)?;
                write_private(&token_path, &t)?;
                steps.ok(format!(
                    "minted a token for {} / {}, saved to {} (valid for a year)",
                    args.workspace,
                    args.user,
                    token_path.display()
                ));
                t
            }
        };
        verify(&url, &token).map_err(|e| {
            anyhow::anyhow!(
                "{e}\n  The server is up but could not recall. If it says the embedder is unreachable, the server's container cannot reach ollama{}",
                if cfg!(target_os = "linux") {
                    ": on Linux, start ollama with OLLAMA_HOST=0.0.0.0 (sudo systemctl edit ollama, add Environment=OLLAMA_HOST=0.0.0.0 under [Service], then sudo systemctl restart ollama)."
                } else {
                    "."
                }
            )
        })?;
        steps.ok("recall works with that token");
    }

    let state = json!({
        "mode": "local",
        "url": url,
        "workspace": args.workspace,
        "user": args.user,
        "repo": repo,
        "compose": compose_files,
        "token_file": token_path,
        "ollama": if args.gpu { Value::Null } else { Value::String(args.ollama.clone()) },
    });
    finish(
        &home,
        &antumbra,
        &args.common,
        &url,
        &args.workspace,
        &state,
        &steps,
    )
}

/// The repository to run the stack from: the one given, or the one the
/// current directory is in.
fn find_repo(given: Option<&Path>) -> anyhow::Result<PathBuf> {
    let start = match given {
        Some(p) => p.to_path_buf(),
        None => std::env::current_dir()?,
    };
    for dir in start.ancestors() {
        if dir.join("docker").join("docker-compose.yml").is_file()
            && dir.join("crates").join("antumbra-mcp").is_dir()
        {
            return Ok(dir.to_path_buf());
        }
    }
    anyhow::bail!(
        "{} is not inside a clone of the Antumbra repository.\n  The local setup builds the server from the repository: clone it, cd into it, and run this again (or pass --repo <path>).",
        start.display()
    )
}

/// ollama answering, the model pulled, and vectors of the store's width.
fn ensure_ollama(base: &str, steps: &Steps) -> anyhow::Result<()> {
    let base = base.trim_end_matches('/');
    let agent = agent(Duration::from_secs(10));
    let tags: Value = agent
        .get(&format!("{base}/api/tags"))
        .call()
        .ok()
        .filter(|r| r.status().is_success())
        .and_then(|mut r| r.body_mut().read_json().ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "ollama is not answering at {base}.\n  Install it from https://ollama.com and start it, then run this again.\n  (Running it somewhere else? Pass --ollama <url>.)"
            )
        })?;
    steps.ok(format!("ollama answers at {base}"));
    let pulled = tags
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("name").and_then(Value::as_str))
        .any(|name| name == MODEL || name.starts_with(&format!("{MODEL}:")));
    if !pulled {
        if steps.dry_run {
            steps.would(format!("pull the {MODEL} embedding model (about 46 MB)"));
            return Ok(());
        }
        steps.info(format!(
            "pulling the {MODEL} embedding model (about 46 MB)..."
        ));
        let mut response = agent_within(Duration::from_secs(1800))
            .post(&format!("{base}/api/pull"))
            .send_json(json!({ "model": MODEL, "stream": false }))
            .map_err(|e| anyhow::anyhow!("pulling {MODEL}: {e}"))?;
        if !response.status().is_success() {
            let said = response.body_mut().read_to_string().unwrap_or_default();
            anyhow::bail!("ollama could not pull {MODEL}: {said}");
        }
    }
    // The first embedding loads the model into memory, which takes a while.
    let mut response = agent_within(Duration::from_secs(120))
        .post(&format!("{base}/v1/embeddings"))
        .send_json(json!({ "model": MODEL, "input": "antumbra" }))
        .map_err(|e| anyhow::anyhow!("embedding a test sentence: {e}"))?;
    let body: Value = response.body_mut().read_json().unwrap_or(Value::Null);
    let width = body
        .pointer("/data/0/embedding")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if width != EMBED_DIM {
        anyhow::bail!(
            "{MODEL} through ollama gave vectors of {width} dimensions; the store needs {EMBED_DIM}"
        );
    }
    steps.ok(format!("{MODEL} embeds ({EMBED_DIM} dimensions)"));

    // Can the server's container reach ollama? Under a native Docker Engine
    // on Linux the container comes in through the bridge gateway, which this
    // machine can probe directly. Anywhere that answer is no (or there is no
    // bridge to see: Docker Desktop keeps it in its own VM, out of a WSL
    // distro's sight), ask from inside a container instead, with the
    // host.docker.internal mapping compose gives the server.
    let port = base.rsplit(':').next().unwrap_or("11434");
    if cfg!(target_os = "linux") && bridge_reaches(port) {
        steps.ok("the server's container can reach ollama");
        return Ok(());
    }
    match container_reaches(port) {
        Reach::Yes => steps.ok("the server's container can reach ollama"),
        Reach::Unknown(why) => steps.info(format!(
            "could not ask from a container whether the server can reach ollama ({why}); recall is checked once the server is up"
        )),
        Reach::No => anyhow::bail!("{}", unreachable_from_container(port)),
    }
    Ok(())
}

/// ollama's tag list through the Docker bridge gateway, from this machine.
fn bridge_reaches(port: &str) -> bool {
    let Ok(gateway) = output(
        "docker",
        &[
            "network",
            "inspect",
            "bridge",
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ],
    ) else {
        return false;
    };
    agent(Duration::from_secs(3))
        .get(&format!("http://{}:{port}/api/tags", gateway.trim()))
        .call()
        .is_ok_and(|r| r.status().is_success())
}

enum Reach {
    Yes,
    No,
    Unknown(String),
}

/// A throwaway container fetches ollama's tag list at host.docker.internal,
/// as the server's container will.
fn container_reaches(port: &str) -> Reach {
    let Some(docker) = find_program("docker") else {
        return Reach::Unknown("docker is not on the PATH".into());
    };
    let url = format!("http://host.docker.internal:{port}/api/tags");
    let out = Command::new(docker)
        .args([
            "run",
            "--rm",
            "--add-host",
            "host.docker.internal:host-gateway",
            "busybox:1.37",
            "wget",
            "-q",
            "-T",
            "5",
            "-O",
            "/dev/null",
            &url,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output();
    match out {
        Err(e) => Reach::Unknown(e.to_string()),
        Ok(o) if o.status.success() => Reach::Yes,
        // 125-127 are docker's own failures (no daemon, no image), not the
        // fetch's.
        Ok(o) if matches!(o.status.code(), Some(125..=127) | None) => {
            Reach::Unknown(String::from_utf8_lossy(&o.stderr).trim().to_string())
        }
        Ok(_) => Reach::No,
    }
}

/// What to do when a container cannot reach ollama, for the Docker this is.
fn unreachable_from_container(port: &str) -> String {
    let desktop = output("docker", &["info", "--format", "{{.OperatingSystem}}"])
        .is_ok_and(|os| os.contains("Docker Desktop"));
    let wsl = std::env::var_os("WSL_DISTRO_NAME").is_some()
        || Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists();
    let head =
        format!("the server's container cannot reach ollama at host.docker.internal:{port}.");
    let elsewhere = "(Running ollama somewhere the container can reach? Set ANTUMBRA_EMBEDDER_URL in docker/.env.)";
    if desktop && wsl {
        format!(
            "{head}\n  Docker Desktop runs containers in its own VM, where host.docker.internal is the Windows host, not this WSL distro.\n  Either:\n    - run ollama on Windows instead (https://ollama.com) and stop the copy in WSL, so only one owns port {port}; or\n    - let this distro share Windows' network: in %UserProfile%\\.wslconfig put\n        [wsl2]\n        networkingMode=mirrored\n      run `wsl --shutdown` from Windows, start ollama here with OLLAMA_HOST=0.0.0.0, and run this again.\n  {elsewhere}"
        )
    } else if cfg!(target_os = "linux") && !desktop {
        format!(
            "{head}\n  ollama only listens on 127.0.0.1, and the container comes in through the Docker bridge.\n  sudo systemctl edit ollama, add these two lines, save:\n    [Service]\n    Environment=OLLAMA_HOST=0.0.0.0\n  then: sudo systemctl restart ollama, and run this again."
        )
    } else {
        format!(
            "{head}\n  Make sure ollama runs on this machine itself (not inside another VM) and answers on port {port}, then run this again.\n  {elsewhere}"
        )
    }
}

/// On Linux the container runs as uid 65532 and needs to own its data
/// directory. Docker can do that without sudo.
#[cfg(unix)]
fn give_to_container(dir: &Path, steps: &Steps) -> anyhow::Result<()> {
    use std::os::unix::fs::MetadataExt;
    if !cfg!(target_os = "linux") || std::fs::metadata(dir)?.uid() == 65532 {
        return Ok(());
    }
    let mount = format!("{}:/data", dir.display());
    let status = Command::new("docker")
        .args([
            "run",
            "--rm",
            "-v",
            &mount,
            "busybox:1.37",
            "chown",
            "-R",
            "65532:65532",
            "/data",
        ])
        .stdout(Stdio::null())
        .status()?;
    if !status.success() {
        anyhow::bail!(
            "could not hand {} to the database's user (uid 65532). Run: sudo chown -R 65532:65532 {}",
            dir.display(),
            dir.display()
        );
    }
    steps.ok(format!("{} belongs to the database's user", dir.display()));
    Ok(())
}

#[cfg(not(unix))]
fn give_to_container(_dir: &Path, _steps: &Steps) -> anyhow::Result<()> {
    Ok(())
}

fn compose_args(files: &[PathBuf]) -> Vec<String> {
    let mut args = vec!["compose".to_string()];
    for file in files {
        args.push("-f".into());
        args.push(file.to_string_lossy().into_owned());
    }
    args
}

/// Waits for the server's dashboard, which answers without a token.
fn wait_for(url: &str) -> anyhow::Result<()> {
    let agent = agent(Duration::from_secs(5));
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(180) {
        if agent
            .get(&format!("{url}/dashboard"))
            .call()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    anyhow::bail!(
        "the server did not answer at {url} within three minutes.\n  See why: docker logs antumbra-mcp"
    )
}

/// A token signed with the stack's own secret, minted by the server's image.
fn mint(compose_files: &[PathBuf], workspace: &str, user: &str) -> anyhow::Result<String> {
    let mut args = compose_args(compose_files);
    args.extend(
        [
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "antumbra-mcp",
            "--mint-token",
            "--tenant",
            workspace,
            "--user",
            user,
            "--token-ttl-days",
            "365",
        ]
        .map(String::from),
    );
    let out = Command::new("docker")
        .args(&args)
        .stderr(Stdio::piped())
        .output()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .map(str::trim)
        .rfind(|line| token::looks_like_jwt(line))
        .map(str::to_string)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "minting a token failed:\n{}",
                String::from_utf8_lossy(&out.stderr).trim()
            )
        })
}

// ---------------------------------------------------------------- hosted

fn hosted(args: &HostedArgs) -> anyhow::Result<()> {
    let steps = Steps {
        dry_run: args.common.dry_run,
    };
    let home = home()?;
    let antumbra = home.join(".antumbra");
    let url = args
        .server
        .trim()
        .trim_end_matches('/')
        .trim_end_matches("/mcp")
        .to_string();
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        anyhow::bail!("the workspace's address starts with https:// (got {url:?})");
    }
    println!("Connecting this machine to the hosted workspace at {url}.\n");

    let pasted = if let Some(t) = &args.token {
        token::read_pasted(t)?
    } else if let Some(file) = &args.token_file {
        token::read_pasted(&std::fs::read_to_string(file)?)?
    } else if let Some(link) = &args.signin_link {
        token::Pasted::Link(link.trim().to_string())
    } else if std::io::stdin().is_terminal() && !args.common.yes {
        token::read_pasted(&prompt(
            "Paste your token, or the sign-in link from your email: ",
        )?)?
    } else {
        anyhow::bail!("give the token with --token-file <file>, --token, ANTUMBRA_TOKEN, or --signin-link <url>")
    };
    let token = match pasted {
        token::Pasted::Token(t) => t,
        token::Pasted::Link(link) => {
            let mut response = agent(Duration::from_secs(30))
                .get(&link)
                .call()
                .map_err(|e| anyhow::anyhow!("opening the sign-in link: {e}"))?;
            let body: Value = response.body_mut().read_json().unwrap_or(Value::Null);
            let t = token::from_signin(&body).ok_or_else(|| {
                anyhow::anyhow!(
                    "the sign-in link did not give a token (it may have been used already, or expired: links last 15 minutes and work once). Ask for a new one."
                )
            })?;
            steps.ok("traded the sign-in link for a token");
            t
        }
    };
    if let Some(exp) = token::expires(&token) {
        if exp < chrono::Utc::now().timestamp() {
            anyhow::bail!("that token has expired; ask for a new one");
        }
    }
    let workspace = token::workspace(&token).unwrap_or_else(|| "ws:default".to_string());

    verify(&url, &token)?;
    steps.ok(format!(
        "the server accepts the token for workspace {workspace}"
    ));
    let token_path = antumbra.join("token.txt");
    if steps.dry_run {
        steps.would(format!("save the token to {}", token_path.display()));
    } else {
        write_private(&token_path, &token)?;
        steps.ok(format!("token saved to {}", token_path.display()));
    }

    let state = json!({
        "mode": "hosted",
        "url": url,
        "workspace": workspace,
        "token_file": token_path,
    });
    finish(
        &home,
        &antumbra,
        &args.common,
        &url,
        &workspace,
        &state,
        &steps,
    )
}

// ---------------------------------------------------------------- both

/// Claude Code wired to `url`, the state written, and what to do next.
fn finish(
    home: &Path,
    antumbra: &Path,
    common: &Common,
    url: &str,
    workspace: &str,
    state: &Value,
    steps: &Steps,
) -> anyhow::Result<()> {
    if common.no_claude {
        steps.info("left Claude Code alone (--no-claude)");
    } else {
        connect_claude(home, antumbra, common, url, workspace, steps)?;
    }
    // After the settings, so the name read is the one the hooks will report.
    let settings: Value = std::fs::read_to_string(
        common
            .claude_dir
            .clone()
            .unwrap_or_else(|| home.join(".claude"))
            .join("settings.json"),
    )
    .ok()
    .and_then(|t| serde_json::from_str(&t).ok())
    .unwrap_or(Value::Null);
    let token = std::fs::read_to_string(antumbra.join("token.txt"))
        .map(|t| t.trim().to_string())
        .unwrap_or_default();
    register_machine(url, &token, &session_host(&settings), steps);
    if !steps.dry_run {
        let mut state = state.clone();
        state["configured_at"] = Value::String(chrono::Utc::now().to_rfc3339());
        std::fs::create_dir_all(antumbra)?;
        std::fs::write(
            antumbra.join("setup.json"),
            serde_json::to_string_pretty(&state)?,
        )?;
    }
    if steps.dry_run {
        println!("\nNothing was changed (--dry-run).");
    } else {
        println!(
            "\nAntumbra is set up. Start a new Claude Code session: it opens with what Antumbra"
        );
        println!("remembers, and what you work on goes back in.");
        println!("Check it any time with: antumbra setup check");
    }
    Ok(())
}

/// The hook scripts this platform runs, and the event each one serves.
fn platform_hooks() -> Vec<(&'static str, &'static str, u32)> {
    let ext_ps = cfg!(windows);
    let pick = |sh: &'static str, ps: &'static str| if ext_ps { ps } else { sh };
    vec![
        (
            "SessionStart",
            pick("antumbra-session-start.sh", "antumbra-session-start.ps1"),
            10,
        ),
        (
            "UserPromptSubmit",
            pick("antumbra-prompt-recall.sh", "antumbra-prompt-recall.ps1"),
            20,
        ),
        (
            "Stop",
            pick("antumbra-capture.sh", "antumbra-capture.ps1"),
            5,
        ),
        (
            "PreCompact",
            pick("antumbra-capture.sh", "antumbra-capture.ps1"),
            5,
        ),
    ]
}

fn headers_helper_name() -> &'static str {
    if cfg!(windows) {
        "antumbra-mcp-headers.ps1"
    } else {
        "antumbra-mcp-headers.sh"
    }
}

/// The command that runs a script: bash, or Windows PowerShell (which every
/// Windows machine has; the hooks need nothing newer).
fn script_command(path: &Path) -> String {
    if cfg!(windows) {
        format!(
            "powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"{}\"",
            path.display()
        )
    } else {
        format!("bash \"{}\"", path.display())
    }
}

fn connect_claude(
    home: &Path,
    antumbra: &Path,
    common: &Common,
    url: &str,
    workspace: &str,
    steps: &Steps,
) -> anyhow::Result<()> {
    // The scripts, from this build.
    let dir = antumbra.join("hooks");
    let mut names: Vec<&str> = platform_hooks().iter().map(|(_, s, _)| *s).collect();
    names.push(headers_helper_name());
    names.dedup();
    if steps.dry_run {
        steps.would(format!("write the hooks to {}", dir.display()));
    } else {
        std::fs::create_dir_all(&dir)?;
        for name in &names {
            let text = hooks::bundled(name)
                .ok_or_else(|| anyhow::anyhow!("{name} is not bundled in this build"))?;
            let path = dir.join(name);
            std::fs::write(&path, text)?;
            make_executable(&path)?;
        }
        steps.ok(format!("hooks written to {}", dir.display()));
    }

    // The settings.
    let claude = common
        .claude_dir
        .clone()
        .unwrap_or_else(|| home.join(".claude"));
    let path = claude.join("settings.json");
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let auto_memory_off = if common.keep_auto_memory {
        false
    } else if common.yes || steps.dry_run || !std::io::stdin().is_terminal() {
        true
    } else {
        let current: Value =
            serde_json::from_str(if text.trim().is_empty() { "{}" } else { &text })
                .unwrap_or(Value::Null);
        current.get("autoMemoryEnabled").is_some()
            || confirm("Turn off Claude Code's own file memory, so Antumbra is the one store?")?
    };
    let wiring = settings::Wiring {
        hooks: platform_hooks()
            .into_iter()
            .map(|(event, script, timeout)| settings::Hook {
                event,
                script,
                command: script_command(&dir.join(script)),
                timeout,
            })
            .collect(),
        env: vec![
            ("ANTUMBRA_URL".into(), url.to_string()),
            ("ANTUMBRA_WORKSPACE_ID".into(), workspace.to_string()),
        ],
        env_default: vec![("ANTUMBRA_HOST_ID".into(), host_id())],
        auto_memory_off,
    };
    let wired = settings::wire(&text, &wiring)?;
    // A hook wired by hand somewhere else is worth a word; setup's own, from an
    // earlier run, is not.
    for (event, command) in &wired.already {
        if !wiring.hooks.iter().any(|h| &h.command == command) {
            steps.info(format!("{event} already runs {command}; left as it is"));
        }
    }
    if !wired.changed() {
        steps.ok(format!("{} already wires Antumbra", path.display()));
    } else if steps.dry_run {
        steps.would(format!("edit {}: {}", path.display(), describe(&wired)));
    } else {
        std::fs::create_dir_all(&claude)?;
        if path.exists() {
            let backup = path.with_file_name(format!(
                "settings.json.antumbra-backup-{}",
                chrono::Utc::now().format("%Y%m%d%H%M%S")
            ));
            std::fs::copy(&path, &backup)?;
            steps.info(format!("backed up the settings to {}", backup.display()));
        }
        std::fs::write(&path, &wired.text)?;
        let reread: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        if reread.get("hooks").is_none() {
            anyhow::bail!("{} did not take the edit", path.display());
        }
        steps.ok(format!("{}: {}", path.display(), describe(&wired)));
    }
    if wired.auto_memory_was == Some(true) {
        steps.warn("Claude Code's own file memory is on (autoMemoryEnabled: true); with it, memories land in two stores");
    }

    // The MCP server, with the token read from its file on every connection.
    let helper = script_command(&dir.join(headers_helper_name()));
    let server =
        json!({ "type": "http", "url": format!("{url}/mcp"), "headersHelper": helper }).to_string();
    match find_program("claude") {
        None => {
            steps.warn("the claude command is not on your PATH, so the MCP server is not registered. Once it is:");
            steps.info(format!(
                "claude mcp add-json antumbra --scope user '{server}'"
            ));
        }
        Some(_) if steps.dry_run => {
            steps.would("register the MCP server \"antumbra\" with Claude Code (user scope)");
        }
        Some(claude) => {
            let _ = Command::new(&claude)
                .args(["mcp", "remove", "antumbra", "--scope", "user"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let out = Command::new(&claude)
                .args(["mcp", "add-json", "antumbra", "--scope", "user", &server])
                .output()?;
            if out.status.success() {
                steps.ok("MCP server \"antumbra\" registered with Claude Code (user scope)");
            } else {
                steps.warn(format!(
                    "registering the MCP server failed: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                ));
                steps.info(format!(
                    "claude mcp add-json antumbra --scope user '{server}'"
                ));
            }
        }
    }
    Ok(())
}

fn describe(wired: &settings::Wired) -> String {
    let mut said = Vec::new();
    if !wired.added.is_empty() {
        said.push(format!("hooks on {}", wired.added.join(", ")));
    }
    if !wired.env_set.is_empty() {
        said.push(wired.env_set.join(", "));
    }
    if wired.auto_memory_set {
        said.push("autoMemoryEnabled false".into());
    }
    said.join("; ")
}

/// One tool call that proves the server is there, takes the token, and can
/// recall (which also proves its embedder answers).
fn verify(url: &str, token: &str) -> anyhow::Result<()> {
    call_tool(
        url,
        token,
        "recall_memories",
        json!({ "query": "antumbra setup", "top_k": 1 }),
        "recall",
    )
    .map(|_| ())
}

/// One tool call through the server's REST shim, its answer on success. `what`
/// names the call in the error the server's refusal becomes.
fn call_tool(
    url: &str,
    token: &str,
    tool: &str,
    arguments: Value,
    what: &str,
) -> anyhow::Result<Value> {
    call_tool_from(url, token, None, tool, arguments, what)
}

/// The header a client names its machine in, which the server stamps on what
/// that machine writes (#193). The MCP headers helper sends it from
/// `ANTUMBRA_HOST_ID`.
const DEVICE_HEADER: &str = "X-Antumbra-Host";

/// [`call_tool`] from the machine `device` names, as the agent's connection
/// calls once the headers helper names it.
fn call_tool_from(
    url: &str,
    token: &str,
    device: Option<&str>,
    tool: &str,
    arguments: Value,
    what: &str,
) -> anyhow::Result<Value> {
    let mut request = agent(Duration::from_secs(30))
        .post(&format!("{url}/mcp/call"))
        .header("authorization", &format!("Bearer {token}"));
    if let Some(device) = device {
        request = request.header(DEVICE_HEADER, device);
    }
    let mut response = request
        .send_json(json!({ "tool": tool, "arguments": arguments }))
        .map_err(|e| anyhow::anyhow!("could not reach {url}: {e}"))?;
    let status = response.status();
    let body: Value = response.body_mut().read_json().unwrap_or(Value::Null);
    if status.as_u16() == 401 || status.as_u16() == 403 {
        anyhow::bail!(
            "the server at {url} refused the token; it may have expired, or be for another server"
        );
    }
    if let Some(said) = body.get("error") {
        anyhow::bail!(
            "{what} failed: {}",
            said.as_str()
                .map_or_else(|| said.to_string(), str::to_string)
        );
    }
    if !status.is_success() {
        anyhow::bail!("the server at {url} answered {status}");
    }
    Ok(body)
}

/// The name this machine's sessions report, which the hooks read from
/// `ANTUMBRA_HOST_ID`: the one Claude Code's settings give them, else this
/// process's own, else the one setup writes when neither says.
fn session_host(settings: &Value) -> String {
    let named = |h: &str| Some(h.trim().to_string()).filter(|h| !h.is_empty());
    settings
        .pointer("/env/ANTUMBRA_HOST_ID")
        .and_then(Value::as_str)
        .and_then(named)
        .or_else(|| {
            std::env::var("ANTUMBRA_HOST_ID")
                .ok()
                .and_then(|h| named(&h))
        })
        .unwrap_or_else(host_id)
}

/// Name this machine in the user's fabric, as every session start does, so it
/// is listed from now rather than from its first session. A server older than
/// device registration does not know the tool; that is a warning, not a
/// failed setup, because the machine's sessions work either way.
fn register_machine(url: &str, token: &str, host: &str, steps: &Steps) {
    if steps.dry_run {
        steps.would(format!("list this machine among your devices as {host}"));
        return;
    }
    match call_tool(
        url,
        token,
        "register_device",
        json!({ "host": host }),
        "registering this machine",
    ) {
        Ok(said) if said["registered"] == true => steps.ok(format!(
            "this machine is listed among your devices as {} (a {} node)",
            said["host"].as_str().unwrap_or(host),
            said["role"].as_str().unwrap_or("memory")
        )),
        Ok(said) => steps.info(format!(
            "{}: {}",
            said["host"].as_str().unwrap_or(host),
            said["note"].as_str().unwrap_or("left as it is")
        )),
        Err(e) => steps.warn(format!(
            "this machine is not listed among your devices yet ({e}); its sessions name it once the server can"
        )),
    }
}

// ---------------------------------------------------------------- check

/// What `check` finds in the MCP headers helper setup wrote, which the agent's
/// connection runs to authenticate and to name this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Helper {
    /// Not there: the agent cannot connect.
    Missing,
    /// A copy from before writes were stamped per machine: it sends the token
    /// and no name, so the server stamps this machine's writes with its own.
    Unnamed,
    /// It names the machine, and differs from this build's copy.
    Edited,
    /// This build's copy.
    Same,
}

fn helper_state(text: Option<&str>) -> Helper {
    let Some(text) = text else {
        return Helper::Missing;
    };
    if !text.contains(DEVICE_HEADER) {
        return Helper::Unnamed;
    }
    let bundled = hooks::bundled(headers_helper_name()).unwrap_or_default();
    if text.replace("\r\n", "\n") == bundled.replace("\r\n", "\n") {
        Helper::Same
    } else {
        Helper::Edited
    }
}

/// Which machine the server stamps this one's writes with, from its answer to
/// `devices` asked under `host`'s name (#193).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Stamp {
    /// This machine's own name.
    Named,
    /// Another name, the one the server fell back to: its own, because it did
    /// not take `host` as one machine's name (`local`, say).
    Other(String),
    /// The server does not say, being older than per-machine stamps: it stamps
    /// every write with its own name.
    Unsaid,
}

fn stamp_of(said: &Value, host: &str) -> Stamp {
    match said.get("this_device").and_then(Value::as_str) {
        Some(stamped) if stamped.trim().eq_ignore_ascii_case(host.trim()) => Stamp::Named,
        Some(stamped) => Stamp::Other(stamped.to_string()),
        None => Stamp::Unsaid,
    }
}

fn check(args: &CheckArgs) -> anyhow::Result<()> {
    let home = home()?;
    let antumbra = home.join(".antumbra");
    let state: Value = match std::fs::read_to_string(antumbra.join("setup.json")) {
        Ok(text) => serde_json::from_str(&text)?,
        Err(_) => {
            anyhow::bail!("Antumbra has not been set up on this machine. Run: antumbra setup")
        }
    };
    let url = state
        .get("url")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let mode = state.get("mode").and_then(Value::as_str).unwrap_or("local");
    println!("Checking the {mode} setup against {url}.\n");
    let mut problems = 0usize;
    let mut fail = |said: String, fix: String| {
        problems += 1;
        println!("  \u{2717} {said}");
        println!("      {fix}");
    };
    let ok = |said: String| println!("  \u{2713} {said}");

    let token_path = antumbra.join("token.txt");
    let token = std::fs::read_to_string(&token_path)
        .map(|t| t.trim().to_string())
        .unwrap_or_default();
    let mut reachable = false;
    if token.is_empty() {
        fail(
            format!("no token in {}", token_path.display()),
            "run setup again".into(),
        );
    } else {
        ok(format!("token in {}", token_path.display()));
        match verify(&url, &token) {
            Ok(()) => {
                reachable = true;
                ok("the server accepts the token, and recall works".into())
            }
            Err(e) => fail(
                format!("{e}"),
                if mode == "local" {
                    "start the stack: antumbra setup local".into()
                } else {
                    "ask for a new token, then: antumbra setup hosted <url> --token-file <file>"
                        .into()
                },
            ),
        }
    }

    if mode == "local" {
        for container in ["antumbra-surrealdb", "antumbra-mcp"] {
            match output(
                "docker",
                &["inspect", "--format", "{{.State.Status}}", container],
            ) {
                Ok(s) if s.trim() == "running" => ok(format!("{container} is running")),
                Ok(s) => fail(
                    format!("{container} is {}", s.trim()),
                    format!("docker start {container}, or: antumbra setup local"),
                ),
                Err(_) => fail(
                    format!("{container} does not exist"),
                    "antumbra setup local".into(),
                ),
            }
        }
        if let Some(ollama) = state.get("ollama").and_then(Value::as_str) {
            let up = agent(Duration::from_secs(5))
                .get(&format!("{ollama}/api/tags"))
                .call()
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            if up {
                ok(format!("ollama answers at {ollama}"));
            } else {
                fail(
                    format!("ollama is not answering at {ollama}"),
                    "start ollama".into(),
                );
            }
        }
    }

    let claude = args
        .claude_dir
        .clone()
        .unwrap_or_else(|| home.join(".claude"));
    let settings_path = claude.join("settings.json");
    let settings: Value = std::fs::read_to_string(&settings_path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null);
    let scripts = hooks::scripts_in(&settings);
    for (event, script, _) in platform_hooks() {
        let stem = script.trim_end_matches(".ps1").trim_end_matches(".sh");
        let wired = settings
            .get("hooks")
            .and_then(|h| h.get(event))
            .map(|e| e.to_string().contains(stem))
            .unwrap_or(false);
        if wired {
            ok(format!("{event} runs {stem}"));
        } else {
            fail(
                format!("{event} does not run {stem}"),
                "antumbra setup (local or hosted) wires it".into(),
            );
        }
    }
    for hook in hooks::compare(&scripts, &home, |p| std::fs::read_to_string(p).ok()) {
        match hook.state {
            hooks::HookState::Same => {}
            hooks::HookState::Differs => println!(
                "  ! {} differs from this build's copy; setup writes this build's to ~/.antumbra/hooks",
                hook.path.display()
            ),
            hooks::HookState::Missing => fail(
                format!("{} is run by the settings and is missing", hook.path.display()),
                "run setup again to write it".into(),
            ),
        }
    }
    // The agent's own connection runs this one, not the settings' hooks, so the
    // comparison above never sees it.
    let helper = antumbra.join("hooks").join(headers_helper_name());
    match helper_state(std::fs::read_to_string(&helper).ok().as_deref()) {
        Helper::Same => ok(format!(
            "the MCP connection's headers helper names this machine to the server ({DEVICE_HEADER})"
        )),
        Helper::Edited => println!(
            "  ! {} differs from this build's copy; setup writes this build's to ~/.antumbra/hooks",
            helper.display()
        ),
        Helper::Unnamed => fail(
            format!(
                "{} does not name this machine, so the server stamps what the agent writes with its own name",
                helper.display()
            ),
            "run setup again to write this build's".into(),
        ),
        Helper::Missing => fail(
            format!(
                "{} is missing, and the MCP connection runs it",
                helper.display()
            ),
            "run setup again to write it".into(),
        ),
    }
    match settings
        .pointer("/env/ANTUMBRA_URL")
        .and_then(Value::as_str)
    {
        Some(set) if set == url => ok(format!("ANTUMBRA_URL is {url}")),
        Some(set) => fail(
            format!("ANTUMBRA_URL is {set}, not {url}"),
            "run setup again".into(),
        ),
        None => fail(
            "ANTUMBRA_URL is not set in the settings".into(),
            "run setup again".into(),
        ),
    }
    match find_program("claude") {
        Some(claude) => {
            let registered = Command::new(claude)
                .args(["mcp", "get", "antumbra"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if registered {
                ok("the MCP server \"antumbra\" is registered with Claude Code".into());
            } else {
                fail(
                    "the MCP server \"antumbra\" is not registered".into(),
                    "run setup again".into(),
                );
            }
        }
        None => println!(
            "  ! the claude command is not on your PATH, so the MCP registration was not checked"
        ),
    }
    if reachable {
        let host = session_host(&settings).to_lowercase();
        // Asked under this machine's name, as the agent's connection asks once
        // its headers helper names it, so the answer says what the server
        // stamps that connection's writes with. Whether the helper names it is
        // the helper line above; the two together are what the agent gets.
        match call_tool_from(
            &url,
            &token,
            Some(&host),
            "devices",
            json!({}),
            "listing your devices",
        ) {
            Ok(said) => {
                match stamp_of(&said, &host) {
                    Stamp::Named => ok(format!(
                        "the server stamps writes that name this machine as {host}"
                    )),
                    Stamp::Other(stamped) => fail(
                        format!("this machine's writes are stamped as {stamped}, not {host}"),
                        "set ANTUMBRA_HOST_ID in the settings' env to this machine's own name (not local or any), then run setup again".into(),
                    ),
                    Stamp::Unsaid => println!(
                        "  ! the server stamps every write with its own name (it is older than per-machine stamps)"
                    ),
                }
                let listed = said["devices"].as_array().and_then(|all| {
                    all.iter().find(|d| {
                        d["host"]
                            .as_str()
                            .is_some_and(|h| h.trim().to_lowercase() == host)
                    })
                });
                match listed {
                    Some(d) => ok(format!(
                        "this machine is listed among your devices as {host} ({} node, last seen {})",
                        d["role"].as_str().unwrap_or("?"),
                        d["last_seen"].as_str().unwrap_or("?")
                    )),
                    None => fail(
                        format!("this machine ({host}) is not listed among your devices"),
                        "run setup again, or start a session: each one names it".into(),
                    ),
                }
            }
            // A server from before device registration: nothing to check yet.
            Err(e)
                if e.to_string().contains("unknown tool")
                    || e.to_string().contains("not in this server's tool profile") =>
            {
                println!("  ! the server does not list devices yet (it is older than device registration)")
            }
            Err(e) => fail(format!("{e}"), "check the server".into()),
        }
    }

    if problems == 0 {
        println!("\nEverything is in place.");
        Ok(())
    } else {
        anyhow::bail!(
            "{problems} thing{} to fix, above",
            if problems == 1 { "" } else { "s" }
        )
    }
}

// ---------------------------------------------------------------- the machine

fn home() -> anyhow::Result<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("no home directory (neither USERPROFILE nor HOME is set)"))
}

/// This machine's name, for the provenance on what it writes.
fn host_id() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| output("hostname", &[]).ok())
        .map(|h| h.trim().to_lowercase())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "local".into())
}

/// `n` random bytes as base64 (URL-safe and unpadded for a password, so it
/// needs no quoting anywhere).
fn secret(n: usize, url_safe: bool) -> String {
    let mut bytes = vec![0u8; n];
    getrandom::fill(&mut bytes).expect("the OS random source");
    if url_safe {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    } else {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }
}

/// A program on the PATH. On Windows, `.exe`, `.cmd` and `.bat` count, and
/// the native Claude Code installer's `~/.local/bin` is looked in too.
fn find_program(name: &str) -> Option<PathBuf> {
    let exts: &[&str] = if cfg!(windows) {
        &[".exe", ".cmd", ".bat"]
    } else {
        &[""]
    };
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    if let Ok(home) = home() {
        dirs.push(home.join(".local").join("bin"));
    }
    for dir in dirs {
        for ext in exts {
            let path = dir.join(format!("{name}{ext}"));
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

/// A command's stdout, or an error when it fails.
fn output(program: &str, args: &[&str]) -> anyhow::Result<String> {
    let path =
        find_program(program).ok_or_else(|| anyhow::anyhow!("{program} is not on the PATH"))?;
    let out = Command::new(path)
        .args(args)
        .stderr(Stdio::piped())
        .output()?;
    if !out.status.success() {
        anyhow::bail!(
            "{program} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn agent(timeout: Duration) -> ureq::Agent {
    agent_within(timeout)
}

fn agent_within(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into()
}

/// Writes a file only its owner can read (a token, the stack's secrets).
fn write_private(path: &Path, text: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(text.as_bytes())?;
        std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, text)?;
        Ok(())
    }
}

#[cfg(unix)]
fn make_executable(path: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))?;
    Ok(())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

fn prompt(question: &str) -> anyhow::Result<String> {
    print!("{question}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

fn confirm(question: &str) -> anyhow::Result<bool> {
    let answer = prompt(&format!("  {question} [Y/n] "))?;
    Ok(!matches!(answer.to_lowercase().as_str(), "n" | "no"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_hook_setup_writes_is_bundled() {
        for (_, script, _) in platform_hooks() {
            assert!(hooks::bundled(script).is_some(), "{script}");
        }
        assert!(hooks::bundled(headers_helper_name()).is_some());
    }

    #[test]
    fn secrets_are_long_and_need_no_quoting() {
        let pass = secret(24, true);
        assert_eq!(pass.len(), 32);
        assert!(pass
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'));
        assert_ne!(pass, secret(24, true));
        assert_eq!(secret(32, false).len(), 44);
    }

    #[test]
    fn a_path_with_spaces_is_quoted_in_the_command() {
        let cmd = script_command(Path::new("/home/a b/.antumbra/hooks/antumbra-capture.sh"));
        assert!(
            cmd.ends_with("\"/home/a b/.antumbra/hooks/antumbra-capture.sh\""),
            "{cmd}"
        );
    }

    #[test]
    fn the_machine_is_registered_under_the_name_its_hooks_report() {
        // A name pinned in the settings before setup is the one the hooks
        // report, so it is the one registered.
        let pinned = json!({ "env": { "ANTUMBRA_HOST_ID": " mac " } });
        assert_eq!(session_host(&pinned), "mac");
        // A blank one names nothing and falls through to a name that is there.
        let blank = json!({ "env": { "ANTUMBRA_HOST_ID": "  " } });
        assert!(!session_host(&blank).trim().is_empty());
        assert!(!session_host(&Value::Null).trim().is_empty());
    }

    #[test]
    fn the_check_tells_a_helper_that_names_the_machine_from_one_that_does_not() {
        let bundled = hooks::bundled(headers_helper_name()).unwrap();
        assert!(
            bundled.contains(DEVICE_HEADER),
            "this build's names the machine"
        );
        assert_eq!(helper_state(Some(bundled)), Helper::Same);
        assert_eq!(
            helper_state(Some(&bundled.replace('\n', "\r\n"))),
            Helper::Same,
            "line endings are not a difference"
        );
        assert_eq!(
            helper_state(Some(&format!("{bundled}\n# edited here\n"))),
            Helper::Edited
        );
        // What setup wrote before writes were stamped per machine.
        let before = "@{ Authorization = \"Bearer $token\" } | ConvertTo-Json -Compress";
        assert_eq!(helper_state(Some(before)), Helper::Unnamed);
        assert_eq!(helper_state(None), Helper::Missing);
    }

    #[test]
    fn the_check_reads_what_the_server_stamps_this_machines_writes_with() {
        let said = |stamped: &str| json!({ "devices": [], "this_device": stamped });
        assert_eq!(stamp_of(&said("windows"), "windows"), Stamp::Named);
        assert_eq!(stamp_of(&said("windows"), " Windows "), Stamp::Named);
        // `local` names no machine, so the server stamps its own name.
        assert_eq!(
            stamp_of(&said("kuskokwim"), "local"),
            Stamp::Other("kuskokwim".into())
        );
        assert_eq!(
            stamp_of(&json!({ "devices": [] }), "windows"),
            Stamp::Unsaid
        );
    }

    #[test]
    fn the_repository_is_found_from_inside_it() {
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        let repo = find_repo(Some(here)).unwrap();
        assert!(repo.join("docker").join("docker-compose.yml").is_file());
        assert!(find_repo(Some(&std::env::temp_dir())).is_err());
    }
}
