//! `antumbra setup`: Antumbra connected to a coding agent in one command.
//!
//! Two ways to run it, and a check:
//!
//! - **`local`**: the store and the MCP server on this machine, in Docker, with
//!   embeddings from ollama. Setup checks what is installed, pulls the
//!   embedding model, fills in `docker/.env` with generated secrets, starts the
//!   stack, mints a token, and proves the server answers with it. Inside a
//!   clone of the repository the server is built from it. Anywhere else setup
//!   writes the compose file it carries to `~/.antumbra/stack` and pulls the
//!   server image published for its own version.
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

use antumbra_http::{SurfaceClient, DEVICE_HEADER};

use crate::agents::{self, Agent};
use crate::claude::{hooks, managed, policy};

/// Where the store's vectors come from: ollama's default address.
const OLLAMA: &str = "http://127.0.0.1:11434";
/// The embedding model, 384 dimensions to match the store.
const MODEL: &str = "all-minilm";
const EMBED_DIM: usize = 384;
/// The MCP server's port in the local stack.
const MCP_PORT: u16 = 8081;
/// The server image, published for every release at the release's version.
/// `setup local` outside a clone runs the one at its own version.
const IMAGE: &str = "ghcr.io/albedosehen/antumbra-mcp";
/// The stack for running without a clone: the compose file and `.env`
/// template this CLI was released with.
const STACK_COMPOSE: &str = include_str!("../../../docker/docker-compose.yml");
const STACK_ENV_EXAMPLE: &str = include_str!("../../../docker/.env.example");

#[derive(Subcommand)]
pub enum SetupMode {
    /// Run the store and the MCP server on this machine, in Docker, and connect
    /// your coding agents to them. Inside a clone of the repository the server
    /// is built from it, and anywhere else the published server image is
    /// pulled.
    Local(LocalArgs),
    /// Connect your coding agents to a hosted workspace, with the token (or
    /// the sign-in link) you were given.
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
    /// The agents to connect, by name: claude, gemini, codex, copilot, goose, opencode, or all. Give it more
    /// than once, or with commas. Without it: Claude Code, and every other
    /// agent found on this machine.
    #[arg(long = "agent", value_delimiter = ',', value_name = "NAME")]
    pub agents: Vec<String>,
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
    /// The repository clone to build the server from (default: the one you are
    /// in). With none, the published server image is pulled instead.
    #[arg(long)]
    pub repo: Option<PathBuf>,
    /// Where ollama answers on this machine.
    #[arg(long, default_value = OLLAMA)]
    pub ollama: String,
    /// The GPU build (Linux, or Windows through WSL2, with an NVIDIA card):
    /// embeddings in-process, answers from trained experts, consolidation on
    /// its own. ollama is not needed with it. It is built from source, so it
    /// runs from a clone.
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
    /// The token, for scripts. Prefer --token-file, or ANTUMBRA_TOKEN, so it
    /// stays out of your shell history.
    #[arg(long, env = "ANTUMBRA_TOKEN", hide_env_values = true, conflicts_with_all = ["token_file", "signin_link"])]
    pub token: Option<String>,
    /// A file holding the token.
    #[arg(long, conflicts_with = "signin_link")]
    pub token_file: Option<PathBuf>,
    /// The sign-in link from your email; setup trades it for the token. For
    /// scripts: run without it and setup asks, so the link stays out of your
    /// shell history.
    #[arg(long)]
    pub signin_link: Option<String>,
    /// Accept a plain http:// address for a server that is not this machine's
    /// own, for a lab. The token then travels in cleartext, and you said so.
    #[arg(long)]
    pub allow_http: bool,
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

impl agents::Report for Steps {
    fn dry_run(&self) -> bool {
        self.dry_run
    }
    fn ok(&self, said: &str) {
        Steps::ok(self, said);
    }
    fn would(&self, said: &str) {
        Steps::would(self, said);
    }
    fn warn(&self, said: &str) {
        Steps::warn(self, said);
    }
    fn info(&self, said: &str) {
        Steps::info(self, said);
    }
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
                // The interactive path is for the address one was given, so
                // the lab exception is a flag, not a question.
                allow_http: false,
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
    // A name --agent does not know is said now, not after the stack is up.
    named_agents(&args.common.agents)?;
    println!("Setting up Antumbra on this machine.\n");

    let stack = find_stack(args.repo.as_deref(), args.gpu)?;
    match &stack {
        Stack::Clone(repo) => steps.ok(format!(
            "repository at {}, which the server is built from",
            repo.display()
        )),
        Stack::Pulled { image } => steps.ok(format!(
            "no clone here, so the server is the published image {image}"
        )),
    }
    let docker = stack.docker(&antumbra);
    let mut compose_files = vec![docker.join("docker-compose.yml")];
    if args.gpu {
        compose_files.push(docker.join("docker-compose.gpu.yml"));
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

    // The stack's compose file, when there is no clone to run it from. It is
    // this release's, so it is replaced when it differs. Settings belong in
    // the .env beside it, which setup only adds to.
    if matches!(stack, Stack::Pulled { .. }) {
        let compose = &compose_files[0];
        if std::fs::read_to_string(compose).ok().as_deref() == Some(STACK_COMPOSE) {
            steps.ok(format!(
                "the stack's compose file is in {}",
                docker.display()
            ));
        } else if steps.dry_run {
            steps.would(format!(
                "write this release's compose file to {}",
                docker.display()
            ));
        } else {
            std::fs::create_dir_all(&docker)?;
            std::fs::write(compose, STACK_COMPOSE)?;
            std::fs::write(docker.join(".env.example"), STACK_ENV_EXAMPLE)?;
            steps.ok(format!(
                "wrote this release's compose file to {}",
                docker.display()
            ));
        }
    }

    // docker/.env.
    let data = antumbra.join("surrealdb");
    let env_path = docker.join(".env");
    let env_name = match &stack {
        Stack::Clone(_) => "docker/.env".to_string(),
        Stack::Pulled { .. } => env_path.display().to_string(),
    };
    let example = match &stack {
        Stack::Clone(_) => std::fs::read_to_string(docker.join(".env.example")).unwrap_or_default(),
        Stack::Pulled { .. } => STACK_ENV_EXAMPLE.to_string(),
    };
    let on_disk = std::fs::read_to_string(&env_path).ok();
    // A stack an earlier setup ran from a clone keeps that clone's secrets:
    // the store's root password is fixed when its data is first written, and
    // the token was signed with the old secret.
    let carried = match (&stack, &on_disk) {
        (Stack::Pulled { .. }, None) => earlier_env(&antumbra),
        _ => None,
    };
    let existing = on_disk.clone().or_else(|| {
        carried
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
    });
    let wants = vec![
        env_file::Want {
            name: "SURREAL_PASS",
            value: secret(24, true)?,
        },
        env_file::Want {
            name: "ANTUMBRA_JWT_SECRET",
            value: secret(32, false)?,
        },
        env_file::Want {
            name: "ANTUMBRA_SURREAL_DATA",
            value: data.to_string_lossy().replace('\\', "/"),
        },
    ];
    let (mut env_text, wrote) = env_file::fill(existing.as_deref(), &example, &wants);
    let mut moved_image = None;
    if let Stack::Pulled { image } = &stack {
        let have = env_file::get(&env_text, "ANTUMBRA_IMAGE");
        if have != Some(image.as_str()) {
            if release_image_moves(have) {
                env_text = env_file::set(&env_text, "ANTUMBRA_IMAGE", image);
                moved_image = Some(image.as_str());
            } else {
                steps.ok(format!(
                    "{env_name} names its own server image, {}, which setup keeps",
                    have.unwrap_or_default()
                ));
            }
        }
    }
    let mut said = Vec::new();
    if let Some(from) = &carried {
        said.push(format!("carried over from {}", from.display()));
    }
    if !wrote.is_empty() {
        said.push(format!("generated {}", wrote.join(", ")));
    }
    if let Some(image) = moved_image {
        said.push(format!("server image {image}"));
    }
    if said.is_empty() {
        steps.ok(format!("{env_name} already holds the stack's secrets"));
    } else if steps.dry_run {
        for part in &said {
            steps.would(format!("write {env_name}: {part}"));
        }
    } else {
        std::fs::create_dir_all(&data)?;
        if let Some(caveat) = write_private(&env_path, &env_text)? {
            steps.warn(caveat);
        }
        for part in &said {
            steps.ok(format!("{env_name}: {part}"));
        }
    }
    let data_dir = env_file::get(&env_text, "ANTUMBRA_SURREAL_DATA")
        .map(PathBuf::from)
        .unwrap_or(data);
    if !steps.dry_run {
        std::fs::create_dir_all(&data_dir)?;
        give_to_container(&data_dir, &steps)?;
    }

    // The stack.
    let image = env_file::get(&env_text, "ANTUMBRA_IMAGE")
        .unwrap_or_default()
        .to_string();
    let build = match &stack {
        Stack::Clone(_) => "--build",
        Stack::Pulled { .. } => "--no-build",
    };
    let mut up: Vec<String> = compose_args(&compose_files);
    up.extend(["up", "-d", build, "surrealdb", "antumbra-mcp"].map(String::from));
    if steps.dry_run {
        if matches!(stack, Stack::Pulled { .. }) {
            steps.would(format!("pull {image} unless Docker has it"));
        }
        steps.would(format!("run docker {}", up.join(" ")));
    } else {
        match &stack {
            Stack::Clone(_) => println!("\n  Starting the store and the server. The first run builds the server image,\n  which takes several minutes; later runs rebuild only what the clone changed.\n"),
            Stack::Pulled { .. } => {
                pull_image(&image, &env_name, &steps)?;
                println!("\n  Starting the store and the server.\n");
            }
        }
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
        // A token from before the operator claim is replaced: on their own
        // stack, the person is its operator.
        let token = match std::fs::read_to_string(&token_path)
            .ok()
            .map(|t| t.trim().to_string())
            .filter(|t| {
                token::looks_like_jwt(t) && token::is_operator(t) && verify(&url, t).is_ok()
            }) {
            Some(t) => {
                steps.ok(format!(
                    "kept the token in {}, which the server accepts",
                    token_path.display()
                ));
                t
            }
            None => {
                let t = mint(&compose_files, &args.workspace, &args.user)?;
                if let Some(caveat) = write_private(&token_path, &t)? {
                    steps.warn(caveat);
                }
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
        "repo": match &stack {
            Stack::Clone(repo) => json!(repo),
            Stack::Pulled { .. } => Value::Null,
        },
        "image": match &stack {
            Stack::Clone(_) => Value::Null,
            Stack::Pulled { .. } => Value::String(image),
        },
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

/// Where the local stack runs from, which decides how the server image is had.
enum Stack {
    /// A clone of the repository: the image is built from it.
    Clone(PathBuf),
    /// No clone: the compose file this CLI carries, written to
    /// `~/.antumbra/stack/docker`, and the release image at this CLI's
    /// version, pulled.
    Pulled { image: String },
}

impl Stack {
    /// The directory holding the compose files and `.env`.
    fn docker(&self, antumbra: &Path) -> PathBuf {
        match self {
            Stack::Clone(repo) => repo.join("docker"),
            Stack::Pulled { .. } => antumbra.join("stack").join("docker"),
        }
    }
}

/// A clone to build from when there is one, the published image when not.
/// The GPU build has no published image, so it needs the clone.
fn find_stack(given: Option<&Path>, gpu: bool) -> anyhow::Result<Stack> {
    if let Some(repo) = find_repo(given)? {
        return Ok(Stack::Clone(repo));
    }
    if gpu {
        anyhow::bail!(
            "{} is not inside a clone of the Antumbra repository, and the GPU build is built from source.\n  Clone it, cd into it, and run this again (or pass --repo <path>).",
            std::env::current_dir()?.display()
        );
    }
    Ok(Stack::Pulled {
        image: format!("{IMAGE}:{}", env!("CARGO_PKG_VERSION")),
    })
}

/// The repository clone to build the server from: the one given, or the one
/// the current directory is in. `None` when no clone was given and there is
/// none here.
fn find_repo(given: Option<&Path>) -> anyhow::Result<Option<PathBuf>> {
    let start = match given {
        Some(p) => p.to_path_buf(),
        None => std::env::current_dir()?,
    };
    for dir in start.ancestors() {
        if dir.join("docker").join("docker-compose.yml").is_file()
            && dir.join("crates").join("antumbra-mcp").is_dir()
        {
            return Ok(Some(dir.to_path_buf()));
        }
    }
    if given.is_some() {
        anyhow::bail!(
            "{} is not inside a clone of the Antumbra repository.\n  Pass --repo the clone's path, or leave --repo out to run the published server image.",
            start.display()
        );
    }
    Ok(None)
}

/// The `.env` of the clone an earlier setup ran the stack from, when it is
/// still there.
fn earlier_env(antumbra: &Path) -> Option<PathBuf> {
    let state: Value =
        serde_json::from_str(&std::fs::read_to_string(antumbra.join("setup.json")).ok()?).ok()?;
    let env = PathBuf::from(state.get("repo")?.as_str()?)
        .join("docker")
        .join(".env");
    env.is_file().then_some(env)
}

/// Whether setup moves the `.env`'s server image to this CLI's release: when
/// it names none, or names a release image. One named by hand stays.
fn release_image_moves(have: Option<&str>) -> bool {
    match have {
        None => true,
        Some(image) => image.is_empty() || image.starts_with(&format!("{IMAGE}:")),
    }
}

/// The server image, pulled unless Docker already has it.
fn pull_image(image: &str, env_name: &str, steps: &Steps) -> anyhow::Result<()> {
    let have = Command::new("docker")
        .args(["image", "inspect", "--format", "{{.Id}}", image])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if have {
        steps.ok(format!("Docker has the server image {image}"));
        return Ok(());
    }
    println!("\n  Downloading the server image {image}.\n");
    let status = Command::new("docker").args(["pull", image]).status()?;
    if !status.success() {
        anyhow::bail!(
            "`docker pull {image}` failed (its output is above).\n  A CLI built from source has no published image at its version: run setup inside your clone to build the server there, or set ANTUMBRA_IMAGE in {env_name} to a released image."
        );
    }
    println!();
    steps.ok(format!("pulled the server image {image}"));
    Ok(())
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
        let mut response = antumbra_http::agent(Duration::from_secs(1800))
            .post(&format!("{base}/api/pull"))
            .send_json(json!({ "model": MODEL, "stream": false }))
            .map_err(|e| anyhow::anyhow!("pulling {MODEL}: {e}"))?;
        if !response.status().is_success() {
            let said = response.body_mut().read_to_string().unwrap_or_default();
            anyhow::bail!("ollama could not pull {MODEL}: {said}");
        }
    }
    // The first embedding loads the model into memory, which takes a while.
    let mut response = antumbra_http::agent(Duration::from_secs(120))
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
    let elsewhere = "(Running ollama somewhere the container can reach? Set ANTUMBRA_EMBEDDER_URL in the stack's .env: docker/.env in a clone, ~/.antumbra/stack/docker/.env without one.)";
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
/// The stack is the person's own, so the token is its operator's and `antumbra
/// tui` may act on it. An image from before the operator claim (one named by
/// hand in the stack's `.env`) mints a plain token instead, and says so.
fn mint(compose_files: &[PathBuf], workspace: &str, user: &str) -> anyhow::Result<String> {
    match mint_with(compose_files, workspace, user, true) {
        Ok(token) => Ok(token),
        Err(e) if e.to_string().contains("--operator") => {
            eprintln!(
                "  this server image cannot mint an operator's token; `antumbra tui` will be read-only"
            );
            mint_with(compose_files, workspace, user, false)
        }
        Err(e) => Err(e),
    }
}

fn mint_with(
    compose_files: &[PathBuf],
    workspace: &str,
    user: &str,
    operator: bool,
) -> anyhow::Result<String> {
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
    if operator {
        args.push("--operator".into());
    }
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
    may_carry_the_token(&url, args.allow_http)?;
    named_agents(&args.common.agents)?;
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
            // The link's answer is the token itself.
            may_carry_the_token(&link, args.allow_http)?;
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
        if let Some(caveat) = write_private(&token_path, &token)? {
            steps.warn(caveat);
        }
        steps.ok(format!("token saved to {}", token_path.display()));
    }

    let state = json!({
        "mode": "hosted",
        "url": url,
        "workspace": workspace,
        "token_file": token_path,
        "allow_http": args.allow_http,
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

/// Refuse `address` unless the token may travel over it: `https://` anywhere,
/// plain `http://` to this machine alone, or plain `http://` anywhere when the
/// operator said so with `--allow-http`.
pub(crate) fn may_carry_the_token(address: &str, allow_http: bool) -> anyhow::Result<()> {
    match cleartext_problem(address, allow_http) {
        None => Ok(()),
        Some(problem) => Err(anyhow::anyhow!("{problem}")),
    }
}

/// What is wrong with the token travelling to `address`, if anything: no
/// scheme, or plain `http://` to a machine other than this one without
/// `allow_http`. `None` when the address is fine.
fn cleartext_problem(address: &str, allow_http: bool) -> Option<String> {
    if address.starts_with("https://") {
        return None;
    }
    let Some(rest) = address.strip_prefix("http://") else {
        return Some(format!(
            "the address must start with https:// (got {address:?})"
        ));
    };
    if allow_http || is_this_machine(host_of(rest)) {
        return None;
    }
    Some(format!(
        "{address:?} is a plain http:// address, so the token would travel in cleartext. \
         Use the https:// address, or pass --allow-http if this is a lab where that is accepted"
    ))
}

/// The host in what follows a URL's scheme: the authority up to the first
/// `/`, `?` or `#`, without userinfo or a port, and a bracketed IPv6 address
/// without its brackets.
pub(crate) fn host_of(rest: &str) -> &str {
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    if let Some(bracketed) = authority.strip_prefix('[') {
        return bracketed.split(']').next().unwrap_or(bracketed);
    }
    match authority.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => host,
        _ => authority,
    }
}

/// Whether `host` names this machine: `localhost`, or a loopback address.
pub(crate) fn is_this_machine(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
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
    let variable = |name: &str| std::env::var(name).ok();
    // A setup run from a session Unity Gateway started inherits what `ug`
    // set for the agent it started, which is not the user's own.
    let env = agents::gateway::outside(home, &variable);
    let find = |name: &str| find_running(name);
    let ground = agents::Ground {
        home,
        antumbra: agents::Program::this(find_program("antumbra").is_some()),
        env: &env,
        find: &find,
    };
    let chosen = chosen(common, &ground)?;
    // A command that is there and does not run is said, so an agent left out
    // for it is not a silence.
    for agent in Agent::ALL {
        if let Some(path) = find_program(agent.program()).filter(|path| !runs(path)) {
            steps.info(format!(
                "{} was found at {} and does not run (`{} --version` fails), so it is not taken as installed",
                agent.title(),
                path.display(),
                agent.program()
            ));
        }
    }
    // Claude Code's own settings, as they stand when they are asked for.
    let settings = || -> Value {
        json_at(
            &common
                .claude_dir
                .clone()
                .unwrap_or_else(|| home.join(".claude"))
                .join("settings.json"),
        )
        .unwrap_or(Value::Null)
    };
    let mut connected: Vec<&str> = Vec::new();
    if chosen.claude {
        connect_claude(home, antumbra, common, url, workspace, steps)?;
        connected.push(CLAUDE);
        // Claude Code as Unity Gateway starts it: a launch that leaves the
        // user's settings out is given Antumbra's in a file of its own.
        if agents::gateway::present(&ground) {
            let launch = json_at(&agents::gateway::claude_settings(&ground))
                .as_ref()
                .and_then(agents::gateway::claude_launch);
            let alone = claude_settings_alone(antumbra, url, workspace, &settings())?;
            agents::gateway::connect_claude(antumbra, &alone, launch, steps)?;
        }
    } else if common.no_claude {
        steps.info("left Claude Code alone (--no-claude)");
    } else {
        steps.info("left Claude Code alone (--agent does not name it)");
    }
    if !chosen.others.is_empty() && !chosen.claude {
        // Claude Code's hooks are the same scripts; without it, they are
        // still what `antumbra hook` runs.
        write_hooks(&antumbra.join("hooks"), &agents::hook::scripts(), steps)?;
    }
    // The ones connected because they were found here, which nobody named.
    let mut found: Vec<Agent> = Vec::new();
    for agent in &chosen.others {
        // One agent's config that cannot be written is that agent's loss,
        // not the setup's.
        match agents::connect(*agent, &ground, steps) {
            Ok(()) => {
                connected.push(agent.name());
                if chosen.found {
                    found.push(*agent);
                }
            }
            Err(e) => steps.warn(format!("{} was not connected: {e}", agent.title())),
        }
    }
    // After the settings, so the name read is the one the hooks will report.
    let settings = settings();
    let token = std::fs::read_to_string(antumbra.join("token.txt"))
        .map(|t| t.trim().to_string())
        .unwrap_or_default();
    let host = session_host(&settings);
    register_machine(url, &token, &host, steps);
    if !steps.dry_run {
        let mut state = state.clone();
        // What `antumbra hook` and `antumbra mcp bridge` read, for an agent
        // whose settings carry no `env` of Antumbra's, and what the check
        // goes by.
        state["host"] = Value::String(host);
        state["agents"] = json!(connected);
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
        match titles(&connected) {
            Some(agents) => {
                println!(
                    "\nAntumbra is set up. Start a new session in {agents}: it opens with what"
                );
                println!("Antumbra remembers, and what you work on goes back in.");
            }
            None => println!("\nAntumbra is set up. No agent was connected to it."),
        }
        if let Some(found) = found_line(&found) {
            println!("{found}");
        }
        println!("Check it any time with: antumbra setup check");
    }
    Ok(())
}

/// Claude Code's name among the agents a setup records.
const CLAUDE: &str = "claude";

/// The agents one run connects.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Chosen {
    claude: bool,
    others: Vec<Agent>,
    /// Whether the others are the ones found on this machine, and not ones
    /// `--agent` named.
    found: bool,
}

/// The agents `--agent` names, or nothing when it names none. A name it does
/// not know is an error that lists the ones it takes.
fn named_agents(names: &[String]) -> anyhow::Result<Option<Chosen>> {
    let mut chosen = Chosen {
        claude: false,
        others: Vec::new(),
        found: false,
    };
    let mut any = false;
    for name in names {
        let name = name.trim().to_lowercase();
        if name.is_empty() {
            continue;
        }
        any = true;
        match name.as_str() {
            "all" => {
                chosen.claude = true;
                chosen.others = Agent::ALL.to_vec();
            }
            CLAUDE => chosen.claude = true,
            other => match Agent::named(other) {
                Some(agent) if !chosen.others.contains(&agent) => chosen.others.push(agent),
                Some(_) => {}
                None => anyhow::bail!(
                    "--agent {other}: no such agent. It takes {CLAUDE}, {}, or all.",
                    Agent::ALL.map(Agent::name).join(", ")
                ),
            },
        }
    }
    Ok(any.then_some(chosen))
}

/// The agents this run connects: the ones `--agent` names, or, when it names
/// none, Claude Code and every other agent found on this machine.
/// `--no-claude` leaves Claude Code out either way.
fn chosen(common: &Common, ground: &agents::Ground<'_>) -> anyhow::Result<Chosen> {
    let mut chosen = match named_agents(&common.agents)? {
        Some(named) => named,
        None => Chosen {
            claude: true,
            others: Agent::ALL
                .into_iter()
                .filter(|agent| agents::present(*agent, ground))
                .collect(),
            found: true,
        },
    };
    if common.no_claude {
        chosen.claude = false;
    }
    Ok(chosen)
}

/// The agents `names` record, as a person reads them: "Claude Code and
/// Gemini CLI". Nothing for none.
fn titles(names: &[&str]) -> Option<String> {
    let titles: Vec<&str> = names
        .iter()
        .filter_map(|name| match *name {
            CLAUDE => Some("Claude Code"),
            other => Agent::named(other).map(Agent::title),
        })
        .collect();
    match titles.as_slice() {
        [] => None,
        [one] => Some((*one).to_string()),
        [first @ .., last] => Some(format!("{} and {last}", first.join(", "))),
    }
}

/// What the summary says of the agents connected because they were found on
/// this machine, so that one nobody asked for is seen there, and not first in
/// a check that fails. Nothing for none.
fn found_line(found: &[Agent]) -> Option<String> {
    let names: Vec<&str> = found.iter().map(|agent| agent.name()).collect();
    let agents = titles(&names)?;
    let were_found = match found {
        [_] => "was connected because it was found",
        _ => "were connected because they were found",
    };
    Some(format!(
        "{agents} {were_found} on this machine.\nTo choose the agents yourself, name them with --agent."
    ))
}

/// Write the bundled hook scripts `names` into `dir`, replacing what is there.
fn write_hooks(dir: &Path, names: &[String], steps: &Steps) -> anyhow::Result<()> {
    if steps.dry_run {
        steps.would(format!("write the hooks to {}", dir.display()));
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    for name in names {
        let text = hooks::bundled(name)
            .ok_or_else(|| anyhow::anyhow!("{name} is not bundled in this build"))?;
        let path = dir.join(name);
        std::fs::write(&path, text)?;
        make_executable(&path)?;
    }
    steps.ok(format!("hooks written to {}", dir.display()));
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
        // Ten seconds: the hook's own work is bounded (two for its report to
        // the server, a few git calls), but starting PowerShell alone takes
        // three on a machine that is busy building, and a hook cut off at
        // five skipped its nudge and left its sentinel behind.
        (
            "Stop",
            pick("antumbra-capture.sh", "antumbra-capture.ps1"),
            10,
        ),
        (
            "PreCompact",
            pick("antumbra-capture.sh", "antumbra-capture.ps1"),
            10,
        ),
    ]
}

/// The two hooks that count which skills are used. A skill is used two ways
/// and each hook sees one: the agent calls the `Skill` tool, or a person
/// types `/name`, which never touches the tool. Both run `antumbra claude
/// skill-used`, which reads the hook's input itself, so there is no script
/// and nothing differs between shells. It never fails and never speaks, and
/// it is not waited for. Counting was a snippet for a person to paste into
/// their settings, which nobody did.
fn skill_hooks() -> Vec<settings::Hook> {
    let command = format!("{} claude skill-used", antumbra_command());
    [
        ("PostToolUse", Some("Skill")),
        ("UserPromptExpansion", None),
    ]
    .into_iter()
    .map(|(event, tool)| settings::Hook {
        event,
        script: SKILL_USED,
        command: command.clone(),
        timeout: 10,
        tool,
        waits: false,
    })
    .collect()
}

/// The words that name the skill counter in a hook's command.
const SKILL_USED: &str = "claude skill-used";

/// How a hook runs this program: by its name where that is on the path,
/// which survives an upgrade that moves it, else by where it is now.
fn antumbra_command() -> String {
    if find_program("antumbra").is_some() {
        return "antumbra".to_string();
    }
    match std::env::current_exe() {
        Ok(path) => format!("\"{}\"", path.display()),
        Err(_) => "antumbra".to_string(),
    }
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

/// What setup wires into Claude Code's settings, but for whether its file
/// memory is turned off: the hooks, run from `dir` (`~/.antumbra/hooks`),
/// where the server is, and this machine's name where the settings give none.
fn claude_wiring(dir: &Path, url: &str, workspace: &str) -> settings::Wiring {
    settings::Wiring {
        hooks: platform_hooks()
            .into_iter()
            .map(|(event, script, timeout)| {
                settings::Hook::script(event, script, script_command(&dir.join(script)), timeout)
            })
            .chain(skill_hooks())
            .collect(),
        env: vec![
            ("ANTUMBRA_URL".into(), url.to_string()),
            ("ANTUMBRA_WORKSPACE_ID".into(), workspace.to_string()),
        ],
        env_default: vec![("ANTUMBRA_HOST_ID".into(), host_id())],
        auto_memory_off: false,
        file: "settings.json",
    }
}

/// Claude Code settings that hold what setup wires and nothing else, for a
/// launch that leaves the user's own out (see [`agents::gateway`]): the
/// hooks, where the server is, the name `settings` (the user's own) has this
/// machine's sessions report, and Claude Code's file memory off where
/// `settings` has it off.
fn claude_settings_alone(
    antumbra: &Path,
    url: &str,
    workspace: &str,
    settings: &Value,
) -> anyhow::Result<String> {
    let wiring = settings::Wiring {
        env_default: vec![("ANTUMBRA_HOST_ID".into(), session_host(settings))],
        auto_memory_off: settings.get("autoMemoryEnabled") == Some(&Value::Bool(false)),
        file: agents::gateway::CLAUDE_SETTINGS,
        ..claude_wiring(&antumbra.join("hooks"), url, workspace)
    };
    Ok(settings::wire("", &wiring)?.text)
}

/// The JSON a file holds, or nothing when it is not there or does not parse.
fn json_at(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
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
    let mut names: Vec<String> = platform_hooks()
        .iter()
        .map(|(_, s, _)| (*s).to_string())
        .collect();
    names.push(headers_helper_name().to_string());
    names.dedup();
    write_hooks(&dir, &names, steps)?;

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
        auto_memory_off,
        ..claude_wiring(&dir, url, workspace)
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
    )
    .map(|_| ())
}

/// One tool call through the server's REST shim, its answer on success; a
/// refusal is the server's own words, as every door shows them.
fn call_tool(url: &str, token: &str, tool: &str, arguments: Value) -> anyhow::Result<Value> {
    call_tool_from(url, token, None, tool, arguments)
}

/// [`call_tool`] from the machine `device` names, as the agent's connection
/// calls once the headers helper names it.
fn call_tool_from(
    url: &str,
    token: &str,
    device: Option<&str>,
    tool: &str,
    arguments: Value,
) -> anyhow::Result<Value> {
    Ok(
        SurfaceClient::new(url, Some(token), Duration::from_secs(30))
            .from_machine(device.map(str::to_owned))
            .call(tool, arguments)?,
    )
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

/// The name setup registers the MCP server under with Claude Code.
const CLAUDE_SERVER: &str = "antumbra";

/// What is said of a file that is the organization's to change.
const THEIRS: &str = "that file is your organization's: ask whoever manages it";

/// What is said of a file Unity Gateway writes. It writes it from the
/// workspace's configuration and writes it again, so what it holds is the
/// workspace's to change and not the user's to edit.
const WORKSPACES: &str =
    "that file is Unity Gateway's, written from your workspace's configuration: ask whoever manages the workspace";

/// Whose a file that stops something is to change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Whose {
    Organization,
    Workspace,
    User,
}

impl Whose {
    /// `by` is the organization's when `managed`, and the workspace's when it
    /// is `gateway`, the settings file Unity Gateway passes Claude Code.
    fn of(by: &Path, managed: bool, gateway: Option<&Path>) -> Self {
        if managed {
            Whose::Organization
        } else if gateway == Some(by) {
            Whose::Workspace
        } else {
            Whose::User
        }
    }
}

/// What the check says when a file stops the hooks setup wired into Claude
/// Code, and what to do about it. Nothing when they run. `gateway` is the
/// settings file Unity Gateway passes, where the launch judged is its.
fn hooks_held_back(hooks: &policy::Hooks, gateway: Option<&Path>) -> Option<(String, String)> {
    let policy::Hooks::Stopped { by, key, managed } = hooks else {
        return None;
    };
    let why = match *key {
        "allowManagedHooksOnly" => "Claude Code runs only your organization's hooks",
        "strictPluginOnlyCustomization" => "Claude Code runs no hook from a settings file of yours",
        _ => "Claude Code's hooks are turned off",
    };
    let said = format!(
        "{why} ({key} in {}), so Antumbra's do not run: nothing is recalled for a prompt, and nothing asks for what a turn learned",
        by.display()
    );
    let tools = "The memory tools still work when they are asked for";
    let fix = match Whose::of(by, *managed, gateway) {
        Whose::Organization => format!("{THEIRS}. {tools}"),
        Whose::Workspace => format!("{WORKSPACES}. {tools}"),
        Whose::User => format!("remove {key} from that file"),
    };
    Some((said, fix))
}

/// What the check says when a file keeps Claude Code from loading the MCP
/// server setup registered at `url`, and what to do about it. Nothing when it
/// loads. `gateway` is the settings file Unity Gateway passes, where the
/// launch judged is its.
fn server_held_back(
    server: &policy::Server,
    url: &str,
    gateway: Option<&Path>,
) -> Option<(String, String)> {
    let policy::Server::Blocked { by, managed, why } = server else {
        return None;
    };
    let whose = Whose::of(by, *managed, gateway);
    let by = by.display();
    let (said, fix) = match why {
        policy::Why::Fixed => (
            format!("{by} fixes the MCP servers Claude Code loads, and \"{CLAUDE_SERVER}\" is not one of them, so it is not connected"),
            format!("{THEIRS}, to add it there"),
        ),
        policy::Why::PluginsOnly => (
            format!("Claude Code loads no MCP server a user registered (strictPluginOnlyCustomization in {by}), so \"{CLAUDE_SERVER}\" is not connected"),
            THEIRS.to_string(),
        ),
        policy::Why::Denied => (
            format!("a deniedMcpServers entry in {by} blocks the MCP server \"{CLAUDE_SERVER}\" ({url}), so it is not connected"),
            match whose {
                Whose::Organization => THEIRS.to_string(),
                Whose::Workspace => WORKSPACES.to_string(),
                Whose::User => "remove that entry from that file".to_string(),
            },
        ),
        policy::Why::Unlisted { entry, open } => (
            format!("{by} lists the MCP servers Claude Code may load (allowedMcpServers), and \"{CLAUDE_SERVER}\" ({url}) is not one, so it is not connected"),
            match (whose, *open) {
                (Whose::User, _) => format!("add {entry} to allowedMcpServers in that file"),
                (Whose::Workspace, _) => format!("{WORKSPACES}, to add {entry}"),
                (Whose::Organization, false) => format!("{THEIRS}, to add {entry}"),
                // Without `allowManagedMcpServersOnly` the list is one the
                // user's own settings add to, which the vendor's page says
                // in as many words. Whether to is between them and whoever
                // set the list, so it is said and not done.
                (Whose::Organization, true) => format!(
                    "{THEIRS}, to add {entry}. Claude Code also counts an allowedMcpServers of your own settings as part of that list (allowManagedMcpServersOnly is not set)"
                ),
            },
        ),
    };
    Some((said, fix))
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
    // One thing found about an agent: in place, or what to do about it. What
    // works, or waits on something setup cannot change, is said as a
    // differing hook of Claude Code's is, and is not a failed check.
    let say = |finding: agents::Finding, fail: &mut dyn FnMut(String, String)| match finding.fix {
        None => println!("  \u{2713} {}", finding.said),
        Some(fix) if finding.works => println!("  ! {}; {fix}", finding.said),
        Some(fix) => fail(finding.said, fix),
    };
    let variable = |name: &str| std::env::var(name).ok();
    let env = agents::gateway::outside(&home, &variable);
    let find = |name: &str| find_running(name);
    let ground = agents::Ground {
        home: &home,
        antumbra: agents::Program::this(find_program("antumbra").is_some()),
        env: &env,
        find: &find,
    };

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
        if let Some(problem) = private_file_problem(&token_path) {
            fail(
                problem,
                "run setup again, which rewrites it readable by you alone".into(),
            );
        }
        if let Some(exp) = token::expires(&token) {
            let left = exp - chrono::Utc::now().timestamp();
            if left <= 3 * 86_400 {
                println!(
                    "  ! the token expires in {} hours; ask for a new sign-in link and run `antumbra setup hosted <url>` again before then",
                    left.max(0) / 3600
                );
            }
        }
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
    let settings: Value = json_at(&settings_path).unwrap_or(Value::Null);
    // What decides whether Claude Code runs what setup wired: the
    // organization's files, then the user's own. Reading back what setup
    // wrote cannot see a policy that stops it.
    let theirs: Vec<policy::Source> = managed::files(&managed::dir())
        .into_iter()
        .map(|(path, object)| policy::Source::managed(path, Value::Object(object)))
        .collect();
    let own = policy::Source::user(settings_path.clone(), settings.clone());
    let plain: Vec<policy::Source> = theirs.iter().cloned().chain([own.clone()]).collect();
    let fixed = policy::fixed(&managed::dir());
    let server_url = format!("{url}/mcp");
    let hooks_run = policy::hooks(&plain);
    let server_loads = policy::server(&plain, fixed.as_ref(), CLAUDE_SERVER, &server_url);
    // A setup that recorded its agents is checked for those. One from before
    // setup recorded them connected Claude Code, and is checked for that.
    let recorded: Option<Vec<String>> = state.get("agents").and_then(Value::as_array).map(|all| {
        all.iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()
    });
    let claude_connected = recorded
        .as_ref()
        .is_none_or(|all| all.iter().any(|name| name == CLAUDE));
    if !claude_connected {
        println!(
            "    Claude Code is not one of this setup's agents, so its settings were not checked"
        );
    }
    if claude_connected {
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
        if let Some((said, fix)) = hooks_held_back(&hooks_run, None) {
            fail(said, fix);
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
    }
    // A plain http:// address to another machine carries the token in
    // cleartext; setup refuses it unless told --allow-http, which it records.
    let allow_http = state
        .get("allow_http")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(problem) = cleartext_problem(&url, allow_http) {
        fail(
            problem,
            "use the https:// address, or run setup again with --allow-http if this is a lab"
                .into(),
        );
    }
    if claude_connected {
        // A server the organization keeps out is gone from Claude Code's own
        // list, and registering it again is refused, so that is what is said.
        match (
            server_held_back(&server_loads, &server_url, None),
            find_program("claude"),
        ) {
            (Some((said, fix)), _) => fail(said, fix),
            (None, Some(claude)) => {
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
            (None, None) => println!(
            "  ! the claude command is not on your PATH, so the MCP registration was not checked"
        ),
        }
    }
    // Claude Code as Unity Gateway starts it: the settings of Antumbra's own
    // for a launch that leaves the user's out, and what such a launch's own
    // settings stop that a plain launch's do not.
    if claude_connected && agents::gateway::present(&ground) {
        let passed_path = agents::gateway::claude_settings(&ground);
        let passed = json_at(&passed_path);
        let launch = passed.as_ref().and_then(agents::gateway::claude_launch);
        let workspace = state
            .get("workspace")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match claude_settings_alone(&antumbra, &url, workspace, &settings) {
            Ok(alone) => {
                for finding in agents::gateway::check_claude(&antumbra, &alone, launch) {
                    say(finding, &mut fail);
                }
            }
            Err(e) => fail(format!("{e}"), "run setup again".into()),
        }
        if let Some(passed) = passed {
            let with_own = launch != Some(agents::gateway::ClaudeLaunch::WithoutUserSettings);
            let through: Vec<policy::Source> = theirs
                .iter()
                .cloned()
                .chain([policy::Source::user(passed_path.clone(), passed)])
                .chain(with_own.then_some(own))
                .collect();
            let through_ug = |said: String| format!("started by `ug claude`: {said}");
            let gateway = Some(passed_path.as_path());
            let hooks_there = policy::hooks(&through);
            if let Some((said, fix)) =
                hooks_held_back(&hooks_there, gateway).filter(|_| hooks_there != hooks_run)
            {
                fail(through_ug(said), fix);
            }
            let server_there = policy::server(&through, fixed.as_ref(), CLAUDE_SERVER, &server_url);
            if let Some((said, fix)) = server_held_back(&server_there, &server_url, gateway)
                .filter(|_| server_there != server_loads)
            {
                fail(through_ug(said), fix);
            }
        }
    }
    // The other agents: what setup wrote for each, read back, and the scripts
    // their hooks run.
    let others: Vec<Agent> = recorded
        .iter()
        .flatten()
        .filter_map(|name| Agent::named(name))
        .collect();
    if !others.is_empty() {
        for name in agents::hook::scripts() {
            let path = antumbra.join("hooks").join(&name);
            if !path.is_file() {
                fail(
                    format!("{} is missing, and `antumbra hook` runs it", path.display()),
                    "run setup again to write it".into(),
                );
            }
        }
        for agent in others {
            for finding in agents::check(agent, &ground) {
                say(finding, &mut fail);
            }
            // What an organization's requirements stop of it, which what
            // setup wrote does not show.
            if agent == Agent::Codex {
                let requirements = agents::codex::requirements(&variable);
                if let Ok(text) = std::fs::read_to_string(&requirements) {
                    for finding in agents::codex::held_back(&text, &requirements) {
                        say(finding, &mut fail);
                    }
                }
            }
        }
    }
    if reachable {
        let host = session_host(&settings).to_lowercase();
        // Asked under this machine's name, as the agent's connection asks once
        // its headers helper names it, so the answer says what the server
        // stamps that connection's writes with. Whether the helper names it is
        // the helper line above; the two together are what the agent gets.
        match call_tool_from(&url, &token, Some(&host), "devices", json!({})) {
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
    crate::machine::home()
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
/// needs no quoting anywhere). Without the OS random source there is no
/// secret worth the name, so that failure is setup's to report, not to paper
/// over.
fn secret(n: usize, url_safe: bool) -> anyhow::Result<String> {
    let mut bytes = vec![0u8; n];
    getrandom::fill(&mut bytes).map_err(|e| {
        anyhow::anyhow!("the OS random source failed ({e}); the stack's secrets cannot be made")
    })?;
    Ok(if url_safe {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    } else {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    })
}

/// A program on the PATH. On Windows, `.exe`, `.cmd` and `.bat` count, and
/// the native Claude Code installer's `~/.local/bin` is looked in too.
/// How long a program is given to say its version before it is taken to run.
const VERSION_WAIT: Duration = Duration::from_secs(5);

/// Whether the program at `path` runs: `--version` exits with success. A
/// file by an agent's name on the PATH is not always the agent: a wrapper
/// another tool installs can be there and fail for want of the real thing.
/// One that has not answered in [`VERSION_WAIT`] is stopped and taken to run:
/// slowness is not a failure. Asked once for a path in a run.
pub(crate) fn runs(path: &Path) -> bool {
    static ASKED: std::sync::Mutex<Option<std::collections::HashMap<PathBuf, bool>>> =
        std::sync::Mutex::new(None);
    // A poisoned lock holds a whole map: each entry is written in one step.
    let known = |path: &Path| {
        ASKED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_or_insert_with(Default::default)
            .get(path)
            .copied()
    };
    if let Some(known) = known(path) {
        return known;
    }
    let answer = match Command::new(path)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Err(_) => false,
        Ok(mut child) => {
            let started = std::time::Instant::now();
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => break status.success(),
                    Ok(None) if started.elapsed() < VERSION_WAIT => {
                        std::thread::sleep(Duration::from_millis(25));
                    }
                    Ok(None) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break true;
                    }
                    Err(_) => break false,
                }
            }
        }
    };
    ASKED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get_or_insert_with(Default::default)
        .insert(path.to_path_buf(), answer);
    answer
}

/// [`find_program`], for a program that also runs ([`runs`]): how an agent
/// is found on this machine.
pub(crate) fn find_running(name: &str) -> Option<PathBuf> {
    find_program(name).filter(|path| runs(path))
}

pub(crate) fn find_program(name: &str) -> Option<PathBuf> {
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

/// The agent every call setup makes is made with: a refusal's body comes back
/// rather than becoming a bare status.
fn agent(timeout: Duration) -> ureq::Agent {
    antumbra_http::agent(timeout)
}

/// Writes a file only its owner can read (a token, the stack's secrets). The
/// directory it lives in is made the owner's alone; the text goes to a
/// temporary name beside it, with the final permissions, and is renamed into
/// place, so a crash mid-write leaves the old file or none, never a truncated
/// token. On Windows the file is given an owner-only DACL after the rename.
/// What could not be tightened comes back as a caveat for the caller to show.
pub(crate) fn write_private(path: &Path, text: &str) -> anyhow::Result<Option<String>> {
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let temporary = {
        let mut name = path
            .file_name()
            .map(|n| n.to_os_string())
            .unwrap_or_else(|| "private".into());
        name.push(".tmp");
        parent.join(name)
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)?;
        // A directory that was there before, however it was made, holds the
        // same secrets: it is the owner's alone too.
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o600))?;
        std::fs::rename(&temporary, path)?;
        Ok(None)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(parent)?;
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)?;
        // The file inherits the directory's ACL; cut the inheritance and grant
        // the owner alone. icacls ships with Windows, so a failure here is a
        // restricted account or a filesystem without ACLs, worth a word.
        let Ok(user) = std::env::var("USERNAME") else {
            return Ok(Some(format!(
                "{} keeps the permissions it inherited; USERNAME is not set, so it could not be made yours alone",
                path.display()
            )));
        };
        let tightened = std::process::Command::new("icacls")
            .arg(path)
            .args(["/inheritance:r", "/grant:r", &format!("{user}:F")])
            .output();
        match tightened {
            Ok(out) if out.status.success() => Ok(None),
            Ok(out) => Ok(Some(format!(
                "{} keeps the permissions it inherited; icacls refused: {}",
                path.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            ))),
            Err(e) => Ok(Some(format!(
                "{} keeps the permissions it inherited; icacls could not run: {e}",
                path.display()
            ))),
        }
    }
}

/// Why `path` is not the owner's alone, when it is not: a token readable by
/// another user of the machine is that user's token too.
#[cfg(unix)]
fn private_file_problem(path: &Path) -> Option<String> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path).ok()?.permissions().mode() & 0o777;
    (mode & 0o077 != 0).then(|| {
        format!(
            "{} is readable by other users of this machine (mode {mode:o})",
            path.display()
        )
    })
}

/// The DACL is set when the file is written; it is not read back here.
#[cfg(not(unix))]
fn private_file_problem(_path: &Path) -> Option<String> {
    None
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
    fn the_agents_to_connect_are_the_ones_named() -> anyhow::Result<()> {
        let names =
            |given: &[&str]| -> Vec<String> { given.iter().map(|n| n.to_string()).collect() };
        // None named: setup finds them itself.
        assert_eq!(named_agents(&[])?, None);
        assert_eq!(named_agents(&names(&[" "]))?, None);
        assert_eq!(
            named_agents(&names(&["Gemini"]))?,
            Some(Chosen {
                claude: false,
                others: vec![Agent::Gemini],
                found: false
            })
        );
        assert_eq!(
            named_agents(&names(&["claude", "gemini", "gemini"]))?,
            Some(Chosen {
                claude: true,
                others: vec![Agent::Gemini],
                found: false
            })
        );
        assert_eq!(
            named_agents(&names(&["all"]))?,
            Some(Chosen {
                claude: true,
                others: Agent::ALL.to_vec(),
                found: false
            })
        );
        let Err(problem) = named_agents(&names(&["cursor"])) else {
            anyhow::bail!("an agent setup does not know was taken");
        };
        let said = problem.to_string();
        assert!(said.contains("--agent cursor: no such agent"), "{said}");
        assert!(
            said.contains("claude, gemini, codex, copilot, goose, opencode"),
            "{said}"
        );
        Ok(())
    }

    #[test]
    fn an_agent_that_left_only_its_directory_is_not_connected_unless_named() -> anyhow::Result<()> {
        let home = agents::scratch::Scratch::new("setup-chosen")?;
        let nothing = |_: &str| None;
        let nowhere = |_: &str| None;
        let ground = agents::Ground {
            home: home.path(),
            antumbra: agents::Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        // What Codex leaves behind when it is removed, and Gemini CLI's
        // settings on a machine whose PATH does not have it.
        std::fs::create_dir_all(agents::codex::home(&ground))?;
        let gemini = agents::gemini::home(&ground);
        std::fs::create_dir_all(&gemini)?;
        std::fs::write(gemini.join("settings.json"), "{}\n")?;
        let common = |named: &[&str]| Common {
            yes: true,
            dry_run: false,
            no_claude: false,
            agents: named.iter().map(|name| name.to_string()).collect(),
            keep_auto_memory: false,
            claude_dir: None,
        };

        assert_eq!(
            chosen(&common(&[]), &ground)?,
            Chosen {
                claude: true,
                others: vec![Agent::Gemini],
                found: true
            }
        );
        // Named, it is connected, and not as one that was found.
        assert_eq!(
            chosen(&common(&["codex"]), &ground)?,
            Chosen {
                claude: false,
                others: vec![Agent::Codex],
                found: false
            }
        );
        // With its program on the PATH it is found.
        let codex = |name: &str| (name == "codex").then(|| PathBuf::from("codex"));
        let installed = agents::Ground {
            home: home.path(),
            antumbra: agents::Program::Named,
            env: &nothing,
            find: &codex,
        };
        assert_eq!(
            chosen(&common(&[]), &installed)?,
            Chosen {
                claude: true,
                others: vec![Agent::Gemini, Agent::Codex],
                found: true
            }
        );
        Ok(())
    }

    #[test]
    fn the_summary_names_the_agents_connected_because_they_were_found() {
        assert_eq!(found_line(&[]), None);
        assert_eq!(
            found_line(&[Agent::Codex]).as_deref(),
            Some(
                "Codex was connected because it was found on this machine.\n\
                 To choose the agents yourself, name them with --agent."
            )
        );
        assert_eq!(
            found_line(&[Agent::Gemini, Agent::Codex, Agent::Opencode]).as_deref(),
            Some(
                "Gemini CLI, Codex and OpenCode were connected because they were found on this machine.\n\
                 To choose the agents yourself, name them with --agent."
            )
        );
    }

    #[test]
    fn the_agents_connected_are_named_as_a_person_reads_them() {
        assert_eq!(titles(&[]), None);
        assert_eq!(titles(&["claude"]).as_deref(), Some("Claude Code"));
        assert_eq!(
            titles(&["claude", "gemini"]).as_deref(),
            Some("Claude Code and Gemini CLI")
        );
        // A name from a newer setup that this build does not know is left out.
        assert_eq!(
            titles(&["gemini", "someday"]).as_deref(),
            Some("Gemini CLI")
        );
    }

    #[test]
    fn the_scripts_another_agents_hooks_run_are_the_ones_claude_code_runs() {
        // Setup writes them once, for Claude Code, and `antumbra hook` finds
        // them there.
        let claude: Vec<&str> = platform_hooks().iter().map(|(_, s, _)| *s).collect();
        for script in agents::hook::scripts() {
            assert!(claude.contains(&script.as_str()), "{script}");
        }
    }

    #[test]
    fn every_hook_setup_writes_is_bundled() -> anyhow::Result<()> {
        for (_, script, _) in platform_hooks() {
            assert!(hooks::bundled(script).is_some(), "{script}");
        }
        assert!(hooks::bundled(headers_helper_name()).is_some());
        Ok(())
    }

    #[test]
    fn the_token_travels_only_over_https_or_to_this_machine() -> anyhow::Result<()> {
        for fine in [
            "https://workspace.example",
            "https://workspace.example:8443/mcp",
            "http://127.0.0.1:8081",
            "http://127.5.6.7",
            "http://localhost:8081",
            "http://LOCALHOST/mcp",
            "http://[::1]:8081/mcp",
            "http://user@localhost:1",
        ] {
            anyhow::ensure!(cleartext_problem(fine, false).is_none(), "{fine}");
        }
        for elsewhere in [
            "http://workspace.example:8081",
            "http://localhost.example",
            "http://127.0.0.1.example",
            "http://user@workspace.example",
        ] {
            let Some(said) = cleartext_problem(elsewhere, false) else {
                anyhow::bail!("{elsewhere} was accepted");
            };
            anyhow::ensure!(
                said.contains("cleartext") && said.contains("--allow-http"),
                "{elsewhere}: {said}"
            );
            anyhow::ensure!(
                cleartext_problem(elsewhere, true).is_none(),
                "{elsewhere} with --allow-http"
            );
        }
        let Some(said) = cleartext_problem("workspace.example", false) else {
            anyhow::bail!("an address without a scheme was accepted");
        };
        anyhow::ensure!(said.contains("https://"), "{said}");
        anyhow::ensure!(
            cleartext_problem("workspace.example", true).is_some(),
            "--allow-http does not excuse a missing scheme"
        );
        Ok(())
    }

    #[test]
    fn secrets_are_long_and_need_no_quoting() -> anyhow::Result<()> {
        let pass = secret(24, true)?;
        assert_eq!(pass.len(), 32);
        assert!(pass
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'));
        assert_ne!(pass, secret(24, true)?);
        assert_eq!(secret(32, false)?.len(), 44);
        Ok(())
    }

    #[test]
    fn a_path_with_spaces_is_quoted_in_the_command() -> anyhow::Result<()> {
        let cmd = script_command(Path::new("/home/a b/.antumbra/hooks/antumbra-capture.sh"));
        assert!(
            cmd.ends_with("\"/home/a b/.antumbra/hooks/antumbra-capture.sh\""),
            "{cmd}"
        );
        Ok(())
    }

    #[test]
    fn the_machine_is_registered_under_the_name_its_hooks_report() -> anyhow::Result<()> {
        // A name pinned in the settings before setup is the one the hooks
        // report, so it is the one registered.
        let pinned = json!({ "env": { "ANTUMBRA_HOST_ID": " mac " } });
        assert_eq!(session_host(&pinned), "mac");
        // A blank one names nothing and falls through to a name that is there.
        let blank = json!({ "env": { "ANTUMBRA_HOST_ID": "  " } });
        assert!(!session_host(&blank).trim().is_empty());
        assert!(!session_host(&Value::Null).trim().is_empty());
        Ok(())
    }

    #[test]
    fn the_check_tells_a_helper_that_names_the_machine_from_one_that_does_not() -> anyhow::Result<()>
    {
        let Some(bundled) = hooks::bundled(headers_helper_name()) else {
            anyhow::bail!("the headers helper is not bundled");
        };
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
        Ok(())
    }

    #[test]
    fn the_check_reads_what_the_server_stamps_this_machines_writes_with() -> anyhow::Result<()> {
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
        Ok(())
    }

    #[test]
    fn the_settings_for_a_launch_without_the_users_hold_what_setup_wires_and_no_more(
    ) -> anyhow::Result<()> {
        let antumbra = Path::new("/home/me/.antumbra");
        // The user's own settings: a name for this machine, file memory off,
        // and things that are theirs alone.
        let theirs = json!({
            "env": { "ANTUMBRA_HOST_ID": "laptop", "EDITOR": "vi" },
            "autoMemoryEnabled": false,
            "model": "opus",
            "permissions": { "defaultMode": "auto" }
        });
        let text = claude_settings_alone(antumbra, "https://memory.example", "ws:team", &theirs)?;
        let alone: Value = serde_json::from_str(&text)?;
        let Some(object) = alone.as_object() else {
            anyhow::bail!("not an object: {text}");
        };
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["autoMemoryEnabled", "env", "hooks"]);
        assert_eq!(
            alone["env"],
            json!({
                "ANTUMBRA_URL": "https://memory.example",
                "ANTUMBRA_WORKSPACE_ID": "ws:team",
                "ANTUMBRA_HOST_ID": "laptop"
            })
        );
        assert_eq!(alone["autoMemoryEnabled"], false);
        // The hooks Claude Code's own settings get, run from Antumbra's
        // directory, and the two that count skills.
        for (event, script, _) in platform_hooks() {
            let hook = alone["hooks"][event].to_string();
            assert!(
                hook.contains(script) && hook.contains(".antumbra"),
                "{event}: {hook}"
            );
        }
        for event in ["PostToolUse", "UserPromptExpansion"] {
            let hook = alone["hooks"][event].to_string();
            assert!(hook.contains(SKILL_USED), "{event}: {hook}");
        }

        // Where the user's settings leave file memory alone, so do these, and
        // a machine nobody named has the name setup gives it.
        let plain = claude_settings_alone(
            antumbra,
            "http://127.0.0.1:8081",
            "ws:default",
            &Value::Null,
        )?;
        let alone: Value = serde_json::from_str(&plain)?;
        assert!(alone.get("autoMemoryEnabled").is_none());
        assert!(alone["env"]["ANTUMBRA_HOST_ID"]
            .as_str()
            .is_some_and(|host| !host.is_empty()));
        // The same in, the same out, so the check can compare the file with it.
        assert_eq!(
            plain,
            claude_settings_alone(
                antumbra,
                "http://127.0.0.1:8081",
                "ws:default",
                &Value::Null
            )?
        );
        Ok(())
    }

    #[test]
    fn what_a_policy_stops_is_said_with_whose_it_is_to_change() -> anyhow::Result<()> {
        use policy::{Hooks, Server, Why};
        let theirs = PathBuf::from("managed-settings.json");
        let mine = PathBuf::from("settings.json");
        // The settings Unity Gateway passes, where the launch judged is its.
        let passed = PathBuf::from("ucode-settings.json");
        let gateway = Some(passed.as_path());
        let url = "https://memory.example/mcp";

        assert_eq!(hooks_held_back(&Hooks::Run, None), None);
        let stopped = |by: &Path, key: &'static str, managed: bool| {
            hooks_held_back(
                &Hooks::Stopped {
                    by: by.to_path_buf(),
                    key,
                    managed,
                },
                gateway,
            )
        };
        let Some((said, fix)) = stopped(&theirs, "allowManagedHooksOnly", true) else {
            anyhow::bail!("hooks the organization stops were not said");
        };
        assert!(
            said.contains("only your organization's hooks")
                && said.contains("allowManagedHooksOnly in managed-settings.json"),
            "{said}"
        );
        // Theirs to change, and what still works meanwhile.
        assert!(
            fix.contains("your organization's") && fix.contains("memory tools"),
            "{fix}"
        );
        let Some((said, fix)) = stopped(&mine, "disableAllHooks", false) else {
            anyhow::bail!("hooks the user turned off were not said");
        };
        assert!(said.contains("disableAllHooks in settings.json"), "{said}");
        assert_eq!(fix, "remove disableAllHooks from that file");
        // The file ug writes is not the user's to edit: ug writes it again.
        let Some((_, fix)) = stopped(&passed, "disableAllHooks", false) else {
            anyhow::bail!("hooks ug's settings turn off were not said");
        };
        assert!(
            fix.contains("Unity Gateway's") && !fix.contains("remove"),
            "{fix}"
        );

        assert_eq!(server_held_back(&Server::Loads, url, None), None);
        let entry = json!({ "serverUrl": url }).to_string();
        let blocked = |by: &Path, managed: bool, why: Why| {
            server_held_back(
                &Server::Blocked {
                    by: by.to_path_buf(),
                    managed,
                    why,
                },
                url,
                gateway,
            )
        };
        let unlisted = |by: &Path, managed: bool, open: bool| {
            blocked(
                by,
                managed,
                Why::Unlisted {
                    entry: entry.clone(),
                    open,
                },
            )
        };
        // The organization's list, kept to its own: only they can add to it.
        let Some((said, fix)) = unlisted(&theirs, true, false) else {
            anyhow::bail!("a server the list leaves out was not said");
        };
        assert!(
            said.contains("managed-settings.json lists the MCP servers")
                && said.contains(url)
                && said.contains("not connected"),
            "{said}"
        );
        assert!(
            fix.contains("your organization's") && fix.contains(&entry),
            "{fix}"
        );
        assert!(!fix.contains("your own settings"), "{fix}");
        // One the user's own settings add to: both are said, theirs first.
        let Some((_, fix)) = unlisted(&theirs, true, true) else {
            anyhow::bail!("a server the list leaves out was not said");
        };
        assert!(
            fix.starts_with("that file is your organization's")
                && fix.contains("allowManagedMcpServersOnly is not set"),
            "{fix}"
        );
        // A list of the user's own is theirs to add to.
        let Some((_, fix)) = unlisted(&mine, false, true) else {
            anyhow::bail!("a server the list leaves out was not said");
        };
        assert_eq!(
            fix,
            format!("add {entry} to allowedMcpServers in that file")
        );
        // One in the file ug writes is the workspace's.
        let Some((said, fix)) = unlisted(&passed, false, true) else {
            anyhow::bail!("a server ug's list leaves out was not said");
        };
        assert!(said.contains("ucode-settings.json lists"), "{said}");
        assert!(
            fix.contains("Unity Gateway's") && fix.contains(&entry) && !fix.starts_with("add "),
            "{fix}"
        );

        for (by, why, managed, in_said, in_fix) in [
            (
                &theirs,
                Why::Denied,
                true,
                "a deniedMcpServers entry in managed-settings.json",
                "your organization's",
            ),
            (
                &mine,
                Why::Denied,
                false,
                "a deniedMcpServers entry in settings.json",
                "remove that entry",
            ),
            (
                &passed,
                Why::Denied,
                false,
                "a deniedMcpServers entry in ucode-settings.json",
                "whoever manages the workspace",
            ),
            (
                &theirs,
                Why::Fixed,
                true,
                "fixes the MCP servers",
                "your organization's",
            ),
            (
                &theirs,
                Why::PluginsOnly,
                true,
                "strictPluginOnlyCustomization",
                "your organization's",
            ),
        ] {
            let Some((said, fix)) = blocked(by, managed, why.clone()) else {
                anyhow::bail!("{why:?} was not said");
            };
            assert!(
                said.contains(in_said) && said.contains("\"antumbra\""),
                "{said}"
            );
            assert!(fix.contains(in_fix), "{fix}");
        }
        Ok(())
    }

    #[test]
    fn the_repository_is_found_from_inside_it() -> anyhow::Result<()> {
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        let Some(repo) = find_repo(Some(here))? else {
            anyhow::bail!("the crate's own directory is not seen as inside the repository");
        };
        assert!(repo.join("docker").join("docker-compose.yml").is_file());
        assert!(
            find_repo(Some(&std::env::temp_dir())).is_err(),
            "a directory given that is not a clone is an error, not the published image"
        );
        Ok(())
    }

    #[test]
    fn the_stack_without_a_clone_runs_the_image_setup_names() -> anyhow::Result<()> {
        assert!(STACK_COMPOSE.contains("image: ${ANTUMBRA_IMAGE:-antumbra-mcp:local}"));
        for name in [
            "SURREAL_PASS=",
            "ANTUMBRA_JWT_SECRET=",
            "ANTUMBRA_SURREAL_DATA=",
        ] {
            assert!(STACK_ENV_EXAMPLE.contains(name), "{name}");
        }
        Ok(())
    }

    #[test]
    fn an_upgrade_moves_a_release_image_and_keeps_one_named_by_hand() -> anyhow::Result<()> {
        assert!(release_image_moves(None));
        assert!(release_image_moves(Some("")));
        assert!(release_image_moves(Some(
            "ghcr.io/albedosehen/antumbra-mcp:0.2.0"
        )));
        assert!(!release_image_moves(Some(
            "registry.example/antumbra-mcp:mine"
        )));
        assert!(!release_image_moves(Some("antumbra-mcp:local")));
        assert!(!release_image_moves(Some(
            "ghcr.io/albedosehen/antumbra-mcp-gpu:0.2.0"
        )));
        Ok(())
    }

    #[test]
    fn an_earlier_clone_setup_lends_its_env_to_the_stack() -> anyhow::Result<()> {
        let dir = std::env::temp_dir().join(format!("antumbra-earlier-{}", std::process::id()));
        let clone = dir.join("clone");
        std::fs::create_dir_all(clone.join("docker"))?;
        std::fs::write(clone.join("docker").join(".env"), "SURREAL_PASS=old\n")?;
        std::fs::write(
            dir.join("setup.json"),
            serde_json::to_string(&json!({ "mode": "local", "repo": clone }))?,
        )?;
        assert_eq!(earlier_env(&dir), Some(clone.join("docker").join(".env")));
        std::fs::write(
            dir.join("setup.json"),
            serde_json::to_string(&json!({ "mode": "local", "repo": null }))?,
        )?;
        assert_eq!(
            earlier_env(&dir),
            None,
            "a pulled stack has no clone to lend"
        );
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// A private file is the owner's alone, as is its directory, and is
    /// replaced whole: a second write leaves the new text, no temporary, and
    /// the same permissions.
    #[cfg(unix)]
    #[test]
    fn a_private_file_is_the_owners_alone_and_replaced_whole() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("antumbra-private-{}", std::process::id()));
        let path = dir.join("nested").join("token.txt");
        let caveat = write_private(&path, "first\n")?;
        anyhow::ensure!(caveat.is_none(), "{caveat:?}");
        let second = write_private(&path, "second\n")?;
        anyhow::ensure!(second.is_none(), "{second:?}");
        let mode = |p: &Path| -> anyhow::Result<u32> {
            Ok(std::fs::metadata(p)?.permissions().mode() & 0o777)
        };
        anyhow::ensure!(std::fs::read_to_string(&path)? == "second\n");
        anyhow::ensure!(mode(&path)? == 0o600, "file mode {:o}", mode(&path)?);
        anyhow::ensure!(
            mode(&dir.join("nested"))? == 0o700,
            "dir mode {:o}",
            mode(&dir.join("nested"))?
        );
        anyhow::ensure!(
            !dir.join("nested").join("token.txt.tmp").exists(),
            "the temporary is gone"
        );
        anyhow::ensure!(private_file_problem(&path).is_none());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))?;
        let Some(problem) = private_file_problem(&path) else {
            anyhow::bail!("a file readable by others was not a problem");
        };
        anyhow::ensure!(problem.contains("644"), "{problem}");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// A file by an agent's name that fails is not the agent; one that
    /// answers is; and one that is not a program at all is not.
    #[cfg(unix)]
    #[test]
    fn a_command_that_is_there_and_fails_does_not_run() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let dir = crate::agents::scratch::Scratch::new("runs")?;
        let program = |name: &str, body: &str| -> anyhow::Result<PathBuf> {
            let path = dir.path().join(name);
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n"))?;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
            Ok(path)
        };
        let wrapper = program(
            "wrapper",
            "echo 'Error: codex not found in PATH' >&2; exit 1",
        )?;
        let real = program("real", "echo 'codex 1.2.3'")?;
        let text = dir.path().join("notes.txt");
        std::fs::write(&text, "not a program")?;
        assert!(!runs(&wrapper));
        assert!(runs(&real));
        assert!(!runs(&text));
        assert!(!runs(&dir.path().join("nothing-here")));
        // Asked once: the answer stands though the file has changed.
        std::fs::write(&wrapper, "#!/bin/sh\nexit 0\n")?;
        assert!(!runs(&wrapper));
        Ok(())
    }
}
