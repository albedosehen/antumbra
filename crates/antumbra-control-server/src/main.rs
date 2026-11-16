//! The hosted **control plane** HTTP server: invite-gated signup, magic-link
//! login, and RS256 token issuance, run as its own service (separate from the
//! data-plane MCP server, which only *verifies* the tokens this server signs).
//!
//! The whole signup -> token -> login flow lives in `antumbra-control`; this
//! binary is the thin HTTP + config + mailer shell over it (see `http` and
//! `mailer`), plus the operator's invite lifecycle (mint / list / revoke) and a
//! self-probe for the container healthcheck.

mod http;
mod mailer;

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use clap::{Args, Parser, Subcommand};

use antumbra_control::Issuer;
use antumbra_store::repo::invite;
use antumbra_store::{Store, EMBED_DIM};

use crate::http::{router, AppState, Cooldown};
use crate::mailer::build_mailer;

#[derive(Parser)]
#[command(
    name = "antumbra-control-server",
    about = "Antumbra hosted control plane"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the HTTP control plane (signup / login / magic-link verify).
    Serve(ServeArgs),
    /// Mint a single-use invite code and print it (hand it to a new user).
    MintInvite(MintInviteArgs),
    /// List the outstanding (un-redeemed) invite codes.
    ListInvites(StoreArgs),
    /// Revoke an outstanding invite code.
    RevokeInvite(RevokeInviteArgs),
    /// Probe a running server's `/healthz` and exit 0/1: the container
    /// healthcheck (distroless has no shell or curl, so the binary checks
    /// itself).
    Probe(ProbeArgs),
}

#[derive(Args)]
struct StoreArgs {
    /// The SurrealDB this service works in: a server beside it, by default.
    #[arg(
        long,
        env = "ANTUMBRA_DB_URL",
        default_value = "ws://127.0.0.1:8000/rpc"
    )]
    url: String,
    /// The credentials, as every Antumbra binary takes them.
    #[command(flatten)]
    auth: antumbra_store::args::DbAuthArgs,
}

#[derive(Args)]
struct MintInviteArgs {
    #[command(flatten)]
    store: StoreArgs,
    /// Days until the invite expires; `0` mints a non-expiring code. Bounded
    /// at a century, so the date arithmetic behind it cannot overflow.
    #[arg(long, default_value_t = 14, value_parser = clap::value_parser!(i64).range(0..=36_500))]
    ttl_days: i64,
}

#[derive(Args)]
struct RevokeInviteArgs {
    #[command(flatten)]
    store: StoreArgs,
    /// The invite code to revoke.
    code: String,
}

#[derive(Args)]
struct ProbeArgs {
    /// `host:port` the server listens on.
    #[arg(long, default_value = "127.0.0.1:8090")]
    addr: String,
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    store: StoreArgs,
    #[arg(long, default_value = "0.0.0.0:8090")]
    addr: String,
    /// External base URL the magic links point back to (this server).
    #[arg(long, env = "ANTUMBRA_CONTROL_BASE_URL")]
    base_url: String,
    /// Path to the RS256 PRIVATE key (PEM) issued tokens are signed with. The MCP
    /// server verifies with the matching public key (its `--jwt-public-key`).
    #[arg(long, env = "ANTUMBRA_SIGNING_KEY")]
    signing_key: String,
    /// Secret signing the short-lived magic-link tokens (distinct from the RS256 key).
    #[arg(long, env = "ANTUMBRA_MAGIC_SECRET")]
    magic_secret: String,
    /// Audience claim on issued tokens (must match the MCP server's verifier).
    #[arg(long, env = "ANTUMBRA_AUDIENCE", default_value = "antumbra")]
    audience: String,
    /// Issued bearer-token lifetime, in days.
    #[arg(long, default_value_t = 30)]
    token_ttl_days: u64,
    /// Magic-link lifetime, in minutes.
    #[arg(long, default_value_t = 15)]
    magic_ttl_mins: u64,
    /// Minimum seconds between magic links to the same address (the email-
    /// cannon brake); `0` disables the cooldown.
    #[arg(long, default_value_t = 60)]
    link_cooldown_secs: u64,
    /// Log sign-in links to stderr instead of emailing them, for development:
    /// anyone who reads the logs can sign in. Ignored when `SMTP_HOST` is set
    /// (then it is email, or the server refuses to start), and required when
    /// it is not. From the environment, a switch: empty, `0`, `false`, `no`
    /// and `off` are off, anything else is on, so a compose file can pass the
    /// variable through unset.
    #[arg(
        long,
        env = "ANTUMBRA_DEV_MAILER",
        value_parser = clap::builder::FalseyValueParser::new()
    )]
    dev_mailer: bool,
}

async fn connect(s: &StoreArgs) -> anyhow::Result<Store> {
    Ok(s.auth.connect(&s.url, EMBED_DIM).await?)
}

/// The `.env.example` placeholder: a magic secret anyone can read out of the
/// repo, so refusing it outright beats serving with forgeable links.
const PLACEHOLDER_SECRET: &str = "change-me-base64-32-bytes";

fn validate_magic_secret(secret: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        secret != PLACEHOLDER_SECRET,
        "ANTUMBRA_MAGIC_SECRET is still the .env.example placeholder; \
         generate a real one: openssl rand -base64 32"
    );
    anyhow::ensure!(
        secret.len() >= 32,
        "ANTUMBRA_MAGIC_SECRET is too short ({} chars, want >= 32); \
         generate one: openssl rand -base64 32",
        secret.len()
    );
    Ok(())
}

/// Resolve on SIGTERM (what `docker stop` sends PID 1) or ctrl-c, so axum can
/// stop accepting and drain in-flight requests instead of dropping them.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install ctrl-c handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    eprintln!("antumbra-control-server: shutdown signal; draining");
}

/// One HTTP/1.0 GET against `/healthz`, std-only so the healthcheck needs
/// nothing but this binary.
fn probe(addr: &str) -> anyhow::Result<()> {
    use std::io::{Read, Write};
    let target = addr
        .parse::<std::net::SocketAddr>()
        .map_err(|e| anyhow::anyhow!("bad --addr {addr}: {e}"))?;
    let mut stream = std::net::TcpStream::connect_timeout(&target, Duration::from_secs(3))?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;
    write!(stream, "GET /healthz HTTP/1.0\r\nHost: {addr}\r\n\r\n")?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    let status = response.lines().next().unwrap_or_default();
    anyhow::ensure!(
        status.starts_with("HTTP/1.1 200") || status.starts_with("HTTP/1.0 200"),
        "unhealthy: {status}"
    );
    Ok(())
}

async fn serve(a: ServeArgs) -> anyhow::Result<()> {
    validate_magic_secret(&a.magic_secret)?;
    let store = connect(&a.store).await?;
    let private_pem = std::fs::read(&a.signing_key)
        .map_err(|e| anyhow::anyhow!("read signing key {}: {e}", a.signing_key))?;
    let issuer = Issuer::new(
        private_pem,
        a.audience,
        Duration::from_secs(a.token_ttl_days * 86_400),
    );
    let state = AppState {
        store,
        issuer: Arc::new(issuer),
        magic_secret: Arc::new(a.magic_secret.into_bytes()),
        base_url: Arc::new(a.base_url),
        magic_ttl: Duration::from_secs(a.magic_ttl_mins * 60),
        mailer: build_mailer(a.dev_mailer)?,
        cooldown: Arc::new(Cooldown::new(Duration::from_secs(a.link_cooldown_secs))),
        limits: Arc::new(antumbra_core::limit::Limiter::new(
            crate::http::LINKS_PER_CLIENT,
            crate::http::LINKS_EVERYONE,
        )),
    };
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(&a.addr).await?;
    eprintln!(
        "antumbra-control-server: listening on http://{}/ (signup / login / magic)",
        a.addr
    );
    // With the peer's address on each request, so link requests are budgeted
    // per client.
    let app = app.into_make_service_with_connect_info::<std::net::SocketAddr>();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // The shape of the CLI and the MCP server: SurrealDB query evaluation
    // recurses deep, and an embedded SurrealDB 4 engine overflows both the
    // thread that blocks on the run (1 MiB on Windows) and the runtime's
    // workers (2 MiB; SurrealDB recommends 10 MiB). A `ws://` store runs its
    // queries on its server, but `--url` may name an embedded one.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(16 * 1024 * 1024)
                .build()?
                .block_on(run())
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("antumbra-control-server worker thread panicked"))?
}

async fn run() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Command::Serve(a) => serve(a).await,
        Command::MintInvite(a) => {
            let store = connect(&a.store).await?;
            let code = antumbra_control::new_invite_code();
            let expires_at =
                (a.ttl_days > 0).then(|| Utc::now() + chrono::Duration::days(a.ttl_days));
            invite::mint(&store, &code, expires_at).await?;
            match expires_at {
                Some(exp) => eprintln!("expires {}", exp.to_rfc3339()),
                None => eprintln!("never expires"),
            }
            println!("{code}");
            Ok(())
        }
        Command::ListInvites(s) => {
            let store = connect(&s).await?;
            let mut invites = invite::all(&store).await?;
            invites.sort_by_key(|inv| inv.created_at);
            for inv in &invites {
                let expiry = inv
                    .expires_at
                    .map_or_else(|| "never expires".to_string(), |e| e.to_rfc3339());
                println!(
                    "{}  minted {}  {}",
                    inv.code,
                    inv.created_at.to_rfc3339(),
                    expiry
                );
            }
            eprintln!("{} outstanding invite(s)", invites.len());
            Ok(())
        }
        Command::RevokeInvite(a) => {
            let store = connect(&a.store).await?;
            if invite::revoke(&store, &a.code).await? {
                eprintln!("revoked {}", a.code);
                Ok(())
            } else {
                anyhow::bail!("no outstanding invite {}", a.code)
            }
        }
        Command::Probe(a) => probe(&a.addr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dev_mailer_is_a_flag_that_is_off_unless_given() -> anyhow::Result<()> {
        let Command::Serve(off) = Cli::try_parse_from(SERVE)?.cmd else {
            anyhow::bail!("not the serve command");
        };
        anyhow::ensure!(!off.dev_mailer, "off unless asked for");
        let with_flag = SERVE.iter().copied().chain(["--dev-mailer"]);
        let Command::Serve(on) = Cli::try_parse_from(with_flag)?.cmd else {
            anyhow::bail!("not the serve command");
        };
        anyhow::ensure!(on.dev_mailer);
        Ok(())
    }

    /// A `serve` command line with every required argument and nothing else.
    const SERVE: [&str; 8] = [
        "antumbra-control-server",
        "serve",
        "--base-url",
        "https://control.example",
        "--signing-key",
        "signing.pem",
        "--magic-secret",
        "0123456789abcdef0123456789abcdef",
    ];

    // The compose file passes ANTUMBRA_DEV_MAILER through unset, as an empty
    // string, so the variable has to read as a switch rather than as a strict
    // `true`/`false`. The one test that touches the variable; it restores it.
    #[test]
    fn the_dev_mailer_variable_reads_like_a_switch() -> anyhow::Result<()> {
        for (value, expected) in [
            ("", false),
            ("0", false),
            ("false", false),
            ("no", false),
            ("1", true),
            ("true", true),
            ("yes", true),
        ] {
            std::env::set_var("ANTUMBRA_DEV_MAILER", value);
            let parsed = Cli::try_parse_from(SERVE);
            std::env::remove_var("ANTUMBRA_DEV_MAILER");
            let Command::Serve(serve) = parsed?.cmd else {
                anyhow::bail!("not the serve command");
            };
            anyhow::ensure!(
                serve.dev_mailer == expected,
                "ANTUMBRA_DEV_MAILER={value:?} read as {}",
                serve.dev_mailer
            );
        }
        Ok(())
    }

    // A day count is bounded at the parser, so the date arithmetic behind it
    // cannot overflow on a typo: a usage error naming the range, not a panic.
    #[test]
    fn an_invite_ttl_past_a_century_is_a_usage_error() -> anyhow::Result<()> {
        let mint = |ttl: &str| {
            Cli::try_parse_from(["antumbra-control-server", "mint-invite", "--ttl-days", ttl])
        };
        let Err(e) = mint("999999999999999") else {
            anyhow::bail!("a ttl past a century was accepted");
        };
        anyhow::ensure!(e.to_string().contains("0..=36500"), "{e}");
        anyhow::ensure!(mint("-1").is_err(), "a negative ttl was accepted");
        mint("0")?;
        mint("36500")?;
        Ok(())
    }

    // The store flags are the shared ones (#303), on every subcommand that
    // reaches the store: the password's environment value is hidden in --help
    // and the url reads its variable, as in every binary.
    #[test]
    fn the_store_flags_read_as_in_every_binary() -> anyhow::Result<()> {
        use clap::CommandFactory;
        let cmd = Cli::command();
        cmd.clone().debug_assert();
        for name in ["serve", "mint-invite", "list-invites", "revoke-invite"] {
            let sub = cmd
                .find_subcommand(name)
                .ok_or_else(|| anyhow::anyhow!("no subcommand {name}"))?;
            let flag = |id: &str| {
                sub.get_arguments().find(|a| a.get_id() == id).map(|a| {
                    (
                        a.get_env().map(|e| e.to_string_lossy().into_owned()),
                        a.is_hide_env_values_set(),
                    )
                })
            };
            anyhow::ensure!(
                flag("db_pass") == Some((Some("ANTUMBRA_DB_PASS".into()), true)),
                "{name}: --db-pass"
            );
            anyhow::ensure!(
                flag("url") == Some((Some("ANTUMBRA_DB_URL".into()), false)),
                "{name}: --url"
            );
            anyhow::ensure!(
                flag("db_auth").map(|f| f.0) == Some(Some("ANTUMBRA_DB_AUTH".into())),
                "{name}: --db-auth"
            );
        }
        Ok(())
    }

    #[test]
    fn the_placeholder_and_short_magic_secrets_are_refused() {
        assert!(validate_magic_secret(PLACEHOLDER_SECRET).is_err());
        assert!(validate_magic_secret("short").is_err());
        assert!(validate_magic_secret("oqRzkA0sZ1Yx9fJ2mB7cD4eF6gH8iK0l").is_ok());
    }

    // The probe round-trip against a real listener: healthy store -> exit Ok,
    // and a refused connection -> Err.
    #[tokio::test]
    async fn probe_round_trips_against_a_live_server() {
        use crate::http::{router, AppState, Cooldown};
        use std::sync::Arc;

        const RS_PRIV: &[u8] = include_bytes!("../../antumbra-control/tests/test_jwt_priv.pem");
        let store = antumbra_store::Store::connect_memory(EMBED_DIM)
            .await
            .unwrap();
        let state = AppState {
            store,
            issuer: Arc::new(Issuer::new(
                RS_PRIV.to_vec(),
                "antumbra",
                Duration::from_secs(60),
            )),
            magic_secret: Arc::new(b"test-magic".to_vec()),
            base_url: Arc::new("https://app".to_string()),
            magic_ttl: Duration::from_secs(60),
            mailer: Arc::new(crate::mailer::StderrMailer),
            cooldown: Arc::new(Cooldown::new(Duration::ZERO)),
            limits: Arc::new(antumbra_core::limit::Limiter::new(
                crate::http::LINKS_PER_CLIENT,
                crate::http::LINKS_EVERYONE,
            )),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            axum::serve(listener, router(state)).await.unwrap();
        });

        let probed = tokio::task::spawn_blocking(move || probe(&addr))
            .await
            .unwrap();
        assert!(probed.is_ok(), "healthy server probes ok: {probed:?}");

        // Nothing listens on this port: the probe must fail, not hang.
        let dead = tokio::task::spawn_blocking(|| probe("127.0.0.1:9"))
            .await
            .unwrap();
        assert!(dead.is_err());
    }
}
