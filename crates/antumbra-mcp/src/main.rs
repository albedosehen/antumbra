//! `antumbra-mcp`: the Model Context Protocol server that exposes Antumbra's
//! Penumbra memory to an agent. The runtime surface that lets a client (Claude
//! Code, any MCP host) talk to Antumbra instead of a separate memory service.
//!
//! Two transports. **stdio** (default) serves a single `(tenant, user)` passed
//! on the command line: connect as owner, provision the principal, sign the
//! session in so every tool is engine-isolated. **HTTP** (`--http <addr>`) is
//! the networked, per-request multi-tenant surface: each request carries a
//! signed JWT whose `tenant`/`user` claims become `$auth`, and the server keeps
//! one signed-in session per identity (see [`http`]).
//!
//! Either way the server logs to stderr, and with `OTEL_EXPORTER_OTLP_ENDPOINT`
//! set it also exports a span per HTTP request and per tool call to that
//! collector over OTLP/HTTP (see [`telemetry`]).

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use rmcp::ServiceExt;

use antumbra_core::ports::Embedder;
use antumbra_core::{Compartment, CompartmentId, TenantId, UserId};
use antumbra_store::repo::{compartment, principal};
use antumbra_store::{ConnectionConfig, Store, EMBED_DIM};

/// The JWT token contract now lives in the shared `antumbra-auth` crate (so the
/// hosted control plane can mint what this server verifies); re-exported here so
/// `crate::auth::…` keeps resolving.
mod auth {
    pub use antumbra_auth::*;
}
mod dependencies;
mod hardware;
mod http;
mod notify;
mod pdf;
mod profile;
mod server;
mod session;
mod telemetry;
mod warmup;
use server::McpServer;

#[derive(Parser)]
#[command(
    name = "antumbra-mcp",
    about = "Antumbra MCP server over the Penumbra memory store",
    version
)]
struct Cli {
    /// SurrealDB url: `surrealkv://./data/antumbra.skv` (persistent),
    /// `mem://` (ephemeral), or `ws://host:8000/rpc`.
    #[arg(
        long,
        env = "ANTUMBRA_DB_URL",
        default_value = "surrealkv://./data/antumbra.skv"
    )]
    url: String,
    /// The workspace (tenant) this stdio server serves. Ignored with `--http`,
    /// where the tenant comes from each request's verified JWT.
    #[arg(long, default_value = "ws:default")]
    tenant: String,
    /// The user this stdio session acts as (the compartment-ownership / sharing
    /// actor). Ignored with `--http`.
    #[arg(long, default_value = "user:default")]
    user: String,
    /// The host/device this session runs on: stamped as memory provenance, and
    /// the name this machine registers under in its user's fabric.
    /// Defaults to the machine name, which inside a container is the container
    /// ID -- ephemeral, so a recreate would register the same machine as a new
    /// node every time. Set it to the real host there.
    #[arg(long, env = "ANTUMBRA_HOST")]
    host: Option<String>,
    /// Serve the networked multi-tenant HTTP surface on this address
    /// (e.g. `0.0.0.0:8081`) instead of stdio. Requires a JWT key.
    #[arg(long)]
    http: Option<String>,
    /// HS256 shared secret for verifying request JWTs (symmetric).
    #[arg(long, env = "ANTUMBRA_JWT_SECRET", hide_env_values = true)]
    jwt_secret: Option<String>,
    /// Read the HS256 secret from this file instead (a Docker secret, a
    /// Kubernetes secret, a Key Vault mount): it then appears in neither the
    /// process arguments nor the environment.
    #[arg(long, env = "ANTUMBRA_JWT_SECRET_FILE", conflicts_with = "jwt_secret")]
    jwt_secret_file: Option<std::path::PathBuf>,
    /// Path to a PEM RSA public key for verifying request JWTs (RS256).
    #[arg(long)]
    jwt_public_key: Option<std::path::PathBuf>,
    /// Required JWT audience claim (this server's identifier), if set.
    #[arg(long)]
    jwt_audience: Option<String>,
    #[command(flatten)]
    db: antumbra_store::args::DbAuthArgs,
    /// Enable the autonomous propose trigger: once a user's unorganized inbox
    /// reaches this many memories, a write auto-clusters it into proposed
    /// compartments (reversible; the user curates). Off when unset.
    #[arg(long)]
    auto_propose: Option<usize>,
    /// Retired: a reinforced memory no longer trains its compartment
    /// into an expert, which taught experts to echo memories. Standing experts
    /// are trained from accepted behaviors by the keeper, on any node that can
    /// train. Still accepted, so a deployment that passes it keeps starting.
    #[arg(long, default_value_t = false, hide = true)]
    auto_consolidate: bool,
    /// The chunk index, on the networked surface: how many pieces
    /// of a memory to embed at once while cutting memories into the pieces
    /// recall searches. Zero leaves the index as it is; recall then reads whole
    /// memories for anything not in it.
    #[arg(long, env = "ANTUMBRA_CHUNK_IN_FLIGHT", default_value_t = 4)]
    chunk_in_flight: usize,
    /// The training every standing expert on this node gets: the adapter's
    /// LoRA rank. The serving engine is built at it, so lowering it below an
    /// expert already minted here means retraining that expert. Nine behaviors
    /// were all learned at the default; 32 in one expert were not.
    #[arg(long, env = "ANTUMBRA_BEHAVE_RANK", default_value_t = 16)]
    behave_rank: usize,
    /// Epochs over each behavior's examples and the base model's replay.
    #[arg(long, env = "ANTUMBRA_BEHAVE_ROUNDS", default_value_t = 3)]
    behave_rounds: usize,
    /// The learning rate. Above 1.5e-4, nine behaviors taught together pulled
    /// the expert off the base model's answers everywhere else.
    #[arg(long, env = "ANTUMBRA_BEHAVE_LR", default_value_t = 1.5e-4)]
    behave_lr: f64,
    /// How sure admission has to be, from a behavior's held-out counts, that
    /// the expert answers it worse than the base model before that refuses
    /// the set: a probability above 0.5 and below 1. One lost task of three
    /// is about 0.74; three lost of three is near certain.
    #[arg(
        long,
        env = "ANTUMBRA_BEHAVE_REGRESSION_CONFIDENCE",
        default_value_t = 0.9
    )]
    behave_regression_confidence: f64,
    /// Let the loop tune the regression confidence from the audits of the
    /// experts it admitted: per workspace, one step of 0.02 at a time, only
    /// when two readings in a row agree, never below 0.6 or above 0.98. Off,
    /// the recipe's value stands and the audits are only noted.
    #[arg(long, env = "ANTUMBRA_BEHAVE_TUNE_CONFIDENCE")]
    behave_tune_confidence: bool,
    /// Which quantized forms of the base an expert is measured on when it is
    /// admitted: `auto` (Q8_0 and Q6_K, the two forms a server with no GPU
    /// serves by default), `off`, `q8_0`, `q6_k`, `q4_k`, or several with
    /// commas (`q8_0,q4_k`). An expert is trained
    /// over the dense base and may be served over a quantized one; the
    /// measurement reads its held-out examples through both and records the
    /// difference on its card. It costs one more load of the base per form,
    /// per admitted training, and nothing is refused on it.
    #[arg(long, env = "ANTUMBRA_BEHAVE_DRIFT", default_value = "auto")]
    behave_drift: String,
    /// How the base model is held to serve experts: `auto`, `off`, `q8_0`,
    /// `q6_k` or `q4_k`. On a build with no GPU backend, where the dense base
    /// is too slow to answer with, `auto` serves from a Q6_K base where the
    /// CPU has its fast kernels (AVX2 on x86-64, or an Arm CPU's NEON with
    /// its dot product) and from a Q8_0 base on any other CPU; on a GPU
    /// build it serves from the dense base. Every quantized form rewords an
    /// expert's answers, and a smaller one decodes faster where a token is
    /// bound by how fast its weights are read. `q8_0` moves answers least.
    /// On a CPU with AVX-512 VNNI, `q6_k` decodes about 1.3 times as fast
    /// and shifts an expert's log-probabilities half again as far, and
    /// `q4_k` about 1.7 times as fast and four times as far. On held-out
    /// behavior tasks 82 answers of 113 passed their checks at `q8_0` and
    /// 79 at `q4_k`, against the dense base's 85. `off` keeps the base
    /// dense anywhere. Experts are never quantized.
    #[arg(long, env = "ANTUMBRA_SERVE_QUANT", default_value = "auto")]
    serve_quant: String,
    /// Whether an expert's answer is decoded several tokens a pass: `on` or
    /// `off`. On, a pass over the weights carries, after the token in
    /// hand, the tokens guessed to follow it, and keeps each that is the
    /// model's own next token: the answer is the one a token at a time
    /// writes, in fewer passes. The guesses come from what the model wrote
    /// for earlier requests through the same experts, which the server
    /// remembers while it runs, and from the request's own prompt. So a
    /// request is answered faster where its answer resembles an earlier one
    /// through the same experts, and several times faster where the very
    /// prompt was answered before; a caller who can time answers through
    /// those experts can tell that it was. `off` remembers nothing and
    /// decodes a token a pass. It is read ahead where the base is served
    /// quantized on a CPU with the fast kernels, and from the dense base on
    /// a CUDA GPU whose library computes a row the same way whatever is
    /// beside it, which is asked of it at the first answer. On such a GPU
    /// reading ahead decodes through passes of sixteen rows, which the GPU
    /// rounds otherwise than a pass of one: what is guessed still decides
    /// nothing, but an answer can differ in a word from the one `off`
    /// writes, where two tokens were all but tied. Experts are trained and
    /// audited here decoding as this is set.
    #[arg(
        long,
        env = "ANTUMBRA_SERVE_AHEAD",
        default_value_t = true,
        action = clap::ArgAction::Set,
        value_parser = clap::builder::BoolishValueParser::new()
    )]
    serve_ahead: bool,
    /// Shift every seeded stream a training draws from: the same set trained
    /// under another value is another draw, for measuring how much a training
    /// varies. 0, the default, is the stream every training has used.
    #[arg(long, env = "ANTUMBRA_GENERATION_SEED", default_value_t = 0)]
    generation_seed: u64,
    /// Serve only a workspace's members (HTTP transport only): a signed token
    /// for a (workspace, user) the workspace does not hold is refused, where
    /// it was provisioned on its first call. People are let in by a join code
    /// one of the workspace's admins makes. Turn it on once the workspaces
    /// here have their owners; a workspace nobody has called yet is still
    /// opened by its first caller.
    #[arg(long, env = "ANTUMBRA_MEMBERS_ONLY")]
    members_only: bool,

    /// The address of the control plane that signs this server's tokens
    /// (HTTP transport only, with `--jwt-public-key`). With it, a workspace's
    /// admins make join codes here, and someone who asks to join is sent
    /// there: it verifies their address by a link and signs their token once
    /// an admin approved. Both read this server's store.
    #[arg(long, env = "ANTUMBRA_CONTROL_PLANE", value_name = "URL")]
    control_plane: Option<String>,

    /// Print a long-lived hook token for `--tenant`/`--user`, signed with
    /// `--jwt-secret` (HS256), and exit -- the credential a non-interactive
    /// lifecycle hook presents on the offline / self-hosted tier (the server
    /// otherwise only verifies, never mints).
    #[arg(long)]
    mint_token: bool,
    /// With `--mint-token`: the token may also read the substrate's own state
    /// (shadows, boundaries, the loop) and act on it from `antumbra tui`. For
    /// whoever runs this server, never for a workspace's ordinary members.
    #[arg(long, requires = "mint_token")]
    operator: bool,
    /// Validity, in days, of a `--mint-token` token.
    #[arg(long, default_value_t = 365)]
    token_ttl_days: u64,
    #[command(flatten)]
    embedder: antumbra_embed::args::EmbedderArgs,
    /// Rerank hybrid-recall candidates with a cross-encoder via a TEI/Cohere-style
    /// `/rerank` endpoint (the precision stage after RRF). Off when unset.
    #[arg(long, env = "ANTUMBRA_RERANK_URL")]
    rerank_url: Option<String>,
    /// Model name sent to `--rerank-url` (omit for text-embeddings-inference,
    /// which ignores it; Cohere/Jina require it).
    #[arg(long, env = "ANTUMBRA_RERANK_MODEL")]
    rerank_model: Option<String>,
    /// The relevance floor's calibration as `a,b` in
    /// `P(relevant) = sigmoid(a * log10(score) + b)`, for a reranker fitted by
    /// hand. Without it the floor uses the fit for the model the endpoint
    /// serves, or the configured `--rerank-model`, and runs without a floor for
    /// a model nobody has fitted.
    #[arg(long, env = "ANTUMBRA_FLOOR_CALIBRATION")]
    floor_calibration: Option<String>,
    /// Optional bearer key for `--rerank-url`.
    #[arg(long, env = "ANTUMBRA_RERANK_KEY", hide_env_values = true)]
    rerank_key: Option<String>,
    #[command(flatten)]
    copal: antumbra_copal::args::CopalArgs,
    /// The origin copal's render listener answers on (its
    /// `COPAL_RENDER_ORIGIN`), e.g. `http://10.0.0.130:8083`: where relics are
    /// displayed, and the one origin the relic pages may frame. Relics need it
    /// and an archive (`--copal-addr`); without either, publishing a relic is
    /// refused and the gallery is empty.
    #[arg(long, env = "ANTUMBRA_COPAL_RENDER_ORIGIN")]
    copal_render_origin: Option<String>,
    /// Let the networked server name copal tenants in a request header the
    /// archive trusts as given (copal's header auth mode, the tenancy default).
    /// Many identities write through one networked server, so isolation at the
    /// archive then rests on the network between them alone; say so, or use
    /// --copal-key or --copal-keys, which prove the tenant with a credential.
    #[arg(
        long,
        env = "ANTUMBRA_COPAL_TRUST_HEADERS",
        default_value_t = false,
        value_parser = clap::builder::FalseyValueParser::new()
    )]
    copal_trust_headers: bool,
    /// A Gotenberg service (gotenberg.dev) that prints relics as PDF for the
    /// viewer's download, e.g. `http://pdf:3000` with the Compose `pdf`
    /// profile. Without one, relics still download as HTML and Markdown.
    #[arg(long, env = "ANTUMBRA_PDF_RENDERER")]
    pdf_renderer: Option<String>,
    /// The address of the Antumbra server that trains this user's experts,
    /// e.g. `http://trainer:8081`. An expert's weights stay on the machine
    /// that trained it; with this set, a server that routes to an expert it
    /// does not hold pulls the adapter from there once, checks it against
    /// the sha256 on the expert's row, and serves it from then on without
    /// that server. Needs `--http` and `--jwt-secret`, the same secret as
    /// the server named: each pull is signed for the user being answered.
    /// Without it such an expert is not served here.
    #[arg(long, env = "ANTUMBRA_EXPERT_SOURCE")]
    expert_source: Option<String>,
    /// A remote SurrealDB to keep this server's store in step with, e.g.
    /// `ws://host:8000/rpc`: the collector `antumbra sync` runs, inside the
    /// server. A server on an embedded store is that store's one writer, so
    /// nothing else can sync it while the server runs. Rows are carried both
    /// ways, last write wins, as owner. Needs `--http`.
    #[arg(long, env = "ANTUMBRA_SYNC_REMOTE")]
    sync_remote: Option<String>,
    /// Root username for the sync remote. With `--sync-remote-pass`, or
    /// neither for a remote with no authentication.
    #[arg(long, env = "ANTUMBRA_SYNC_REMOTE_USER")]
    sync_remote_user: Option<String>,
    /// Root password for the sync remote.
    #[arg(long, env = "ANTUMBRA_SYNC_REMOTE_PASS", hide_env_values = true)]
    sync_remote_pass: Option<String>,
    /// Read the sync remote's password from this file instead (see
    /// --jwt-secret-file).
    #[arg(
        long,
        env = "ANTUMBRA_SYNC_REMOTE_PASS_FILE",
        conflicts_with = "sync_remote_pass"
    )]
    sync_remote_pass_file: Option<std::path::PathBuf>,
    /// Seconds between two sync passes.
    #[arg(long, env = "ANTUMBRA_SYNC_INTERVAL_SECS", default_value_t = 15)]
    sync_interval_secs: u64,
    /// Which tools this server advertises and serves: `all` (default), `agent`
    /// (the developer-agent profile: recall/store/reinforce/penalize memories,
    /// recall/ingest documents, route, answer), or a comma-separated list of
    /// tool names, in which `agent` expands (`agent,population`). Anything
    /// outside the profile is neither listed nor callable, over JSON-RPC or
    /// the REST shim: less description text in every session, and no operator
    /// action (sharing, revoking, forgetting) one agent call away.
    #[arg(long, env = "ANTUMBRA_TOOLS", default_value = "all")]
    tools: String,
    /// Serve GitHub App webhooks at `POST /github/webhook` (HTTP transport
    /// only), each delivery verified against this HMAC secret (the App's
    /// webhook secret). A merged pull request re-anchors the merged branch's
    /// memories to the merge commit and becomes a memory itself; a deleted
    /// branch marks its memories orphaned. Needs --github-tenant or
    /// --github-repos to say which workspace a repository's memories live in.
    #[arg(long, env = "ANTUMBRA_GITHUB_WEBHOOK_SECRET", hide_env_values = true)]
    github_webhook_secret: Option<String>,
    /// Read the webhook secret from this file instead (see --jwt-secret-file).
    #[arg(
        long,
        env = "ANTUMBRA_GITHUB_WEBHOOK_SECRET_FILE",
        conflicts_with = "github_webhook_secret"
    )]
    github_webhook_secret_file: Option<std::path::PathBuf>,
    /// Every repository the App delivers events for lives in this ONE
    /// workspace (tenant).
    #[arg(long, env = "ANTUMBRA_GITHUB_TENANT", conflicts_with = "github_repos")]
    github_tenant: Option<String>,
    /// Path to a JSON file mapping repository slug -> workspace tenant (e.g.
    /// `{"github.com/acme/orders": "ws:acme"}`). A repository absent from the
    /// map is acknowledged and ignored rather than landing somewhere else.
    #[arg(long, env = "ANTUMBRA_GITHUB_REPOS")]
    github_repos: Option<std::path::PathBuf>,
    /// The GitHub App's id, with --github-app-key-file: lets the receiver read
    /// repository contents as the App (installation tokens), which is what
    /// ingesting a merged pull request's documents and cold-starting a newly
    /// installed repository need. Without it the receiver still re-anchors,
    /// orphans, and remembers pull requests.
    #[arg(long, env = "ANTUMBRA_GITHUB_APP_ID")]
    github_app_id: Option<String>,
    /// Path to the App's private key (PEM), read once at startup.
    #[arg(long, env = "ANTUMBRA_GITHUB_APP_KEY_FILE")]
    github_app_key_file: Option<std::path::PathBuf>,
    /// The GitHub API base (`https://<host>/api/v3` for Enterprise Server).
    #[arg(
        long,
        env = "ANTUMBRA_GITHUB_API_URL",
        default_value = "https://api.github.com"
    )]
    github_api_url: String,
    /// Post a knowledge diff on every pull request in a mapped repository: a
    /// neutral check run listing the memories and documents anchored to the
    /// paths it changes and to its branch. Needs the App (--github-app-id)
    /// with the Checks permission (write). Off by default.
    #[arg(long, env = "ANTUMBRA_GITHUB_KNOWLEDGE_DIFF")]
    github_knowledge_diff: bool,
}

/// The owner connection, with the schema applied: the credentials are the
/// server's own login, and it signs in per request as each tenant on top.
async fn connect(url: &str, db: &antumbra_store::args::DbAuthArgs) -> Result<Store> {
    Ok(db.connect(url, EMBED_DIM).await?)
}

/// A credential-less **serving** connection to an already-provisioned remote: it
/// connects without applying the schema and only ever holds a per-request record
/// session, so the engine ACL is enforced (a root connection would bypass
/// it). Used by the networked HTTP surface for the actual request work.
pub(crate) async fn connect_serving(url: &str) -> Result<Store> {
    let config = ConnectionConfig::builder()
        .url(url)
        .namespace("antumbra")
        .database("main")
        .build()?;
    Ok(Store::connect_without_schema(config, EMBED_DIM).await?)
}

/// The built-in embedder when the flags name none: the real candle BERT under
/// `--features models`. It produces `EMBED_DIM`-wide vectors.
#[cfg(feature = "models")]
fn builtin_embedder() -> Result<Box<dyn Embedder>> {
    Ok(Box::new(antumbra_serve::BertEmbedder::load()?))
}

/// Without `models` there is no built-in embedder: the operator brings one
/// (`--embedder-url`) or opts into the byte-histogram stand-in explicitly.
/// Refusing to start beats a server whose recall silently matches character
/// statistics instead of meaning.
#[cfg(not(feature = "models"))]
fn builtin_embedder() -> Result<Box<dyn Embedder>> {
    Err(antumbra_embed::args::NoEmbedder.into())
}

/// Owner-side provisioning for an identity: ensure the principal exists and the
/// user's default (inbox) compartment is present. Idempotent (safe across
/// restarts). The caller must be in **owner mode** (not signed in as a tenant),
/// since it writes the principal/compartment tables. Returns the default
/// compartment id. Shared by both transports.
pub(crate) async fn provision_identity(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<CompartmentId> {
    principal::provision(store, tenant, user).await?;
    let default_compartment = CompartmentId::new(format!(
        "comp:{}:{}:default",
        tenant.as_str(),
        user.as_str()
    ));
    let exists = compartment::list_owned(store, tenant, user)
        .await?
        .iter()
        .any(|c| c.id == default_compartment);
    if !exists {
        compartment::create(
            store,
            &Compartment::new(
                default_compartment.clone(),
                tenant.clone(),
                user.clone(),
                "default",
                Utc::now(),
            ),
        )
        .await?;
    }
    Ok(default_compartment)
}

/// Register this machine in the user's fabric: what backend the build can
/// drive, how much video memory the device it would train on has, and the
/// role that follows. Re-registering on every start is the point -- the
/// answer changes when the binary is rebuilt with a GPU backend, or the card is
/// pulled -- and the row is keyed per (tenant, user, host), so it updates.
///
/// A failed registration is reported and not fatal. The fabric is not why the
/// operator started a memory server, and a node that cannot say where it is
/// still recalls and stores perfectly well; it just will not be dispatched to.
/// Reported rather than swallowed, because a fabric that quietly has no genesis
/// node looks exactly like a fabric whose genesis node is a laptop.
pub(crate) async fn register_node(store: &Store, tenant: &TenantId, user: &UserId, host: &str) {
    let profile = antumbra_core::DeviceProfile::detected(
        tenant.clone(),
        user.clone(),
        host,
        hardware::backend(),
        hardware::vram_mib(),
        Utc::now(),
    )
    .with_os(
        antumbra_http::os_label()
            .as_deref()
            .and_then(antumbra_core::device::os_of),
    );
    match antumbra_store::repo::device::upsert(store, &profile).await {
        Ok(()) => eprintln!(
            "antumbra-mcp: registered {host} as a {} node ({}{})",
            profile.role.as_str(),
            profile.backend,
            match profile.vram_mib {
                Some(mib) => format!(", {mib} MiB"),
                None => String::new(),
            }
        ),
        Err(e) => eprintln!("antumbra-mcp: could not register {host} in the fabric: {e}"),
    }
}

/// Build a signed-in stdio session: one connection, provisioned and bound as
/// `(tenant, user)` for the life of the process. (The HTTP transport instead
/// shares one connection across identities (see [`http`]) because an embedded
/// engine is single-writer.)
#[allow(clippy::too_many_arguments)]
async fn build_session(
    url: &str,
    db: &antumbra_store::args::DbAuthArgs,
    tenant: TenantId,
    user: UserId,
    host: String,
    embedder: Arc<dyn Embedder>,
    recipe: antumbra_core::behavior::Recipe,
    serve_quant: &str,
    serve_ahead: bool,
) -> Result<McpServer> {
    let store = connect(url, db).await?;
    let default_compartment = provision_identity(&store, &tenant, &user).await?;
    store.signin(&tenant, &user).await?;
    register_node(&store, &tenant, &user, &host).await;
    // The `answer` tool serves through the routed expert. Build the engine from
    // the session's visible population (one connection; embedded is single-writer).
    let serve = build_serve(&store, &host, &recipe, serve_quant, serve_ahead).await?;
    // The process outlives the record session, so a keeper re-signs the
    // connection in before it expires (checked at every tool call).
    let keeper = session::SessionKeeper::new(store.clone(), tenant.clone(), user.clone());
    Ok(McpServer::new(
        store,
        embedder,
        tenant,
        user,
        host,
        default_compartment,
        serve,
    )
    .with_session_keeper(keeper))
}

/// Build the serving engine the `answer` tool drives: a resident
/// [`MultiAdapterServe`](antumbra_serve::MultiAdapterServe) over the shared base,
/// registered with every expert the `store` can currently see (routing scopes
/// which a session may actually pick). `None` only when the build has no model
/// backend.
///
/// An empty population still gets an engine. It used to get `None`, so a fresh
/// node had nothing for the consolidation trigger to hot-register its first
/// expert into: the train succeeded, the log said "now servable", and `answer`
/// reported that serving was not configured until someone restarted the
/// server. The engine is lazy, so an empty one costs no VRAM: the base loads
/// on the first `answer` that has an adapter to serve.
#[cfg(feature = "models")]
pub(crate) async fn build_serve(
    store: &Store,
    host: &str,
    recipe: &antumbra_core::behavior::Recipe,
    serve_quant: &str,
    serve_ahead: bool,
) -> Result<Option<Arc<dyn antumbra_core::ports::Serve>>> {
    use antumbra_serve::{MultiAdapterServe, RaftConfig, ServeQuant};

    // Dormant experts stay registered, since they are served when named;
    // archived ones are not.
    let experts = antumbra_store::repo::lifecycle::servable(store).await?;
    // With no expert to name a base, use the one consolidation trains on, so the
    // first minted adapter fits the resident model.
    let base = serving_base(
        experts.first().map(|e| e.base_model.as_str()),
        &RaftConfig::default().base_model,
    );
    // Serve the learned mode (greedy, as admission decodes), not the
    // training-time exploration draw, at the rank the keeper trains standing
    // experts at, so a freshly minted adapter fits.
    let trained = antumbra_serve::behave::standing_config(recipe);
    // The base is held as the operator said, or as this build serves: a node
    // with no GPU backend answers from a quantized base, since the dense one
    // reads about 6 GB a token there.
    let quant =
        ServeQuant::from_setting(serve_quant).map_err(|e| anyhow::anyhow!("--serve-quant: {e}"))?;
    let cfg = RaftConfig {
        lora_rank: trained.lora_rank,
        lora_alpha: trained.lora_alpha,
        serve_quant: quant,
        read_ahead: serve_ahead,
        ..RaftConfig::for_serving(RaftConfig::default().max_new_tokens, 0.0)
    };
    eprintln!(
        "antumbra-mcp: experts are served on the {} backend from a {} base, {}",
        hardware::backend(),
        quant.map_or("dense", ServeQuant::name),
        match serve_ahead {
            true => "several tokens a pass where the device reads ahead",
            false => "a token a pass",
        }
    );
    let mut engine = MultiAdapterServe::new(base, cfg);
    // Only the adapters this machine trained. An expert's row reaches every
    // server on its database and every store that syncs with it, and the
    // adapter does not come with it: it is a file on the node that trained
    // the expert. Registering them all would route to an adapter that is not
    // here and fail at serve time, on a path that looks perfectly valid in
    // the row. One trained elsewhere is pulled and registered when it is
    // first served, where this server has a source or an archive for it.
    let (mine, elsewhere): (Vec<_>, Vec<_>) = experts.iter().partition(|e| e.is_placed_on(host));
    for e in &mine {
        engine.register(e.id.clone(), e.artifact_uri.clone());
    }
    if !elsewhere.is_empty() {
        eprintln!(
            "antumbra-mcp: {} of {} experts were trained on another node; each is pulled when it is first served, where this server has a source or an archive for it",
            elsewhere.len(),
            experts.len()
        );
    }
    Ok(Some(Arc::new(engine)))
}

#[cfg(not(feature = "models"))]
pub(crate) async fn build_serve(
    _store: &Store,
    _host: &str,
    _recipe: &antumbra_core::behavior::Recipe,
    _serve_quant: &str,
    _serve_ahead: bool,
) -> Result<Option<Arc<dyn antumbra_core::ports::Serve>>> {
    Ok(None)
}

/// Where this server gets the adapters it does not hold: the server it was
/// told trains them (`--expert-source`), when it can sign for them; the
/// copal archive it keeps, which that server archives them in; or both, the
/// archive asked when the source does not answer. `None` with neither.
///
/// A pull from the source is signed with the HS256 secret, so a server
/// verifying with a public key, which takes precedence, has nothing to sign
/// with: it says so and pulls from no source.
fn expert_source(
    url: Option<&str>,
    secret: Option<&str>,
    verifies_with_a_public_key: bool,
    archive: Option<Arc<antumbra_copal::CopalArchive>>,
) -> Option<Arc<server::pull::ExpertSource>> {
    use server::pull::ExpertSource;
    let dir = antumbra_core::adapter::DIR;
    let source = match (url.map(str::trim).filter(|u| !u.is_empty()), secret) {
        (Some(url), Some(secret)) if !verifies_with_a_public_key => {
            Some(ExpertSource::new(url, secret.as_bytes(), dir))
        }
        (Some(_), _) => {
            eprintln!(
                "antumbra-mcp: --expert-source is not in effect: a pull is signed with the HS256 \
                 secret (--jwt-secret), and this server verifies with a public key or has none"
            );
            None
        }
        (None, _) => None,
    };
    match (source, archive) {
        (Some(source), Some(archive)) => Some(Arc::new(source.with_archive(archive))),
        (Some(source), None) => Some(Arc::new(source)),
        (None, Some(archive)) => Some(Arc::new(ExpertSource::archive_only(archive, dir))),
        (None, None) => None,
    }
}

/// The base model the serving engine loads: the population's, or `default` when
/// there is no population yet.
#[cfg_attr(not(feature = "models"), allow(dead_code))]
fn serving_base(population_base: Option<&str>, default: &str) -> String {
    population_base.unwrap_or(default).to_string()
}

/// The machine name, or whatever the operator named it.
///
/// Inside a container the hostname is the container ID, which changes on every
/// recreate. Since a node's row is keyed on (tenant, user, host), taking that
/// as the name would register the same machine as a new node each time and
/// leave the old rows behind still claiming to be trainers. `--host` /
/// `ANTUMBRA_HOST` is how a containerised deployment says what it really is.
fn default_host(explicit: Option<String>) -> String {
    explicit.unwrap_or_else(|| {
        std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "local".into())
    })
}

fn main() -> Result<()> {
    // Before the runtime and held until it is gone, so the spans still queued
    // are flushed on the way out, by an exporter whose blocking HTTP client
    // must not be dropped inside the runtime.
    let _telemetry = telemetry::init();
    // SurrealDB's engine-enforced ACL subqueries for tenant isolation and memory compartments recurse deep;
    // host the runtime on a large-stack thread so the 1 MB Windows main-thread
    // stack does not overflow (see the matching note in the CLI). Requests run
    // on the worker threads, where an embedded SurrealDB 4 engine overflows the
    // 2 MiB default, so they get room too (SurrealDB recommends 10 MiB).
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(16 * 1024 * 1024)
                .build()?;
            let outcome = runtime.block_on(run());
            // Whatever still runs on the blocking pool (an embed call, a
            // train) gets a moment and is then left behind: waiting on a train
            // would outlast the grace period, and the flush comes after this.
            runtime.shutdown_timeout(std::time::Duration::from_secs(2));
            outcome
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("antumbra-mcp worker thread panicked"))?
}

/// Header-mode tenancy tells the archive the tenant in a request header it
/// trusts as given, so isolation between workspaces at the archive rests on
/// the network path alone. On the networked surface, where many identities
/// write through one server, that trust has to be stated (`--copal-trust-
/// headers`); the keyed tenancies prove the tenant and need nothing. A stdio
/// server serves one identity and is trusted as that identity already.
fn copal_headers_allowed(
    networked: bool,
    copal: Option<&antumbra_copal::CopalArchive>,
    trusted: bool,
) -> Result<()> {
    match copal {
        Some(archive) if networked && archive.trusts_headers() && !trusted => anyhow::bail!(
            "the copal archive is told each workspace's tenant in a header it trusts as given, \
             which on a networked server leaves isolation at the archive to the network between \
             them; pass --copal-trust-headers if that network is yours alone, or use --copal-key \
             or --copal-keys, which prove the tenant with a credential"
        ),
        _ => Ok(()),
    }
}

impl Cli {
    /// The optional endpoints as the operator meant them. docker compose
    /// passes an unset variable through as an empty one (`${VAR:-}`), and
    /// clap reads an empty variable as a value, so a stack without a reranker
    /// or copal configured one at the address "": every recall paid a failing
    /// rerank call, and every document ingest would fail on the archive. An
    /// empty setting is an unset one.
    fn without_empty_settings(mut self) -> Self {
        let set = |value: Option<String>| value.filter(|v| !v.trim().is_empty());
        self.rerank_url = set(self.rerank_url);
        self.rerank_key = set(self.rerank_key);
        self.copal = self.copal.without_empty_settings();
        self.copal_render_origin = set(self.copal_render_origin);
        self.pdf_renderer = set(self.pdf_renderer);
        self.sync_remote = set(self.sync_remote);
        self.sync_remote_user = set(self.sync_remote_user);
        self.sync_remote_pass = set(self.sync_remote_pass);
        self.sync_remote_pass_file = self
            .sync_remote_pass_file
            .filter(|file| !file.as_os_str().is_empty());
        self
    }
}

/// What `--sync-remote` and the flags beside it ask for: `None` with no
/// remote. A user and a password come together, or neither does for a
/// remote with no authentication; one without the other is a mistake said
/// here, not a collector that quietly signs in as nobody.
fn sync_settings(
    remote: Option<String>,
    user: Option<String>,
    pass: Option<String>,
    interval_secs: u64,
) -> Result<Option<http::sync::Settings>> {
    let Some(remote) = remote else {
        if user.is_some() || pass.is_some() {
            anyhow::bail!("--sync-remote-user and --sync-remote-pass need --sync-remote");
        }
        return Ok(None);
    };
    let credentials = match (user, pass) {
        (Some(user), Some(pass)) => Some((user, pass)),
        (None, None) => None,
        _ => anyhow::bail!(
            "--sync-remote-user and --sync-remote-pass are given together, or neither for a remote with no authentication"
        ),
    };
    Ok(Some(http::sync::Settings {
        remote,
        credentials,
        interval: std::time::Duration::from_secs(interval_secs.max(1)),
    }))
}

async fn run() -> Result<()> {
    let cli = Cli::parse().without_empty_settings();
    #[cfg(feature = "models")]
    antumbra_serve::seed_generation(cli.generation_seed);
    // Secrets resolve once, from the inline flag or its file, and only the
    // resolved value is used from here on.
    let jwt_secret = antumbra_core::secrets::resolve(
        cli.jwt_secret.clone(),
        cli.jwt_secret_file.as_deref(),
        "JWT secret",
    )?;
    let db = cli.db.resolved()?;

    // Mint a hook token and exit -- no DB or embedder needed. HS256 only: the
    // server holds the symmetric secret; an RS256 deployment mints via its auth
    // service's private key.
    if cli.mint_token {
        let secret = jwt_secret.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "--mint-token needs --jwt-secret (HS256); RS256 tokens are minted by your auth service"
            )
        })?;
        // A token minted under a weak secret is one anyone can mint.
        validate_jwt_secret(secret)?;
        let ttl = std::time::Duration::from_secs(cli.token_ttl_days * 24 * 60 * 60);
        let mint = if cli.operator {
            auth::mint_hs256_operator
        } else {
            auth::mint_hs256
        };
        println!("{}", mint(secret.as_bytes(), &cli.tenant, &cli.user, ttl)?);
        return Ok(());
    }

    if cli.auto_consolidate {
        eprintln!(
            "antumbra-mcp: --auto-consolidate is retired and does nothing; standing experts \
             are trained from accepted behaviors"
        );
    }
    let host = default_host(cli.host);
    // A configured endpoint embeds on the tenant's side; otherwise the
    // built-in embedder (candle BERT under `models`; the byte-histogram stand-in
    // only when asked for explicitly). Either way the vectors are EMBED_DIM-wide.
    let embedder: Arc<dyn Embedder> = match cli.embedder.configured() {
        Some(embedder) => Arc::from(embedder),
        None => Arc::from(builtin_embedder()?),
    };

    // Optional cross-encoder rerank stage. Operator-configured endpoint; the
    // precision stage runs after hybrid recall and degrades to RRF order on error.
    // Built once as the concrete type and then viewed two ways. The same
    // endpoint answers both questions, and calling it twice to get an order and
    // then a magnitude would double the latency of every recall.
    let rerank_model = cli.rerank_model.filter(|m| !m.trim().is_empty());
    let explicit_calibration =
        match cli.floor_calibration.as_deref().map(str::trim) {
            None | Some("") => None,
            Some(text) => Some(antumbra_rerank::floor::parse_calibration(text).ok_or_else(
                || anyhow::anyhow!("--floor-calibration wants `a,b` with a > 0, got `{text}`"),
            )?),
        };
    // The model the endpoint serves, asked once: it decides the floor's
    // calibration, since a fit belongs to one model's scores.
    let rerank_url_for_recheck = cli.rerank_url.clone();
    let rerank_key_for_recheck = cli.rerank_key.clone();
    let served = match cli.rerank_url.clone() {
        Some(url) => {
            let key = cli.rerank_key.clone();
            tokio::task::spawn_blocking(move || antumbra_rerank::served_model(&url, key.as_deref()))
                .await
                .ok()
                .flatten()
        }
        None => None,
    };
    let http_reranker: Option<Arc<antumbra_rerank::HttpReranker>> = cli.rerank_url.map(|url| {
        Arc::new(antumbra_rerank::HttpReranker::new(
            url,
            rerank_model.clone(),
            cli.rerank_key,
        ))
    });
    let reranker: Option<Arc<dyn antumbra_core::ports::Reranker>> = http_reranker
        .clone()
        .map(|r| r as Arc<dyn antumbra_core::ports::Reranker>);
    let scorer: Option<Arc<dyn antumbra_core::ports::RelevanceScorer>> =
        http_reranker.map(|r| r as Arc<dyn antumbra_core::ports::RelevanceScorer>);
    // The relevance floor. The same cross-encoder that orders the
    // pool also answers "does this answer the query" once its score is mapped
    // through the fitted calibration, so the floor costs no extra model and
    // appears exactly when a reranker is configured.
    let mut running_calibration = None;
    let warm_scorer = scorer.clone();
    let decider: Option<Arc<dyn antumbra_core::ports::TypedDecider>> = scorer.and_then(|s| {
        use antumbra_rerank::floor::{choose, CalibratedFloor, FloorChoice};
        match choose(
            explicit_calibration,
            served.as_deref(),
            rerank_model.as_deref(),
        ) {
            FloorChoice::Calibrated {
                calibration,
                because,
            } => {
                eprintln!("antumbra-mcp: relevance floor on, calibrated: {because}");
                running_calibration = Some(calibration);
                Some(Arc::new(CalibratedFloor::with_calibration(s, calibration))
                    as Arc<dyn antumbra_core::ports::TypedDecider>)
            }
            FloorChoice::Off { because } => {
                eprintln!("antumbra-mcp: relevance floor off: {because}");
                None
            }
        }
    });
    // Pay for the models' first use now rather than in the first recall, and
    // check the floor's calibration again once the reranker answers, when the
    // endpoint did not say at startup what it serves. A networked server
    // waits for its embedder before it listens, so no call waits on the
    // warm-up; a stdio session's handshake does not wait for it.
    let warm =
        cli.http.is_some() && warmup::embedder_now(embedder.as_ref(), warmup::EMBED_PATIENCE).await;
    warmup::spawn(
        (!warm).then(|| embedder.clone()),
        warm_scorer,
        rerank_url_for_recheck
            .filter(|_| served.is_none())
            .map(|url| warmup::Recheck {
                url,
                key: rerank_key_for_recheck,
                calibration: running_calibration,
            }),
    );

    // Optional copal document-of-record archive. Operator-configured; absent,
    // ingest keeps only the chunks (the v0 behavior, unchanged).
    let copal = cli.copal.archive(cli.copal_render_origin)?;
    copal_headers_allowed(
        cli.http.is_some(),
        copal.as_deref(),
        cli.copal_trust_headers,
    )?;
    // The relic viewer's PDF download, printed by a renderer beside the
    // server; only the networked surface has a viewer.
    let pdf = cli
        .pdf_renderer
        .as_deref()
        .map(pdf::PdfRenderer::new)
        .transpose()?
        .map(Arc::new);
    // The tool profile, validated against the real tool list so a typo
    // refuses at startup instead of silently hiding a tool.
    let profile = profile::ToolProfile::parse(&cli.tools, &server::McpServer::all_tool_names())?
        .map(Arc::new);
    // The GitHub App webhook receiver: its secret resolves like every other
    // secret, and a repository must map to a workspace before an event can
    // touch anything. Webhooks arrive over the network, so HTTP only.
    let github_secret = antumbra_core::secrets::resolve(
        cli.github_webhook_secret.clone(),
        cli.github_webhook_secret_file.as_deref(),
        "GitHub webhook secret",
    )?;
    let github = http::GithubConfig::from_flags(
        github_secret,
        cli.github_tenant.clone(),
        cli.github_repos.as_deref(),
        cli.github_app_id.clone(),
        cli.github_app_key_file.as_deref(),
        &cli.github_api_url,
    )?;
    if cli.github_knowledge_diff && !github.as_ref().is_some_and(|g| g.reads_contents()) {
        anyhow::bail!(
            "--github-knowledge-diff posts check runs as the App: it needs the webhook receiver              and --github-app-id with --github-app-key-file"
        );
    }
    let github = github.map(|g| g.with_knowledge_diff(cli.github_knowledge_diff));
    if github.is_some() && cli.http.is_none() {
        anyhow::bail!("the GitHub webhook receiver needs --http: deliveries arrive over the networked surface");
    }

    let recipe = antumbra_core::behavior::Recipe::new(
        cli.behave_rank,
        cli.behave_rounds,
        cli.behave_lr,
        cli.behave_regression_confidence,
    )
    .map_err(|e| {
        anyhow::anyhow!(
            "--behave-rank/--behave-rounds/--behave-lr/--behave-regression-confidence: {e}"
        )
    })?;

    if let Some(addr) = cli.http {
        // Networked multi-tenant surface: identity per request from a verified JWT.
        let verifier = build_verifier(&jwt_secret, &cli.jwt_public_key, &cli.jwt_audience)?;
        // Said at start, not at the first training an hour in.
        #[cfg(feature = "models")]
        antumbra_serve::ServeQuant::drift_from_setting(&cli.behave_drift)
            .map_err(|e| anyhow::anyhow!("--behave-drift: {e}"))?;
        let source = expert_source(
            cli.expert_source.as_deref(),
            jwt_secret.as_deref(),
            cli.jwt_public_key.is_some(),
            copal.clone(),
        );
        let sync = sync_settings(
            cli.sync_remote.clone(),
            cli.sync_remote_user.clone(),
            antumbra_core::secrets::resolve(
                cli.sync_remote_pass.clone(),
                cli.sync_remote_pass_file.as_deref(),
                "sync remote password",
            )?,
            cli.sync_interval_secs,
        )?;
        return http::serve(
            addr,
            cli.url,
            db,
            host,
            embedder,
            verifier,
            cli.auto_propose,
            cli.chunk_in_flight,
            reranker,
            decider,
            copal,
            pdf,
            profile,
            github,
            recipe,
            cli.behave_tune_confidence,
            cli.behave_drift,
            source,
            &cli.serve_quant,
            cli.serve_ahead,
            sync,
            // A server that verifies with a public key holds nothing to sign
            // with.
            jwt_secret
                .as_deref()
                .filter(|_| cli.jwt_public_key.is_none())
                .map(|secret| secret.as_bytes().to_vec()),
            cli.members_only,
            cli.control_plane.clone(),
        )
        .await;
    }

    if cli.sync_remote.is_some() {
        anyhow::bail!(
            "--sync-remote needs --http: the stdio server is one user's process and does not sync"
        );
    }
    // stdio: one fixed identity for the life of the process.
    let mut service = build_session(
        &cli.url,
        &db,
        TenantId::new(cli.tenant),
        UserId::new(cli.user),
        host,
        embedder,
        recipe,
        &cli.serve_quant,
        cli.serve_ahead,
    )
    .await?;
    if let Some(threshold) = cli.auto_propose {
        service = service.with_auto_propose(threshold);
    }
    if let Some(r) = reranker {
        service = service.with_reranker(r);
    }
    if let Some(d) = decider {
        service = service.with_decider(d);
    }
    if let Some(c) = copal {
        service = service.with_copal_archive(c);
    }
    if let Some(p) = profile {
        service = service.with_tool_profile(p);
    }
    let running = service
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    running.waiting().await?;
    Ok(())
}

/// The `.env.example` placeholder: a secret anyone can read out of the repo,
/// so refusing it outright beats serving with tokens anyone can mint.
const PLACEHOLDER_JWT_SECRET: &str = "change-me-base64-32-bytes";

/// The fewest bytes an HS256 secret may have: the key is the only thing between
/// a token's claims and anyone who wants to write them, and below this an
/// offline search finds it.
const MIN_JWT_SECRET_BYTES: usize = 32;

/// Refuse an HS256 secret that is the placeholder or too short, naming the fix.
fn validate_jwt_secret(secret: &str) -> Result<()> {
    anyhow::ensure!(
        secret != PLACEHOLDER_JWT_SECRET,
        "ANTUMBRA_JWT_SECRET is still the .env.example placeholder; generate a real one: \
         openssl rand -base64 32"
    );
    anyhow::ensure!(
        secret.len() >= MIN_JWT_SECRET_BYTES,
        "the HS256 secret (--jwt-secret / ANTUMBRA_JWT_SECRET) is too short ({} bytes, want at \
         least {MIN_JWT_SECRET_BYTES}); generate one: openssl rand -base64 32",
        secret.len()
    );
    Ok(())
}

/// Resolve the JWT verifier from the configured key material. RS256 (a PEM
/// public key) takes precedence over an HS256 secret; one is required, and an
/// HS256 secret must be one worth the name ([`validate_jwt_secret`]).
fn build_verifier(
    secret: &Option<String>,
    public_key: &Option<std::path::PathBuf>,
    audience: &Option<String>,
) -> Result<auth::JwtVerifier> {
    let mut v = match (public_key, secret) {
        (Some(pem_path), _) => {
            let pem = std::fs::read(pem_path)?;
            auth::JwtVerifier::rs256_pem(&pem)?
        }
        (None, Some(secret)) => {
            validate_jwt_secret(secret)?;
            auth::JwtVerifier::hs256(secret.as_bytes())
        }
        (None, None) => anyhow::bail!(
            "--http needs a JWT key: pass --jwt-secret (HS256) or --jwt-public-key <pem> (RS256)"
        ),
    };
    if let Some(aud) = audience {
        v = v.with_audience(aud);
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_store::testing::mem_store;

    // The store flags are the shared ones (#303): the password's environment
    // value is hidden in --help, and the url reads its variable, as in every
    // binary; --db-pass-file still excludes --db-pass across the flatten.
    #[test]
    fn the_store_flags_read_as_in_every_binary() -> Result<()> {
        use clap::CommandFactory;
        let cmd = Cli::command();
        cmd.clone().debug_assert();
        let flag = |id: &str| {
            cmd.get_arguments().find(|a| a.get_id() == id).map(|a| {
                (
                    a.get_env().map(|e| e.to_string_lossy().into_owned()),
                    a.is_hide_env_values_set(),
                )
            })
        };
        assert_eq!(
            flag("db_pass"),
            Some((Some("ANTUMBRA_DB_PASS".into()), true))
        );
        assert_eq!(flag("url"), Some((Some("ANTUMBRA_DB_URL".into()), false)));
        assert!(
            Cli::try_parse_from(["antumbra-mcp", "--db-pass", "x", "--db-pass-file", "f"]).is_err()
        );
        Ok(())
    }

    // The verifier refuses a secret an offline search would find, and the
    // placeholder anyone can read out of the repository, and says what to do.
    #[test]
    fn the_verifier_refuses_a_short_or_placeholder_secret() -> Result<()> {
        for weak in [
            "short",
            "0123456789abcdef0123456789abcde",
            PLACEHOLDER_JWT_SECRET,
        ] {
            let Err(e) = build_verifier(&Some(weak.to_string()), &None, &None) else {
                anyhow::bail!("{weak:?} was accepted as an HS256 secret");
            };
            anyhow::ensure!(e.to_string().contains("openssl rand -base64 32"), "{e}");
        }
        let Err(e) = validate_jwt_secret("short") else {
            anyhow::bail!("a short secret passed validation");
        };
        anyhow::ensure!(e.to_string().contains("5 bytes"), "{e}");
        Ok(())
    }

    #[test]
    fn the_verifier_takes_a_secret_of_32_bytes_or_more() -> Result<()> {
        build_verifier(
            &Some("0123456789abcdef0123456789abcdef".to_string()),
            &None,
            &None,
        )?;
        build_verifier(
            &Some("oqRzkA0sZ1Yx9fJ2mB7cD4eF6gH8iK0lMnOpQrStUvWx".to_string()),
            &None,
            &Some("antumbra".to_string()),
        )?;
        Ok(())
    }

    /// What compose hands over for an unset endpoint is an empty string, and
    /// clap keeps it; the server treats it as no endpoint at all.
    #[test]
    fn an_empty_endpoint_setting_is_an_unset_one() -> Result<()> {
        use anyhow::Context as _;

        let cli = Cli::try_parse_from([
            "antumbra-mcp",
            "--rerank-url",
            "",
            "--copal-addr",
            " ",
            "--copal-tenant",
            "",
            "--copal-render-origin",
            "",
        ])
        .context("empty values parse")?
        .without_empty_settings();
        assert_eq!(cli.rerank_url, None);
        assert_eq!(cli.copal.copal_addr, None);
        assert_eq!(cli.copal.copal_tenant, None);
        assert_eq!(cli.copal_render_origin, None);

        let cli = Cli::try_parse_from([
            "antumbra-mcp",
            "--rerank-url",
            "http://rerank/rerank",
            "--copal-addr",
            "http://copal:8080",
        ])
        .context("set values parse")?
        .without_empty_settings();
        assert_eq!(cli.rerank_url.as_deref(), Some("http://rerank/rerank"));
        assert_eq!(cli.copal.copal_addr.as_deref(), Some("http://copal:8080"));
        Ok(())
    }

    /// Header-mode tenancy on the networked surface is the operator's to
    /// trust, by name; the keyed modes and the stdio server need nothing.
    #[test]
    fn header_mode_with_http_needs_the_flag() -> anyhow::Result<()> {
        use antumbra_copal::{CopalArchive, CopalTenancy};
        let header_mode = CopalArchive::new("127.0.0.1:9010", CopalTenancy::PerWorkspace);
        let keyed = CopalArchive::new("127.0.0.1:9010", CopalTenancy::SharedKey("ck1_x".into()));
        anyhow::ensure!(
            copal_headers_allowed(true, Some(&header_mode), false).is_err(),
            "header mode on the networked server without the flag is refused"
        );
        copal_headers_allowed(true, Some(&header_mode), true)?;
        copal_headers_allowed(true, Some(&keyed), false)?;
        copal_headers_allowed(false, Some(&header_mode), false)?;
        copal_headers_allowed(true, None, false)?;
        Ok(())
    }

    #[test]
    fn an_empty_population_still_names_a_base() -> Result<()> {
        assert_eq!(serving_base(None, "org/base"), "org/base");
        assert_eq!(
            serving_base(Some("org/trained-on"), "org/base"),
            "org/trained-on"
        );
        Ok(())
    }

    /// A node puts itself in the user's fabric on start, under its own record
    /// session -- which is the whole point of the fabric's write rule: the row a
    /// node writes is its own user's, and it writes it as that user rather than
    /// as the owner.
    #[tokio::test]
    async fn a_node_registers_itself_as_the_user_it_serves() -> Result<()> {
        let store = mem_store().await?;
        let tenant = TenantId::new("ws:fabric");
        let user = UserId::new("user:lily");
        provision_identity(&store, &tenant, &user).await?;
        store.signin(&tenant, &user).await?;

        register_node(&store, &tenant, &user, "her-laptop").await;

        let fabric = antumbra_store::repo::device::list_for_user(&store, &tenant, &user).await?;
        let [node] = fabric.as_slice() else {
            anyhow::bail!("one machine, one row, got {}", fabric.len());
        };
        assert_eq!(node.host, "her-laptop");
        assert_eq!(node.backend, hardware::backend());
        // The role is the one the hardware earns, not one this test asserts
        // into being: on a CPU build that is Memory, on a CUDA build with a big
        // enough card it is Genesis, and either way it is derived.
        assert_eq!(
            node.role,
            antumbra_core::role_for(hardware::backend(), hardware::vram_mib())
        );

        // Starting again is not a second machine.
        register_node(&store, &tenant, &user, "her-laptop").await;
        assert_eq!(
            antumbra_store::repo::device::list_for_user(&store, &tenant, &user)
                .await?
                .len(),
            1
        );
        Ok(())
    }

    /// The cold start: a node with no experts yet must still come up with a
    /// serving engine, or the first expert it mints has nowhere to be registered.
    #[cfg(feature = "models")]
    #[tokio::test]
    async fn a_fresh_node_gets_a_serving_engine() -> Result<()> {
        let store = mem_store().await?;
        let Some(serve) = build_serve(
            &store,
            "test-host",
            &antumbra_core::behavior::Recipe::default(),
            "auto",
            true,
        )
        .await?
        else {
            anyhow::bail!("an empty population must still get a serving engine");
        };
        let minted = antumbra_core::ExpertId::new("expert:user:test:comp:fresh");
        assert!(!serve.can_serve(&minted));
        serve.register_expert(&minted, "adapters/fresh_g0.safetensors");
        assert!(serve.can_serve(&minted), "hot-registration has a target");
        Ok(())
    }

    // The embedder and copal flags are the shared ones (#304): the model reads
    // its variable (a server started with it set no longer ignores it), the
    // keys hide their environment values in --help, every flag keeps its name
    // and default.
    #[test]
    fn the_embedder_and_copal_flags_read_as_in_every_binary() -> Result<()> {
        use clap::CommandFactory;
        let cmd = Cli::command();
        cmd.clone().debug_assert();
        let flag = |id: &str| {
            cmd.get_arguments().find(|a| a.get_id() == id).map(|a| {
                (
                    a.get_long().map(str::to_owned),
                    a.get_env().map(|e| e.to_string_lossy().into_owned()),
                    a.is_hide_env_values_set(),
                    a.get_default_values()
                        .iter()
                        .map(|v| v.to_string_lossy().into_owned())
                        .collect::<Vec<_>>(),
                )
            })
        };
        assert_eq!(
            flag("embedder_model"),
            Some((
                Some("embedder-model".into()),
                Some("ANTUMBRA_EMBEDDER_MODEL".into()),
                false,
                vec!["all-MiniLM-L6-v2".into()],
            ))
        );
        for (id, long, env) in [
            ("embedder_url", "embedder-url", "ANTUMBRA_EMBEDDER_URL"),
            ("embedder_key", "embedder-key", "ANTUMBRA_EMBEDDER_KEY"),
            ("fake_embedder", "fake-embedder", "ANTUMBRA_FAKE_EMBEDDER"),
            ("copal_addr", "copal-addr", "ANTUMBRA_COPAL_ADDR"),
            ("copal_tenant", "copal-tenant", "ANTUMBRA_COPAL_TENANT"),
            ("copal_key", "copal-key", "ANTUMBRA_COPAL_KEY"),
            ("copal_keys", "copal-keys", "ANTUMBRA_COPAL_KEYS"),
        ] {
            let Some(got) = flag(id) else {
                anyhow::bail!("{id} is a flag");
            };
            assert_eq!(got.0.as_deref(), Some(long), "{id}");
            assert_eq!(got.1.as_deref(), Some(env), "{id}");
            assert_eq!(
                got.2,
                id.ends_with("_key"),
                "{id} hides its value iff a key"
            );
        }
        Ok(())
    }

    // What the sync flags become (#393): a remote, and how to sign in to it.
    #[test]
    fn the_sync_flags_name_a_remote_and_how_to_sign_in_to_it() -> Result<()> {
        let remote = || Some("ws://db:8000/rpc".to_string());
        assert_eq!(sync_settings(None, None, None, 15)?, None);
        assert_eq!(
            sync_settings(remote(), None, None, 15)?,
            Some(http::sync::Settings {
                remote: "ws://db:8000/rpc".into(),
                credentials: None,
                interval: std::time::Duration::from_secs(15),
            })
        );
        // An interval of none is one second, not a loop that never waits.
        assert_eq!(
            sync_settings(remote(), Some("root".into()), Some("a password".into()), 0)?,
            Some(http::sync::Settings {
                remote: "ws://db:8000/rpc".into(),
                credentials: Some(("root".into(), "a password".into())),
                interval: std::time::Duration::from_secs(1),
            })
        );
        // One without the other, or either without a remote, is said: not a
        // collector that quietly signs in as nobody.
        assert!(sync_settings(remote(), Some("root".into()), None, 15).is_err());
        assert!(sync_settings(remote(), None, Some("a password".into()), 15).is_err());
        assert!(sync_settings(None, Some("root".into()), None, 15).is_err());
        assert!(sync_settings(None, None, Some("a password".into()), 15).is_err());
        Ok(())
    }

    // The remote's password is a secret like the server's others: its
    // environment value stays out of --help, it may come from a file and not
    // from both, and an empty setting, as docker compose passes an unset
    // variable, is an unset one.
    #[test]
    fn the_sync_password_is_hidden_and_an_empty_setting_is_unset() -> Result<()> {
        use clap::CommandFactory;
        let cmd = Cli::command();
        let flag = |id: &str| {
            cmd.get_arguments().find(|a| a.get_id() == id).map(|a| {
                (
                    a.get_env().map(|e| e.to_string_lossy().into_owned()),
                    a.is_hide_env_values_set(),
                )
            })
        };
        assert_eq!(
            flag("sync_remote_pass"),
            Some((Some("ANTUMBRA_SYNC_REMOTE_PASS".into()), true))
        );
        assert_eq!(
            flag("sync_remote"),
            Some((Some("ANTUMBRA_SYNC_REMOTE".into()), false))
        );
        assert!(Cli::try_parse_from([
            "antumbra-mcp",
            "--sync-remote-pass",
            "x",
            "--sync-remote-pass-file",
            "f"
        ])
        .is_err());
        let cli = Cli::try_parse_from([
            "antumbra-mcp",
            "--sync-remote",
            "",
            "--sync-remote-user",
            " ",
            "--sync-remote-pass",
            "",
        ])?
        .without_empty_settings();
        assert_eq!(
            (cli.sync_remote, cli.sync_remote_user, cli.sync_remote_pass),
            (None, None, None)
        );
        Ok(())
    }
}
