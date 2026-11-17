//! The surface pair: the flags every `antumbra claude` command that calls the
//! Antumbra MCP surface takes, as the hooks know them.

use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct SurfaceArgs {
    /// The Antumbra MCP surface, as the hooks know it.
    #[arg(
        long = "surface",
        env = "ANTUMBRA_URL",
        default_value = "http://127.0.0.1:8081"
    )]
    pub surface: String,
    /// The bearer token for that surface. Prefer the environment variable: a
    /// flag ends up in the shell's history.
    #[arg(long, env = "ANTUMBRA_TOKEN", hide_env_values = true)]
    pub token: Option<String>,
}
