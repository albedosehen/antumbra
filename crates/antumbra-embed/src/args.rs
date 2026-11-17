//! The embedder flags every binary that embeds takes, declared once. The CLI
//! and the MCP server each declared them, and they drifted: the CLI read
//! `ANTUMBRA_EMBEDDER_MODEL` and the server did not, so a server started with
//! that variable set ignored it; the server hid the key's environment value in
//! `--help` and the CLI printed it. Flattened into each binary's command, the
//! flags read the same everywhere.
//!
//! What the flags name outright is built here: the endpoint, or the stand-in.
//! The built-in model (candle BERT under `--features models`) stays with the
//! binaries, which hold that feature; without it, a binary refuses with
//! [`NoEmbedder`] rather than embed by character statistics in silence.

use clap::Args;

use antumbra_core::ports::Embedder;
use antumbra_core::testing::FixedEmbedder;
use antumbra_store::EMBED_DIM;

use crate::HttpEmbedder;

/// The embedder flags, as every binary takes them. `global`, so a command
/// with subcommands accepts them anywhere on the line; on a command without
/// subcommands that changes nothing.
#[derive(Args, Clone, Debug)]
pub struct EmbedderArgs {
    /// Embed via an OpenAI-compatible `/embeddings` endpoint (for example
    /// Ollama serving `all-minilm`, or text-embeddings-inference) instead of
    /// the built-in embedder. It must return vectors of the store's dimension.
    /// Use the SAME endpoint the population and memories were built with, or
    /// recall and routing lose coherence. Off when unset.
    #[arg(long, global = true, env = "ANTUMBRA_EMBEDDER_URL")]
    pub embedder_url: Option<String>,
    /// Model name sent to `--embedder-url` (it must produce the index
    /// dimension).
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_EMBEDDER_MODEL",
        default_value = "all-MiniLM-L6-v2"
    )]
    pub embedder_model: String,
    /// Optional bearer key for `--embedder-url`.
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_EMBEDDER_KEY",
        hide_env_values = true
    )]
    pub embedder_key: Option<String>,
    /// Use the deterministic byte-histogram stand-in instead of a real
    /// embedder. Recall is then NOT semantic (it matches character
    /// statistics), so this is for smoke tests and demos only. Without
    /// `--features models`, a binary that embeds refuses to run with neither
    /// this nor `--embedder-url` rather than silently degrading.
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_FAKE_EMBEDDER",
        default_value_t = false
    )]
    pub fake_embedder: bool,
}

impl EmbedderArgs {
    /// The embedder the flags name outright: the endpoint when
    /// `--embedder-url` is set (it always wins: the same endpoint the
    /// population was built with), the byte-histogram stand-in when
    /// `--fake-embedder` asks for it, and `None` when neither does, for the
    /// binary to fill with its built-in model or refuse with [`NoEmbedder`].
    /// Every path produces `EMBED_DIM`-wide vectors, so the gate and the
    /// store stay dimension-consistent.
    pub fn configured(&self) -> Option<Box<dyn Embedder>> {
        if let Some(url) = &self.embedder_url {
            return Some(Box::new(HttpEmbedder::new(
                url.clone(),
                self.embedder_model.clone(),
                self.embedder_key.clone(),
            )));
        }
        self.fake_embedder
            .then(|| Box::new(FixedEmbedder::new(EMBED_DIM)) as Box<dyn Embedder>)
    }
}

/// The refusal when the flags name no embedder and the binary has no built-in
/// one: a command whose recall silently matched character statistics instead
/// of meaning would be worse than one that stops and says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoEmbedder;

impl std::fmt::Display for NoEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "no embedder configured: pass --embedder-url <OpenAI-compatible /embeddings endpoint \
             returning {EMBED_DIM}-d vectors> (for example Ollama serving all-minilm), or \
             --fake-embedder to accept the non-semantic byte-histogram stand-in (demos only)"
        )
    }
}

impl std::error::Error for NoEmbedder {}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::*;

    /// A command with nothing but these flags.
    #[derive(Parser, Debug)]
    struct Probe {
        #[command(flatten)]
        embedder: EmbedderArgs,
    }

    fn parsed(argv: &[&str]) -> anyhow::Result<EmbedderArgs> {
        Ok(Probe::try_parse_from(argv)?.embedder)
    }

    #[test]
    fn the_flags_read_the_same_in_every_binary() -> anyhow::Result<()> {
        Probe::command().debug_assert();
        let cmd = Probe::command();
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
        anyhow::ensure!(
            flag("embedder_model")
                == Some((
                    Some("embedder-model".into()),
                    Some("ANTUMBRA_EMBEDDER_MODEL".into()),
                    false,
                    vec!["all-MiniLM-L6-v2".into()],
                )),
            "--embedder-model reads its variable and keeps its default"
        );
        anyhow::ensure!(
            flag("embedder_key")
                == Some((
                    Some("embedder-key".into()),
                    Some("ANTUMBRA_EMBEDDER_KEY".into()),
                    true,
                    vec![],
                )),
            "--embedder-key hides its environment value"
        );
        anyhow::ensure!(
            flag("embedder_url").map(|f| f.1) == Some(Some("ANTUMBRA_EMBEDDER_URL".into())),
            "--embedder-url reads its variable"
        );
        anyhow::ensure!(
            flag("fake_embedder").map(|f| (f.1, f.3))
                == Some((Some("ANTUMBRA_FAKE_EMBEDDER".into()), vec!["false".into()])),
            "--fake-embedder reads its variable and is off by default"
        );
        Ok(())
    }

    #[test]
    fn the_endpoint_wins_the_stand_in_is_opted_into_and_nothing_is_none() -> anyhow::Result<()> {
        let none = parsed(&["probe"])?;
        anyhow::ensure!(none.configured().is_none(), "no flags name no embedder");
        anyhow::ensure!(none.embedder_model == "all-MiniLM-L6-v2");

        let Some(fake) = parsed(&["probe", "--fake-embedder"])?.configured() else {
            anyhow::bail!("--fake-embedder names the stand-in");
        };
        anyhow::ensure!(fake.dim() == EMBED_DIM);

        let Some(endpoint) = parsed(&[
            "probe",
            "--embedder-url",
            "http://embed.example/v1/embeddings",
            "--fake-embedder",
        ])?
        .configured() else {
            anyhow::bail!("--embedder-url names the endpoint");
        };
        anyhow::ensure!(endpoint.dim() == EMBED_DIM);
        Ok(())
    }

    #[test]
    fn the_refusal_names_both_ways_out() -> anyhow::Result<()> {
        let shown = NoEmbedder.to_string();
        assert!(shown.contains("--embedder-url"), "{shown}");
        assert!(shown.contains("--fake-embedder"), "{shown}");
        assert!(shown.contains(&EMBED_DIM.to_string()), "{shown}");
        Ok(())
    }
}
