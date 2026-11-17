//! The copal archive flags every binary that ingests takes, declared once.
//! The CLI and the MCP server each declared them and they drifted: the server
//! hid the key's environment value in `--help` and the CLI printed it.
//! Flattened into each binary's command, the flags read the same everywhere,
//! and [`CopalArchive::from_flags`] is reached one way.
//!
//! The render origin stays the MCP server's own flag: only the networked
//! surface publishes relics, so only it has links to build.

use std::sync::Arc;

use clap::Args;

use crate::{CopalArchive, CopalConfigError};

/// The copal flags, as every binary takes them. `global`, so a command with
/// subcommands accepts them anywhere on the line; on a command without
/// subcommands that changes nothing.
#[derive(Args, Clone, Debug, Default)]
pub struct CopalArgs {
    /// Archive each ingested document's ORIGINAL content to a copal file
    /// service (the document of record): bare `host:port` or a full URL base.
    /// The upload happens before any chunk is stored (a dead copal fails the
    /// ingest), and every chunk carries the copal file id + digest. Off when
    /// unset: ingest keeps only the chunks.
    #[arg(long, global = true, env = "ANTUMBRA_COPAL_ADDR")]
    pub copal_addr: Option<String>,
    /// Land every workspace's documents under this ONE copal tenant, named by
    /// the `x-copal-tenant` header (copal `header` auth mode). Omit all three
    /// tenancy args for the default: per-workspace tenancy over header auth,
    /// where each workspace presents ITSELF as the copal tenant (its own
    /// quotas, listings, and search scope). Either way the archive identity is
    /// (workspace, title). At most one of --copal-tenant / --copal-key /
    /// --copal-keys.
    #[arg(long, global = true, env = "ANTUMBRA_COPAL_TENANT")]
    pub copal_tenant: Option<String>,
    /// One `ck1` copal API key for every workspace: shared tenancy under
    /// copal's deployed `keys` auth mode, where the tenant is bound to the
    /// credential (the key's tenant is THE tenant).
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_COPAL_KEY",
        hide_env_values = true
    )]
    pub copal_key: Option<String>,
    /// Path to a JSON file mapping workspace tenant -> `ck1` copal API key
    /// (e.g. `{"ws:acme": "ck1..."}`): per-workspace tenancy under copal's
    /// `keys` auth mode. Loaded once at startup (restart to pick up newly
    /// minted keys); a workspace absent from the map fails its ingest rather
    /// than landing in another tenant.
    #[arg(long, global = true, env = "ANTUMBRA_COPAL_KEYS")]
    pub copal_keys: Option<std::path::PathBuf>,
    /// Allow a plain `http://` archive address that is not this machine: the
    /// network between this process and the archive is trusted (a compose
    /// stack's own network). Without it such an address is refused, and a bare
    /// `host:port` away from this machine gets `https://`.
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_COPAL_ALLOW_HTTP",
        default_value_t = false,
        value_parser = clap::builder::FalseyValueParser::new()
    )]
    pub copal_allow_http: bool,
}

impl CopalArgs {
    /// The flags as the operator meant them: docker compose passes an unset
    /// variable through as an empty one (`${VAR:-}`), and clap reads an empty
    /// variable as a value, so a stack without copal configured one at the
    /// address "" and every document ingest failed on the archive. An empty
    /// setting is an unset one.
    pub fn without_empty_settings(mut self) -> Self {
        let set = |value: Option<String>| value.filter(|v| !v.trim().is_empty());
        self.copal_addr = set(self.copal_addr);
        self.copal_tenant = set(self.copal_tenant);
        self.copal_key = set(self.copal_key);
        self.copal_keys = self.copal_keys.filter(|p| !p.as_os_str().is_empty());
        self
    }

    /// The archive these flags configure: `None` when no address is set, so
    /// ingest keeps only the chunks. `render_origin` is the MCP server's; a
    /// binary that publishes no relics passes `None`.
    pub fn archive(
        &self,
        render_origin: Option<String>,
    ) -> Result<Option<Arc<CopalArchive>>, CopalConfigError> {
        CopalArchive::from_flags(
            self.copal_addr.as_deref(),
            self.copal_tenant.clone(),
            self.copal_key.clone(),
            self.copal_keys.as_deref(),
            render_origin,
            self.copal_allow_http,
        )
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::*;

    /// A command with nothing but these flags.
    #[derive(Parser, Debug)]
    struct Probe {
        #[command(flatten)]
        copal: CopalArgs,
    }

    fn parsed(argv: &[&str]) -> Result<CopalArgs, clap::Error> {
        Probe::try_parse_from(argv).map(|p| p.copal)
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
                )
            })
        };
        for (id, long, env, hidden) in [
            ("copal_addr", "copal-addr", "ANTUMBRA_COPAL_ADDR", false),
            (
                "copal_tenant",
                "copal-tenant",
                "ANTUMBRA_COPAL_TENANT",
                false,
            ),
            ("copal_key", "copal-key", "ANTUMBRA_COPAL_KEY", true),
            ("copal_keys", "copal-keys", "ANTUMBRA_COPAL_KEYS", false),
        ] {
            let want = Some((Some(long.to_owned()), Some(env.to_owned()), hidden));
            if flag(id) != want {
                anyhow::bail!("{id}: {:?} != {want:?}", flag(id));
            }
        }
        Ok(())
    }

    // (clap itself refuses an empty path on the command line; the filter on
    // `--copal-keys` is for the environment, where compose hands one over.)
    /// Plain HTTP across a network is the operator's to allow, by name.
    #[test]
    fn plain_http_across_a_network_needs_the_flag() -> Result<(), String> {
        let refused = parsed(&["probe", "--copal-addr", "http://archive.example:9000"])
            .map_err(|e| e.to_string())?;
        if !refused.copal_allow_http {
            match refused.archive(None) {
                Err(CopalConfigError::Cleartext(_)) => {}
                Err(other) => return Err(format!("the wrong refusal: {other}")),
                Ok(_) => return Err("plain HTTP across a network was accepted".into()),
            }
        }
        let allowed = parsed(&[
            "probe",
            "--copal-addr",
            "http://archive.example:9000",
            "--copal-allow-http",
        ])
        .map_err(|e| e.to_string())?;
        if !allowed.copal_allow_http || allowed.archive(None).map_err(|e| e.to_string())?.is_none()
        {
            return Err("--copal-allow-http did not allow it".into());
        }
        Ok(())
    }

    #[test]
    fn an_empty_setting_is_an_unset_one() -> anyhow::Result<()> {
        let args = parsed(&[
            "probe",
            "--copal-addr",
            " ",
            "--copal-tenant",
            "",
            "--copal-key",
            "",
        ])?
        .without_empty_settings();
        if args.copal_addr.is_some()
            || args.copal_tenant.is_some()
            || args.copal_key.is_some()
            || args.copal_keys.is_some()
        {
            anyhow::bail!("blank settings kept: {args:?}");
        }
        let args = parsed(&["probe", "--copal-addr", "http://copal.example:8080"])?
            .without_empty_settings();
        if args.copal_addr.as_deref() != Some("http://copal.example:8080") {
            anyhow::bail!("a set address was dropped: {args:?}");
        }
        Ok(())
    }

    #[test]
    fn no_address_is_no_archive_and_two_tenancies_are_refused() -> anyhow::Result<()> {
        let none = parsed(&["probe"])?;
        if none.archive(None)?.is_some() {
            anyhow::bail!("an archive from no address");
        }
        let two = parsed(&[
            "probe",
            "--copal-addr",
            "copal.example:8080",
            "--copal-tenant",
            "ws:acme",
            "--copal-key",
            "ck1_test",
        ])?;
        match two.archive(None) {
            Err(CopalConfigError::ConflictingTenancy) => Ok(()),
            Err(other) => anyhow::bail!("the wrong refusal: {other}"),
            Ok(_) => anyhow::bail!("two tenancies were accepted"),
        }
    }

    #[test]
    fn one_tenancy_configures_an_archive() -> anyhow::Result<()> {
        let one = parsed(&[
            "probe",
            "--copal-addr",
            "copal.example:8080",
            "--copal-tenant",
            "ws:acme",
        ])?;
        match one.archive(Some("https://render.example".into())) {
            Ok(Some(_)) => Ok(()),
            Ok(None) => anyhow::bail!("an address configured no archive"),
            Err(e) => anyhow::bail!("refused: {e}"),
        }
    }
}
