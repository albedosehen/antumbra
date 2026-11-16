//! The database credential flags every Antumbra binary takes, and the one
//! connection they make with them. Three binaries declared these flags apart
//! and drifted: one hid the password's environment value in `--help` and the
//! others did not, one offered a database-scoped login and the others did
//! not. Declared once here, flattened into each binary's command, the flags
//! read the same everywhere and connect the same way.
//!
//! The URL stays each binary's own: its default differs (an embedded store for
//! the CLI and the MCP server, a server beside it for the control server),
//! and clap cannot override a flattened field's default.

use clap::{Args, ValueEnum};

use antumbra_core::Result;

use crate::{ConnectionConfig, Store};

/// How `--db-user` / `--db-pass` authenticate to SurrealDB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum DbAuth {
    /// Instance-root credentials.
    #[default]
    Root,
    /// A `DEFINE USER ... ON DATABASE ... ROLES OWNER` user: least privilege,
    /// containing a compromise of the process to the one database it works in.
    Database,
}

/// The credential flags, as every binary takes them. `global`, so a command
/// with subcommands accepts them anywhere on the line; on a command without
/// subcommands that changes nothing.
#[derive(Args, Clone, Debug, Default)]
pub struct DbAuthArgs {
    /// Username for an authenticated remote SurrealDB (`ws://`). Omit for an
    /// embedded store or an unauthenticated server. This credential is the
    /// process's own login, distinct from the tokens that authenticate its
    /// clients.
    #[arg(long, global = true, env = "ANTUMBRA_DB_USER")]
    pub db_user: Option<String>,
    /// The password that goes with `--db-user`.
    #[arg(long, global = true, env = "ANTUMBRA_DB_PASS", hide_env_values = true)]
    pub db_pass: Option<String>,
    /// The authentication level of `--db-user` / `--db-pass`.
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_DB_AUTH",
        value_enum,
        default_value = "root"
    )]
    pub db_auth: DbAuth,
}

impl DbAuthArgs {
    /// Connect to `url` with these credentials and apply the schema: as root
    /// when both are given (none for an embedded or unauthenticated store), or
    /// as the database user when `--db-auth database`, which needs both.
    pub async fn connect(&self, url: &str, embed_dim: usize) -> Result<Store> {
        let builder = ConnectionConfig::builder()
            .url(url)
            .namespace("antumbra")
            .database("main");
        match self.db_auth {
            DbAuth::Database => {
                let (Some(user), Some(pass)) = (self.db_user.as_deref(), self.db_pass.as_deref())
                else {
                    return Err(antumbra_core::AntumbraError::other(
                        "--db-auth database requires --db-user and --db-pass",
                    ));
                };
                Store::connect_with_db_user(
                    builder.build().map_err(crate::error::map)?,
                    user,
                    pass,
                    embed_dim,
                )
                .await
            }
            DbAuth::Root => {
                let builder = match (self.db_user.as_deref(), self.db_pass.as_deref()) {
                    (Some(user), Some(pass)) => builder.username(user).password(pass),
                    _ => builder,
                };
                Store::connect(builder.build().map_err(crate::error::map)?, embed_dim).await
            }
        }
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
        db: DbAuthArgs,
    }

    #[test]
    fn the_flags_parse_hide_the_password_and_default_to_root() -> Result<()> {
        Probe::command().debug_assert();
        let probe = Probe::try_parse_from(["probe"]).map_err(|e| {
            antumbra_core::AntumbraError::other(format!("no flags did not parse: {e}"))
        })?;
        if probe.db.db_user.is_some() || probe.db.db_pass.is_some() {
            return Err(antumbra_core::AntumbraError::other(
                "credentials from nowhere",
            ));
        }
        if probe.db.db_auth != DbAuth::Root {
            return Err(antumbra_core::AntumbraError::other("the default is root"));
        }
        let probe = Probe::try_parse_from([
            "probe",
            "--db-user",
            "svc",
            "--db-pass",
            "pw",
            "--db-auth",
            "database",
        ])
        .map_err(|e| antumbra_core::AntumbraError::other(format!("flags did not parse: {e}")))?;
        if probe.db.db_auth != DbAuth::Database || probe.db.db_user.as_deref() != Some("svc") {
            return Err(antumbra_core::AntumbraError::other(
                "the flags were not read",
            ));
        }
        let hidden = Probe::command()
            .get_arguments()
            .find(|a| a.get_id() == "db_pass")
            .is_some_and(|a| a.is_hide_env_values_set() && a.get_env().is_some());
        if !hidden {
            return Err(antumbra_core::AntumbraError::other(
                "--db-pass must read the environment and hide its value in --help",
            ));
        }
        Ok(())
    }

    #[tokio::test]
    async fn root_without_credentials_connects_to_an_embedded_store() -> Result<()> {
        let store = DbAuthArgs::default()
            .connect("mem://", crate::EMBED_DIM)
            .await?;
        store.signin_root().await?;
        Ok(())
    }

    #[tokio::test]
    async fn the_database_level_needs_both_credentials() -> Result<()> {
        let args = DbAuthArgs {
            db_auth: DbAuth::Database,
            ..DbAuthArgs::default()
        };
        let Err(e) = args.connect("mem://", crate::EMBED_DIM).await else {
            return Err(antumbra_core::AntumbraError::other(
                "a database-level connection was made with no credentials",
            ));
        };
        if !e.to_string().contains("--db-user and --db-pass") {
            return Err(antumbra_core::AntumbraError::other(format!("{e}")));
        }
        Ok(())
    }
}
