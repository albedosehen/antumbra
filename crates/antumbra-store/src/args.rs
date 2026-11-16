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
    /// Read the password from this file instead (a Docker secret, a Kubernetes
    /// secret, a Key Vault mount): it then appears in neither the process
    /// arguments nor the environment.
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_DB_PASS_FILE",
        conflicts_with = "db_pass"
    )]
    pub db_pass_file: Option<std::path::PathBuf>,
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
    /// The password, from the flag or its file, whichever was given.
    pub fn password(&self) -> Result<Option<String>> {
        antumbra_core::secrets::resolve(
            self.db_pass.clone(),
            self.db_pass_file.as_deref(),
            "database password",
        )
    }

    /// These flags with the password resolved into `db_pass` and the file
    /// forgotten, for a caller that reads the credentials themselves (the MCP
    /// server chooses its serving mode by whether there are any).
    pub fn resolved(&self) -> Result<Self> {
        Ok(Self {
            db_pass: self.password()?,
            db_pass_file: None,
            ..self.clone()
        })
    }

    /// Connect to `url` with these credentials and apply the schema: as root
    /// when both are given (none for an embedded or unauthenticated store), or
    /// as the database user when `--db-auth database`, which needs both.
    pub async fn connect(&self, url: &str, embed_dim: usize) -> Result<Store> {
        let builder = ConnectionConfig::builder()
            .url(url)
            .namespace("antumbra")
            .database("main");
        let password = self.password()?;
        match self.db_auth {
            DbAuth::Database => {
                let (Some(user), Some(pass)) = (self.db_user.as_deref(), password.as_deref())
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
                let builder = match (self.db_user.as_deref(), password.as_deref()) {
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

    /// The password's file is the flag's twin: read and trimmed, and never
    /// given beside the flag.
    #[test]
    fn the_password_may_come_from_a_file_but_not_from_both() -> Result<()> {
        let path = std::env::temp_dir().join(format!("antumbra-db-pass-{}", std::process::id()));
        std::fs::write(&path, "pw-from-file\n")
            .map_err(|e| antumbra_core::AntumbraError::other(e.to_string()))?;
        let probe = Probe::try_parse_from([
            "probe",
            "--db-user",
            "svc",
            "--db-pass-file",
            &path.to_string_lossy(),
        ])
        .map_err(|e| antumbra_core::AntumbraError::other(format!("{e}")))?;
        let resolved = probe.db.resolved()?;
        // The file is read once the flags are parsed; the guard below removes
        // it whatever the assertions say.
        let _ = std::fs::remove_file(&path);
        if resolved.db_pass.as_deref() != Some("pw-from-file") || resolved.db_pass_file.is_some() {
            return Err(antumbra_core::AntumbraError::other(format!(
                "the password was not resolved from its file: {resolved:?}"
            )));
        }
        if Probe::try_parse_from(["probe", "--db-pass", "x", "--db-pass-file", "f"]).is_ok() {
            return Err(antumbra_core::AntumbraError::other(
                "the flag and its file were accepted together",
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
