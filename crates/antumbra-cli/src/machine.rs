//! This machine, as every command reads it.

use std::path::PathBuf;

/// The user's home directory: `USERPROFILE` on Windows, else `HOME`; `None`
/// when neither is set, which each command says in its own words.
pub fn home() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}
