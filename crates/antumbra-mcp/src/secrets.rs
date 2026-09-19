//! Secrets from files: the shape a Docker secret, a Kubernetes secret, or a
//! Key Vault CSI mount takes. A secret given as a file appears in neither the
//! process arguments (`docker inspect .Args`, `ps`) nor the environment
//! (`docker inspect .Config.Env`), which is where an inline flag or an env var
//! lands for anyone who can inspect the container.

use std::path::Path;

use anyhow::{bail, Context, Result};

/// Resolve one secret from its inline form or its file, whichever the operator
/// gave. The file's content is trimmed of surrounding whitespace (a trailing
/// newline is how every secret store writes a file), and an empty file is
/// refused rather than becoming an empty credential. clap already refuses both
/// forms together; this refuses again so the rule holds for every caller.
pub fn resolve(inline: Option<String>, file: Option<&Path>, what: &str) -> Result<Option<String>> {
    match (inline, file) {
        (Some(_), Some(path)) => bail!(
            "{what}: given both inline and as the file {}; pass one",
            path.display()
        ),
        (None, Some(path)) => {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("cannot read the {what} file {}", path.display()))?;
            let value = raw.trim();
            if value.is_empty() {
                bail!("the {what} file {} is empty", path.display());
            }
            Ok(Some(value.to_string()))
        }
        (inline, None) => Ok(inline),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static SEQ: AtomicU32 = AtomicU32::new(0);

    fn secret_file(content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "antumbra-secret-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn a_file_is_read_and_trimmed_and_inline_passes_through() {
        let path = secret_file("s3cret\n");
        assert_eq!(
            resolve(None, Some(&path), "JWT secret").unwrap().as_deref(),
            Some("s3cret")
        );
        assert_eq!(
            resolve(Some("inline".into()), None, "JWT secret")
                .unwrap()
                .as_deref(),
            Some("inline")
        );
        assert_eq!(resolve(None, None, "JWT secret").unwrap(), None);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn an_empty_or_missing_file_and_both_forms_are_refused() {
        let empty = secret_file("  \n");
        let err = resolve(None, Some(&empty), "database password").unwrap_err();
        assert!(err.to_string().contains("is empty"), "{err}");
        std::fs::remove_file(&empty).ok();

        let missing = std::env::temp_dir().join("antumbra-secret-that-does-not-exist");
        let err = resolve(None, Some(&missing), "database password").unwrap_err();
        assert!(err.to_string().contains("cannot read"), "{err}");

        let present = secret_file("x");
        let err = resolve(Some("y".into()), Some(&present), "database password").unwrap_err();
        assert!(err.to_string().contains("pass one"), "{err}");
        std::fs::remove_file(present).ok();
    }
}
