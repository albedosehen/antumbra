//! Secrets from files: the shape a Docker secret, a Kubernetes secret, or a
//! Key Vault CSI mount takes. A secret given as a file appears in neither the
//! process arguments (`docker inspect .Args`, `ps`) nor the environment
//! (`docker inspect .Config.Env`), which is where an inline flag or an env var
//! lands for anyone who can inspect the container. Every binary that takes a
//! secret offers the file beside the flag, and resolves it here.

use std::path::Path;

use crate::{AntumbraError, Result};

/// Resolve one secret from its inline form or its file, whichever the operator
/// gave. The file's content is trimmed of surrounding whitespace (a trailing
/// newline is how every secret store writes a file), and an empty file is
/// refused rather than becoming an empty credential. clap already refuses both
/// forms together; this refuses again so the rule holds for every caller.
pub fn resolve(inline: Option<String>, file: Option<&Path>, what: &str) -> Result<Option<String>> {
    match (inline, file) {
        (Some(_), Some(path)) => Err(AntumbraError::other(format!(
            "{what}: given both inline and as the file {}; pass one",
            path.display()
        ))),
        (None, Some(path)) => {
            let raw = std::fs::read_to_string(path).map_err(|e| {
                AntumbraError::other(format!(
                    "cannot read the {what} file {}: {e}",
                    path.display()
                ))
            })?;
            let value = raw.trim();
            if value.is_empty() {
                return Err(AntumbraError::other(format!(
                    "the {what} file {} is empty",
                    path.display()
                )));
            }
            Ok(Some(value.to_string()))
        }
        (inline, None) => Ok(inline),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    static SEQ: AtomicU32 = AtomicU32::new(0);

    /// A file of this test's own, removed when the guard drops.
    struct SecretFile(std::path::PathBuf);

    impl SecretFile {
        fn holding(content: &str) -> Result<Self> {
            let path = std::env::temp_dir().join(format!(
                "antumbra-secret-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&path, content).map_err(|e| AntumbraError::other(e.to_string()))?;
            Ok(Self(path))
        }
    }

    impl Drop for SecretFile {
        fn drop(&mut self) {
            // A file left behind is a leak in the temp directory, not a test
            // failure, so the outcome is not read.
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn a_file_is_read_and_trimmed_and_inline_passes_through() -> Result<()> {
        let file = SecretFile::holding("s3cret\n")?;
        assert_eq!(
            resolve(None, Some(&file.0), "JWT secret")?.as_deref(),
            Some("s3cret")
        );
        assert_eq!(
            resolve(Some("inline".into()), None, "JWT secret")?.as_deref(),
            Some("inline")
        );
        assert_eq!(resolve(None, None, "JWT secret")?, None);
        Ok(())
    }

    #[test]
    fn an_empty_or_missing_file_and_both_forms_are_refused() -> Result<()> {
        let empty = SecretFile::holding("  \n")?;
        let Err(e) = resolve(None, Some(&empty.0), "database password") else {
            return Err(AntumbraError::other("an empty file was accepted"));
        };
        assert!(e.to_string().contains("is empty"), "{e}");

        let missing = std::env::temp_dir().join("antumbra-secret-that-does-not-exist");
        let Err(e) = resolve(None, Some(&missing), "database password") else {
            return Err(AntumbraError::other("a missing file was accepted"));
        };
        assert!(e.to_string().contains("cannot read"), "{e}");

        let present = SecretFile::holding("x")?;
        let Err(e) = resolve(Some("y".into()), Some(&present.0), "database password") else {
            return Err(AntumbraError::other("both forms were accepted"));
        };
        assert!(e.to_string().contains("pass one"), "{e}");
        Ok(())
    }
}
