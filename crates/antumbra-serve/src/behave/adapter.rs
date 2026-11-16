//! Where a standing expert's adapter rests on disk.
//!
//! A training writes a candidate, in a directory of its own. Admission
//! promotes the candidate to a file named for the expert and the sha256 of
//! its bytes, and nothing writes to that file again. So the file an expert's
//! row names is the one it was admitted with for as long as the row names
//! it, whatever is trained after.

use std::io::Read;
use std::path::Path;

use antumbra_core::{AntumbraError, Result};
use sha2::{Digest, Sha256};

/// The key on an expert's card that holds its adapter's [`Seal`].
pub const CARD_ADAPTER: &str = "adapter";

/// An adapter file's extension.
const EXTENSION: &str = "safetensors";

/// What an admitted adapter is, apart from where it is kept: the sha256 of
/// its file and the file's size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seal {
    /// Lowercase hex.
    pub sha256: String,
    pub bytes: u64,
}

impl Seal {
    /// As an expert's card holds it, under [`CARD_ADAPTER`].
    pub fn to_card(&self) -> serde_json::Value {
        serde_json::json!({ "sha256": self.sha256, "bytes": self.bytes })
    }
}

/// The directory a training of a standing expert saves into: under
/// `adapter_dir`, and apart from every admitted adapter in it.
pub fn candidates(adapter_dir: &str) -> String {
    format!("{adapter_dir}/candidates")
}

/// Where a training's adapter rests once its verdict is in.
///
/// Admitted, the `candidate` is promoted into `adapter_dir` under a name its
/// content determines, and that path and its [`Seal`] come back. The same
/// bytes admitted twice are one file: the one already there is left as it is.
///
/// Refused, the candidate stays where the training wrote it, for the next
/// training of the expert to write over, and no admitted adapter is touched.
pub fn place(candidate: &str, adapter_dir: &str, admitted: bool) -> Result<(String, Option<Seal>)> {
    if !admitted {
        return Ok((candidate.to_string(), None));
    }
    let from = Path::new(candidate);
    let stem = from.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
        AntumbraError::other(format!(
            "the adapter candidate {candidate} has no file name"
        ))
    })?;
    let seal = seal(from)?;
    let uri = format!("{adapter_dir}/{stem}.{}.{EXTENSION}", seal.sha256);
    let to = Path::new(&uri);
    if to.exists() {
        std::fs::remove_file(from).map_err(|e| failed("remove", candidate, &e))?;
    } else {
        std::fs::rename(from, to).map_err(|e| failed("promote", candidate, &e))?;
    }
    Ok((uri, Some(seal)))
}

/// Remove `superseded`, the adapter of the expert that the one served from
/// `kept` replaced. Whether a file went.
///
/// The path comes from a stored row, so only an adapter file beside `kept`
/// is removed: anything else, and `kept` itself, is left where it is.
pub fn discard(superseded: &str, kept: &str) -> Result<bool> {
    let (old, new) = (Path::new(superseded), Path::new(kept));
    let an_adapter = old.extension().and_then(|e| e.to_str()) == Some(EXTENSION);
    if old == new || old.parent() != new.parent() || !an_adapter {
        return Ok(false);
    }
    match std::fs::remove_file(old) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(failed("remove", superseded, &e)),
    }
}

/// Read `path` through once: its sha256 and its length.
fn seal(path: &Path) -> Result<Seal> {
    let shown = path.display().to_string();
    let mut file = std::fs::File::open(path).map_err(|e| failed("open", &shown, &e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut bytes = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| failed("read", &shown, &e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        bytes += n as u64;
    }
    let sha256 = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok(Seal { sha256, bytes })
}

fn failed(what: &str, path: &str, e: &std::io::Error) -> AntumbraError {
    AntumbraError::other(format!("could not {what} the adapter {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of this test's own, removed when it ends.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(name: &str) -> anyhow::Result<Self> {
            let dir = std::env::temp_dir()
                .join(format!("antumbra-adapter-{name}-{}", std::process::id()));
            if dir.exists() {
                std::fs::remove_dir_all(&dir)?;
            }
            std::fs::create_dir_all(&dir)?;
            Ok(Self(dir))
        }

        /// The adapter directory, as a training's configuration names it.
        fn adapters(&self) -> String {
            self.0.join("adapters").to_string_lossy().replace('\\', "/")
        }

        /// Write `bytes` as the candidate a training of `expert` leaves.
        fn train(&self, expert: &str, bytes: &[u8]) -> anyhow::Result<String> {
            let dir = candidates(&self.adapters());
            std::fs::create_dir_all(&dir)?;
            let path = format!("{dir}/{expert}.{EXTENSION}");
            std::fs::write(&path, bytes)?;
            Ok(path)
        }

        /// Train `expert` to `bytes` and admit it: where its adapter rests.
        fn admit(&self, expert: &str, bytes: &[u8]) -> anyhow::Result<String> {
            let candidate = self.train(expert, bytes)?;
            Ok(place(&candidate, &self.adapters(), true)?.0)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // Best effort: a leftover scratch directory fails no test.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// sha256 of the three bytes `abc`, the standard's first test vector.
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn an_admitted_adapter_is_named_by_its_content() -> anyhow::Result<()> {
        let scratch = Scratch::new("named")?;
        let candidate = scratch.train("expert_a", b"abc")?;

        let (uri, seal) = place(&candidate, &scratch.adapters(), true)?;

        assert_eq!(
            uri,
            format!("{}/expert_a.{ABC}.safetensors", scratch.adapters())
        );
        assert_eq!(
            seal,
            Some(Seal {
                sha256: ABC.into(),
                bytes: 3
            })
        );
        assert_eq!(std::fs::read(&uri)?, b"abc");
        assert!(
            !Path::new(&candidate).exists(),
            "the candidate was promoted"
        );
        Ok(())
    }

    /// The regression: a retraining that was refused wrote over the file the
    /// admitted expert's row named.
    #[test]
    fn a_refused_training_leaves_the_admitted_adapter_as_it_was() -> anyhow::Result<()> {
        let scratch = Scratch::new("refused")?;
        let admitted = scratch.admit("expert_a", b"admitted weights")?;

        let again = scratch.train("expert_a", b"refused weights")?;
        let (rests, seal) = place(&again, &scratch.adapters(), false)?;

        assert_eq!(std::fs::read(&admitted)?, b"admitted weights");
        assert_eq!(rests, again, "a refused adapter stays a candidate");
        assert_eq!(seal, None);
        assert_ne!(rests, admitted);
        Ok(())
    }

    #[test]
    fn an_admitted_retraining_rests_beside_the_adapter_it_replaces() -> anyhow::Result<()> {
        let scratch = Scratch::new("retrained")?;
        let before = scratch.admit("expert_a", b"first weights")?;
        let after = scratch.admit("expert_a", b"second weights")?;

        assert_ne!(before, after);
        assert_eq!(std::fs::read(&before)?, b"first weights");
        assert_eq!(std::fs::read(&after)?, b"second weights");
        Ok(())
    }

    #[test]
    fn the_same_bytes_admitted_twice_are_one_file() -> anyhow::Result<()> {
        let scratch = Scratch::new("twice")?;
        let before = scratch.admit("expert_a", b"abc")?;

        let again = scratch.train("expert_a", b"abc")?;
        let (after, _) = place(&again, &scratch.adapters(), true)?;

        assert_eq!(before, after);
        assert_eq!(std::fs::read(&after)?, b"abc");
        assert!(!Path::new(&again).exists(), "the second candidate is gone");
        Ok(())
    }

    #[test]
    fn two_experts_with_the_same_bytes_keep_their_own_files() -> anyhow::Result<()> {
        let scratch = Scratch::new("two")?;
        let a = scratch.admit("expert_a", b"abc")?;
        let b = scratch.admit("expert_b", b"abc")?;

        assert_ne!(a, b);
        assert!(discard(&a, &b)?, "one goes");
        assert_eq!(std::fs::read(&b)?, b"abc", "and the other stays");
        Ok(())
    }

    #[test]
    fn a_superseded_adapter_is_removed_and_nothing_else() -> anyhow::Result<()> {
        let scratch = Scratch::new("discard")?;
        let old = scratch.admit("expert_a", b"old")?;
        let new = scratch.admit("expert_a", b"new")?;
        let elsewhere = scratch.0.join("notes.safetensors");
        std::fs::write(&elsewhere, b"an adapter of another directory")?;
        let beside = format!("{}/notes.txt", scratch.adapters());
        std::fs::write(&beside, b"not an adapter")?;

        assert!(!discard(&new, &new)?, "the adapter kept is never removed");
        assert!(!discard(&elsewhere.to_string_lossy(), &new)?);
        assert!(!discard(&beside, &new)?);
        assert!(discard(&old, &new)?);
        assert!(!discard(&old, &new)?, "already gone is not an error");

        assert!(!Path::new(&old).exists());
        assert!(Path::new(&new).exists());
        assert!(elsewhere.exists());
        assert!(Path::new(&beside).exists());
        Ok(())
    }

    #[test]
    fn a_candidate_that_is_not_there_is_an_error() -> anyhow::Result<()> {
        let scratch = Scratch::new("missing")?;
        let missing = format!("{}/expert_a.safetensors", candidates(&scratch.adapters()));

        let err = match place(&missing, &scratch.adapters(), true) {
            Ok(placed) => anyhow::bail!("a missing candidate was placed: {placed:?}"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("could not open the adapter"), "{err}");
        Ok(())
    }
}
