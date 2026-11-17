//! What an admitted adapter is, wherever its file is kept.
//!
//! An expert's card carries its adapter's [`Seal`]: the sha256 of the file
//! and its size. The file is kept under a name the expert and that sha256
//! determine ([`file_name`]), on the machine that trained it and on any that
//! pulled a copy. So a copy is checked against the row before it is served,
//! and found again by name without asking anyone.

use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{AntumbraError, ExpertId, Result};

/// The key on an expert's card that holds its adapter's [`Seal`].
pub const CARD_ADAPTER: &str = "adapter";

/// An adapter file's extension.
pub const EXTENSION: &str = "safetensors";

/// The directory adapters are kept in, under a server's working directory.
pub const DIR: &str = "adapters";

/// The key, beside the seal on an expert's card, of the copal file the
/// adapter is archived in.
pub const CARD_COPAL_FILE: &str = "copal_file";

/// The copal file `card`'s adapter is archived in, when it was archived.
pub fn archived_in(card: &serde_json::Value) -> Option<&str> {
    card.get(CARD_ADAPTER)?.get(CARD_COPAL_FILE)?.as_str()
}

/// How many hex digits a sha256 is written in.
const SHA256_HEX: usize = 64;

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

    /// The seal on `card`. `None` when the card has none, as an expert
    /// admitted before seals were kept, or when what it has is not a sha256
    /// and a size: the sha256 becomes part of a file name, so nothing else is
    /// let through.
    pub fn of_card(card: &serde_json::Value) -> Option<Self> {
        let held = card.get(CARD_ADAPTER)?;
        let sha256 = held.get("sha256")?.as_str()?;
        let bytes = held.get("bytes")?.as_u64()?;
        is_sha256(sha256).then(|| Self {
            sha256: sha256.to_string(),
            bytes,
        })
    }

    /// The seal of `bytes`.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self {
            sha256: hex(&Sha256::digest(bytes)),
            bytes: bytes.len() as u64,
        }
    }

    /// Read the file at `path` through once: its seal.
    pub fn of_file(path: &Path) -> Result<Self> {
        let failed = |what: &str, e: std::io::Error| {
            AntumbraError::other(format!(
                "could not {what} the adapter {}: {e}",
                path.display()
            ))
        };
        let mut file = std::fs::File::open(path).map_err(|e| failed("open", e))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        let mut bytes = 0u64;
        loop {
            let n = file.read(&mut buf).map_err(|e| failed("read", e))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            bytes += n as u64;
        }
        Ok(Self {
            sha256: hex(&hasher.finalize()),
            bytes,
        })
    }
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn is_sha256(text: &str) -> bool {
    text.len() == SHA256_HEX
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// An expert's id as the stem of its adapter's file name: the id with the
/// characters a path would read as separators replaced, as a training names
/// the adapter it saves.
pub fn stem(expert: &ExpertId) -> String {
    expert.as_str().replace([':', '/', '\\'], "_")
}

/// The name `expert`'s adapter is kept under when its file has `sha256`.
pub fn file_name(expert: &ExpertId, sha256: &str) -> String {
    format!("{}.{sha256}.{EXTENSION}", stem(expert))
}

/// The files in `dir` that are adapters of `expert` under another sha256
/// than `keep`: the copies an expert retrained since has left behind. A
/// directory that cannot be read has none.
pub fn others(dir: &Path, expert: &ExpertId, keep: &str) -> Vec<PathBuf> {
    let prefix = format!("{}.", stem(expert));
    let suffix = format!(".{EXTENSION}");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.strip_prefix(&prefix))
                .and_then(|rest| rest.strip_suffix(&suffix))
                .is_some_and(|sha256| is_sha256(sha256) && sha256 != keep)
        })
        .map(|entry| entry.path())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sha256 of the three bytes `abc`, the standard's first test vector.
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    /// A directory of this test's own, removed when it ends.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> std::io::Result<Self> {
            let dir = std::env::temp_dir().join(format!(
                "antumbra-core-adapter-{name}-{}",
                std::process::id()
            ));
            if dir.exists() {
                std::fs::remove_dir_all(&dir)?;
            }
            std::fs::create_dir_all(&dir)?;
            Ok(Self(dir))
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // Best effort: a leftover scratch directory fails no test.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_file_and_its_bytes_have_one_seal() -> Result<()> {
        let scratch = Scratch::new("seal").map_err(|e| AntumbraError::other(e.to_string()))?;
        let path = scratch.0.join("a.safetensors");
        std::fs::write(&path, b"abc").map_err(|e| AntumbraError::other(e.to_string()))?;

        let seal = Seal::of_file(&path)?;

        assert_eq!(
            seal,
            Seal {
                sha256: ABC.into(),
                bytes: 3
            }
        );
        assert_eq!(seal, Seal::of_bytes(b"abc"));
        Ok(())
    }

    #[test]
    fn a_seal_is_read_back_from_the_card_it_was_put_on() {
        let seal = Seal::of_bytes(b"abc");
        let card = serde_json::json!({ CARD_ADAPTER: seal.to_card(), "scope": "everywhere" });
        assert_eq!(Seal::of_card(&card), Some(seal));
    }

    #[test]
    fn a_card_names_the_archived_copy_beside_its_seal() {
        let seal = Seal::of_bytes(b"abc");
        let mut held = seal.to_card();
        assert_eq!(
            archived_in(&serde_json::json!({ CARD_ADAPTER: held.clone() })),
            None
        );
        held[CARD_COPAL_FILE] = serde_json::json!("01file");
        let card = serde_json::json!({ CARD_ADAPTER: held });
        assert_eq!(archived_in(&card), Some("01file"));
        // And the seal reads as it did: the copy is beside it, not in it.
        assert_eq!(Seal::of_card(&card), Some(seal));
        assert_eq!(archived_in(&serde_json::json!({})), None);
    }

    #[test]
    fn a_card_without_a_sha256_and_a_size_has_no_seal() {
        let of =
            |held: serde_json::Value| Seal::of_card(&serde_json::json!({ CARD_ADAPTER: held }));
        assert_eq!(Seal::of_card(&serde_json::json!({})), None);
        assert_eq!(of(serde_json::json!({ "sha256": ABC })), None);
        assert_eq!(of(serde_json::json!({ "bytes": 3 })), None);
        assert_eq!(of(serde_json::json!({ "sha256": ABC, "bytes": -3 })), None);
        // What is not a sha256 would be a file name of someone's choosing.
        for not_a_digest in ["", "aa", "../../elsewhere", &ABC.to_uppercase(), &ABC[1..]] {
            assert_eq!(
                of(serde_json::json!({ "sha256": not_a_digest, "bytes": 3 })),
                None,
                "{not_a_digest:?}"
            );
        }
    }

    #[test]
    fn an_adapter_is_named_for_its_expert_and_its_content() {
        let expert = ExpertId::new("expert:ws:a:user:b:behavior:github.com/owner/repo");
        assert_eq!(
            stem(&expert),
            "expert_ws_a_user_b_behavior_github.com_owner_repo"
        );
        assert_eq!(
            file_name(&expert, ABC),
            format!("expert_ws_a_user_b_behavior_github.com_owner_repo.{ABC}.safetensors")
        );
    }

    #[test]
    fn the_copies_left_behind_are_an_expert_s_own_under_another_sha256() -> std::io::Result<()> {
        let scratch = Scratch::new("others")?;
        let expert = ExpertId::new("expert:a");
        let other = Seal::of_bytes(b"other").sha256;
        let kept = scratch.0.join(file_name(&expert, ABC));
        let left = scratch.0.join(file_name(&expert, &other));
        // Another expert whose id begins as this one's does, a file with no
        // sha256 in its name, and a file that is not an adapter.
        let neighbor = scratch
            .0
            .join(file_name(&ExpertId::new("expert:a.b"), &other));
        let plain = scratch.0.join("expert_a.safetensors");
        let notes = scratch.0.join(format!("expert_a.{other}.txt"));
        for path in [&kept, &left, &neighbor, &plain, &notes] {
            std::fs::write(path, b"x")?;
        }

        assert_eq!(others(&scratch.0, &expert, ABC), vec![left]);
        assert!(others(&scratch.0.join("absent"), &expert, ABC).is_empty());
        Ok(())
    }
}
