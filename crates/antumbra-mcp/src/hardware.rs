//! What this node is, as far as the node can tell.
//!
//! Two questions, answered separately because they fail separately: which
//! backend this build can drive, and how much video memory the device it would
//! train on has. The first is a property of the binary and is always knowable;
//! the second is a reading that may not be available, and `None` is a real
//! answer to it rather than a zero.

/// The backend this build can actually drive.
///
/// Compiled-in, not probed, and deliberately so: the role derived from this
/// decides where genesis is dispatched, and a binary built without the CUDA
/// backend cannot train on a CUDA box however many GPUs are plugged into it. A
/// node that reported the silicon rather than its own reach would collect
/// dispatches it could only escalate.
pub fn backend() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "metal") {
        "metal"
    } else {
        "cpu"
    }
}

/// Total video memory in MiB on the device training would use, or `None` when
/// the node cannot tell. Only asked on a CUDA build, where `nvidia-smi` ships
/// with the driver; a Metal node's memory is unified and is not the same
/// quantity, so it is left unanswered rather than answered wrongly.
///
/// Asked once. The answer does not change while the process runs, and the
/// standing-expert keeper consults the role when it starts.
pub fn vram_mib() -> Option<u64> {
    static VRAM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *VRAM.get_or_init(probe_vram_mib)
}

/// What this node is, from what it found. The same derivation the registry
/// records, so the role a node advertises and the role it acts on are one
/// answer rather than two that can drift.
#[cfg_attr(not(feature = "models"), allow(dead_code))]
pub fn role() -> antumbra_core::DeviceRole {
    antumbra_core::role_for(backend(), vram_mib())
}

fn probe_vram_mib() -> Option<u64> {
    if backend() != "cuda" {
        return None;
    }
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    first_reported_mib(&String::from_utf8_lossy(out.stdout.as_slice()))
}

/// The first GPU `nvidia-smi` lists, which is the one training runs on
/// (`antumbra_train::device::best_device` takes CUDA device 0). Not the largest:
/// on a mixed box the biggest card is not necessarily the one that gets the
/// work, and claiming its memory would put the node over the genesis floor on
/// the strength of a card it will not use.
fn first_reported_mib(out: &str) -> Option<u64> {
    out.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .and_then(|line| line.parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::{role_for, DeviceRole};

    #[test]
    fn the_first_card_listed_is_the_one_training_would_use() {
        // One GPU, as nvidia-smi prints it (nounits, so bare MiB).
        assert_eq!(first_reported_mib("24564\n"), Some(24_564));
        // Two, the smaller first: that is the one device 0 is, so that is the
        // figure, even though the box also holds a bigger card.
        assert_eq!(first_reported_mib("8192\n24564\n"), Some(8_192));
        assert_eq!(
            role_for("cuda", first_reported_mib("8192\n24564\n")),
            DeviceRole::Memory,
            "a big second card does not make device 0 a trainer"
        );
    }

    #[test]
    fn a_reading_that_is_not_a_reading_is_not_a_zero() {
        assert_eq!(first_reported_mib(""), None);
        assert_eq!(first_reported_mib("\n  \n"), None);
        // What a driver-less box prints on stdout before failing.
        assert_eq!(first_reported_mib("[N/A]\n"), None);
        assert_eq!(first_reported_mib("No devices were found\n"), None);
        // And an unreadable figure leaves the backend's own verdict standing,
        // rather than demoting the node on a non-answer.
        assert_eq!(
            role_for("cuda", first_reported_mib("[N/A]\n")),
            DeviceRole::Genesis
        );
    }

    /// The default build drives no accelerator, so it must not claim one: this
    /// is what keeps a CPU-only binary on a GPU box from advertising itself as
    /// somewhere to train.
    #[test]
    fn a_build_with_no_gpu_backend_says_so_and_asks_nothing_further() {
        if cfg!(any(feature = "cuda", feature = "metal")) {
            return;
        }
        assert_eq!(backend(), "cpu");
        assert_eq!(vram_mib(), None);
        assert_eq!(role_for(backend(), vram_mib()), DeviceRole::Memory);
    }
}
