//! The capture-SFT block: the training options `teach`, `memory-import`,
//! `consolidate` and `metabolize` share, in one place so each command's
//! `--help` says the same thing about them. Each command keeps its own
//! `--run`, whose default names the command.

use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct CaptureSftArgs {
    /// SFT epochs over the verified corrections. Overriding a strong base
    /// prior from one example needs many (try 40-80).
    #[arg(long, default_value_t = 40)]
    pub rounds: usize,
    /// Completions sampled to measure whether the correction internalized.
    #[arg(long, default_value_t = 8)]
    pub samples: usize,
    /// Max tokens generated per sampled completion.
    #[arg(long, default_value_t = 32)]
    pub max_new_tokens: usize,
    /// LoRA learning rate. Capturing a correction wants it higher than RAFT.
    #[arg(long, default_value_t = 3e-4)]
    pub lr: f64,
}
