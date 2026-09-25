//! What `antumbra train` is given: the options of the generational loop's
//! command, in one place for the parser and the command both.

use clap::Args;

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to the JSON corpus of verifiable tasks ({id,prompt,verify}).
    #[arg(long)]
    pub corpus: String,
    #[arg(long, default_value_t = 1)]
    pub generations: u32,
    #[arg(long, default_value = "run:train")]
    pub run: String,
    /// Completions sampled per task per round (RAFT's K).
    #[arg(long, default_value_t = 8)]
    pub samples: usize,
    /// Rounds per shadow.
    #[arg(long, default_value_t = 4)]
    pub rounds: usize,
    /// Max tokens generated per completion.
    #[arg(long, default_value_t = 256)]
    pub max_new_tokens: usize,
    /// Algorithm: `raft` (reward-ranked SFT) or `grpo`.
    #[arg(long, default_value = "raft")]
    pub algo: String,
    /// A critic adapter (`antumbra critic train`) to shape GRPO's advantages
    /// inside the verifier's parts (ADR-0022 S-2). GRPO only; fitness still
    /// reads the verifier alone.
    #[arg(long)]
    pub critic: Option<String>,
    /// How far the critic may shape, before the clamp that keeps every pass
    /// above every fail.
    #[arg(long, default_value_t = 0.5)]
    pub critic_weight: f32,
    /// Quantize the frozen base to 4-bit Q4_K (QLoRA).
    #[arg(long)]
    pub quantize_base: bool,
    /// Warm-start the LoRA from this saved adapter (continual fine-tune)
    /// instead of fresh factors. Monolithic arm.
    #[arg(long)]
    pub parent: Option<String>,
    /// Withhold a held-out and an audit slice of the corpus from training
    /// (ADR-0022), so each generation reports the visible-minus-held-out
    /// gap. Changes what is learned: about three tasks in ten are measured
    /// and never trained on. Refused when the corpus is too small to leave
    /// any task visible, as several shipped demo corpora are.
    #[arg(long)]
    pub holdout: bool,
    /// Search the training recipe (ADR-0022 S-1): each generation trains
    /// `--cohort` shadows under recipes the search proposes, each from the
    /// base, and carries the best forward. Learning rate and batch size
    /// are searched, and the KL weight too under `--algo grpo`. Multiplies
    /// each generation's training time by the cohort size.
    #[arg(long)]
    pub search: bool,
    /// Shadows per searched generation.
    #[arg(long, default_value_t = 4)]
    pub cohort: usize,
    /// Members of the cohort in the slow cohort, whose recipes are held
    /// for `--slow-interval` generations and cannot be displaced by what
    /// the fast members score (ADR-0022 S-1). Defaults to a third of the
    /// cohort, rounded down; 0 runs the fast cohort alone.
    #[arg(long)]
    pub slow: Option<usize>,
    /// Generations a slow member keeps its recipe.
    #[arg(long, default_value_t = 3)]
    pub slow_interval: u32,
    /// Generations over which the fast members' ready interval lengthens
    /// from one toward the slow interval. Defaults to the run's
    /// `--generations`; 0 keeps it at one.
    #[arg(long)]
    pub anneal: Option<u32>,
    /// Measure every shared expert's leave-one-out contribution
    /// (ADR-0022 S-5) in every N-th generation: each is masked, the live
    /// tasks are routed again, and both ways are scored under the same
    /// seeds. About two evaluations of the live tasks each time. 0 (the
    /// default) leaves it unmeasured.
    #[arg(long, default_value_t = 0)]
    pub contribution_every: u32,
    /// Live tasks a contribution measurement samples: 32, or 64 with --grow,
    /// whose census needs enough of every region. Set it to compare runs on
    /// the same sample.
    #[arg(long)]
    pub contribution_tasks: Option<usize>,
    /// Gate admission (ADR-0022 S-5): a graduate whose capability vector
    /// is at least this similar (cosine) to an active shared expert's is a
    /// twin. It joins only if it beats that expert head to head on the
    /// live tasks, and then replaces it, which is archived; otherwise it
    /// is not admitted. Above 1 admits every graduate.
    #[arg(long, default_value_t = 0.95)]
    pub duplicate_above: f32,
    /// Retirement as the loop's job (ADR-0022 S-5): demote an expert to
    /// dormant once its contribution on the tasks routed to it has been
    /// at or below nothing in this many consecutive measurements, each on
    /// at least two tasks. It reads the contribution stream, so it acts
    /// only with --contribution-every. 0 leaves every move to a person.
    #[arg(long, default_value_t = 3)]
    pub retire_after: u32,
    /// Merge sibling experts (ADR-0022 S-5): at each generation boundary,
    /// the most similar pair of active shared experts is merged at the
    /// population's rank when their adapters share enough of their
    /// subspace (--merge-retained) and the merge scores on the live tasks
    /// at least as well as the better of them. Both are then archived,
    /// so reviving them undoes it.
    #[arg(long)]
    pub merge: bool,
    /// Share of the two adapters' averaged delta the population's rank
    /// must keep for them to count as siblings.
    #[arg(long, default_value_t = 0.9)]
    pub merge_retained: f32,
    /// The grow step (ADR-0022 S-3): each generation learns from the
    /// region (skill) the latest census makes most learnable, plus a
    /// quarter sampled from the whole visible slice. It reads the census
    /// the contribution measurement takes, which it turns on every
    /// generation over 64 live tasks unless --contribution-every is set.
    #[arg(long)]
    pub grow: bool,
    /// With --grow, how a region is chosen among those that pass the gate:
    /// credit (the record's objective: expected realized improvement,
    /// learnability its prior), learnability (the most learnable,
    /// regardless of credit) or uniform (at random, the baseline).
    #[arg(long, default_value = "credit")]
    pub grow_by: String,
    /// With --grow, where the region's shadow starts: incumbent (the adapter
    /// of the expert that serves the region now, so it refines what is there)
    /// or base (fresh factors, as every shadow did before). Incumbent falls
    /// back to fresh factors when no expert serves the region.
    #[arg(long, default_value = "incumbent")]
    pub grow_from: String,
    /// Judge graduation on this many re-measurements of the carried-forward
    /// shadow, each under a fresh seed, on the held-out slice under
    /// `--holdout` (ADR-0022 S-1). Defaults to 3 with `--search` and to off
    /// otherwise; 0 turns it off.
    #[arg(long)]
    pub remeasure: Option<u32>,
}
