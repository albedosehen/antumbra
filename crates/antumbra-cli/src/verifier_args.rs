//! `antumbra verifier`: the verifier namespace and its trust protocol
//! (ADR-0022 S-4).

use clap::Subcommand;

#[derive(Subcommand)]
pub enum VerifierAction {
    /// Add a verifier to the namespace and print its address. A synthesized
    /// one starts with no power: it grants reward only once `measure` has
    /// found it sound. Proposing the same content again returns the verifier
    /// already there.
    Propose {
        /// The spec `CommandVerifier` runs: JSON, or @path to a JSON file.
        #[arg(long)]
        spec: String,
        /// The region of tasks it is measured in and may grant reward in.
        #[arg(long)]
        domain: String,
        /// The one task it checks; every task in its domain when omitted.
        #[arg(long)]
        task: Option<String>,
        /// What decides it: reducible (something frozen) or partial (a
        /// property or metamorphic relation). Derived oracles are refused.
        #[arg(long, default_value = "reducible")]
        tier: String,
        /// Register it as authored: the ground, trusted by authorship.
        #[arg(long)]
        authored: bool,
        /// Who or what proposed it, for the audit trail.
        #[arg(long)]
        by: Option<String>,
    },
    /// Build the cases a verifier is measured on from a corpus. Each task's
    /// reference solution is known-good. Each completion given for it is
    /// labeled by the task's own authored verifier, which is registered as
    /// authored and named as the label's anchor. Every completion for an
    /// impossible task is labeled impossible without being run.
    Cases {
        /// The corpus: a JSON array of tasks with `id`, `verify`, and
        /// optionally `completion` (the reference), `skill` and `impossible`.
        #[arg(long)]
        corpus: String,
        /// Only this task.
        #[arg(long)]
        task: Option<String>,
        /// Only tasks of this skill.
        #[arg(long)]
        skill: Option<String>,
        /// A JSON array of `{task, completion}` to label.
        #[arg(long)]
        completions: Option<String>,
        /// Where to write the cases; standard output when omitted.
        #[arg(long)]
        out: Option<String>,
    },
    /// Measure a verifier on labeled cases and record the measurement. A
    /// proposed verifier found sound is trusted; one found flaky or taking a
    /// shortcut is revoked; a trusted one found unsound is quarantined.
    Measure {
        /// The verifier: its address, or a unique prefix of it.
        verifier: String,
        /// A JSON array of cases, as `cases` writes them.
        #[arg(long)]
        cases: String,
        /// Runs of every case, for the determinism gate.
        #[arg(long, default_value_t = 3)]
        repeats: u32,
        /// Of the one-sided bound on the false-positive rate.
        #[arg(long, default_value_t = 0.95)]
        confidence: f64,
        /// The largest false-positive bound that is trusted.
        #[arg(long, default_value_t = 0.10)]
        max_false_positive: f64,
        /// The share of known-good cases it must accept.
        #[arg(long, default_value_t = 0.5)]
        min_accepted: f64,
        /// How long a sound measurement keeps it trusted.
        #[arg(long, default_value_t = 7)]
        ttl_days: i64,
    },
    /// The decisive test: every trusted synthesized verifier in a domain runs
    /// on the known-bad and impossible cases it checks, and one pass
    /// quarantines it.
    Challenge {
        #[arg(long)]
        domain: String,
        /// A JSON array of cases, as `cases` writes them.
        #[arg(long)]
        cases: String,
    },
    /// Every verifier, with its state and whether it grants reward now.
    List {
        #[arg(long)]
        domain: Option<String>,
    },
    /// One verifier: its record, its moves, and its measurements.
    Show { verifier: String },
    /// Stop a trusted verifier granting reward, keeping its rows.
    Quarantine {
        verifier: String,
        #[arg(long)]
        note: Option<String>,
    },
    /// Take a verifier out of use for good, keeping its rows.
    Revoke {
        verifier: String,
        #[arg(long)]
        note: Option<String>,
    },
}
