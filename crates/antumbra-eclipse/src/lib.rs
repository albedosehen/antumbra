//! Eclipse: the standing instruments of the self-improving loop.
//!
//! The loop opens every recursive seam except one and holds a single thing
//! fixed while doing it:
//!
//! > **The anchor invariant.** Reward may never originate from a signal that
//! > has never been checked against something outside the loop.
//!
//! The order is just as fixed: the standing instruments land first, before
//! any seam, because they are how a seam is judged. This crate is those
//! instruments and, for now, only those.
//!
//! - [`slice`] is the partition everything else is built on: which tasks the
//!   loop may see, which are frozen away from it, which are touched by no
//!   decision at all, and which cannot be passed.
//! - [`instrument`] measures one generation: the visible-minus-held-out gap,
//!   banded by task size because the gap grows with size and an average over
//!   sizes hides exactly that, and the impossible-task set, where a single
//!   pass fails the generation whole.
//! - [`trend`] reads the audit slice across generations, and says whether a
//!   climbing search score is carrying any competence with it.
//! - [`isomorphic`] re-verifies a graduation candidate under a transform that
//!   changes a task's surface and not its meaning.
//!
//! **Nothing here grants reward, and nothing here is a training target.** That
//! is a property of the crate, not a convention: it depends on no reward type
//! and returns no reward, so there is no signature through which a number
//! measured here could reach the optimizer. Trace monitors need this in
//! particular, because optimizing against a detector has been shown to
//! produce obfuscated hacking rather than less hacking, and the same holds
//! for every instrument in this crate. A measurement that becomes something to
//! improve stops measuring.
//!
//! The instruments are worth having even with every seam shut: they measure
//! whether today's loop is sound, which is a question that predates the seams.

pub mod instrument;
pub mod isomorphic;
pub mod slice;
pub mod trend;

pub use instrument::{GenerationReport, Outcome, Rate, SizeBand, SizeBands};
pub use isomorphic::{Pair, Reverification};
pub use slice::{Holdout, Partition, Slice};
pub use trend::{Point, Trend, Watch};
