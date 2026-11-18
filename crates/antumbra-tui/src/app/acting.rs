//! What an operator does from the console, and what the console says back.
//!
//! An action is staged behind a confirm ([`Pending`]), carried out through
//! the [`Backend`], and answered on screen: a [`Notice`] says what was done,
//! what was refused and why, or why nothing happened. The event stream keeps
//! the history; the notice is what the main view shows.
//!
//! A child module of `app`, so these are the same inherent methods on the same
//! `App`; they live here to keep `app.rs` to the state and its reading.

use antumbra_core::behavior;
use antumbra_core::console::Action as ConsoleAction;
use antumbra_core::{BoundaryId, ExpertId, MemoryPolicy, RunId, ShadowId};

use super::{App, Focus, LayoutMode, Mode, Page, Snapshot};
use crate::backend::Backend;
use crate::events::{Event, EventKind};
use crate::theme::Tone;
use crate::words;

/// An operator mutation awaiting a yes/no confirmation.
pub enum Pending {
    /// Mark the shadow pruned (collapse it).
    PruneShadow(ShadowId),
    /// Mark the shadow graduated (promote it into the population).
    GraduateShadow(ShadowId),
    /// Freeze (`true`) or thaw (`false`) the expert.
    FreezeExpert(ExpertId, bool),
    /// Delete the boundary from the antumbra.
    DeleteBoundary(BoundaryId),
    /// Ask the generational loop to halt gracefully at the next generation.
    HaltLoop(RunId),
    /// Withdraw a halt the loop has not reached yet.
    CancelHalt(RunId),
    /// Replace the workspace's memory knobs with these.
    SetMemoryPolicy(MemoryPolicy),
}

impl Pending {
    /// Whether going ahead takes something away: a shadow collapsed, a
    /// boundary deleted. The confirm prompt marks the key that does it.
    pub fn removes(&self) -> bool {
        matches!(self, Pending::PruneShadow(_) | Pending::DeleteBoundary(_))
    }
}

/// What the console last did, declined to do, or could not do, shown above
/// the footer until it ages out.
pub struct Notice {
    pub text: String,
    /// Done is settled, refused or failed is failed, and anything else open.
    pub tone: Tone,
    /// The animation clock (ms) when it was said.
    pub at_ms: f64,
}

/// How long a notice stays on screen.
pub const NOTICE_MS: f64 = 8000.0;

impl App {
    /// Say `text` on the main view, in `tone`. Nothing is added to the event
    /// stream: for a result the stream will record itself, from the reload
    /// that follows.
    pub fn note(&mut self, tone: Tone, text: impl Into<String>) {
        self.notice = Some(Notice {
            text: text.into(),
            tone,
            at_ms: self.clock_ms,
        });
    }

    /// Say `text` on the main view and keep it in the event stream.
    pub fn say(&mut self, tone: Tone, text: impl Into<String>) {
        let text = text.into();
        let kind = if tone == Tone::Failed {
            EventKind::Problem
        } else {
            EventKind::System
        };
        self.push_event(kind, text.clone());
        self.note(tone, text);
    }

    /// Say that something was refused or went wrong.
    pub fn problem(&mut self, text: impl Into<String>) {
        self.say(Tone::Failed, text);
    }

    /// Record an operator-initiated event in the stream, and say it on the
    /// main view.
    pub fn operator_event(&mut self, text: String) {
        self.say(Tone::Open, text);
    }

    /// The notice to show now: the last one while it is fresh, and a failed
    /// read's for as long as reads go on failing.
    pub fn current_notice(&self) -> Option<&Notice> {
        self.notice.as_ref().filter(|n| {
            self.clock_ms - n.at_ms < NOTICE_MS || (self.reads_failing && n.tone == Tone::Failed)
        })
    }

    /// Take a read of the server that has come back: apply it, or say that it
    /// failed. A run of failed reads is said once, when it starts, and its
    /// end is said when a read succeeds again.
    pub fn read_finished(&mut self, read: anyhow::Result<Option<Snapshot>>) {
        match read {
            Ok(snapshot) => {
                if self.reads_failing {
                    self.reads_failing = false;
                    let source = self.source.clone();
                    self.say(Tone::Settled, format!("reading {source} again"));
                }
                match snapshot {
                    Some(snapshot) => self.apply_snapshot(snapshot),
                    // Nothing changed since the last read.
                    None => self.since_reload_ms = 0.0,
                }
            }
            Err(e) if !self.reads_failing => {
                self.reads_failing = true;
                self.problem(format!("reload failed · {e:#}"));
            }
            Err(_) => {}
        }
    }

    /// Whether this console may act. A read-only one says how to become able.
    pub(super) fn may_act(&mut self) -> bool {
        if !self.operator {
            self.problem(words::READ_ONLY);
        }
        self.operator
    }

    /// Open the confirm prompt on `pending`.
    fn stage(&mut self, pending: Pending) {
        self.pending = Some(pending);
        self.mode = Mode::Confirm;
    }

    /// Bring `focus`'s list onto the screen: the population page, in a layout
    /// that draws that list. A command that acts on a list's selection goes
    /// here first, so what it is about to act on is in view.
    pub fn show(&mut self, focus: Focus) {
        self.page = Page::Population;
        let drawn = match self.layout {
            LayoutMode::Focused | LayoutMode::Dashboard => true,
            LayoutMode::Graph | LayoutMode::Table => focus == Focus::Experts,
        };
        if !drawn {
            self.layout = LayoutMode::Focused;
        }
        self.focus = focus;
    }

    /// Stage the action `x` asks for, which follows from the page and the
    /// list: freeze or thaw the selected expert, prune the selected shadow,
    /// delete the selected boundary, or halt the selected loop run (cancel its
    /// halt, when it has one pending). Where nothing can be done it says why.
    pub fn request_action(&mut self) {
        if !self.may_act() {
            return;
        }
        match self.page {
            Page::Loop => self.request_halt(),
            Page::Population => match self.focus {
                Focus::Experts => self.stage_freeze(),
                Focus::Shadows => self.stage_prune(),
                Focus::Boundaries => match self.selected_boundary().map(|b| b.id.clone()) {
                    Some(id) => self.stage(Pending::DeleteBoundary(id)),
                    None => self.note(Tone::Open, "no boundary is selected"),
                },
            },
            Page::Memory | Page::Evals | Page::Sovereign => {
                self.note(Tone::Open, "nothing on this page can be acted on")
            }
        }
    }

    /// Stage halting the selected run, or cancelling its halt when it has one
    /// pending.
    fn request_halt(&mut self) {
        let Some(run) = self.selected_loop_head().map(|h| h.run_id.clone()) else {
            self.note(
                Tone::Open,
                "no training loop has run, so there is none to halt",
            );
            return;
        };
        if self.halt_pending(&run) {
            self.stage(Pending::CancelHalt(run));
        } else {
            self.stage(Pending::HaltLoop(run));
        }
    }

    /// Whether the selected shadow is a standing expert's training, said on
    /// the main view when it is. Pruning and graduating steer the
    /// generational loop, which reads them back; the standing-experts keeper
    /// settles its own trainings and never does, so on one of its shadows
    /// either would only misstate what is serving.
    fn selected_is_kept(&mut self) -> bool {
        let Some(scope) = self
            .selected_shadow()
            .and_then(|s| behavior::shadow_scope(&s.id))
            .map(str::to_owned)
        else {
            return false;
        };
        self.operator_event(format!(
            "{scope}: a standing expert's training is settled by the server · freeze the expert to stop it serving"
        ));
        true
    }

    /// Stage pruning the selected shadow, unless it is already pruned.
    fn stage_prune(&mut self) {
        if self.selected_is_kept() {
            return;
        }
        match self
            .selected_shadow()
            .map(|s| (s.id.clone(), s.status.as_str() == "pruned"))
        {
            None => self.note(Tone::Open, "no candidate is selected"),
            Some((_, true)) => self.note(Tone::Open, "that candidate is already pruned"),
            Some((id, false)) => self.stage(Pending::PruneShadow(id)),
        }
    }

    /// Stage graduating the selected shadow: promote it into the population,
    /// unless it already graduated. Asked from anywhere, it brings the shadows
    /// into view first.
    pub fn request_graduate(&mut self) {
        if !self.may_act() {
            return;
        }
        self.show(Focus::Shadows);
        if self.selected_is_kept() {
            return;
        }
        match self
            .selected_shadow()
            .map(|s| (s.id.clone(), s.status.as_str() == "graduated"))
        {
            None => self.note(Tone::Open, "no candidate is selected"),
            Some((_, true)) => self.note(Tone::Open, "that candidate has already graduated"),
            Some((id, false)) => self.stage(Pending::GraduateShadow(id)),
        }
    }

    /// Stage freezing the selected expert, or thawing it if already frozen.
    /// Asked from anywhere, it brings the experts into view first.
    pub fn request_freeze(&mut self) {
        if !self.may_act() {
            return;
        }
        self.show(Focus::Experts);
        self.stage_freeze();
    }

    fn stage_freeze(&mut self) {
        match self
            .selected_expert()
            .map(|e| (e.id.clone(), !e.is_frozen()))
        {
            Some((id, freeze)) => self.stage(Pending::FreezeExpert(id, freeze)),
            None => self.note(Tone::Open, "no expert is selected"),
        }
    }

    /// Stage cancelling the selected run's pending halt (the palette's
    /// command), bringing the loop into view first. It says so when there is
    /// no halt to cancel.
    pub fn request_cancel_halt(&mut self) {
        if !self.may_act() {
            return;
        }
        self.page = Page::Loop;
        match self.selected_loop_head().map(|h| h.run_id.clone()) {
            None => self.note(
                Tone::Open,
                "no training loop has run, so there is no halt to cancel",
            ),
            Some(run) if self.halt_pending(&run) => self.stage(Pending::CancelHalt(run)),
            Some(run) => self.note(
                Tone::Open,
                format!("no halt is pending for {}", run.as_str()),
            ),
        }
    }

    /// The confirmation prompt for the staged action, if any.
    pub fn pending_prompt(&self) -> Option<String> {
        self.pending.as_ref().map(|p| match p {
            Pending::PruneShadow(id) => format!("Prune candidate {}?", id.as_str()),
            Pending::GraduateShadow(id) => format!("Graduate candidate {}?", id.as_str()),
            Pending::FreezeExpert(id, true) => format!("Freeze expert {}?", id.as_str()),
            Pending::FreezeExpert(id, false) => format!("Thaw expert {}?", id.as_str()),
            Pending::HaltLoop(run) => format!("Halt {}?", run.as_str()),
            Pending::CancelHalt(run) => format!("Cancel the halt of {}?", run.as_str()),
            Pending::DeleteBoundary(id) => {
                let behavior = self
                    .boundaries
                    .iter()
                    .find(|b| b.id.as_str() == id.as_str())
                    .map_or_else(|| id.as_str().to_string(), |b| b.behavior.clone());
                format!("Delete boundary \"{behavior}\"?")
            }
            Pending::SetMemoryPolicy(policy) => match self.knobs_changed(policy) {
                1 => format!("Apply {}?", self.knob_changes(policy)),
                n => format!("Apply {n} knob changes?"),
            },
        })
    }

    /// What going ahead with the staged action does, in a sentence under the
    /// prompt, so the answer is to a question that was understood.
    pub fn pending_effect(&self) -> Option<&'static str> {
        self.pending.as_ref().map(|p| match p {
            Pending::PruneShadow(_) => "It is dropped from training.",
            Pending::GraduateShadow(_) => "It becomes an expert.",
            Pending::FreezeExpert(_, true) => "Training will no longer change it.",
            Pending::FreezeExpert(_, false) => "Training may change it again.",
            Pending::HaltLoop(_) => "It stops at its next generation.",
            Pending::CancelHalt(_) => "It keeps running.",
            Pending::DeleteBoundary(_) => "Routing will no longer avoid it.",
            Pending::SetMemoryPolicy(_) => "It applies from the next recall.",
        })
    }

    /// Cancel the staged action.
    pub fn cancel_action(&mut self) {
        self.pending = None;
        self.mode = Mode::Normal;
    }

    /// Carry out the staged operator action through `backend`, then reload so
    /// the change shows immediately. What was done, a refusal, or a target
    /// that is gone is said on the main view. Returns to the live view whether
    /// or not anything was staged.
    pub async fn apply_pending(&mut self, backend: &Backend) {
        self.mode = Mode::Normal;
        let Some(pending) = self.pending.take() else {
            return;
        };
        // What to say once it is done, and whether the event stream needs
        // telling: a shadow's or an expert's change is recorded there by the
        // reload that follows, so saying it too would list it twice.
        let (action, said, recorded) = match pending {
            Pending::PruneShadow(shadow) => {
                let said = format!("pruned · {}", shadow.as_str());
                (ConsoleAction::PruneShadow { shadow }, said, true)
            }
            Pending::GraduateShadow(shadow) => {
                let said = format!("graduated · {}", shadow.as_str());
                (ConsoleAction::GraduateShadow { shadow }, said, true)
            }
            Pending::FreezeExpert(expert, true) => {
                let said = format!("frozen · {}", expert.as_str());
                (ConsoleAction::FreezeExpert { expert }, said, true)
            }
            Pending::FreezeExpert(expert, false) => {
                let said = format!("thawed · {}", expert.as_str());
                (ConsoleAction::ThawExpert { expert }, said, true)
            }
            Pending::DeleteBoundary(boundary) => {
                let said = self
                    .boundaries
                    .iter()
                    .find(|b| b.id.as_str() == boundary.as_str())
                    .map_or_else(
                        || "boundary removed".to_string(),
                        |b| format!("boundary removed · {}", b.behavior),
                    );
                (ConsoleAction::DeleteBoundary { boundary }, said, false)
            }
            Pending::HaltLoop(run) => {
                let said = format!("halt requested · {}", run.as_str());
                (ConsoleAction::HaltLoop { run }, said, false)
            }
            Pending::CancelHalt(run) => {
                let said = format!("halt canceled · {}", run.as_str());
                (ConsoleAction::CancelHalt { run }, said, false)
            }
            Pending::SetMemoryPolicy(policy) => {
                let said = format!("memory knobs set · {}", self.knob_changes(&policy));
                (ConsoleAction::SetMemoryPolicy { policy }, said, false)
            }
        };
        let done = backend.act(action).await;
        // Reload first, so the notice is the action's and not the reload's.
        self.refresh(backend).await;
        match done {
            Ok(true) if recorded => self.note(Tone::Settled, said),
            Ok(true) => self.say(Tone::Settled, said),
            Ok(false) => self.problem("not done · it is no longer there"),
            Err(e) => self.problem(format!("not done · {e:#}")),
        }
    }

    /// Record an event at the head of the stream (newest first), capped.
    pub(super) fn push_event(&mut self, kind: EventKind, text: String) {
        self.events.insert(
            0,
            Event {
                at_ms: self.clock_ms,
                kind,
                text,
            },
        );
        self.events.truncate(crate::events::MAX_EVENTS);
    }
}
