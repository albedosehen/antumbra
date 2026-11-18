//! The console's keys, in one place: which key asks for what in the view the
//! console is in, and the few words that say so.
//!
//! The terminal loop turns an event into a [`Key`] and asks [`action_for`];
//! the footer draws [`bindings`] and the help draws [`page_bindings`]. A key
//! takes effect because it is in the list, and the screen says what a key does
//! by reading the same list, so the two cannot come apart.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::app::{App, Focus, LayoutMode, Mode, Page};
use crate::command::Action;

/// A key press, as far as the console tells keys apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    Char(char),
    /// A letter held with Ctrl, lowercased.
    Ctrl(char),
    Up,
    Down,
    Left,
    Right,
    Enter,
    Esc,
    Tab,
    Backspace,
    Home,
    End,
    PageUp,
    PageDown,
}

impl Key {
    /// The key a terminal event is, or `None` for one the console has no use
    /// for. Ctrl with Alt is how some keyboards type a character (AltGr), so
    /// only Ctrl alone makes a [`Key::Ctrl`].
    pub fn from_event(event: KeyEvent) -> Option<Key> {
        let ctrl = event.modifiers.contains(KeyModifiers::CONTROL)
            && !event.modifiers.contains(KeyModifiers::ALT);
        Some(match event.code {
            KeyCode::Char(c) if ctrl => Key::Ctrl(c.to_ascii_lowercase()),
            KeyCode::Char(c) => Key::Char(c),
            KeyCode::Up => Key::Up,
            KeyCode::Down => Key::Down,
            KeyCode::Left => Key::Left,
            KeyCode::Right => Key::Right,
            KeyCode::Enter => Key::Enter,
            KeyCode::Esc => Key::Esc,
            KeyCode::Tab => Key::Tab,
            KeyCode::Backspace => Key::Backspace,
            KeyCode::Home => Key::Home,
            KeyCode::End => Key::End,
            KeyCode::PageUp => Key::PageUp,
            KeyCode::PageDown => Key::PageDown,
            _ => return None,
        })
    }
}

/// The part of the help a binding is listed under.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    /// Moving the selection.
    Move,
    /// Changing what is on screen.
    View,
    /// Acting on what is selected.
    Act,
    /// The console itself.
    Console,
}

impl Group {
    /// Every group, in the order the help lists them.
    pub const ALL: [Group; 4] = [Group::Move, Group::Act, Group::View, Group::Console];

    /// The group's heading in the help.
    pub fn title(self) -> &'static str {
        match self {
            Group::Move => "move",
            Group::Act => "act on the selection",
            Group::View => "view",
            Group::Console => "console",
        }
    }
}

/// One thing the keys do in a view: the keys, and the words for it.
pub struct Binding {
    /// The key as the footer's chip writes it: `↑↓`.
    pub chip: &'static str,
    /// Other keys that do the same, which only the help has room for: `j k`.
    pub also: &'static str,
    /// What it does in a word or two, for the footer: `move`.
    pub hint: &'static str,
    /// What it does, for the help.
    pub does: &'static str,
    pub group: Group,
    /// Its place in the footer: the lower, the longer it is kept as the row
    /// narrows. `None` is listed in the help alone.
    pub rank: Option<u8>,
    /// Each key and what it asks for.
    pub keys: Vec<(Key, Action)>,
}

/// The footer keeps these however narrow the row: how to get help, and how to
/// leave.
pub const KEPT: u8 = 0;

fn bind(
    chip: &'static str,
    also: &'static str,
    hint: &'static str,
    does: &'static str,
    group: Group,
    rank: Option<u8>,
    keys: &[(Key, Action)],
) -> Binding {
    Binding {
        chip,
        also,
        hint,
        does,
        group,
        rank,
        keys: keys.to_vec(),
    }
}

/// What `x` does to the selection on this page, as `(hint, does)`, or `None`
/// where nothing on the page can be acted on.
fn act_words(app: &App) -> Option<(&'static str, &'static str)> {
    match app.page {
        Page::Population => Some(match app.focus {
            Focus::Experts if app.selected_expert().is_some_and(|e| e.is_frozen()) => (
                "thaw",
                "thaw the selected expert, or freeze one that is not",
            ),
            Focus::Experts => ("freeze", "freeze the selected expert, or thaw a frozen one"),
            Focus::Shadows => ("prune", "prune the selected candidate"),
            Focus::Boundaries => ("delete", "delete the selected boundary"),
        }),
        Page::Loop if app.selected_halt_pending() => {
            Some(("cancel halt", "cancel the selected run's halt"))
        }
        Page::Loop => Some(("halt", "halt the selected run, or cancel its halt")),
        Page::Memory | Page::Evals | Page::Sovereign => None,
    }
}

/// The main view's bindings on the page the console is on: what the keys do
/// when no overlay is open. The help lists these whatever is open over them.
pub fn page_bindings(app: &App) -> Vec<Binding> {
    use Action as A;
    use Key::{Char, Ctrl, Down, End, Enter, Home, PageDown, PageUp, Tab, Up};
    let population = app.page == Page::Population;
    // Only an operator's console acts, so only an operator's footer offers to.
    let acting = app.operator.then_some(3);
    let mut b = vec![
        bind(
            "↑↓",
            "j k",
            "move",
            "move the selection",
            Group::Move,
            Some(1),
            &[
                (Up, A::Move(-1)),
                (Char('k'), A::Move(-1)),
                (Down, A::Move(1)),
                (Char('j'), A::Move(1)),
            ],
        ),
        bind(
            "home end",
            "g G",
            "ends",
            "jump to the first or the last",
            Group::Move,
            None,
            &[
                (Home, A::First),
                (Char('g'), A::First),
                (End, A::Last),
                (Char('G'), A::Last),
            ],
        ),
        bind(
            "pgup pgdn",
            "",
            "page",
            "move ten at a time",
            Group::Move,
            None,
            &[(PageUp, A::PageMove(-1)), (PageDown, A::PageMove(1))],
        ),
    ];
    // Enter opens a drill-down where the page has one; the Memory page shows
    // its detail beside the list, and the Sovereign page has none.
    let opens = match app.page {
        Page::Population => Some("open everything about what is selected"),
        Page::Evals | Page::Loop => Some("open everything about the selected run"),
        Page::Memory | Page::Sovereign => None,
    };
    if let Some(does) = opens {
        b.push(bind(
            "enter",
            "",
            "details",
            does,
            Group::Move,
            Some(2),
            &[(Enter, A::Submit)],
        ));
    }
    // Tab moves the keys to the next list, where that list is drawn.
    if population && matches!(app.layout, LayoutMode::Focused | LayoutMode::Dashboard) {
        b.push(bind(
            "tab",
            "",
            "next list",
            "move to the next list: experts, candidates, boundaries",
            Group::Move,
            Some(4),
            &[(Tab, A::FocusNext)],
        ));
    }
    b.push(bind(
        "/",
        "",
        "find",
        "find in this list, and jump to it",
        Group::Move,
        Some(5),
        &[(Char('/'), A::Filter)],
    ));

    if let Some((hint, does)) = act_words(app) {
        b.push(bind(
            "x",
            "",
            hint,
            does,
            Group::Act,
            acting,
            &[(Char('x'), A::Act)],
        ));
    }
    if population && app.focus == Focus::Shadows {
        b.push(bind(
            "p",
            "",
            "graduate",
            "graduate the selected candidate into an expert",
            Group::Act,
            acting,
            &[(Char('p'), A::GraduateShadow)],
        ));
    }

    let mut pages = vec![(Char(']'), A::PageStep(1)), (Char('['), A::PageStep(-1))];
    for page in Page::ALL {
        if let Some(digit) = page.key() {
            pages.push((Char(digit), A::Page(page)));
        }
    }
    b.push(bind(
        "1-5",
        "[ ]",
        "pages",
        "pages: Population, Memory, Training, Evaluations, Agent",
        Group::View,
        Some(7),
        &pages,
    ));
    if population {
        b.push(bind(
            "l",
            "L",
            "layout",
            "change the layout: focused, dashboard, graph, table",
            Group::View,
            Some(9),
            &[(Char('l'), A::LayoutNext), (Char('L'), A::LayoutNext)],
        ));
        b.push(bind(
            "s",
            "",
            "sort",
            "sort the experts: by fitness, name or generation",
            Group::View,
            None,
            &[(Char('s'), A::CycleSort)],
        ));
    }
    b.extend([
        bind(
            "a",
            "",
            "ask",
            "ask which expert would take a task",
            Group::Console,
            Some(8),
            &[(Char('a'), A::Ask)],
        ),
        bind(
            "K",
            "",
            "knobs",
            "the memory knobs: how memories are kept",
            Group::Console,
            None,
            &[(Char('K'), A::Knobs)],
        ),
        bind(
            "t",
            "",
            "theme",
            "change the theme: night, mist, mono",
            Group::View,
            None,
            &[(Char('t'), A::CycleTheme)],
        ),
        bind(
            "+ -",
            "0",
            "frame rate",
            "raise or lower the frame-rate cap; 0 follows the monitor",
            Group::View,
            None,
            &[
                (Char('+'), A::FpsUp),
                (Char('='), A::FpsUp),
                (Char('-'), A::FpsDown),
                (Char('_'), A::FpsDown),
                (Char('0'), A::FollowMonitor),
            ],
        ),
        bind(
            ":",
            "",
            "commands",
            "run a command by its name",
            Group::Console,
            Some(6),
            &[(Char(':'), A::Palette)],
        ),
        bind(
            "e",
            "",
            "history",
            "what changed since the console opened, and what it did",
            Group::Console,
            None,
            &[(Char('e'), A::Events)],
        ),
        bind(
            "c",
            "",
            "connect",
            "how to connect an agent, and what this console reads",
            Group::Console,
            None,
            &[(Char('c'), A::Connect)],
        ),
        bind(
            "r",
            "",
            "reload",
            "read the server again now",
            Group::Console,
            None,
            &[(Char('r'), A::Reload)],
        ),
        bind(
            "?",
            "",
            "help",
            "this help",
            Group::Console,
            Some(KEPT),
            &[(Char('?'), A::Help)],
        ),
        bind(
            "q",
            "ctrl-c",
            "quit",
            "quit",
            Group::Console,
            Some(KEPT),
            &[(Char('q'), A::Quit), (Ctrl('c'), A::Quit)],
        ),
    ]);
    b
}

/// A binding that scrolls an overlay, a line or a page at a time.
fn scrolls() -> Binding {
    use Action as A;
    use Key::{Char, Down, PageDown, PageUp, Up};
    bind(
        "↑↓",
        "j k pgup pgdn",
        "scroll",
        "scroll",
        Group::Move,
        Some(1),
        &[
            (Up, A::Move(-1)),
            (Char('k'), A::Move(-1)),
            (Down, A::Move(1)),
            (Char('j'), A::Move(1)),
            (PageUp, A::PageMove(-1)),
            (PageDown, A::PageMove(1)),
        ],
    )
}

/// A binding that closes an overlay with Esc and `also`.
fn closes(also: &'static str, keys: &[Key]) -> Binding {
    let mut all = vec![(Key::Esc, Action::Close)];
    all.extend(keys.iter().map(|&k| (k, Action::Close)));
    bind(
        "esc",
        also,
        "close",
        "close",
        Group::Console,
        Some(KEPT),
        &all,
    )
}

/// A binding that moves the highlight in a list a query narrows. Letters are
/// typed into the query there, so `j` and `k` are not offered.
fn chooses() -> Binding {
    use Action as A;
    use Key::{Ctrl, Down, Up};
    bind(
        "↑↓",
        "ctrl-p ctrl-n",
        "choose",
        "choose",
        Group::Move,
        Some(1),
        &[
            (Up, A::Move(-1)),
            (Ctrl('p'), A::Move(-1)),
            (Down, A::Move(1)),
            (Ctrl('n'), A::Move(1)),
        ],
    )
}

/// The bindings of the view the console is in: the page's when no overlay is
/// open, else the overlay's own.
pub fn bindings(app: &App) -> Vec<Binding> {
    use Action as A;
    use Key::{Char, Enter, Left, Right};
    let submit = |hint: &'static str| {
        bind(
            "enter",
            "",
            hint,
            hint,
            Group::Act,
            Some(2),
            &[(Enter, A::Submit)],
        )
    };
    match app.mode {
        Mode::Normal => page_bindings(app),
        Mode::Help => vec![scrolls(), closes("? q", &[Char('?'), Char('q')])],
        Mode::Palette => vec![chooses(), submit("run"), closes("", &[])],
        Mode::Filter => vec![chooses(), submit("jump"), closes("", &[])],
        Mode::Ask => vec![submit("route"), closes("", &[])],
        Mode::Events => vec![scrolls(), closes("e q", &[Char('e'), Char('q')])],
        Mode::Detail | Mode::Gate | Mode::Connect => {
            vec![scrolls(), closes("enter q", &[Enter, Char('q')])]
        }
        Mode::Confirm => vec![
            bind(
                "y",
                "",
                "confirm",
                "confirm",
                Group::Act,
                Some(1),
                &[(Char('y'), A::Confirm), (Char('Y'), A::Confirm)],
            ),
            bind(
                "n",
                "esc q",
                "cancel",
                "cancel",
                Group::Console,
                Some(KEPT),
                &[
                    (Char('n'), A::Decline),
                    (Char('N'), A::Decline),
                    (Key::Esc, A::Decline),
                    (Char('q'), A::Decline),
                ],
            ),
        ],
        Mode::Knobs => {
            let mut pick = scrolls();
            pick.hint = "pick";
            pick.also = "j k";
            pick.keys.retain(|(_, action)| matches!(action, A::Move(_)));
            vec![
                pick,
                bind(
                    "←→",
                    "h l",
                    "turn",
                    "turn",
                    Group::Act,
                    Some(2),
                    &[
                        (Left, A::Turn(-1)),
                        (Char('h'), A::Turn(-1)),
                        (Right, A::Turn(1)),
                        (Char('l'), A::Turn(1)),
                    ],
                ),
                submit("apply"),
                closes("K q", &[Char('K'), Char('q')]),
            ]
        }
    }
}

/// What `key` asks for in the view the console is in, or `None` when it asks
/// for nothing there. Ctrl-C quits from every view; in a view that takes text
/// any other character is typed.
pub fn action_for(app: &App, key: Key) -> Option<Action> {
    if key == Key::Ctrl('c') {
        return Some(Action::Quit);
    }
    let bound = bindings(app)
        .into_iter()
        .flat_map(|b| b.keys)
        .find(|(k, _)| *k == key)
        .map(|(_, action)| action);
    bound.or(match (app.mode, key) {
        (Mode::Palette | Mode::Filter | Mode::Ask, Key::Char(c)) => Some(Action::Type(c)),
        (Mode::Palette | Mode::Filter | Mode::Ask, Key::Backspace) => Some(Action::Backspace),
        _ => None,
    })
}
