//! The names `antumbra tui snapshot` takes for what a frame shows: a page, a
//! layout, a list and a panel on top. Each is the word the console draws. The
//! name a thing had before the console was reworded is taken too, and any
//! other is refused with the names that are taken, so a frame is never of
//! something that was not asked for.

use anyhow::{anyhow, Result};

use crate::app::{Focus, LayoutMode, Page};

/// A panel a frame can be drawn with on top.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Panel {
    None,
    Help,
    Palette,
    Events,
    Detail,
    Router,
    Connect,
    Knobs,
    Ask,
    Confirm,
}

/// What a flag can name: the thing, the word the console draws for it, and
/// the name it had before, where it had another.
type Names<T> = [(T, &'static str, Option<&'static str>)];

pub(crate) const PAGES: &Names<Page> = &[
    (Page::Population, "population", None),
    (Page::Memory, "memory", None),
    (Page::Loop, "training", Some("loop")),
    (Page::Evals, "evaluations", Some("evals")),
    (Page::Sovereign, "agent", Some("sovereign")),
];

pub(crate) const LAYOUTS: &Names<LayoutMode> = &[
    (LayoutMode::List, "list", Some("focused")),
    (LayoutMode::Dashboard, "dashboard", None),
    (LayoutMode::Graph, "graph", None),
    (LayoutMode::Table, "table", None),
];

pub(crate) const LISTS: &Names<Focus> = &[
    (Focus::Experts, "experts", None),
    (Focus::Shadows, "candidates", Some("shadows")),
    (Focus::Boundaries, "boundaries", None),
];

pub(crate) const PANELS: &Names<Panel> = &[
    (Panel::None, "none", None),
    (Panel::Help, "help", None),
    (Panel::Palette, "palette", None),
    (Panel::Events, "events", None),
    (Panel::Detail, "detail", None),
    (Panel::Router, "router", Some("gate")),
    (Panel::Connect, "connect", None),
    (Panel::Knobs, "knobs", None),
    (Panel::Ask, "ask", None),
    (Panel::Confirm, "confirm", None),
];

/// The page `--page` names.
pub(crate) fn page(given: &str) -> Result<Page> {
    pick("page", given, PAGES)
}

/// The layout `--layout` names.
pub(crate) fn layout(given: &str) -> Result<LayoutMode> {
    pick("layout", given, LAYOUTS)
}

/// The list `--focus` names.
pub(crate) fn list(given: &str) -> Result<Focus> {
    pick("focus", given, LISTS)
}

/// The panel `--overlay` names.
pub(crate) fn panel(given: &str) -> Result<Panel> {
    pick("overlay", given, PANELS)
}

/// The thing `given` names among `names`, in any case. A name none of them
/// has is refused, and the refusal lists the ones the flag takes.
fn pick<T: Copy>(flag: &str, given: &str, names: &Names<T>) -> Result<T> {
    let asked = given.trim().to_lowercase();
    names
        .iter()
        .find(|(_, name, former)| *name == asked || *former == Some(asked.as_str()))
        .map(|(thing, ..)| *thing)
        .ok_or_else(|| {
            let taken: Vec<&str> = names.iter().map(|(_, name, _)| *name).collect();
            anyhow!(
                "--{flag} {given}: no such name. It takes {}.",
                one_of(&taken)
            )
        })
}

/// `a, b or c`.
fn one_of(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [only] => (*only).to_string(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_and_a_layout_are_named_by_the_word_the_console_draws() -> Result<()> {
        // A page by its tab's label, every page and in tab order, so one
        // added to the tabs cannot be left without its name.
        let named: Vec<(Page, String)> = PAGES
            .iter()
            .map(|(page, name, _)| (*page, (*name).to_string()))
            .collect();
        let tabs: Vec<(Page, String)> = Page::ALL
            .iter()
            .map(|page| (*page, page.title().to_lowercase()))
            .collect();
        assert_eq!(named, tabs);
        // A layout by the name its notice gives.
        for (layout, name, _) in LAYOUTS {
            assert_eq!(layout.name(), *name);
        }
        Ok(())
    }

    #[test]
    fn a_name_is_taken_in_any_case_and_so_is_the_one_it_had() -> Result<()> {
        assert_eq!(list("candidates")?, Focus::Shadows);
        assert_eq!(list("Candidates")?, Focus::Shadows);
        assert_eq!(list("shadows")?, Focus::Shadows);
        assert_eq!(list("experts")?, Focus::Experts);
        assert_eq!(list("boundaries")?, Focus::Boundaries);
        assert_eq!(layout("list")?, LayoutMode::List);
        assert_eq!(layout("focused")?, LayoutMode::List);
        assert_eq!(layout("table")?, LayoutMode::Table);
        for tab in Page::ALL {
            assert_eq!(page(tab.title())?, tab);
        }
        assert_eq!(page("loop")?, Page::Loop);
        assert_eq!(page("evals")?, Page::Evals);
        assert_eq!(page("sovereign")?, Page::Sovereign);
        assert_eq!(panel("router")?, Panel::Router);
        assert_eq!(panel("gate")?, Panel::Router);
        assert_eq!(panel("none")?, Panel::None);
        Ok(())
    }

    #[test]
    fn a_name_nothing_has_is_refused_with_the_names_that_are_taken() -> Result<()> {
        let Err(refusal) = list("candidate") else {
            anyhow::bail!("a name no list has was taken");
        };
        assert_eq!(
            refusal.to_string(),
            "--focus candidate: no such name. It takes experts, candidates or boundaries."
        );
        let Err(refusal) = layout("grid") else {
            anyhow::bail!("a name no layout has was taken");
        };
        assert_eq!(
            refusal.to_string(),
            "--layout grid: no such name. It takes list, dashboard, graph or table."
        );
        // The refusal gives the words the console draws, never a former one.
        let Err(refusal) = page("nowhere") else {
            anyhow::bail!("a name no page has was taken");
        };
        let said = refusal.to_string();
        assert!(
            said.contains("training") && !said.contains("loop"),
            "{said}"
        );
        assert!(panel("popup").is_err());
        Ok(())
    }
}
