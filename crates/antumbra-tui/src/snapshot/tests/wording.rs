//! What the console says: that every word on screen is one a newcomer can
//! read, that every page says what it is, that the help defines the words it
//! uses, and that a token which may not read something says so instead of
//! looking empty.

use super::*;
use crate::app::Page;
use crate::words;

/// The console with something in every list.
fn full_app() -> App {
    let mut app = demo_app();
    app.shadows = vec![graduated_shadow(), kept_shadow()];
    app.boundaries = vec![actionable_boundary(), missed_boundary()];
    app
}

/// The console as a token that is not an operator's sees it: its workspace's
/// experts and memories, and none of what the server keeps for operators.
fn read_only_app() -> App {
    let mut app = demo_app();
    app.operator = false;
    app.shadows.clear();
    app.boundaries.clear();
    app.loop_heads.clear();
    app.evals.clear();
    app.router = None;
    app
}

/// A workspace that holds nothing yet.
fn empty_app() -> App {
    let mut app = read_only_app();
    app.operator = true;
    app.experts.clear();
    app.memories.clear();
    app.edges.clear();
    app
}

type Open = fn(&mut App);
type Make = fn() -> App;

/// Every frame a person can reach: each page in each layout and on each
/// list, each overlay, and the same for a read-only console and an empty one.
fn every_frame() -> anyhow::Result<Vec<(String, String)>> {
    let overlays: [(&str, Open); 9] = [
        ("help", App::toggle_help),
        ("palette", App::open_palette),
        ("filter", App::open_filter),
        ("history", App::open_events),
        ("detail", App::open_detail),
        ("router", App::open_gate),
        ("connect", App::open_connect),
        ("ask", App::open_ask),
        ("knobs", App::open_knobs),
    ];
    let kinds: [(&str, Make); 3] = [
        ("full", full_app),
        ("read-only", read_only_app),
        ("empty", empty_app),
    ];
    let mut frames = Vec::new();
    for (kind, make) in kinds {
        for page in Page::ALL {
            for layout in [
                LayoutMode::Focused,
                LayoutMode::Dashboard,
                LayoutMode::Graph,
                LayoutMode::Table,
            ] {
                for focus in [Focus::Experts, Focus::Shadows, Focus::Boundaries] {
                    let mut app = make();
                    app.set_page(page);
                    app.set_focus(focus);
                    app.set_layout(layout);
                    frames.push((
                        format!("{kind} · {} · {} · {focus:?}", page.title(), layout.name()),
                        to_text(&render(&mut app, 120, 40, 1600.0)?),
                    ));
                }
            }
            for (name, open) in overlays {
                let mut app = make();
                app.set_page(page);
                app.operator_event("candidate shadow:g4 graduated".into());
                open(&mut app);
                // Tall enough that the help shows its whole glossary.
                frames.push((
                    format!("{kind} · {} · {name}", page.title()),
                    to_text(&render(&mut app, 120, 80, 1600.0)?),
                ));
            }
        }
        // Each confirm prompt, which says what going ahead does.
        for focus in [Focus::Experts, Focus::Shadows, Focus::Boundaries] {
            let mut app = full_app();
            app.shadows = vec![Shadow {
                status: ShadowStatus::Exploring,
                ..graduated_shadow()
            }];
            app.set_focus(focus);
            app.request_action();
            frames.push((
                format!("{kind} · confirm · {focus:?}"),
                to_text(&render(&mut app, 120, 40, 1600.0)?),
            ));
        }
    }
    Ok(frames)
}

#[test]
fn no_frame_says_an_internal_word() -> anyhow::Result<()> {
    // The project's own names for what the console shows. Each has a plain
    // word here, which is the dashboard's.
    const INTERNAL: [&str; 12] = [
        "umbra",
        "penumbra",
        "substrate",
        "shadow",
        "shadows",
        "gate",
        "gates",
        "gating",
        "tripwire",
        "ood",
        "exemplars",
        "keeper",
    ];
    for (frame, text) in every_frame()? {
        // An id keeps its stored form, so a candidate's still begins
        // `shadow:`; the product's name is not one of its regions.
        let lower = text.to_lowercase().replace("shadow:", "");
        for word in lower.split(|c: char| !c.is_alphabetic()) {
            assert!(
                !INTERNAL.contains(&word),
                "{frame}: the frame says `{word}`:\n{text}"
            );
        }
    }
    Ok(())
}

#[test]
fn every_page_says_what_it_is_under_the_tabs() -> anyhow::Result<()> {
    for page in Page::ALL {
        for focus in [Focus::Experts, Focus::Shadows, Focus::Boundaries] {
            let mut app = full_app();
            app.set_page(page);
            app.set_focus(focus);
            let caption = words::caption(&app);
            assert!(
                caption.ends_with('.') && caption.contains(": "),
                "`{caption}` is a name and a sentence"
            );
            let text = to_text(&render(&mut app, 140, 40, 1600.0)?);
            let third = text.lines().nth(2).unwrap_or_default();
            assert_eq!(third.trim(), caption, "{} · {focus:?}", page.title());
        }
        // Narrower than the sentence, it is cut and never wraps into the page.
        let mut app = full_app();
        app.set_page(page);
        let text = to_text(&render(&mut app, 60, 24, 1600.0)?);
        let third = text.lines().nth(2).unwrap_or_default();
        assert!(third.trim_end().ends_with('…'), "`{third}`");
        assert!(
            text.lines().nth(3).unwrap_or_default().starts_with('╭'),
            "the page starts on the next row:\n{text}"
        );
    }
    // The population page's sentence is about the list in front.
    let mut app = full_app();
    app.set_focus(Focus::Shadows);
    assert!(words::caption(&app).starts_with("Candidates: "));
    app.set_focus(Focus::Boundaries);
    assert!(words::caption(&app).starts_with("Boundaries: "));
    Ok(())
}

#[test]
fn the_help_ends_with_what_each_word_means() -> anyhow::Result<()> {
    let mut app = full_app();
    app.toggle_help();
    let help = to_text(&render(&mut app, 120, 80, 1600.0)?);
    for word in words::GLOSSARY {
        assert!(
            help.contains(&format!("  {:<15}{}", word.term, word.means[0])),
            "the help leaves out `{}`:\n{help}",
            word.term
        );
        for line in word.means {
            assert!(help.contains(line), "`{line}` is cut:\n{help}");
        }
    }
    Ok(())
}

#[test]
fn every_word_the_glossary_defines_is_used_on_screen() -> anyhow::Result<()> {
    // A glossary of words nobody meets is noise: each is somewhere other
    // than the help.
    let frames: Vec<String> = every_frame()?
        .into_iter()
        .filter(|(frame, _)| !frame.ends_with("help"))
        .map(|(_, text)| text.to_lowercase())
        .collect();
    for word in words::GLOSSARY {
        assert!(
            frames.iter().any(|text| text.contains(word.term)),
            "nothing on screen says `{}`",
            word.term
        );
    }
    Ok(())
}

#[test]
fn a_read_only_console_says_so_and_does_not_look_empty() -> anyhow::Result<()> {
    let needs = "Only an operator's token reads this";
    // The top bar says it on every page.
    for page in Page::ALL {
        let mut app = read_only_app();
        app.set_page(page);
        let text = to_text(&render(&mut app, 120, 40, 1600.0)?);
        let bar = text.lines().next().unwrap_or_default();
        assert!(bar.contains("read-only"), "{}: `{bar}`", page.title());
        assert!(!bar.contains("operator"), "`{bar}`");
    }
    // What only an operator reads says that, not that nothing has happened.
    let page = |page: Page, focus: Focus| -> anyhow::Result<String> {
        let mut app = read_only_app();
        app.set_page(page);
        app.set_focus(focus);
        Ok(to_text(&render(&mut app, 120, 40, 1600.0)?))
    };
    let training = page(Page::Loop, Focus::Experts)?;
    assert!(training.contains(needs), "{training}");
    assert!(!training.contains("No training loop has run"), "{training}");
    let evaluations = page(Page::Evals, Focus::Experts)?;
    assert!(evaluations.contains(needs), "{evaluations}");
    assert!(!evaluations.contains("No evaluations yet"), "{evaluations}");
    for focus in [Focus::Shadows, Focus::Boundaries] {
        let list = page(Page::Population, focus)?;
        assert!(list.contains(needs), "{focus:?}:\n{list}");
        assert!(!list.contains(" yet:"), "{focus:?}:\n{list}");
    }
    // What every token reads is unaffected.
    let experts = page(Page::Population, Focus::Experts)?;
    assert!(experts.contains("arith-specialist"), "{experts}");
    assert!(
        experts.contains("needs an operator's token"),
        "the router is said to be the operator's:\n{experts}"
    );

    // An operator with the same empty lists is told they are empty.
    let mut app = read_only_app();
    app.operator = true;
    app.set_page(Page::Loop);
    let text = to_text(&render(&mut app, 120, 40, 1600.0)?);
    assert!(text.contains("No training loop has run yet"), "{text}");
    assert!(!text.contains(needs), "{text}");
    Ok(())
}

#[test]
fn the_top_bar_says_what_is_read_as_whom_and_whether_it_is_current() -> anyhow::Result<()> {
    let bar = |app: &mut App| -> anyhow::Result<String> {
        let text = to_text(&render(app, 120, 40, 1600.0)?);
        Ok(text.lines().next().unwrap_or_default().to_string())
    };
    let mut app = full_app();
    let live = bar(&mut app)?;
    assert!(live.contains("Antumbra"), "`{live}`");
    assert!(
        live.contains("http://127.0.0.1:8081 · operator"),
        "`{live}`"
    );
    assert!(live.trim_end().ends_with("● live"), "`{live}`");

    app.read_finished(Err(anyhow::anyhow!("connection refused")));
    let failing = bar(&mut app)?;
    assert!(
        failing.trim_end().ends_with("▲ not answering"),
        "`{failing}`"
    );
    assert!(!failing.contains("live"), "`{failing}`");
    app.read_finished(Ok(None));
    assert!(bar(&mut app)?.trim_end().ends_with("● live"));

    // The demo reads no server: it says it is one, and that nothing is kept.
    app.demo = true;
    let demo = bar(&mut app)?;
    assert!(demo.contains("demo"), "`{demo}`");
    assert!(demo.contains("nothing here is saved"), "`{demo}`");
    assert!(
        !demo.contains("live") && !demo.contains("http://"),
        "`{demo}`"
    );
    Ok(())
}

#[test]
fn the_top_bar_fits_a_narrow_terminal() -> anyhow::Result<()> {
    let mut app = full_app();
    app.source = "https://antumbra.example.com/a/long/path/to/the/server".into();
    for width in [120, 80, 60, 40] {
        let text = to_text(&render(&mut app, width, 24, 1600.0)?);
        let bar = text.lines().next().unwrap_or_default();
        assert!(
            bar.chars().count() <= usize::from(width),
            "{width}: `{bar}`"
        );
        assert!(bar.contains("Antumbra"), "{width}: `{bar}`");
        assert!(bar.trim_end().ends_with("live"), "{width}: `{bar}`");
    }
    Ok(())
}

#[test]
fn an_empty_list_says_why_and_what_fills_it() -> anyhow::Result<()> {
    let page = |page: Page, focus: Focus| -> anyhow::Result<String> {
        let mut app = empty_app();
        // One memory, so the first-run card is not over the page.
        app.memories = demo_memories(chrono::Utc::now());
        app.set_page(page);
        app.set_focus(focus);
        Ok(to_text(&render(&mut app, 120, 40, 1600.0)?))
    };
    let cases = [
        (Page::Population, Focus::Experts, "No experts yet: "),
        (Page::Population, Focus::Shadows, "No candidates yet: "),
        (Page::Population, Focus::Boundaries, "No boundaries yet: "),
        (Page::Loop, Focus::Experts, "No training loop has run yet: "),
        (Page::Evals, Focus::Experts, "No evaluations yet: "),
        (Page::Sovereign, Focus::Experts, "No rules kept yet: "),
    ];
    for (which, focus, why) in cases {
        let text = page(which, focus)?;
        assert!(text.contains(why), "{} · {focus:?}:\n{text}", which.title());
    }
    // No empty state is a bare parenthesis, as they used to be.
    for (frame, text) in every_frame()? {
        assert!(
            !text.contains("(no ") && !text.contains("(select "),
            "{frame}:\n{text}"
        );
    }
    Ok(())
}

#[test]
fn a_page_is_found_by_its_name_or_the_name_it_had() -> anyhow::Result<()> {
    for page in Page::ALL {
        assert_eq!(Page::from_name(page.title()), page);
        assert_eq!(Page::from_name(&page.title().to_lowercase()), page);
    }
    assert_eq!(Page::from_name("loop"), Page::Loop);
    assert_eq!(Page::from_name("evals"), Page::Evals);
    assert_eq!(Page::from_name("sovereign"), Page::Sovereign);
    assert_eq!(Page::from_name("nowhere"), Page::Population);
    Ok(())
}
