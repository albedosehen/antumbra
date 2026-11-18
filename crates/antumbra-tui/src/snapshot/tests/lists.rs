//! The lists a selection moves in: that the population and memory pages open
//! on one, that the selection is seen in it, that its title names the lists
//! beside it, and that a narrow terminal keeps a name before a lesser column.

use super::*;
use crate::app::Page;
use crate::theme::NIGHT;
use ratatui::buffer::Buffer;

/// An expert named `name`, as a workspace would hold one.
fn expert(name: &str, fitness: f32, gen: u32, frozen: bool) -> Expert {
    let now = Utc::now();
    Expert {
        id: ExpertId::new(format!("expert:{name}")),
        name: name.into(),
        base_model: "Qwen2.5-Coder-1.5B".into(),
        artifact_uri: "mem://x".into(),
        capability_card: serde_json::json!({}),
        capability_vec: None,
        fitness,
        frozen_at: frozen.then_some(now),
        generation: Generation(gen),
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: now,
    }
}

/// The console with a few of everything the population page lists.
fn listed_app() -> App {
    let mut app = demo_app();
    app.experts = vec![
        expert("arith-specialist", 0.92, 0, true),
        expert("datetime-specialist", 0.81, 1, true),
        expert("json-shaper", 0.55, 2, false),
    ];
    app.sort_experts();
    app.shadows = vec![graduated_shadow(), kept_shadow()];
    app.boundaries = vec![actionable_boundary(), missed_boundary()];
    app
}

/// `per_kind` memories of each kind, in the order the server sends them: by
/// kind, strongest first within each.
fn memories(per_kind: usize) -> Vec<Memory> {
    let now = Utc::now();
    let mut all = Vec::new();
    for (network, name) in [
        (MemoryNetwork::Bank, "experience"),
        (MemoryNetwork::Opinion, "judgment"),
        (MemoryNetwork::World, "fact"),
    ] {
        for i in 0..per_kind {
            all.push(Memory::new(
                format!("memory:{name}-{i}"),
                "ws:demo",
                network,
                format!("{name} number {i}"),
                0.9 - (i as f32) * 0.001,
                now,
            ));
        }
    }
    all
}

/// The rows of `text` that carry the selection's mark.
fn marked(text: &str) -> Vec<&str> {
    text.lines().filter(|row| row.contains('▸')).collect()
}

/// The background behind the first character of `word` on row `y`.
fn fill_behind(buf: &Buffer, y: u16, word: &str) -> anyhow::Result<ratatui::style::Color> {
    let row: String = (0..buf.area.width)
        .map(|x| buf[(x, y)].symbol().to_string())
        .collect();
    let at = row
        .find(word)
        .ok_or_else(|| anyhow::anyhow!("no `{word}` on row {y}: `{row}`"))?;
    // Every cell before it is one column wide, so its column is its place
    // among the characters.
    let x = u16::try_from(row[..at].chars().count())?;
    Ok(buf[(x, y)].bg)
}

#[test]
fn the_population_page_opens_on_a_list_with_the_selection_marked() -> anyhow::Result<()> {
    let mut app = listed_app();
    assert_eq!(app.layout, LayoutMode::List, "the default layout");
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert_eq!(marked(&text).len(), 1, "one row is marked:\n{text}");
    assert!(marked(&text)[0].contains("▸ arith-specialist"), "{text}");

    // The rows are in the order the keys move through them.
    let place = |name: &str| text.find(name);
    assert!(place("arith-specialist") < place("datetime-specialist"));
    assert!(place("datetime-specialist") < place("json-shaper"));
    app.select_next();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        marked(&text)[0].contains("▸ datetime-specialist"),
        "down moves the mark to the next row:\n{text}"
    );
    // The detail beside it is of the row that is marked.
    assert!(text.contains("╭ Expert ─"), "{text}");
    assert_eq!(
        text.matches("datetime-specialist").count(),
        2,
        "its row in the list, and the detail's heading:\n{text}"
    );
    assert_eq!(
        text.matches("arith-specialist").count(),
        1,
        "the one above is in the list alone:\n{text}"
    );
    Ok(())
}

#[test]
fn the_lists_title_names_all_three_and_marks_the_one_in_front() -> anyhow::Result<()> {
    let mut app = listed_app();
    let buf = render(&mut app, 120, 36, 1600.0)?;
    let text = to_text(&buf);
    // The fourth row is the top of the body, where the title is.
    let title = text.lines().nth(3).unwrap_or_default();
    assert!(
        title.contains("Experts 3 │ Candidates 2 │ Boundaries 2"),
        "`{title}`"
    );
    assert_eq!(fill_behind(&buf, 3, "Experts")?, NIGHT.accent_soft);
    assert_eq!(fill_behind(&buf, 3, "Candidates")?, NIGHT.ground);

    // Tab moves the mark along the title, and the list under it follows.
    app.toggle_focus();
    let buf = render(&mut app, 120, 36, 1600.0)?;
    assert_eq!(fill_behind(&buf, 3, "Experts")?, NIGHT.ground);
    assert_eq!(fill_behind(&buf, 3, "Candidates")?, NIGHT.accent_soft);
    let text = to_text(&buf);
    assert!(
        text.contains("CANDIDATE"),
        "the candidates' columns:\n{text}"
    );
    assert!(marked(&text)[0].contains("shadow:g3"), "{text}");
    app.toggle_focus();
    let buf = render(&mut app, 120, 36, 1600.0)?;
    assert_eq!(fill_behind(&buf, 3, "Boundaries")?, NIGHT.accent_soft);
    assert!(to_text(&buf).contains("BOUNDARY"));
    Ok(())
}

#[test]
fn each_list_shows_a_state_as_a_shape_and_a_word() -> anyhow::Result<()> {
    let mut app = listed_app();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("● frozen") && text.contains("○ open"),
        "{text}"
    );
    assert!(text.contains("shared"), "who may route to it:\n{text}");
    app.set_focus(Focus::Shadows);
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("0.20 → 0.93 ● graduated"), "{text}");
    app.set_focus(Focus::Boundaries);
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("● in force") && text.contains("○ recorded"));
    Ok(())
}

#[test]
fn every_memory_is_reached_and_seen_selected_whatever_their_number() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.memories = memories(100);
    app.edges.clear();
    app.set_page(Page::Memory);
    app.select_first();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    let title = text.lines().nth(3).unwrap_or_default();
    assert!(
        title.contains("All 300 │ World 100 │ Bank 100 │ Opinion 100"),
        "`{title}`"
    );
    // It opens on the first kind the title names, strongest first.
    assert!(marked(&text)[0].contains("▸ fact number 0"), "{text}");

    // The last of three hundred is reached, in view and marked.
    app.select_last();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert_eq!(marked(&text).len(), 1, "{text}");
    assert!(marked(&text)[0].contains("▸ judgment number 99"), "{text}");
    // And the detail beside it is that memory's.
    assert!(text.contains("opinion · a judgment"), "{text}");

    // One step at a time, the mark is always on screen.
    app.select_first();
    for step in 0..120 {
        app.select_next();
        let text = to_text(&render(&mut app, 120, 30, 1600.0)?);
        assert_eq!(marked(&text).len(), 1, "after {step} steps:\n{text}");
    }
    Ok(())
}

#[test]
fn a_kind_in_front_keeps_the_list_and_the_keys_within_it() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.memories = memories(40);
    app.edges.clear();
    app.set_page(Page::Memory);
    app.select_first();
    // Tab twice: from every kind, past the facts, to the experiences.
    app.next_memory_kind();
    app.next_memory_kind();
    let buf = render(&mut app, 120, 36, 1600.0)?;
    let text = to_text(&buf);
    assert_eq!(fill_behind(&buf, 3, "Bank")?, NIGHT.accent_soft);
    assert_eq!(fill_behind(&buf, 3, "All")?, NIGHT.ground);
    assert!(marked(&text)[0].contains("▸ experience number 0"), "{text}");
    assert!(!text.contains("fact number"), "only experiences:\n{text}");
    // With the kind in the title, its column is not repeated on every row.
    assert!(!text.contains("KIND"), "{text}");

    app.select_last();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        marked(&text)[0].contains("▸ experience number 39"),
        "{text}"
    );
    // Past the last is the first of the same kind.
    app.select_next();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(marked(&text)[0].contains("▸ experience number 0"), "{text}");
    Ok(())
}

#[test]
fn a_kind_with_no_memory_says_so() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.memories.retain(|m| m.network == MemoryNetwork::World);
    app.edges.clear();
    app.set_page(Page::Memory);
    app.select_first();
    let world = app.selected_memory;
    app.next_memory_kind(); // the facts
    app.next_memory_kind(); // the experiences, of which there are none
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("Bank 0"), "{text}");
    assert!(text.contains("No memories of this kind yet."), "{text}");
    assert_eq!(app.selected_memory, world, "the selection is left alone");
    // The keys do nothing in an empty list, and nothing breaks.
    app.select_next();
    app.select_last();
    assert_eq!(app.selected_memory, world);
    Ok(())
}

#[test]
fn a_narrow_terminal_drops_a_lesser_column_before_it_cuts_a_name() -> anyhow::Result<()> {
    let mut app = listed_app();
    let wide = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        wide.contains("GEN") && wide.contains("VISIBLE TO"),
        "{wide}"
    );
    let narrow = to_text(&render(&mut app, 80, 24, 1600.0)?);
    assert!(
        !narrow.contains("GEN") && !narrow.contains("VISIBLE TO"),
        "the lesser columns go:\n{narrow}"
    );
    for name in ["arith-specialist", "datetime-specialist", "json-shaper"] {
        assert!(narrow.contains(name), "`{name}` is whole:\n{narrow}");
    }
    assert!(narrow.contains("● frozen"), "its state stays:\n{narrow}");
    // The title still names all three, with their counts.
    let title = narrow.lines().nth(3).unwrap_or_default();
    assert!(
        title.contains("Experts 3 │ Candidates 2 │ Boundaries 2"),
        "`{title}`"
    );

    app.set_focus(Focus::Boundaries);
    let wide = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(wide.contains("WHERE"), "{wide}");
    let narrow = to_text(&render(&mut app, 80, 24, 1600.0)?);
    assert!(!narrow.contains("WHERE"), "{narrow}");
    assert!(narrow.contains("● in force"), "{narrow}");

    app.set_page(Page::Memory);
    let wide = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(wide.contains("KIND"), "{wide}");
    let narrow = to_text(&render(&mut app, 80, 24, 1600.0)?);
    assert!(!narrow.contains("KIND"), "{narrow}");
    assert!(narrow.contains("STRENGTH"), "{narrow}");
    Ok(())
}

#[test]
fn the_memory_graph_is_one_key_from_the_list() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.set_page(Page::Memory);
    let list = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(list.contains("MEMORY") && !list.contains("Memory · by kind"));
    assert!(list.lines().last().unwrap_or_default().contains("graph"));
    app.toggle_memory_graph();
    let graph = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(graph.contains("Memory · by kind"), "{graph}");
    assert!(!graph.contains("STRENGTH  "), "{graph}");
    // The footer offers the way back, and no Tab, which has no list to move
    // along.
    let footer = graph.lines().last().unwrap_or_default();
    assert!(footer.contains("list") && !footer.contains("next kind"));
    Ok(())
}

// Golden the population page's three lists, each a whole frame: there is no
// animated graph in the list layout, and nothing on it tells the time.
#[test]
fn golden_population_lists() -> anyhow::Result<()> {
    let mut app = listed_app();
    assert_golden(
        "list_experts",
        &to_text(&render(&mut app, 120, 36, 1600.0)?),
    )?;
    app.set_focus(Focus::Shadows);
    assert_golden(
        "list_candidates",
        &to_text(&render(&mut app, 120, 36, 1600.0)?),
    )?;
    app.set_focus(Focus::Boundaries);
    assert_golden(
        "list_boundaries",
        &to_text(&render(&mut app, 120, 36, 1600.0)?),
    )?;
    Ok(())
}

// Golden the population list at 80 columns by 24 rows, the size a terminal
// opens at: the columns that go, and the footer that fits.
#[test]
fn golden_population_list_at_80_columns() -> anyhow::Result<()> {
    let mut app = listed_app();
    assert_golden(
        "list_experts_80",
        &to_text(&render(&mut app, 80, 24, 1600.0)?),
    )
}

// Golden the memory page's graph, which the list replaced as what the page
// opens on. It is statically laid out, so the frame is stable.
#[test]
fn golden_memory_graph() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.set_page(Page::Memory);
    app.toggle_memory_graph();
    assert_golden(
        "memory_graph",
        &to_text(&render(&mut app, 120, 36, 1600.0)?),
    )
}
