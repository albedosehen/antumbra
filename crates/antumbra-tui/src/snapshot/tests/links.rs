//! The Memory page's graph: the selected memory's links, drawn, in a store
//! where most memories have none.

use super::*;
use crate::app::Page;

/// One memory of the fixture.
fn memory(id: &str, content: &str) -> Memory {
    Memory::new(
        id,
        "ws:demo",
        MemoryNetwork::World,
        content,
        0.8,
        Utc::now(),
    )
}

/// One link of the fixture.
fn link(from: &str, to: &str, kind: EdgeType) -> MemoryEdge {
    MemoryEdge::new("ws:demo", from, to, kind, 1.0, Utc::now())
}

/// The console on the Memory page of a store with seven memories: one with
/// three links, three linked to it, one two links away, and two with none.
/// The selection is on one of the unlinked ones.
fn linked_app() -> App {
    let mut app = demo_app();
    app.memories = vec![
        memory("memory:lone", "a note linked to nothing"),
        memory("memory:broke", "a raw query broke the migration"),
        memory("memory:rule", "use the query builders"),
        memory("memory:helper", "the builders gained a join"),
        memory("memory:old", "an older rule on queries"),
        memory("memory:next", "then the helper was documented"),
        memory("memory:other", "another unlinked note"),
    ];
    app.edges = vec![
        link("memory:broke", "memory:rule", EdgeType::References),
        link("memory:rule", "memory:helper", EdgeType::References),
        link("memory:rule", "memory:old", EdgeType::Supersedes),
        link("memory:helper", "memory:next", EdgeType::Follows),
    ];
    app.selected_memory = 0;
    app.set_page(Page::Memory);
    app
}

/// The content of the memory selected.
fn selected(app: &App) -> &str {
    app.selected_memory().map_or("", |m| m.content.as_str())
}

#[test]
fn the_graph_opens_on_the_most_linked_memory_and_draws_its_links() -> anyhow::Result<()> {
    let mut app = linked_app();
    app.toggle_memory_graph();
    // The one selected had no links, so the selection went to the most linked.
    assert_eq!(selected(&app), "use the query builders");
    let text = to_text(&render(&mut app, 220, 36, 1600.0)?);
    assert!(
        text.contains("Links · 5 of 7 memories have one · 4 links"),
        "{text}"
    );
    // It, what links to it, what it links to, and what one of those leads
    // to, each by how its content starts; nothing that is not linked.
    for content in [
        "a raw query broke",
        "use the query",
        "the builders gained",
        "an older rule on",
        "then the helper",
    ] {
        assert!(text.contains(content), "`{content}`:\n{text}");
    }
    assert!(!text.contains("linked to nothing"), "{text}");
    // A memory is drawn before what it links to, and a line arrives at an
    // arrowhead.
    // Where a chip starts, counted in cells: the arrowhead before it tells a
    // chip from the same words in the panel beside.
    let column = |chip: &str| {
        let row = text.lines().find(|row| row.contains(chip))?;
        Some(row[..row.find(chip)?].chars().count())
    };
    let (rule, helper) = (column("▸ use the query"), column("▸ the builders gained"));
    assert!(
        rule.is_some() && rule < helper,
        "{rule:?} {helper:?}:
{text}"
    );
    assert!(text.contains("▸ the builders gained"), "{text}");
    assert!(text.contains("its 3 links and where they lead"), "{text}");
    assert!(
        text.contains("── references   ╌╌ supersedes   ┄┄ follows, caused, contradicts"),
        "{text}"
    );
    Ok(())
}

#[test]
fn the_keys_move_through_the_memories_that_have_a_link() -> anyhow::Result<()> {
    let mut app = linked_app();
    app.toggle_memory_graph();
    let mut seen = vec![selected(&app).to_string()];
    for _ in 0..4 {
        app.select_next();
        seen.push(selected(&app).to_string());
    }
    // The most linked first, then the order the server sent them in; and
    // round again after the last.
    assert_eq!(
        seen,
        [
            "use the query builders",
            "the builders gained a join",
            "a raw query broke the migration",
            "an older rule on queries",
            "then the helper was documented",
        ]
    );
    app.select_next();
    assert_eq!(selected(&app), "use the query builders");
    // Back in the list, every memory is listed again.
    app.toggle_memory_graph();
    assert_eq!(app.listed_memories().len(), 7);
    Ok(())
}

#[test]
fn where_names_would_be_cut_only_the_memorys_own_links_are_drawn() -> anyhow::Result<()> {
    let mut app = linked_app();
    app.toggle_memory_graph();
    // Wide, the memory two links away is drawn too.
    let wide = to_text(&render(&mut app, 220, 36, 1600.0)?);
    assert!(wide.contains("then the helper"), "{wide}");
    // Narrower, a fourth column would cut every name to a few letters, so
    // it is left to the panel beside and the names kept can still be read.
    let narrow = to_text(&render(&mut app, 150, 36, 1600.0)?);
    assert!(!narrow.contains("then the helper"), "{narrow}");
    assert!(narrow.contains("its 3 links · ↑↓"), "{narrow}");
    assert!(narrow.contains("use the query bui"), "{narrow}");
    assert!(narrow.contains("a raw query broke the"), "{narrow}");
    Ok(())
}

#[test]
fn a_memory_or_a_store_with_no_links_says_so() -> anyhow::Result<()> {
    let mut app = linked_app();
    app.toggle_memory_graph();
    // A read can leave the selection on a memory whose links are gone.
    app.selected_memory = 0;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("This memory is not linked to another."),
        "{text}"
    );
    app.edges.clear();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("Links · 0 of 7 memories have one · 0 links"),
        "{text}"
    );
    assert!(
        text.contains("No memory is linked to another yet."),
        "{text}"
    );
    Ok(())
}

// Two memories that each link to the other: one line is drawn, the one that
// points forward, and one arrowhead, where it arrives.
#[test]
fn a_pair_that_link_each_other_is_one_line_and_one_arrowhead() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.set_page(Page::Memory);
    app.toggle_memory_graph();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    let graph: String = text
        .lines()
        .filter(|row| row.contains("npm install") || row.contains("prefer deno"))
        .collect();
    assert_eq!(graph.matches('▸').count(), 1, "{text}");
    Ok(())
}
