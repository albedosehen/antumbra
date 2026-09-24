use super::*;
use serde_json::json;

#[test]
fn property_names_follow_the_pattern_the_api_quotes() {
    let long = "x".repeat(65);
    let cases = [
        ("path", true),
        ("a.b-c_D9", true),
        ("x", true),
        (&"x".repeat(64)[..], true),
        ("", false),
        (long.as_str(), false),
        ("with space", false),
        ("dollar$", false),
        ("naïve", false),
        ("a/b", false),
    ];
    for (name, accepted) in cases {
        assert_eq!(is_acceptable_property_name(name), accepted, "{name:?}");
    }
}

#[test]
fn a_sound_schema_has_nothing_wrong() {
    let schema = json!({
        "type": "object",
        "properties": { "query": { "type": "string" }, "top_k": { "type": "integer" } },
        "required": ["query"]
    });
    assert_eq!(problems(&schema), Vec::new());
    assert_eq!(problems(&json!({ "type": "object" })), Vec::new());
}

#[test]
fn a_refused_property_name_is_named() {
    let schema = json!({ "type": "object", "properties": { "file path": {}, "ok": {} } });
    assert_eq!(
        problems(&schema),
        vec![Problem::PropertyName("file path".to_string())]
    );
}

#[test]
fn only_top_level_names_are_held_to_the_pattern() {
    let schema = json!({
        "type": "object",
        "properties": { "options": { "type": "object", "properties": { "any name": {} } } }
    });
    assert_eq!(problems(&schema), Vec::new());
}

#[test]
fn a_schema_the_meta_schema_refuses_is_invalid() -> anyhow::Result<()> {
    let schema = json!({ "type": "object", "properties": { "a": { "type": "strng" } } });
    let found = problems(&schema);
    let [Problem::InvalidSchema(why)] = found.as_slice() else {
        anyhow::bail!("expected one invalid-schema problem, got {found:?}");
    };
    assert!(why.contains("/properties/a/type"), "{why}");
    Ok(())
}

#[test]
fn a_root_that_is_not_an_object_costs_the_server_its_tools_and_ends_the_matter() {
    let roots = [
        json!(null),
        json!({}),
        json!({ "type": "string" }),
        json!({ "type": ["object"] }),
        json!({ "type": "objekt", "properties": { "bad name": {} } }),
        json!({ "anyOf": [{ "type": "object" }] }),
    ];
    for root in roots {
        assert_eq!(problems(&root), vec![Problem::NotAnObject], "{root}");
    }
    assert!(Problem::NotAnObject.is_failure());
    assert!(!Problem::NotAnObject.breaks_requests());
}

#[test]
fn the_dialect_decides_whether_the_meta_schema_is_consulted() {
    let invalid_anywhere = json!({ "type": "object", "required": "not-a-list" });
    let with = |dialect: &str| {
        let mut schema = invalid_anywhere.clone();
        if let Some(object) = schema.as_object_mut() {
            object.insert("$schema".to_string(), json!(dialect));
        }
        problems(&schema)
    };
    assert!(matches!(
        with("https://json-schema.org/draft/2020-12/schema").as_slice(),
        [Problem::InvalidSchema(_)]
    ));
    assert!(matches!(
        with("https://json-schema.org/draft/2020-12/schema#").as_slice(),
        [Problem::InvalidSchema(_)]
    ));
    assert_eq!(
        with("http://json-schema.org/draft-07/schema#"),
        vec![Problem::OtherDialect(
            "http://json-schema.org/draft-07/schema#".to_string()
        )]
    );
}

#[test]
fn another_dialect_is_still_held_to_the_property_names() {
    let schema = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": { "bad name": {} }
    });
    assert_eq!(
        problems(&schema),
        vec![
            Problem::PropertyName("bad name".to_string()),
            Problem::OtherDialect("http://json-schema.org/draft-07/schema#".to_string()),
        ]
    );
}

#[test]
fn a_root_combinator_is_the_whole_story_and_a_nested_one_is_none() {
    for keyword in ROOT_COMBINATORS {
        let schema = json!({ "type": "object", keyword: [{ "properties": { "bad name": {} } }] });
        assert_eq!(problems(&schema), vec![Problem::RootCombinator(keyword)]);
    }
    let nested = json!({
        "type": "object",
        "properties": { "target": { "anyOf": [{ "type": "string" }, { "type": "integer" }] } }
    });
    assert_eq!(problems(&nested), Vec::new());
}

#[test]
fn only_what_fails_a_request_counts_as_breaking() {
    assert!(Problem::PropertyName("a b".into()).breaks_requests());
    assert!(Problem::InvalidSchema("x".into()).breaks_requests());
    assert!(!Problem::RootCombinator("anyOf").breaks_requests());
    assert!(!Problem::OtherDialect("d".into()).breaks_requests());
    assert!(!Problem::RootCombinator("anyOf").is_failure());
    assert!(!Problem::OtherDialect("d".into()).is_failure());
}

#[test]
fn tools_are_found_at_every_depth_of_the_envelope() -> anyhow::Result<()> {
    let tools = json!([{ "name": "read", "inputSchema": { "type": "object" } }]);
    let shapes = [
        json!({ "jsonrpc": "2.0", "id": 2, "result": { "tools": tools } }),
        json!({ "tools": tools }),
        tools.clone(),
    ];
    for shape in &shapes {
        let found = tools_in(shape)?;
        assert_eq!(found.len(), 1, "{shape}");
        assert_eq!(found.first().map(|t| t.name.as_str()), Some("read"));
    }
    let api_spelling = json!([{ "name": "read", "input_schema": { "type": "object" } }]);
    assert_eq!(
        tools_in(&api_spelling)?.first().map(|t| &t.input_schema),
        Some(&json!({ "type": "object" }))
    );
    assert!(tools_in(&json!({ "error": "nope" })).is_err());
    assert!(tools_in(&json!([{ "inputSchema": {} }])).is_err());
    // A tool that lists no schema at all is one whose root is not an object.
    let bare = tools_in(&json!([{ "name": "bare" }]))?;
    assert_eq!(
        lint(&bare).first().map(|f| f.problems.clone()),
        Some(vec![Problem::NotAnObject])
    );
    Ok(())
}

#[test]
fn a_rule_names_the_tool_the_way_the_agent_does() {
    assert_eq!(rule_name("kushtaka", "recall"), "mcp__kushtaka__recall");
    assert_eq!(
        rule_name("claude.ai Claude Docs", "batch"),
        "mcp__claude_ai_Claude_Docs__batch"
    );
    assert_eq!(
        rule_name("context7", "query-docs"),
        "mcp__context7__query-docs"
    );
}

#[test]
fn the_report_prints_a_deny_rule_for_what_breaks_and_only_that() {
    let tools = [
        Tool {
            name: "fine".into(),
            input_schema: json!({ "type": "object" }),
            output_schema: None,
        },
        Tool {
            name: "breaks".into(),
            input_schema: json!({ "type": "object", "properties": { "a b": {} } }),
            output_schema: None,
        },
        Tool {
            name: "dropped".into(),
            input_schema: json!({ "type": "object", "oneOf": [] }),
            output_schema: None,
        },
        Tool {
            name: "rootless".into(),
            input_schema: json!({ "type": "array" }),
            output_schema: None,
        },
    ];
    let findings = lint(&tools);
    assert_eq!(findings.len(), 3);
    assert_eq!(failures(&findings), 2);
    let text = render("my server", tools.len(), &findings);
    assert!(text.contains("4 tool(s) checked"), "{text}");
    assert!(text.contains("[FAIL] rootless"), "{text}");
    // A deny rule cannot help a list the client refuses before rules apply.
    assert!(!text.contains("mcp__my_server__rootless"), "{text}");
    assert!(text.contains("[FAIL] breaks"), "{text}");
    assert!(text.contains("[note] dropped"), "{text}");
    assert!(text.contains("\"mcp__my_server__breaks\""), "{text}");
    assert!(!text.contains("mcp__my_server__dropped"), "{text}");
    assert!(!text.contains("mcp__my_server__fine"), "{text}");

    let clean = render("s", 1, &[]);
    assert!(clean.contains("nothing the API would refuse"), "{clean}");
}
