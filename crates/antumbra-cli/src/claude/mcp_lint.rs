//! The MCP schema lint (ADR-0021): which of a server's tools will break a
//! session in sovereign mode, or go missing from it.
//!
//! The Claude API checks every tool's input schema and rejects the WHOLE request
//! when one fails. The agent runs two of those checks itself and excludes the
//! tools that fail them, but acting on the result is gated on a fetched flag.
//! With the flags off it still runs the checks, writes the answer to a log, and
//! sends the schema anyway: every request then fails with a 400 that names the
//! tool by its position in a list the user never sees.
//!
//! This runs the same two checks, from outside the agent, which is the only
//! place they are of use once every request inside it fails. It also reports a
//! third thing the vendor's list of gated features leaves out: a tool whose
//! schema has `anyOf`, `oneOf` or `allOf` at its root is normally rewritten into
//! one the API accepts, the rewrite needs remote configuration too, and without
//! it the tool is dropped without a word.
//!
//! Checked against the vendor's pages on [`VERIFIED_ON`](super::VERIFIED_ON):
//! <https://code.claude.com/docs/en/mcp#tools-with-invalid-input-schemas>

use serde_json::Value;

/// One tool as a server lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub input_schema: Value,
}

/// What is wrong with one tool's input schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Problem {
    /// The root `type` is not `"object"`. The agent's MCP client refuses the
    /// server's whole `tools/list` over it, so the server has no tools at all.
    /// Measured, and nothing to do with sovereign mode; it is here because
    /// "where did my tools go" is the question this command answers.
    NotAnObject,
    /// A top-level property name the API refuses. Every request fails.
    PropertyName(String),
    /// Not valid against the draft 2020-12 meta-schema. Every request fails.
    InvalidSchema(String),
    /// A combinator at the root. The tool is dropped from the session.
    RootCombinator(&'static str),
    /// Declares another dialect, so the agent never checks it against a
    /// meta-schema and neither can this. The API may still refuse it.
    OtherDialect(String),
}

impl Problem {
    /// Whether this makes every request fail, which is what a deny rule fixes.
    pub fn breaks_requests(&self) -> bool {
        matches!(self, Problem::PropertyName(_) | Problem::InvalidSchema(_))
    }

    /// Whether this fails the command: a session that cannot make a request,
    /// or a server that arrives with no tools.
    pub fn is_failure(&self) -> bool {
        self.breaks_requests() || matches!(self, Problem::NotAnObject)
    }
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Problem::NotAnObject => write!(
                f,
                "the schema's root `type` is not \"object\": the agent's MCP client refuses the server's whole tool list over it, so none of this server's tools load (in any mode; only the server can fix it)"
            ),
            Problem::PropertyName(name) => write!(
                f,
                "the property name {name:?} is refused: 1 to 64 of ASCII letters, digits, `_`, `.` and `-`"
            ),
            Problem::InvalidSchema(why) => {
                write!(f, "not valid JSON Schema draft 2020-12: {why}")
            }
            Problem::RootCombinator(keyword) => write!(
                f,
                "`{keyword}` at the root of the schema: in sovereign mode the agent skips the tool and says so only in a debug log; the server's other tools load"
            ),
            Problem::OtherDialect(dialect) => write!(
                f,
                "declares the dialect {dialect}, which the agent does not check against a meta-schema; the API may still refuse it"
            ),
        }
    }
}

const ROOT_COMBINATORS: [&str; 3] = ["anyOf", "oneOf", "allOf"];
const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

/// The API's rule for a top-level property name: `^[a-zA-Z0-9_.-]{1,64}$`.
pub fn is_acceptable_property_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

fn declared_dialect(schema: &Value) -> Option<&str> {
    schema.get("$schema").and_then(Value::as_str)
}

fn is_draft_2020_12(dialect: &str) -> bool {
    dialect.trim_end_matches('#') == DRAFT_2020_12
}

/// Everything wrong with one input schema, in the order the agent would meet
/// it. The first two end the matter: a list the client refuses never reaches
/// the agent, and a tool the agent skips never reaches the API.
pub fn problems(schema: &Value) -> Vec<Problem> {
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return vec![Problem::NotAnObject];
    }
    if let Some(keyword) = ROOT_COMBINATORS
        .into_iter()
        .find(|keyword| schema.get(keyword).is_some())
    {
        return vec![Problem::RootCombinator(keyword)];
    }
    let names = schema
        .get("properties")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|properties| properties.keys())
        .filter(|name| !is_acceptable_property_name(name))
        .map(|name| Problem::PropertyName(name.clone()));
    let dialect = match declared_dialect(schema) {
        Some(dialect) if !is_draft_2020_12(dialect) => {
            Some(Problem::OtherDialect(dialect.to_string()))
        }
        _ => jsonschema::draft202012::meta::validate(schema)
            .err()
            .map(|error| {
                let at = error.instance_path().to_string();
                let at = if at.is_empty() {
                    "the root".to_string()
                } else {
                    at
                };
                Problem::InvalidSchema(format!("{error} (at {at})"))
            }),
    };
    names.chain(dialect).collect()
}

/// The tools in a `tools/list` answer, however much of its envelope is left:
/// the JSON-RPC response, its `result`, or the bare list. MCP spells the schema
/// `inputSchema`; the API's spelling is accepted too.
pub fn tools_in(answer: &Value) -> anyhow::Result<Vec<Tool>> {
    let list = [
        answer.pointer("/result/tools"),
        answer.get("tools"),
        Some(answer),
    ]
    .into_iter()
    .flatten()
    .find_map(Value::as_array)
    .ok_or_else(|| anyhow::anyhow!("no list of tools in that: expected a tools/list answer"))?;
    list.iter()
        .map(|tool| {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("a tool without a name: {tool}"))?;
            let input_schema = tool
                .get("inputSchema")
                .or_else(|| tool.get("input_schema"))
                .cloned()
                .unwrap_or(Value::Null);
            Ok(Tool {
                name: name.to_string(),
                input_schema,
            })
        })
        .collect()
}

/// One tool with something wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub tool: String,
    pub problems: Vec<Problem>,
}

/// The tools with something wrong, in the server's order.
pub fn lint(tools: &[Tool]) -> Vec<Finding> {
    tools
        .iter()
        .map(|tool| Finding {
            tool: tool.name.clone(),
            problems: problems(&tool.input_schema),
        })
        .filter(|finding| !finding.problems.is_empty())
        .collect()
}

/// The name the agent gives a server's tool, which is what a permission rule
/// has to say. The agent replaces every character of the server's name outside
/// `[a-zA-Z0-9_-]` with `_` (`claude.ai Docs` becomes `claude_ai_Docs`).
pub fn rule_name(server: &str, tool: &str) -> String {
    let clean = |text: &str| -> String {
        text.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    };
    format!("mcp__{}__{}", clean(server), clean(tool))
}

/// How many tools fail the command: its exit status.
pub fn failures(findings: &[Finding]) -> usize {
    findings
        .iter()
        .filter(|f| f.problems.iter().any(Problem::is_failure))
        .count()
}

/// The report as text. It prints the deny rules and does not write them: the
/// settings file grants the agent its permissions, and is the user's to edit.
pub fn render(server: &str, checked: usize, findings: &[Finding]) -> String {
    let mut out = vec![format!("{server}: {checked} tool(s) checked")];
    if findings.is_empty() {
        out.push("nothing the API would refuse, and nothing the agent would drop".to_string());
        return out.join("\n");
    }
    for finding in findings {
        let mark = if finding.problems.iter().any(Problem::is_failure) {
            "FAIL"
        } else {
            "note"
        };
        out.push(format!("[{mark}] {}", finding.tool));
        out.extend(finding.problems.iter().map(|p| format!("       {p}")));
    }
    let deny: Vec<String> = findings
        .iter()
        .filter(|f| f.problems.iter().any(Problem::breaks_requests))
        .map(|f| format!("\"{}\"", rule_name(server, &f.tool)))
        .collect();
    if !deny.is_empty() {
        out.push(String::new());
        out.push(
            "A tool the API refuses fails every request that carries it, with a 400 that names it \
             only by position: at once when tools load upfront, and from the moment the agent \
             loads it under tool search. A deny rule on the bare tool name keeps it out of the \
             request. Add to permissions.deny in your settings file:"
                .to_string(),
        );
        out.push(format!("  {}", deny.join(", ")));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
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
            let schema =
                json!({ "type": "object", keyword: [{ "properties": { "bad name": {} } }] });
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
            },
            Tool {
                name: "breaks".into(),
                input_schema: json!({ "type": "object", "properties": { "a b": {} } }),
            },
            Tool {
                name: "dropped".into(),
                input_schema: json!({ "type": "object", "oneOf": [] }),
            },
            Tool {
                name: "rootless".into(),
                input_schema: json!({ "type": "array" }),
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
}
