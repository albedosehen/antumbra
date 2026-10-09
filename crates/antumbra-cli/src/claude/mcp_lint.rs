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
    /// The tool's declared output shape, when it publishes one. `tools/list`
    /// carries `outputSchema` beside `inputSchema` in current MCP; a server that
    /// omits it is not linted for shape, because a shape nobody declared cannot
    /// be checked without calling the tool, and a lint must not call tools.
    pub output_schema: Option<Value>,
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
    /// A string field inside a collection row, with neither a `maxLength` nor a
    /// description that says what bounds it (ADR-0023 B-3). One field of prose a
    /// caller sized earlier, times however many rows come back, is the shape
    /// that empties a context window. The string names the path to the field.
    UnboundedText(String),
    /// An array with no `maxItems` and nothing in its description about how many
    /// it returns (ADR-0023 B-3). A caller cannot budget for a collection whose
    /// size is stated nowhere.
    UnboundedCollection(String),
    /// A collection that can come back empty, with nothing beside it to say why
    /// (ADR-0023 B-3, generalizing B-2). An empty array answers "no rows" and
    /// not "no rows BECAUSE", so the agent cannot tell a query that matched
    /// nothing from one whose matches a filter or a threshold removed. The two
    /// want opposite responses — rephrase, or widen — and guessing wrong costs a
    /// round trip each time. The string names the path to the collection.
    IndistinguishableEmpty(String),
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

    /// Whether this is about what a tool RETURNS rather than what it accepts.
    ///
    /// Shape problems cost context; they never fail a request. So they are
    /// reported and never counted as failures, which is the same posture the
    /// schema checks take and the reason this is a lint rather than a refusal.
    pub fn is_shape(&self) -> bool {
        matches!(
            self,
            Problem::UnboundedText(_)
                | Problem::UnboundedCollection(_)
                | Problem::IndistinguishableEmpty(_)
        )
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
            Problem::UnboundedText(path) => write!(
                f,
                "returns `{path}`, prose with no `maxLength` and no description saying what bounds it: one such field per row, times every row, is what empties a context window (give it a bound, or say in its description what does)"
            ),
            Problem::UnboundedCollection(path) => write!(
                f,
                "returns `{path}` without `maxItems` or any description of how many: a caller cannot budget for a collection whose size is stated nowhere"
            ),
            Problem::IndistinguishableEmpty(path) => write!(
                f,
                "returns `{path}` empty with nothing beside it to say why: an agent cannot tell \"nothing matched\" from \"matches were filtered out\", and those want opposite next moves (add a flag, a status or a count alongside it, or say in its description what an empty one means)"
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
            let output_schema = tool
                .get("outputSchema")
                .or_else(|| tool.get("output_schema"))
                .filter(|s| !s.is_null())
                .cloned();
            Ok(Tool {
                name: name.to_string(),
                input_schema,
                output_schema,
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

/// What a tool's declared OUTPUT shape will cost a context window (ADR-0023
/// B-3): unbounded prose inside a collection row, and collections that never
/// say how many rows they return.
///
/// Walks the schema rather than looking only at the root, because the shape that
/// matters is a string nested inside an array's items, and `$defs` indirection
/// is how every generated schema expresses exactly that. Resolves local `$ref`s
/// against the document for the same reason, and bounds the walk so a schema
/// that refers to itself cannot spin.
///
/// A field counts as bounded by a `maxLength`, or by a description that says
/// what bounds it. The second is not a loophole: ADR-0023's rule is that a bound
/// may be lifted by one documented call, and a schema cannot express "900 unless
/// you asked for `full`". Saying so in the description is the accurate form, so
/// the lint accepts it and the operator reads it.
pub fn shape_problems(schema: &Value) -> Vec<Problem> {
    fn described(node: &Value) -> bool {
        node.get("description")
            .and_then(Value::as_str)
            .is_some_and(|d| !d.trim().is_empty())
    }
    // Deliberately NOT `described`. A description that says how many rows come
    // back satisfies the bound rule and says nothing about what no rows means,
    // so reusing it would let one sentence silence two different questions.
    //
    // More than one word, because a server says absence in its own vocabulary
    // and the first version of this rule flagged a field that explains itself
    // perfectly ("only when the autonomous propose trigger is enabled and
    // fired") for not using the word "empty".
    // The size rule reads the description for a SIZE, for the same reason the
    // empty rule reads it for an absence: otherwise adding a sentence about one
    // question silences the other, and a server can quiet the whole report with
    // prose that answers neither. A string field's escape hatch stays lenient,
    // because "cut to a prefix unless you asked for `full`" is the bound.
    fn describes_size(node: &Value) -> bool {
        const SIZE: [&str; 6] = [
            "at most",
            "up to",
            "no more than",
            "limit",
            "top_k",
            "default",
        ];
        node.get("description")
            .and_then(Value::as_str)
            .map(|d| d.to_ascii_lowercase())
            .is_some_and(|d| {
                d.chars().any(|c| c.is_ascii_digit()) || SIZE.iter().any(|w| d.contains(w))
            })
    }
    fn describes_empty(node: &Value) -> bool {
        const ABSENCE: [&str; 5] = ["empty", "absent", "omitted", "none", "only when"];
        node.get("description")
            .and_then(Value::as_str)
            .map(|d| d.to_ascii_lowercase())
            .is_some_and(|d| ABSENCE.iter().any(|w| d.contains(w)))
    }
    fn resolve<'a>(node: &'a Value, root: &'a Value) -> &'a Value {
        node.get("$ref")
            .and_then(Value::as_str)
            .and_then(|r| r.strip_prefix('#'))
            .and_then(|p| root.pointer(p))
            .unwrap_or(node)
    }
    fn walk(
        node: &Value,
        root: &Value,
        path: &str,
        in_row: bool,
        depth: usize,
        out: &mut Vec<Problem>,
    ) {
        if depth > 12 {
            return;
        }
        let node = resolve(node, root);
        let ty = node.get("type").and_then(Value::as_str);

        if ty == Some("array") {
            let unbounded = node.get("maxItems").is_none() && !describes_size(node);
            if unbounded {
                out.push(Problem::UnboundedCollection(path.to_string()));
            }
            if let Some(items) = node.get("items") {
                walk(items, root, &format!("{path}[]"), true, depth + 1, out);
            }
            return;
        }

        // A string is only a context hazard where it repeats: one field in one
        // row is small, the same field across every row is the whole payload.
        if ty == Some("string") && in_row {
            // `enum`, `const` and `format` bound a field as surely as a
            // `maxLength` does: an enumerated network name or an RFC3339
            // timestamp cannot be prose a caller sized. Without these the rule
            // flags every identifier in every row, and a report that names
            // thirty fields gets read as noise and then not read at all.
            let bounded = node.get("maxLength").is_some()
                || node.get("enum").is_some()
                || node.get("const").is_some()
                || node.get("format").is_some()
                || described(node);
            if !bounded {
                out.push(Problem::UnboundedText(path.to_string()));
            }
            return;
        }

        if let Some(props) = node.get("properties").and_then(Value::as_object) {
            // An empty collection explains itself only if something BESIDE it
            // varies with the reason it is empty. A boolean, an enumerated
            // status and a count all can — `nothing_cleared_the_floor` is the
            // shape B-2 landed, and a `total` that disagrees with the rows
            // carries the same news. A second collection cannot.
            let explained_by_sibling = props.values().any(|p| {
                let p = resolve(p, root);
                matches!(
                    p.get("type").and_then(Value::as_str),
                    Some("boolean" | "integer" | "number")
                ) || p.get("enum").is_some()
            });
            for (name, child) in props {
                let child_path = format!("{path}.{name}");
                let resolved = resolve(child, root);
                // Only RESULT collections. An empty `tags` inside a row is an
                // ordinary absence, not an ambiguous answer, and flagging every
                // one of them would bury the case that matters.
                let is_result_collection = !in_row
                    && resolved.get("type").and_then(Value::as_str) == Some("array")
                    && resolved
                        .get("minItems")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        == 0;
                if is_result_collection && !explained_by_sibling && !describes_empty(resolved) {
                    out.push(Problem::IndistinguishableEmpty(child_path.clone()));
                }
                walk(child, root, &child_path, in_row, depth + 1, out);
            }
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(list) = node.get(key).and_then(Value::as_array) {
                for child in list {
                    walk(child, root, path, in_row, depth + 1, out);
                }
            }
        }
    }

    let mut out = Vec::new();
    walk(schema, schema, "", false, 0, &mut out);
    out
}

/// The tools with something wrong, in the server's order.
pub fn lint(tools: &[Tool]) -> Vec<Finding> {
    tools
        .iter()
        .map(|tool| {
            let mut problems = problems(&tool.input_schema);
            if let Some(out) = &tool.output_schema {
                problems.extend(shape_problems(out));
            }
            Finding {
                tool: tool.name.clone(),
                problems,
            }
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

/// The tools whose OUTPUT shape will cost a context window (ADR-0023 B-3).
///
/// Counted apart from [`failures`] on purpose: a shape problem is a bill, not a
/// break. The command's exit status stays tied to what stops a session working,
/// so adding these rules to an existing lint cannot start failing a server that
/// was passing.
pub fn shaped(findings: &[Finding]) -> usize {
    findings
        .iter()
        .filter(|f| f.problems.iter().any(Problem::is_shape))
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
        } else if finding.problems.iter().all(Problem::is_shape) {
            // Marked apart from the schema notes: this one costs context rather
            // than breaking anything, and the remedy is the server's to apply.
            "cost"
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
    let shape = shaped(findings);
    if shape > 0 {
        out.push(String::new());
        out.push(format!(
            "{shape} tool(s) return a shape that costs you something you did not budget: prose \
             no one bounded, a collection that never says how many rows it returns, or a \
             collection that comes back empty without saying why. The first two spend context; \
             the third spends a round trip, because an agent that cannot tell \"nothing matched\" \
             from \"everything was filtered\" has to guess which one to retry. Nothing here fails \
             a request, so nothing here fails this command. Only the server can fix it, and a \
             server that returns rows of unbounded text will empty a window whatever the agent \
             does with it."
        ));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests;

/// ADR-0023 B-3, validation 6: the lint flags a server whose collection rows
/// carry an unbounded text field, demonstrated against a fixture.
#[cfg(test)]
mod output_shape;

#[cfg(test)]
mod shape_counting;

/// The lint pointed at a real server's `tools/list`, when one is on disk.
///
/// ADR-0023 B-3 argues these checks are worth more pointed OUTWARD than inward,
/// at servers Antumbra did not write. This is the inward half of that claim, and
/// it is the one that can regress: it runs against a captured `tools/list` from
/// a live antumbra-mcp and reports what the rules find, so a change to the
/// surface that starts spending a caller's context shows up here.
///
/// Ignored because it needs the capture. Take one with:
///   ANTUMBRA_TOOLS_LIST=/path/to/tools-list.json
/// holding the `result` object from a `tools/list` answer.
#[cfg(test)]
mod against_a_real_server;
