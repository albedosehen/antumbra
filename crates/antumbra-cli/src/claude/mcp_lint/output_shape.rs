use super::*;
use serde_json::json;

/// The shape this lint is about: rows of prose nobody bounded. It is the
/// shape `recall_memories` itself had before its answers were bounded.
#[test]
fn rows_of_unbounded_prose_are_flagged_with_the_path() {
    let schema = json!({
        "type": "object",
        "properties": {
            // Keeps this fixture about the text bound: without a sibling
            // that explains an empty result it would also trip the
            // empty-state rule, and a test that asserts two rules at once
            // stops naming which one broke.
            "matched": { "type": "integer" },
            "memories": {
                "type": "array",
                "maxItems": 50,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "maxLength": 64 },
                        "content": { "type": "string" }
                    }
                }
            }
        }
    });
    let found = shape_problems(&schema);
    assert_eq!(
        found,
        vec![Problem::UnboundedText(".memories[].content".into())],
        "the unbounded field is named by its path, and the bounded id is not"
    );
    assert!(found[0].is_shape(), "a shape problem is a shape problem");
    assert!(
        !found[0].is_failure(),
        "shape costs context; it never fails a request, so it must not fail the command"
    );
}

/// A description counts as a bound, because Antumbra's rule is that a bound
/// may be lifted by one documented call and no schema can express "900
/// unless you asked for full". This is how `MemoryView::content` passes.
#[test]
fn a_field_that_says_what_bounds_it_passes() {
    let schema = json!({
        "type": "object",
        "properties": {
            "matched": { "type": "integer" },
            "memories": {
                "type": "array",
                "maxItems": 50,
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "cut to a 900-character prefix unless the call asked for `full`"
                        }
                    }
                }
            }
        }
    });
    assert!(shape_problems(&schema).is_empty());
}

/// A collection that never says how many it returns.
#[test]
fn a_collection_with_no_stated_size_is_flagged() {
    let schema = json!({
        "type": "object",
        "properties": {
            "matched": { "type": "integer" },
            "rows": {
                "type": "array",
                "items": { "type": "object", "properties": {} }
            }
        }
    });
    assert_eq!(
        shape_problems(&schema),
        vec![Problem::UnboundedCollection(".rows".into())]
    );
}

/// Generated schemas put the row type in `$defs` and point at it, so a walk
/// that does not follow `$ref` sees nothing at all. This is the shape
/// `recall_memories` actually publishes.
#[test]
fn a_row_behind_a_ref_is_still_walked() {
    let schema = json!({
        "$defs": {
            "Row": {
                "type": "object",
                "properties": { "prose": { "type": "string" } }
            }
        },
        "type": "object",
        "properties": {
            "matched": { "type": "integer" },
            "rows": {
                "type": "array",
                "maxItems": 10,
                "items": { "$ref": "#/$defs/Row" }
            }
        }
    });
    assert_eq!(
        shape_problems(&schema),
        vec![Problem::UnboundedText(".rows[].prose".into())],
        "the walk follows a local $ref or it checks nothing a generator emits"
    );
}

/// The lint's third rule, recall's rule for an empty answer applied to any
/// tool. A bare collection answers "no rows" and never "no rows BECAUSE", so
/// the agent cannot tell a query that matched nothing from one whose matches a
/// floor removed.
#[test]
fn a_collection_with_nothing_to_explain_an_empty_one_is_flagged() {
    let schema = json!({
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "maxItems": 50,
                "items": { "type": "object", "properties": {} }
            }
        }
    });
    assert_eq!(
        shape_problems(&schema),
        vec![Problem::IndistinguishableEmpty(".memories".into())]
    );
}

/// The shape recall's relevance floor actually landed in `antumbra-mcp` as: a
/// boolean beside the rows that is true exactly when something was retrieved
/// and then floored. This is the fixture that says what the rule is asking for.
#[test]
fn a_boolean_beside_the_rows_passes() {
    let schema = json!({
        "type": "object",
        "properties": {
            "nothing_cleared_the_floor": { "type": "boolean" },
            "memories": {
                "type": "array",
                "maxItems": 50,
                "items": { "type": "object", "properties": {} }
            }
        }
    });
    assert!(shape_problems(&schema).is_empty());
}

/// An enumerated status carries the same news as a boolean, and is how a
/// server with more than two outcomes would say it.
#[test]
fn an_enumerated_status_passes() {
    let schema = json!({
        "type": "object",
        "properties": {
            "status": { "enum": ["ok", "no_match", "all_filtered"] },
            "rows": {
                "type": "array",
                "maxItems": 20,
                "items": { "type": "object", "properties": {} }
            }
        }
    });
    assert!(shape_problems(&schema).is_empty());
}

/// The documented escape hatch, matching the posture of the rules beside
/// this one: a server may say in prose what a schema cannot express.
#[test]
fn a_description_that_says_what_empty_means_passes() {
    let schema = json!({
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "maxItems": 20,
                "description": "empty only when nothing matched; filtered matches are returned with a lower score",
                "items": { "type": "object", "properties": {} }
            }
        }
    });
    assert!(shape_problems(&schema).is_empty());
}

/// A server says absence in its own vocabulary. This is `store_memory`'s
/// `auto_proposed` verbatim, which explains itself completely and which the
/// first version of this rule flagged for not using the word "empty".
#[test]
fn a_field_that_explains_its_absence_in_other_words_passes() {
    let schema = json!({
        "type": "object",
        "properties": {
            "auto_proposed": {
                "type": "array",
                "maxItems": 8,
                "description": "Compartment ids the antumbra auto-created from the inbox on this write (only when the autonomous propose trigger is enabled and fired).",
                "items": { "type": "string", "maxLength": 64 }
            }
        }
    });
    assert!(shape_problems(&schema).is_empty());
}

/// A description that says HOW MANY must not silence a rule about WHY NONE.
/// The two rules share a keyword and ask different questions, so they read
/// the description differently on purpose.
#[test]
fn a_description_about_size_does_not_answer_the_empty_question() {
    let schema = json!({
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "description": "at most 20 rows",
                "items": { "type": "object", "properties": {} }
            }
        }
    });
    assert_eq!(
        shape_problems(&schema),
        vec![Problem::IndistinguishableEmpty(".rows".into())],
        "the size description satisfies the bound rule and not this one"
    );
}

/// The converse, and the reason each rule reads the description for its own
/// question: saying what an empty result means must not pass off as saying
/// how many a full one returns, or one sentence quiets the whole report.
#[test]
fn a_description_about_emptiness_does_not_answer_the_size_question() {
    let schema = json!({
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "description": "empty when nothing matched",
                "items": { "type": "object", "properties": {} }
            }
        }
    });
    assert_eq!(
        shape_problems(&schema),
        vec![Problem::UnboundedCollection(".rows".into())],
        "the empty-state description satisfies that rule and not the bound"
    );
}

/// Only RESULT collections. An empty `tags` inside a row is an ordinary
/// absence rather than an ambiguous answer, and flagging every one of them
/// would bury the case that matters under noise.
#[test]
fn an_empty_collection_inside_a_row_is_not_flagged() {
    let schema = json!({
        "type": "object",
        "properties": {
            "matched": { "type": "integer" },
            "rows": {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "maxItems": 8,
                            "items": { "type": "string", "maxLength": 32 }
                        }
                    }
                }
            }
        }
    });
    assert!(shape_problems(&schema).is_empty());
}

/// A collection that cannot be empty has no empty state to describe.
#[test]
fn a_collection_with_a_minimum_is_not_flagged() {
    let schema = json!({
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "minItems": 1,
                "maxItems": 10,
                "items": { "type": "object", "properties": {} }
            }
        }
    });
    assert!(shape_problems(&schema).is_empty());
}

/// A string OUTSIDE a collection is one field once, not one per row, so it
/// is not the hazard this rule is about and must not be reported.
#[test]
fn a_scalar_string_outside_a_row_is_not_flagged() {
    let schema = json!({
        "type": "object",
        "properties": { "title": { "type": "string" } }
    });
    assert!(shape_problems(&schema).is_empty());
}

/// A schema that refers to itself must not spin the walk.
#[test]
fn a_self_referential_schema_terminates() {
    let schema = json!({
        "$defs": {
            "Node": {
                "type": "object",
                "properties": { "child": { "$ref": "#/$defs/Node" } }
            }
        },
        "type": "object",
        "properties": { "root": { "$ref": "#/$defs/Node" } }
    });
    let _ = shape_problems(&schema);
}

/// A server that publishes no output schema is not linted for shape: a
/// shape nobody declared cannot be checked without calling the tool.
#[test]
fn a_tool_without_an_output_schema_is_left_alone() {
    let tools = [Tool {
        name: "quiet".into(),
        input_schema: json!({ "type": "object" }),
        output_schema: None,
    }];
    assert!(lint(&tools).is_empty());
}
