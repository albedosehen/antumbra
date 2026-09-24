use super::*;
use serde_json::json;

/// Adding output-shape rules to an existing lint must not start failing a
/// server that was passing: a bill is not a break.
#[test]
fn a_shape_problem_is_counted_apart_from_a_failure() {
    let tools = [Tool {
        name: "chatty".into(),
        input_schema: json!({ "type": "object" }),
        output_schema: Some(json!({
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": { "prose": { "type": "string" } }
                    }
                }
            }
        })),
    }];
    let findings = lint(&tools);
    assert_eq!(findings.len(), 1, "the tool is reported");
    assert_eq!(shaped(&findings), 1, "as a shape problem");
    assert_eq!(failures(&findings), 0, "and not as a failure");
}
