//! Filling in `docker/.env` for the local stack.
//!
//! The file is the user's once it exists, so a value already in it stays, as
//! long as it is a real one. What setup writes is only what is missing and
//! what is still the example's placeholder (`change-me-...`, or the example
//! data path), and it writes it in place, keeping every comment and line
//! around it. With no `.env` yet, it starts from `.env.example`, comments
//! and all.

/// One value setup wants in the file.
#[derive(Debug, Clone)]
pub struct Want {
    pub name: &'static str,
    pub value: String,
}

/// Values in `.env.example` that are there to be replaced.
fn placeholder(value: &str) -> bool {
    let value = value.trim().trim_matches('"').trim_matches('\'');
    value.is_empty() || value.starts_with("change-me") || value == "/srv/antumbra/surrealdb"
}

/// The value a line sets for `name`, when it sets it (comments excluded).
fn value_for<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let line = line.trim_start();
    let rest = line.strip_prefix(name)?;
    rest.trim_start().strip_prefix('=')
}

/// The value `text` sets for `name`, if any line sets it (the last one wins,
/// as it does for compose).
pub fn get<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.lines()
        .rev()
        .find_map(|line| value_for(line, name))
        .map(|v| v.trim().trim_matches('"').trim_matches('\''))
}

/// `existing` (the current `.env`, or `None`) with `wants` filled in, starting
/// from `example` when there is no file yet. Returns the new text and the
/// names it wrote.
pub fn fill(existing: Option<&str>, example: &str, wants: &[Want]) -> (String, Vec<&'static str>) {
    let base = existing.unwrap_or(example);
    let mut lines: Vec<String> = base.lines().map(str::to_string).collect();
    let mut wrote = Vec::new();
    for want in wants {
        let at = lines
            .iter()
            .rposition(|line| value_for(line, want.name).is_some());
        match at {
            Some(i) => {
                if placeholder(value_for(&lines[i], want.name).unwrap_or_default()) {
                    lines[i] = format!("{}={}", want.name, want.value);
                    wrote.push(want.name);
                }
            }
            None => {
                lines.push(format!("{}={}", want.name, want.value));
                wrote.push(want.name);
            }
        }
    }
    let mut text = lines.join("\n");
    text.push('\n');
    (text, wrote)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = "# the stack's secrets\nSURREAL_PASS=change-me-to-a-strong-password\n\n# HS256\nANTUMBRA_JWT_SECRET=change-me-base64-32-bytes\nANTUMBRA_SURREAL_DATA=/srv/antumbra/surrealdb\nANTUMBRA_EMBEDDER_MODEL=all-minilm\n";

    fn wants() -> Vec<Want> {
        vec![
            Want {
                name: "SURREAL_PASS",
                value: "pass1".into(),
            },
            Want {
                name: "ANTUMBRA_JWT_SECRET",
                value: "secret1".into(),
            },
            Want {
                name: "ANTUMBRA_SURREAL_DATA",
                value: "C:/Users/me/.antumbra/surrealdb".into(),
            },
        ]
    }

    #[test]
    fn a_new_file_starts_from_the_example_with_its_comments() {
        let (text, wrote) = fill(None, EXAMPLE, &wants());
        assert!(text.contains("# the stack's secrets\nSURREAL_PASS=pass1\n"));
        assert!(text.contains("ANTUMBRA_JWT_SECRET=secret1\n"));
        assert!(text.contains("ANTUMBRA_SURREAL_DATA=C:/Users/me/.antumbra/surrealdb\n"));
        assert!(text.contains("ANTUMBRA_EMBEDDER_MODEL=all-minilm\n"));
        assert_eq!(
            wrote,
            vec![
                "SURREAL_PASS",
                "ANTUMBRA_JWT_SECRET",
                "ANTUMBRA_SURREAL_DATA"
            ]
        );
    }

    #[test]
    fn a_real_value_already_there_is_kept() {
        let existing = "SURREAL_PASS=mine\nANTUMBRA_JWT_SECRET=change-me-base64-32-bytes\n";
        let (text, wrote) = fill(Some(existing), EXAMPLE, &wants());
        assert!(text.contains("SURREAL_PASS=mine\n"));
        assert!(text.contains("ANTUMBRA_JWT_SECRET=secret1\n"));
        assert!(
            text.contains("ANTUMBRA_SURREAL_DATA=C:/Users/me/.antumbra/surrealdb\n"),
            "appended"
        );
        assert_eq!(wrote, vec!["ANTUMBRA_JWT_SECRET", "ANTUMBRA_SURREAL_DATA"]);
        assert_eq!(get(&text, "SURREAL_PASS"), Some("mine"));
    }

    #[test]
    fn a_second_run_writes_nothing() {
        let (once, _) = fill(None, EXAMPLE, &wants());
        let (twice, wrote) = fill(Some(&once), EXAMPLE, &wants());
        assert_eq!(once, twice);
        assert!(wrote.is_empty());
    }

    #[test]
    fn a_commented_line_is_not_a_value() {
        assert_eq!(get("# SURREAL_PASS=x\n", "SURREAL_PASS"), None);
        assert_eq!(get("SURREAL_PASS_FILE=x\n", "SURREAL_PASS"), None);
        assert_eq!(get("SURREAL_PASS = \"x\"\n", "SURREAL_PASS"), Some("x"));
    }
}
