//! The console finds what the agent's connection is configured with, in a
//! home built for each test, so no real setting is read or written.

use super::*;
use std::collections::HashMap;
use std::path::Path;

/// An empty home of its own under the temp directory.
fn home(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "antumbra-tui-settings-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn places(home: &Path, env: &[(&str, &str)]) -> Places {
    let env: HashMap<String, String> = env
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Places {
        env: Box::new(move |name| env.get(name).cloned()),
        home: Some(home.to_path_buf()),
    }
}

/// A token with these claims; its signature is never checked here.
fn token(claims: &str) -> String {
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    format!(
        "{}.{}.sig",
        engine.encode(r#"{"alg":"HS256"}"#),
        engine.encode(claims)
    )
}

#[test]
fn the_server_is_the_one_the_agent_uses_unless_one_is_named() {
    let h = home("surface");
    let p = places(&h, &[]);
    assert_eq!(surface(None, &p).value, DEFAULT_SURFACE);
    assert_eq!(surface(None, &p).from, DEFAULT_FROM);

    write(
        &h.join(".antumbra").join("setup.json"),
        r#"{"mode":"hosted","url":"https://ant.example.com/"}"#,
    );
    let found = surface(None, &p);
    assert_eq!(
        (found.value.as_str(), found.from.as_str()),
        ("https://ant.example.com", "antumbra setup")
    );

    write(
        &h.join(".claude").join("settings.json"),
        r#"{"env":{"ANTUMBRA_URL":"http://10.0.0.132:8081"}}"#,
    );
    let found = surface(None, &p);
    assert_eq!(
        (found.value.as_str(), found.from.as_str()),
        ("http://10.0.0.132:8081", "Claude Code's settings")
    );

    let with_env = places(&h, &[("ANTUMBRA_URL", "http://env:8081")]);
    assert_eq!(surface(None, &with_env).value, "http://env:8081");
    assert_eq!(
        surface(Some("http://flag:8081".into()), &with_env).from,
        "--surface"
    );
}

#[test]
fn claude_codes_settings_are_read_where_it_keeps_them() {
    let h = home("config-dir");
    let elsewhere = h.join("elsewhere");
    write(
        &elsewhere.join("settings.json"),
        r#"{"env":{"ANTUMBRA_URL":"http://moved:8081"}}"#,
    );
    let p = places(&h, &[("CLAUDE_CONFIG_DIR", elsewhere.to_str().unwrap())]);
    assert_eq!(surface(None, &p).value, "http://moved:8081");
}

#[test]
fn an_operators_token_on_hand_is_preferred() {
    let h = home("operator");
    let plain = token(r#"{"tenant":"ws:a","user":"user:a"}"#);
    let operator = token(r#"{"tenant":"ws:a","user":"user:a","operator":true}"#);
    assert!(is_operator(&operator) && !is_operator(&plain));

    // A hook token in the environment and an operator's saved to the file:
    // the console takes the operator's, so it can act.
    write(
        &h.join(".antumbra").join("token.txt"),
        &format!("{operator}\n"),
    );
    let p = places(&h, &[("ANTUMBRA_TOKEN", &plain)]);
    let found = token_of(&p, None);
    assert_eq!(found.value, operator);
    assert_eq!(found.from, "~/.antumbra/token.txt");

    // A token given on the command line is used as given.
    assert_eq!(token_of(&p, Some(&plain)).from, "--token");
}

#[test]
fn with_no_operators_token_the_first_one_found_is_used() {
    let h = home("plain");
    let env_token = token(r#"{"tenant":"ws:a","user":"user:a"}"#);
    let file_token = token(r#"{"tenant":"ws:a","user":"user:b"}"#);
    write(&h.join(".antumbra").join("token.txt"), &file_token);
    let p = places(&h, &[("ANTUMBRA_TOKEN", &env_token)]);
    assert_eq!(token_of(&p, None).from, "ANTUMBRA_TOKEN");

    let from_settings = home("settings-token");
    write(
        &from_settings.join(".claude").join("settings.json"),
        &format!(r#"{{"env":{{"ANTUMBRA_TOKEN":"{env_token}"}}}}"#),
    );
    let found = token_of(&places(&from_settings, &[]), None);
    assert_eq!(found.from, "ANTUMBRA_TOKEN in Claude Code's settings");
}

#[test]
fn no_token_anywhere_says_how_to_get_one() {
    let h = home("none");
    let err = super::token(None, &places(&h, &[]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("antumbra setup"), "{err}");
}

fn token_of(p: &Places, given: Option<&str>) -> Found {
    super::token(given.map(str::to_string), p).unwrap()
}
