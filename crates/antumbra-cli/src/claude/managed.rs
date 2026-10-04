//! The file-based managed settings tier: `managed-settings.json` and the
//! `managed-settings.d/` drop-ins beside it, an organization's policy, which
//! Claude Code applies above every other settings level. Two things in it
//! matter to the doctor: its `env` block, which outranks the user's, and a
//! sign-in forced through a Claude apps gateway, which skips the flag fetch the
//! way the switches do.
//!
//! Only the files are read. A policy delivered by MDM (a macOS configuration
//! profile, the Windows registry) is not, so on a machine managed that way the
//! doctor reports what the files say and no more.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The settings key a gateway sign-in is forced with. It stands among the
/// variables under this name, so detection and the report treat it like a
/// switch; no environment variable is spelled this way.
pub const GATEWAY_LOGIN: &str = "forceLoginMethod";

/// The system directory the managed files live in on this platform.
pub fn dir() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from(r"C:\Program Files\ClaudeCode")
    } else if cfg!(target_os = "macos") {
        PathBuf::from("/Library/Application Support/ClaudeCode")
    } else {
        PathBuf::from("/etc/claude-code")
    }
}

/// What the managed files say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Managed {
    /// The file the report names: `managed-settings.json`, or the drop-in
    /// directory when only drop-ins exist.
    pub path: PathBuf,
    /// The merged `env` block, and the gateway sign-in under
    /// [`GATEWAY_LOGIN`] when one is forced.
    pub variables: BTreeMap<String, String>,
}

/// Read the managed files in `dir`, merged as Claude Code merges them:
/// `managed-settings.json` first, then every `*.json` in `managed-settings.d/`
/// in name order, hidden files left out, a later file's keys over an earlier
/// one's. `None` when there are none. A file that does not parse contributes
/// nothing, as a settings file that does not parse contributes nothing.
pub fn read(dir: &Path) -> Option<Managed> {
    let main = dir.join("managed-settings.json");
    let drop_ins = dir.join("managed-settings.d");
    let mut parts: Vec<PathBuf> = drop_ins
        .read_dir()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            !name.starts_with('.') && name.ends_with(".json")
        })
        .collect();
    parts.sort();
    let named = if main.is_file() {
        main.clone()
    } else if !parts.is_empty() {
        drop_ins
    } else {
        return None;
    };
    let mut merged = serde_json::Map::new();
    for path in std::iter::once(main).chain(parts) {
        let Some(serde_json::Value::Object(object)) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
        else {
            continue;
        };
        for (key, value) in object {
            match (merged.get_mut(&key), value) {
                // The env block merges name by name, like the files do.
                (Some(serde_json::Value::Object(have)), serde_json::Value::Object(more))
                    if key == "env" =>
                {
                    have.extend(more);
                }
                (_, value) => {
                    merged.insert(key, value);
                }
            }
        }
    }
    Some(Managed {
        path: named,
        variables: variables(&merged),
    })
}

/// The `env` block's string values, and the forced gateway sign-in: a
/// `forceLoginMethod` of `"gateway"` with a gateway URL to sign in to, which
/// is what the vendor's page says a gateway session takes.
fn variables(settings: &serde_json::Map<String, serde_json::Value>) -> BTreeMap<String, String> {
    let mut variables: BTreeMap<String, String> = settings
        .get("env")
        .and_then(serde_json::Value::as_object)
        .map(|block| {
            block
                .iter()
                .filter_map(|(name, value)| value.as_str().map(|v| (name.clone(), v.to_string())))
                .collect()
        })
        .unwrap_or_default();
    let text = |key: &str| settings.get(key).and_then(serde_json::Value::as_str);
    if text(GATEWAY_LOGIN) == Some("gateway")
        && text("forceLoginGatewayUrl").is_some_and(|url| !url.trim().is_empty())
    {
        variables.insert(GATEWAY_LOGIN.to_string(), "gateway".to_string());
    }
    variables
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "antumbra-managed-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(dir.join("managed-settings.d")).expect("temp dir");
        dir
    }

    fn write(path: PathBuf, text: &str) {
        std::fs::write(path, text).expect("write");
    }

    /// A forced gateway sign-in turns the flags off like the switches do, and
    /// so does Mantle, Bedrock's other endpoint; both read from the variables.
    #[test]
    fn a_gateway_session_and_mantle_are_sovereign() {
        let origin = super::super::Origin::Settings(dir().join("managed-settings.json"));
        let vars = |pairs: &[(&str, &str)]| {
            pairs
                .iter()
                .map(|(k, v)| {
                    (
                        k.to_string(),
                        super::super::Variable {
                            value: v.to_string(),
                            origin: origin.clone(),
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };
        let named = |pairs: &[(&str, &str)]| -> Vec<String> {
            super::super::detect(&vars(pairs))
                .into_iter()
                .map(|t| t.variable)
                .collect()
        };
        assert_eq!(named(&[(GATEWAY_LOGIN, "gateway")]), [GATEWAY_LOGIN]);
        assert!(named(&[(GATEWAY_LOGIN, "claudeai")]).is_empty());
        assert_eq!(
            named(&[("CLAUDE_CODE_USE_MANTLE", "1")]),
            ["CLAUDE_CODE_USE_MANTLE"]
        );
        assert!(named(&[
            ("CLAUDE_CODE_USE_MANTLE", "1"),
            ("CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST", "1")
        ])
        .is_empty());
    }

    #[test]
    fn no_managed_files_is_none() {
        let dir = temp();
        assert_eq!(read(&dir), None);
        write(dir.join("managed-settings.d").join("notes.txt"), "{}");
        assert_eq!(read(&dir), None, "only *.json drop-ins count");
    }

    /// The main file first, then the drop-ins in name order, each over the
    /// one before; a hidden file or one that does not parse adds nothing.
    #[test]
    fn the_files_merge_in_order_and_env_merges_by_name() {
        let dir = temp();
        write(
            dir.join("managed-settings.json"),
            r#"{"env": {"DISABLE_TELEMETRY": "1", "A": "main"}}"#,
        );
        let d = dir.join("managed-settings.d");
        write(d.join("20-later.json"), r#"{"env": {"A": "twenty"}}"#);
        write(
            d.join("10-first.json"),
            r#"{"env": {"A": "ten", "B": "ten"}}"#,
        );
        write(d.join(".hidden.json"), r#"{"env": {"A": "hidden"}}"#);
        write(d.join("30-broken.json"), "{ not json");
        let managed = read(&dir).expect("managed");
        assert_eq!(managed.path, dir.join("managed-settings.json"));
        let got: Vec<(&str, &str)> = managed
            .variables
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(
            got,
            [("A", "twenty"), ("B", "ten"), ("DISABLE_TELEMETRY", "1")]
        );
    }

    /// A gateway sign-in counts only with a URL to sign in to: the method
    /// alone leaves the developer at a "contact your administrator" message.
    #[test]
    fn a_forced_gateway_sign_in_is_read_only_with_its_url() {
        let dir = temp();
        let d = dir.join("managed-settings.d");
        write(
            d.join("10-login.json"),
            r#"{"forceLoginMethod": "gateway"}"#,
        );
        let managed = read(&dir).expect("managed");
        assert_eq!(managed.path, d, "only drop-ins: the directory is named");
        assert!(!managed.variables.contains_key(GATEWAY_LOGIN));
        write(
            d.join("20-url.json"),
            r#"{"forceLoginGatewayUrl": "https://gateway.internal.example"}"#,
        );
        assert_eq!(
            read(&dir)
                .expect("managed")
                .variables
                .get(GATEWAY_LOGIN)
                .map(String::as_str),
            Some("gateway")
        );
        write(
            d.join("30-claudeai.json"),
            r#"{"forceLoginMethod": "claudeai"}"#,
        );
        assert!(!read(&dir)
            .expect("managed")
            .variables
            .contains_key(GATEWAY_LOGIN));
    }
}
