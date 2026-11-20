//! OpenCode: one plugin file of Antumbra's own.
//!
//! OpenCode has no hooks that run a command. It has plugins: JavaScript it
//! loads from `plugins/` in its config directory. And it has two release
//! lines that share that directory and a binary name, 1.x and 2.x, whose
//! plugins are written differently.
//!
//! So Antumbra is one file there, `plugins/antumbra.js`, whose default export
//! is both: `server` for 1.x and `setup` for 2.x. It runs
//! `antumbra hook opencode <event>` at the three moments, puts what the hook
//! answers in the session's system prompt, and registers the bridge as an MCP
//! server from inside the plugin, which both lines allow. Nothing is added to
//! `opencode.json`, and connecting again replaces the file.
//!
//! The plugin's source is `scripts/hooks/antumbra-opencode.js`, bundled here;
//! setup writes it with one line changed, the one that says how this machine
//! runs `antumbra`.
//!
//! Started by Unity Gateway (`ug opencode`), OpenCode has another config
//! directory, and loads the plugins there and none of these. So on a machine
//! `ug` is on, the same file is written there too.

use std::path::PathBuf;

use super::{gateway, keep, kept, Finding, Ground, Kept, Report};

/// The plugin, as the repository has it.
const PLUGIN: &str = include_str!("../../../../scripts/hooks/antumbra-opencode.js");

/// The line of the plugin that says how `antumbra` is run, which setup
/// rewrites.
const RUNS: &str = "const ANTUMBRA = [\"antumbra\"]";

/// Where OpenCode keeps its config: `OPENCODE_CONFIG_DIR`, else `opencode` in
/// the XDG config directory, which is `~/.config` on every platform,
/// Windows too.
pub fn config_dir(ground: &Ground<'_>) -> PathBuf {
    let named = |name: &str| {
        (ground.env)(name)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    named("OPENCODE_CONFIG_DIR").unwrap_or_else(|| {
        named("XDG_CONFIG_HOME")
            .unwrap_or_else(|| ground.home.join(".config"))
            .join("opencode")
    })
}

/// The plugin file, for an `antumbra` run as `ground` says.
pub fn files(ground: &Ground<'_>) -> Vec<Kept> {
    // A path as a JavaScript string: JSON's escapes are JavaScript's.
    let program = serde_json::Value::String(ground.antumbra.file()).to_string();
    vec![Kept {
        path: "plugins/antumbra.js",
        text: PLUGIN
            .replace("\r\n", "\n")
            .replace(RUNS, &format!("const ANTUMBRA = [{program}]")),
    }]
}

/// Every directory an OpenCode on this machine loads the plugin from, each
/// with what setup calls it: OpenCode's own, and, on a machine Unity Gateway
/// is on, the one under the config home `ug opencode` starts it with.
fn places(ground: &Ground<'_>) -> Vec<(PathBuf, &'static str)> {
    let own = config_dir(ground);
    let theirs = gateway::opencode_config_home(ground).join("opencode");
    let both = gateway::present(ground) && theirs != own;
    std::iter::once((own, "OpenCode: Antumbra's plugin"))
        .chain(both.then_some((
            theirs,
            "OpenCode started by Unity Gateway: Antumbra's plugin",
        )))
        .collect()
}

pub fn connect(ground: &Ground<'_>, report: &dyn Report) -> anyhow::Result<()> {
    let files = files(ground);
    for (dir, what) in places(ground) {
        keep(
            &dir,
            &files,
            &format!("{what} (plugins/antumbra.js: the MCP bridge and the three touchpoints)"),
            report,
        )?;
    }
    Ok(())
}

pub fn check(ground: &Ground<'_>) -> Vec<Finding> {
    let files = files(ground);
    places(ground)
        .into_iter()
        .flat_map(|(dir, what)| kept(&dir, &files, what))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::scratch::Scratch;
    use crate::agents::tests::Told;
    use crate::agents::Program;

    fn nothing(_: &str) -> Option<String> {
        None
    }

    fn nowhere(_: &str) -> Option<PathBuf> {
        None
    }

    #[test]
    fn the_plugin_has_the_line_setup_rewrites_and_both_lines_halves() {
        // The line is there once, or a rewrite would do nothing or too much.
        assert_eq!(PLUGIN.matches(RUNS).count(), 1);
        // 2.x reads `setup`, 1.x reads `server`, from one default export.
        assert!(PLUGIN.contains("export default {"));
        assert!(PLUGIN.contains("async setup(ctx)"));
        assert!(PLUGIN.contains("async server({ client, directory })"));
        for event in ["session-start", "prompt", "stop"] {
            assert!(PLUGIN.contains(&format!("hook(\"{event}\"")), "{event}");
        }
        assert!(PLUGIN.contains("\"mcp\", \"bridge\", \"--agent\", \"opencode\""));
    }

    #[test]
    fn the_plugin_is_written_with_how_this_machine_runs_antumbra() -> anyhow::Result<()> {
        let home = Scratch::new("opencode-files")?;
        let named = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        let files = files(&named);
        assert_eq!(files[0].path, "plugins/antumbra.js");
        assert!(files[0].text.contains(RUNS));

        // A Windows path, as a JavaScript string.
        let at = PathBuf::from("C:\\Program Files\\my tools\\antumbra.exe");
        let placed = Ground {
            home: home.path(),
            antumbra: Program::At(at),
            env: &nothing,
            find: &nowhere,
        };
        let text = &super::files(&placed)[0].text;
        assert!(
            text.contains("const ANTUMBRA = [\"C:\\\\Program Files\\\\my tools\\\\antumbra.exe\"]"),
            "{}",
            text.lines()
                .find(|l| l.starts_with("const ANTUMBRA"))
                .unwrap_or_default()
        );
        assert!(!text.contains(RUNS));
        Ok(())
    }

    #[test]
    fn connecting_writes_the_one_file_and_again_changes_nothing() -> anyhow::Result<()> {
        let home = Scratch::new("opencode-connect")?;
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        let dir = home.path().join(".config").join("opencode");
        std::fs::create_dir_all(&dir)?;
        let theirs = "{ \"model\": \"anthropic/claude\" }\n";
        std::fs::write(dir.join("opencode.json"), theirs)?;

        let told = Told::default();
        connect(&ground, &told)?;
        assert!(dir.join("plugins").join("antumbra.js").is_file());
        // Their config is not what connects it, and is not touched.
        assert_eq!(std::fs::read_to_string(dir.join("opencode.json"))?, theirs);
        assert!(check(&ground).iter().all(|f| f.fix.is_none()));

        let again = Told::default();
        connect(&ground, &again)?;
        assert!(
            again.said.borrow()[0].contains("already in"),
            "{:?}",
            again.said
        );
        Ok(())
    }

    #[test]
    fn with_unity_gateway_here_the_plugin_is_also_where_ug_has_opencode_look() -> anyhow::Result<()>
    {
        let home = Scratch::new("opencode-gateway")?;
        let ucode = home.path().join(".ucode");
        // What ug keeps there for OpenCode: its config, and a plugin of its own.
        let theirs = ucode.join("opencode-xdg").join("opencode");
        std::fs::create_dir_all(theirs.join("plugin"))?;
        std::fs::write(ucode.join("state.json"), "{}")?;
        let config = "{ \"model\": \"databricks-anthropic/claude\" }\n";
        std::fs::write(theirs.join("opencode.json"), config)?;
        std::fs::write(theirs.join("plugin").join("ucode-auth.js"), "// ug's\n")?;
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        let own = home.path().join(".config").join("opencode");

        let told = Told::default();
        connect(&ground, &told)?;
        let said = told.said.borrow().clone();
        assert!(
            said[0].starts_with("ok OpenCode: Antumbra's plugin"),
            "{said:?}"
        );
        assert!(
            said[1].starts_with("ok OpenCode started by Unity Gateway: Antumbra's plugin")
                && said[1].contains("written to"),
            "{said:?}"
        );
        assert_eq!(
            std::fs::read_to_string(own.join("plugins").join("antumbra.js"))?,
            std::fs::read_to_string(theirs.join("plugins").join("antumbra.js"))?
        );
        // Nothing of ug's is touched.
        assert_eq!(
            std::fs::read_to_string(theirs.join("opencode.json"))?,
            config
        );
        assert_eq!(
            std::fs::read_to_string(theirs.join("plugin").join("ucode-auth.js"))?,
            "// ug's\n"
        );
        assert!(check(&ground).iter().all(|f| f.fix.is_none()));

        // ug's copy gone: the check says which is missing.
        std::fs::remove_file(theirs.join("plugins").join("antumbra.js"))?;
        let found = check(&ground);
        assert!(
            found.iter().any(|f| !f.works
                && f.said.starts_with("OpenCode started by Unity Gateway")
                && f.said.contains("missing")),
            "{found:?}"
        );
        Ok(())
    }

    #[test]
    fn a_setup_run_from_inside_ug_opencode_still_finds_opencodes_own_directory(
    ) -> anyhow::Result<()> {
        let home = Scratch::new("opencode-inside")?;
        let xdg = home.path().join(".ucode").join("opencode-xdg");
        let named = xdg.display().to_string();
        // What `ug opencode` gives the session, and so a setup run from it.
        let inside = move |name: &str| (name == "XDG_CONFIG_HOME").then(|| named.clone());
        let ug = |name: &str| (name == "ug").then(|| PathBuf::from(name));
        let outside = gateway::outside(home.path(), &inside);
        let ground = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &outside,
            find: &ug,
        };
        assert_eq!(
            config_dir(&ground),
            home.path().join(".config").join("opencode")
        );
        connect(&ground, &Told::default())?;
        for dir in [
            home.path().join(".config").join("opencode"),
            xdg.join("opencode"),
        ] {
            assert!(
                dir.join("plugins").join("antumbra.js").is_file(),
                "{}",
                dir.display()
            );
        }
        Ok(())
    }

    #[test]
    fn the_config_is_where_opencode_says() -> anyhow::Result<()> {
        let home = Scratch::new("opencode-home")?;
        let plain = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &nothing,
            find: &nowhere,
        };
        // `~/.config/opencode` on every platform, Windows too.
        assert_eq!(
            config_dir(&plain),
            home.path().join(".config").join("opencode")
        );
        let xdg = |name: &str| (name == "XDG_CONFIG_HOME").then(|| "/elsewhere/config".to_string());
        let moved = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &xdg,
            find: &nowhere,
        };
        assert_eq!(
            config_dir(&moved),
            PathBuf::from("/elsewhere/config").join("opencode")
        );
        let own = |name: &str| match name {
            "OPENCODE_CONFIG_DIR" => Some("/own/opencode".to_string()),
            "XDG_CONFIG_HOME" => Some("/elsewhere/config".to_string()),
            _ => None,
        };
        let named = Ground {
            home: home.path(),
            antumbra: Program::Named,
            env: &own,
            find: &nowhere,
        };
        assert_eq!(config_dir(&named), PathBuf::from("/own/opencode"));
        Ok(())
    }
}
