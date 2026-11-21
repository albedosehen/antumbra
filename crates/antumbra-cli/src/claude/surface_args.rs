//! The surface pair: the flags every `antumbra claude` command that calls the
//! Antumbra MCP surface takes, and where each is found when it is not given.

use std::path::Path;

use clap::Args;

use crate::agents::Reach;

/// Where a server `antumbra setup local` started answers, and where the hook
/// scripts look when nothing names another.
pub const LOCAL_SURFACE: &str = "http://127.0.0.1:8081";

#[derive(Args, Debug, Clone)]
pub struct SurfaceArgs {
    /// The Antumbra MCP surface. Without it, the one `antumbra setup`
    /// recorded, else this machine's own.
    #[arg(long = "surface", env = "ANTUMBRA_URL")]
    pub surface: Option<String>,
    /// The bearer token for that surface. Without it, the one in the token
    /// file `antumbra setup` keeps. Prefer the environment variable to the
    /// flag: a flag ends up in the shell's history.
    #[arg(long, env = "ANTUMBRA_TOKEN", hide_env_values = true)]
    pub token: Option<String>,
}

impl SurfaceArgs {
    /// Where to call and what to present there, for the user whose home is
    /// `home`; `env` gives the value of an environment variable, or nothing.
    ///
    /// The flag wins, then the environment (clap has folded both into the
    /// fields), then what `antumbra setup` recorded, then this machine's own
    /// surface. `antumbra setup` keeps the token in a file and the address in
    /// Claude Code's settings, so a hook is run with the address in its
    /// environment and no token, and a terminal with neither: both are found
    /// here the way the hook scripts, `antumbra hook` and the bridge find
    /// them.
    ///
    /// The file's token is presented only to the server it is for: the one
    /// the environment or setup's record names, or this machine's own where
    /// neither names any. A `--surface` that is another server's gets the
    /// token it is given and no other, so a stored token is never sent to an
    /// address typed on a command line.
    pub fn reach(
        &self,
        home: Option<&Path>,
        env: impl Fn(&str) -> Option<String>,
    ) -> (String, Option<String>) {
        let given = |value: &Option<String>| {
            value
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
        };
        let recorded = home.and_then(|home| Reach::read(home, &env).ok());
        let named = given(&self.surface).map(|named| bare(&named).to_owned());
        let its_own = match (&named, &recorded) {
            (None, _) => true,
            (Some(named), Some(recorded)) => same_server(named, &recorded.url),
            (Some(named), None) => same_server(named, LOCAL_SURFACE),
        };
        let surface = named
            .or_else(|| recorded.as_ref().map(|reach| reach.url.clone()))
            .unwrap_or_else(|| LOCAL_SURFACE.to_owned());
        let token = given(&self.token).or_else(|| {
            if !its_own {
                return None;
            }
            // Where nothing records an address there is still the file the
            // hook scripts read, so the address found above stands in for the
            // one `Reach` wants.
            let reach = Reach::read(home?, |name| match name {
                "ANTUMBRA_URL" => Some(surface.clone()),
                other => env(other),
            })
            .ok()?;
            reach.token().ok()
        });
        (surface, token)
    }
}

/// An address as the calls want it: without a trailing slash or `/mcp`, which
/// is how setup's record holds it too.
fn bare(url: &str) -> &str {
    url.trim()
        .trim_end_matches('/')
        .trim_end_matches("/mcp")
        .trim_end_matches('/')
}

/// Whether two addresses name one server, however each was written: with or
/// without a trailing slash or `/mcp`, in either case.
fn same_server(a: &str, b: &str) -> bool {
    bare(a).eq_ignore_ascii_case(bare(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::scratch::Scratch;

    fn args(surface: Option<&str>, token: Option<&str>) -> SurfaceArgs {
        SurfaceArgs {
            surface: surface.map(str::to_owned),
            token: token.map(str::to_owned),
        }
    }

    /// A home as `antumbra setup hosted` leaves one: the address in
    /// `setup.json`, the token in its file.
    fn set_up(home: &Path, url: &str, token: &str) -> anyhow::Result<()> {
        let antumbra = home.join(".antumbra");
        std::fs::create_dir_all(&antumbra)?;
        std::fs::write(
            antumbra.join("setup.json"),
            serde_json::json!({ "mode": "hosted", "url": url }).to_string(),
        )?;
        std::fs::write(antumbra.join("token.txt"), format!("{token}\n"))?;
        Ok(())
    }

    fn nothing(_: &str) -> Option<String> {
        None
    }

    /// What the issue was: a machine that was set up, with no token in the
    /// environment. A terminal has neither the address nor the token, and a
    /// hook has the address alone; both find what setup recorded.
    #[test]
    fn a_machine_that_was_set_up_is_reached_with_its_token_file() -> anyhow::Result<()> {
        let home = Scratch::new("surface-set-up")?;
        set_up(
            home.path(),
            "http://workspace.example:8081",
            "from-the-file",
        )?;

        let in_a_terminal = args(None, None).reach(Some(home.path()), nothing);
        assert_eq!(
            in_a_terminal,
            (
                "http://workspace.example:8081".to_owned(),
                Some("from-the-file".to_owned())
            )
        );

        // A hook: Claude Code gives it the address from its settings.
        let address = |name: &str| {
            (name == "ANTUMBRA_URL").then(|| "http://workspace.example:8081/".to_owned())
        };
        let as_a_hook =
            args(Some("http://workspace.example:8081/"), None).reach(Some(home.path()), address);
        assert_eq!(
            as_a_hook,
            (
                "http://workspace.example:8081".to_owned(),
                Some("from-the-file".to_owned())
            )
        );
        Ok(())
    }

    /// What is given wins over what is recorded, and a token given is the
    /// one presented wherever the call goes.
    #[test]
    fn what_is_given_wins_over_what_was_recorded() -> anyhow::Result<()> {
        let home = Scratch::new("surface-given")?;
        set_up(
            home.path(),
            "http://workspace.example:8081",
            "from-the-file",
        )?;
        let got =
            args(Some("http://other.example"), Some("given")).reach(Some(home.path()), nothing);
        assert_eq!(
            got,
            ("http://other.example".to_owned(), Some("given".to_owned()))
        );
        let got = args(None, Some("given")).reach(Some(home.path()), nothing);
        assert_eq!(
            got,
            (
                "http://workspace.example:8081".to_owned(),
                Some("given".to_owned())
            )
        );
        Ok(())
    }

    /// A stored token is for the server it was stored for. An address that is
    /// another server's gets no token it was not given.
    #[test]
    fn the_files_token_is_not_sent_to_another_server() -> anyhow::Result<()> {
        let home = Scratch::new("surface-another")?;
        set_up(
            home.path(),
            "http://workspace.example:8081",
            "from-the-file",
        )?;
        let got = args(Some("http://other.example:8081"), None).reach(Some(home.path()), nothing);
        assert_eq!(got, ("http://other.example:8081".to_owned(), None));

        // The same server, written another way, is the same server.
        for written in [
            "http://workspace.example:8081/",
            "http://workspace.example:8081/mcp",
            "HTTP://Workspace.Example:8081",
        ] {
            let got = args(Some(written), None).reach(Some(home.path()), nothing);
            assert_eq!(got.1.as_deref(), Some("from-the-file"), "{written}");
        }
        Ok(())
    }

    /// A machine wired by hand has the token file and no record. With nothing
    /// named it is this machine's own surface and the file's token, as the
    /// hook scripts have it; `ANTUMBRA_TOKEN_FILE` names another file.
    #[test]
    fn a_machine_wired_by_hand_is_reached_as_the_hooks_reach_it() -> anyhow::Result<()> {
        let home = Scratch::new("surface-by-hand")?;
        let antumbra = home.path().join(".antumbra");
        std::fs::create_dir_all(&antumbra)?;
        std::fs::write(antumbra.join("token.txt"), "by-hand")?;

        let got = args(None, None).reach(Some(home.path()), nothing);
        assert_eq!(got, (LOCAL_SURFACE.to_owned(), Some("by-hand".to_owned())));
        let got = args(Some(LOCAL_SURFACE), None).reach(Some(home.path()), nothing);
        assert_eq!(got.1.as_deref(), Some("by-hand"));
        let got = args(Some("http://other.example"), None).reach(Some(home.path()), nothing);
        assert_eq!(got.1, None);

        // Wired by hand, the address is in the settings' `env`: a hook has it
        // in its environment, and the file's token is for that server.
        let address = |name: &str| {
            (name == "ANTUMBRA_URL").then(|| "http://workspace.example:8081".to_owned())
        };
        let got =
            args(Some("http://workspace.example:8081"), None).reach(Some(home.path()), address);
        assert_eq!(
            got,
            (
                "http://workspace.example:8081".to_owned(),
                Some("by-hand".to_owned())
            )
        );

        let elsewhere = home.path().join("elsewhere.txt");
        std::fs::write(&elsewhere, "named")?;
        let names_it = |name: &str| {
            (name == "ANTUMBRA_TOKEN_FILE").then(|| elsewhere.to_string_lossy().into_owned())
        };
        let got = args(None, None).reach(Some(home.path()), names_it);
        assert_eq!(got.1.as_deref(), Some("named"));
        Ok(())
    }

    /// Nothing given and nothing recorded is this machine's own surface with
    /// no token, as it was: the server says whether it wants one.
    #[test]
    fn with_nothing_to_find_it_is_this_machines_surface_and_no_token() -> anyhow::Result<()> {
        let home = Scratch::new("surface-nothing")?;
        let got = args(None, None).reach(Some(home.path()), nothing);
        assert_eq!(got, (LOCAL_SURFACE.to_owned(), None));
        assert_eq!(
            args(None, None).reach(None, nothing),
            (LOCAL_SURFACE.to_owned(), None)
        );
        // An empty value is no value: a variable set to nothing names nothing.
        let got = args(Some("  "), Some("")).reach(Some(home.path()), nothing);
        assert_eq!(got, (LOCAL_SURFACE.to_owned(), None));
        Ok(())
    }
}
