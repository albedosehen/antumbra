//! `antumbra bring`: copy the memories you chose, on the Bring page of your
//! own dashboard, to a workspace you joined.
//!
//! It runs on a machine that holds both ends: its own setup's token for your
//! own server, and the token `antumbra join` kept for the workspace. Each
//! chosen memory is read whole from your server, stored again through the
//! workspace's as you there (so it is embedded by the workspace's own model
//! and is yours to share or forget there), and marked brought on your server
//! as soon as it has arrived. What it copies stays where it was.
//!
//! It lands in a compartment of yours in the workspace, named for the machine
//! it came from. A compartment is its owner's until it is shared: your
//! teammates read what you brought once you share that compartment with them.

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Args;
use serde_json::{json, Value};

/// How many chosen memories are read from your server at a time.
const PAGE: u64 = 50;

#[derive(Args, Debug)]
pub struct BringArgs {
    /// The workspace to bring to, by its id (`ws:...`) or its address, when
    /// this machine has joined more than one.
    #[arg(long)]
    pub to: Option<String>,
    /// Say what would be brought, and bring nothing.
    #[arg(long)]
    pub dry_run: bool,
    /// Also offer the compartment they land in to the whole workspace: one of
    /// its admins decides, and until then it stays yours alone.
    #[arg(long)]
    pub offer: bool,
}

/// One call of a tool on a server: its name and arguments, and its answer.
type Call<'a> = &'a dyn Fn(&str, Value) -> anyhow::Result<Value>;

/// A workspace this machine joined, as `antumbra join` recorded it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Joined {
    pub(crate) url: String,
    pub(crate) workspace: String,
    pub(crate) user: String,
    pub(crate) token_file: PathBuf,
}

/// Every workspace this machine has joined, by directory name.
pub(crate) fn joined(antumbra: &Path) -> Vec<Joined> {
    let Ok(dirs) = std::fs::read_dir(antumbra.join("workspaces")) else {
        return Vec::new();
    };
    let mut all: Vec<(String, Joined)> = dirs
        .flatten()
        .filter_map(|dir| {
            let text = std::fs::read_to_string(dir.path().join("workspace.json")).ok()?;
            let kept: Value = serde_json::from_str(&text).ok()?;
            let field = |name: &str| kept.get(name).and_then(Value::as_str).map(str::to_string);
            Some((
                dir.file_name().to_string_lossy().into_owned(),
                Joined {
                    url: field("url")?.trim_end_matches('/').to_string(),
                    workspace: field("workspace")?,
                    user: field("user")?,
                    token_file: PathBuf::from(field("token_file")?),
                },
            ))
        })
        .collect();
    all.sort_by(|a, b| a.0.cmp(&b.0));
    all.into_iter().map(|(_, joined)| joined).collect()
}

/// The workspace `--to` names among those joined, or the only one.
pub(crate) fn target(all: &[Joined], to: Option<&str>) -> anyhow::Result<Joined> {
    let named = |j: &&Joined| {
        to.is_none_or(|to| {
            let to = to.trim().trim_end_matches('/');
            j.workspace == to || j.url == to || j.workspace.trim_start_matches("ws:") == to
        })
    };
    let found: Vec<&Joined> = all.iter().filter(named).collect();
    match (found.as_slice(), to) {
        ([one], _) => Ok((*one).clone()),
        ([], None) => anyhow::bail!(
            "this machine has joined no workspace. Ask to join one with: antumbra join <server> --code <code> --handle <handle>"
        ),
        ([], Some(to)) => anyhow::bail!("this machine has not joined `{to}`"),
        (many, _) => anyhow::bail!(
            "this machine has joined {} workspaces; say which with --to: {}",
            many.len(),
            many.iter()
                .map(|j| format!("{} at {}", j.workspace, j.url))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The id of the compartment of the caller's in the workspace that holds
/// what was brought from `host`, made if it is not there.
fn compartment(team: Call<'_>, host: &str) -> anyhow::Result<String> {
    let name = format!("brought from {host}");
    let listed = team("list_compartments", json!({}))?;
    let held = listed
        .get("compartments")
        .and_then(Value::as_array)
        .and_then(|all| {
            all.iter()
                .find(|c| c.get("name").and_then(Value::as_str) == Some(&name))
        })
        .and_then(|c| c.get("id").and_then(Value::as_str));
    if let Some(id) = held {
        return Ok(id.to_string());
    }
    team("create_compartment", json!({ "name": name }))?
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("the workspace made a compartment and sent no id"))
}

/// What a run came to.
#[derive(Debug, Default, PartialEq)]
struct Brought {
    brought: usize,
    /// What could not be stored in the workspace, with why: left chosen.
    failed: Vec<(String, String)>,
}

/// Copy every chosen memory from `own` to `team`, marking each brought on
/// `own` as it arrives. A memory the workspace refuses is left chosen and
/// said; the rest go on.
fn copy(own: Call<'_>, team: Call<'_>, into: &str) -> anyhow::Result<Brought> {
    let mut done = Brought::default();
    loop {
        // What is brought leaves the chosen, so the next page starts past
        // only what failed.
        let page = own(
            "bring_candidates",
            json!({
                "filter": { "state": "chosen", "include_held": true },
                "full": true,
                "limit": PAGE,
                "offset": done.failed.len(),
            }),
        )?;
        let memories = page
            .get("memories")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if memories.is_empty() {
            return Ok(done);
        }
        for memory in memories {
            let id = memory
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let mut arguments = json!({
                "content": memory.get("content"),
                "network": memory.get("network"),
                "compartment": into,
            });
            for field in ["confidence", "volatile", "provenance"] {
                if let Some(value) = memory.get(field).filter(|v| !v.is_null()) {
                    arguments[field] = value.clone();
                }
            }
            match team("store_memory", arguments) {
                // Marked at once: a run that stops here has brought this one
                // and will not bring it twice.
                Ok(_) => {
                    own("bring_choose", json!({ "choice": "brought", "ids": [id] }))?;
                    done.brought += 1;
                }
                Err(why) => done.failed.push((id, why.to_string())),
            }
        }
    }
}

/// Offer a compartment to the workspace's shared pool: what the answer said.
type Offer<'a> = &'a dyn Fn(&str) -> anyhow::Result<String>;

/// The flow, with both servers given.
fn run_with(
    args: &BringArgs,
    to: &Joined,
    host: &str,
    own: Call<'_>,
    team: Call<'_>,
    offer: Offer<'_>,
) -> anyhow::Result<Vec<String>> {
    let chosen = own(
        "bring_candidates",
        json!({ "filter": { "state": "chosen", "include_held": true }, "limit": 1 }),
    )?
    .get("total")
    .and_then(Value::as_u64)
    .unwrap_or(0);
    if chosen == 0 {
        return Ok(vec![
            "Nothing is chosen. Choose what to bring on the Bring page of your own dashboard."
                .to_string(),
        ]);
    }
    let plural = |n: u64| if n == 1 { "memory" } else { "memories" };
    if args.dry_run {
        return Ok(vec![format!(
            "{chosen} chosen {} would be copied to {} at {}, as {}. Nothing was sent.",
            plural(chosen),
            to.workspace,
            to.url,
            to.user
        )]);
    }
    let into = compartment(team, host)?;
    let done = copy(own, team, &into)?;
    let mut lines = vec![format!(
        "Brought {} {} to {} at {}, as {}.",
        done.brought,
        plural(done.brought as u64),
        to.workspace,
        to.url,
        to.user
    )];
    if done.brought > 0 {
        lines.push(format!(
            "They are in your compartment \"brought from {host}\" there, and are copies: yours stay where they were."
        ));
        if args.offer {
            // Brought either way: an offer that could not be made is said.
            lines.push(match offer(&into) {
                Ok(status) if status == "accepted" => {
                    "That compartment is already shared with the workspace: your teammates read them now.".to_string()
                }
                Ok(_) => "It is offered to the workspace: your teammates read them once one of its admins accepts.".to_string(),
                Err(why) => format!("It could not be offered to the workspace ({why}); it is yours alone there."),
            });
        } else {
            lines.push(
                "A compartment is its owner's until it is shared: offer it to the workspace with --offer, or on the Knowledge page of the workspace's dashboard."
                    .to_string(),
            );
        }
    }
    if !done.failed.is_empty() {
        lines.push(format!(
            "{} could not be stored there and are still chosen:",
            done.failed.len()
        ));
        lines.extend(
            done.failed
                .iter()
                .take(5)
                .map(|(id, why)| format!("  {id}: {why}")),
        );
    }
    Ok(lines)
}

pub fn run(args: BringArgs) -> anyhow::Result<()> {
    let home = crate::machine::home().ok_or_else(|| anyhow::anyhow!("no home directory"))?;
    let antumbra = home.join(".antumbra");
    let to = target(&joined(&antumbra), args.to.as_deref())?;
    let reach = crate::agents::Reach::read(&home, |name| std::env::var(name).ok())?;
    if reach.url.trim_end_matches('/') == to.url {
        anyhow::bail!(
            "this machine's own setup is {} itself: there is nothing of its own to bring there",
            to.url
        );
    }
    let host = antumbra_http::host_id().unwrap_or_else(|| "this machine".to_string());
    let team_token = std::fs::read_to_string(&to.token_file)
        .map(|t| t.trim().to_string())
        .map_err(|e| {
            anyhow::anyhow!(
                "no token for {} in {}: {e}",
                to.workspace,
                to.token_file.display()
            )
        })?;
    let patience = Duration::from_secs(60);
    let own_client = antumbra_http::SurfaceClient::new(&reach.url, Some(&reach.token()?), patience)
        .from_machine(Some(host.clone()));
    let team_client = antumbra_http::SurfaceClient::new(&to.url, Some(&team_token), patience)
        .from_machine(Some(host.clone()));
    let own = |tool: &str, arguments: Value| {
        own_client
            .call(tool, arguments)
            .map_err(anyhow::Error::from)
    };
    let team = |tool: &str, arguments: Value| {
        team_client
            .call(tool, arguments)
            .map_err(anyhow::Error::from)
    };
    let agent = antumbra_http::agent(patience);
    let offer = |compartment: &str| -> anyhow::Result<String> {
        let mut response = agent
            .post(format!("{}/pool/offer", to.url))
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {team_token}"))
            .send_json(json!({ "compartment": compartment }))
            .map_err(|e| anyhow::anyhow!("could not reach {}: {e}", to.url))?;
        let status = response.status().as_u16();
        let answer: Value = response.body_mut().read_json().unwrap_or(Value::Null);
        let said = |name: &str| answer.get(name).and_then(Value::as_str).map(str::to_string);
        match status {
            200 => Ok(said("status").unwrap_or_default()),
            _ => anyhow::bail!(
                "{}",
                said("error").unwrap_or_else(|| format!("the workspace answered {status}"))
            ),
        }
    };
    for line in run_with(&args, &to, &host, &own, &team, &offer)? {
        println!("{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Your own server: the chosen memories, and which were marked brought.
    struct Own {
        chosen: RefCell<Vec<Value>>,
        marked: RefCell<Vec<String>>,
    }

    impl Own {
        fn with(n: usize) -> Self {
            Self {
                chosen: RefCell::new(
                    (0..n)
                        .map(|i| {
                            json!({
                                "id": format!("memory:{i}"), "content": format!("note {i}"), "network": "bank",
                                "confidence": 0.8, "volatile": false,
                                "provenance": { "repo": "github.com/acme/billing", "commit": "3f9c2ab" },
                            })
                        })
                        .collect(),
                ),
                marked: RefCell::new(Vec::new()),
            }
        }

        fn call(&self, tool: &str, arguments: Value) -> anyhow::Result<Value> {
            match tool {
                "bring_candidates" => {
                    let chosen = self.chosen.borrow();
                    let (limit, offset) = (
                        arguments["limit"].as_u64().unwrap_or(50) as usize,
                        arguments["offset"].as_u64().unwrap_or(0) as usize,
                    );
                    let page: Vec<Value> =
                        chosen.iter().skip(offset).take(limit).cloned().collect();
                    Ok(json!({ "total": chosen.len(), "memories": page }))
                }
                "bring_choose" => {
                    let id = arguments["ids"][0].as_str().unwrap_or_default().to_string();
                    self.chosen.borrow_mut().retain(|m| m["id"] != id.as_str());
                    self.marked.borrow_mut().push(id);
                    Ok(json!({ "changed": 1 }))
                }
                other => anyhow::bail!("your own server was asked for {other}"),
            }
        }
    }

    /// The workspace: its compartments and what was stored in it.
    #[derive(Default)]
    struct Team {
        compartments: RefCell<Vec<(String, String)>>,
        stored: RefCell<Vec<Value>>,
        refuses: Option<&'static str>,
    }

    impl Team {
        fn call(&self, tool: &str, arguments: Value) -> anyhow::Result<Value> {
            match tool {
                "list_compartments" => Ok(json!({
                    "compartments": self.compartments.borrow().iter()
                        .map(|(id, name)| json!({ "id": id, "name": name })).collect::<Vec<_>>()
                })),
                "create_compartment" => {
                    let id = format!("comp:{}", self.compartments.borrow().len());
                    let name = arguments["name"].as_str().unwrap_or_default().to_string();
                    self.compartments.borrow_mut().push((id.clone(), name));
                    Ok(json!({ "id": id }))
                }
                "store_memory" => {
                    if self
                        .refuses
                        .is_some_and(|text| arguments["content"] == text)
                    {
                        anyhow::bail!("the workspace is full");
                    }
                    self.stored.borrow_mut().push(arguments);
                    Ok(json!({ "id": "memory:there" }))
                }
                other => anyhow::bail!("the workspace was asked for {other}"),
            }
        }
    }

    fn crew() -> Joined {
        Joined {
            url: "https://team.example".to_string(),
            workspace: "ws:crew".to_string(),
            user: "user:bodil".to_string(),
            token_file: PathBuf::from("/nowhere"),
        }
    }

    fn args(dry_run: bool) -> BringArgs {
        BringArgs {
            to: None,
            dry_run,
            offer: false,
        }
    }

    /// No offer is asked for in these: one that is made fails the test.
    fn no_offer(_: &str) -> anyhow::Result<String> {
        anyhow::bail!("an offer was made that nobody asked for")
    }

    /// Every chosen memory arrives whole, in one compartment named for the
    /// machine, across more than one page, and is marked brought at home.
    #[test]
    fn what_was_chosen_is_copied_and_marked_brought() -> anyhow::Result<()> {
        let (own, team) = (Own::with(PAGE as usize + 7), Team::default());
        let said = run_with(
            &args(false),
            &crew(),
            "laptop",
            &|t, a| own.call(t, a),
            &|t, a| team.call(t, a),
            &no_offer,
        )?;
        assert!(
            said[0].contains("Brought 57 memories to ws:crew"),
            "{said:?}"
        );
        assert!(
            said.iter().any(|l| l.contains("until it is shared")),
            "{said:?}"
        );
        let stored = team.stored.borrow();
        assert_eq!(stored.len(), 57);
        assert_eq!(
            *team.compartments.borrow(),
            [("comp:0".to_string(), "brought from laptop".to_string())]
        );
        let first = &stored[0];
        assert_eq!(first["compartment"], "comp:0");
        assert_eq!(first["content"], "note 0");
        assert_eq!(first["network"], "bank");
        assert_eq!(first["provenance"]["repo"], "github.com/acme/billing");
        assert_eq!(own.marked.borrow().len(), 57);
        assert!(own.chosen.borrow().is_empty());

        // A second run brings nothing and makes no second compartment.
        let again = run_with(
            &args(false),
            &crew(),
            "laptop",
            &|t, a| own.call(t, a),
            &|t, a| team.call(t, a),
            &no_offer,
        )?;
        assert!(again[0].contains("Nothing is chosen"), "{again:?}");
        assert_eq!(team.compartments.borrow().len(), 1);
        Ok(())
    }

    /// A memory the workspace refuses is left chosen and said; the others go.
    #[test]
    fn a_refused_memory_stays_chosen_and_the_rest_arrive() -> anyhow::Result<()> {
        let own = Own::with(4);
        let team = Team {
            refuses: Some("note 1"),
            ..Team::default()
        };
        let said = run_with(
            &args(false),
            &crew(),
            "laptop",
            &|t, a| own.call(t, a),
            &|t, a| team.call(t, a),
            &no_offer,
        )?;
        assert!(said[0].contains("Brought 3 memories"), "{said:?}");
        assert!(
            said.iter()
                .any(|l| l.contains("memory:1: the workspace is full")),
            "{said:?}"
        );
        assert_eq!(own.chosen.borrow().len(), 1);
        assert_eq!(*own.marked.borrow(), ["memory:0", "memory:2", "memory:3"]);
        Ok(())
    }

    /// A dry run says how many and sends nothing.
    #[test]
    fn a_dry_run_sends_nothing() -> anyhow::Result<()> {
        let (own, team) = (Own::with(3), Team::default());
        let said = run_with(
            &args(true),
            &crew(),
            "laptop",
            &|t, a| own.call(t, a),
            &|t, a| team.call(t, a),
            &no_offer,
        )?;
        assert!(
            said[0].contains("3 chosen memories would be copied")
                && said[0].contains("Nothing was sent"),
            "{said:?}"
        );
        assert!(team.stored.borrow().is_empty() && team.compartments.borrow().is_empty());
        assert_eq!(own.chosen.borrow().len(), 3);
        Ok(())
    }

    /// Which workspace: the only one joined, the one named, and said when it
    /// cannot be told.
    #[test]
    fn the_workspace_is_the_one_joined_or_the_one_named() -> anyhow::Result<()> {
        let home = crate::agents::scratch::Scratch::new("bring")?;
        let antumbra = home.path().join(".antumbra");
        assert!(target(&joined(&antumbra), None)
            .is_err_and(|e| e.to_string().contains("joined no workspace")));
        for (dir, url, workspace) in [
            ("a__ws_crew", "https://team.example/", "ws:crew"),
            ("b__ws_lab", "https://lab.example", "ws:lab"),
        ] {
            let at = antumbra.join("workspaces").join(dir);
            std::fs::create_dir_all(&at)?;
            std::fs::write(
                at.join("workspace.json"),
                json!({ "url": url, "workspace": workspace, "user": "user:bodil", "token_file": at.join("token.txt") }).to_string(),
            )?;
        }
        let all = joined(&antumbra);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].url, "https://team.example");
        assert!(target(&all, None).is_err_and(|e| e.to_string().contains("say which with --to")));
        assert_eq!(target(&all, Some("ws:lab"))?.workspace, "ws:lab");
        assert_eq!(target(&all, Some("crew"))?.workspace, "ws:crew");
        assert_eq!(
            target(&all, Some("https://team.example/"))?.workspace,
            "ws:crew"
        );
        assert!(target(&all, Some("ws:other")).is_err());
        assert_eq!(target(&all[..1], None)?.workspace, "ws:crew");
        Ok(())
    }

    /// With `--offer` the compartment they landed in is offered, and what
    /// came of it is said; an offer that fails leaves them brought.
    #[test]
    fn the_compartment_is_offered_when_asked() -> anyhow::Result<()> {
        let offering = BringArgs {
            offer: true,
            ..args(false)
        };
        let (own, team) = (Own::with(2), Team::default());
        let asked = RefCell::new(Vec::new());
        let offer = |compartment: &str| {
            asked.borrow_mut().push(compartment.to_string());
            Ok("offered".to_string())
        };
        let said = run_with(
            &offering,
            &crew(),
            "laptop",
            &|t, a| own.call(t, a),
            &|t, a| team.call(t, a),
            &offer,
        )?;
        assert_eq!(*asked.borrow(), ["comp:0"]);
        assert!(
            said.iter()
                .any(|l| l.contains("once one of its admins accepts")),
            "{said:?}"
        );

        let (own, team) = (Own::with(1), Team::default());
        let refused = |_: &str| -> anyhow::Result<String> { anyhow::bail!("the pool is closed") };
        let said = run_with(
            &offering,
            &crew(),
            "laptop",
            &|t, a| own.call(t, a),
            &|t, a| team.call(t, a),
            &refused,
        )?;
        assert!(said[0].contains("Brought 1 memory"), "{said:?}");
        assert!(
            said.iter()
                .any(|l| l.contains("could not be offered") && l.contains("the pool is closed")),
            "{said:?}"
        );
        Ok(())
    }
}
