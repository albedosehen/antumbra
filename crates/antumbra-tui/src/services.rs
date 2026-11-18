//! The services a workspace's dependency claims name, and what a change to
//! one of them reaches.
//!
//! A dependency is recorded as a memory: which service depends on which, the
//! source of the claim (declared in a manifest, observed, learned, or claimed
//! by someone) and where it came from. The console reads every memory, so it
//! has every claim, and works the graph out here with the arithmetic the
//! server's `blast_radius` uses (`antumbra_core::depgraph`): a pair's sources
//! combine as independent evidence, a claim fades when nobody records it
//! again, and a walk follows the strongest path to each service it reaches.

use antumbra_core::depgraph::{self, Direction, Edge, Hop, Reached, Source};
use antumbra_core::Memory;
use chrono::{DateTime, Utc};

use crate::netgraph::Layout;

/// The weakest pair a walk follows, which is the server's own default: a
/// claim alone does not count until something corroborates it.
pub const WALKED_FROM: f32 = 0.4;

/// How far a walk may go, in the order `h` steps through: the next service,
/// a few, or as far as the server's walk goes.
pub const HOPS: [u32; 3] = [1, 3, 6];

/// A workspace's dependency graph as the console draws it.
#[derive(Debug, Clone, Default)]
pub struct Services {
    /// Every claim read from the memories.
    edges: Vec<Edge>,
    /// Every pair of services with its claims and their combined weight,
    /// strongest first.
    pub pairs: Vec<Hop>,
    /// Every service named, by its slug, in the order the keys move through
    /// them: the graph's reading order, layer by layer.
    pub names: Vec<String>,
    /// Where each service of `names` goes in the drawing. A pair's
    /// dependent is drawn before what it depends on.
    pub layout: Layout,
}

impl Services {
    /// Read the dependency claims among `memories`.
    pub fn read(memories: &[Memory], now: DateTime<Utc>) -> Self {
        let edges: Vec<Edge> = memories.iter().filter_map(Edge::of).collect();
        let pairs = depgraph::hops(&edges, None, now);
        // By name first, so the layout starts from the same order whatever
        // order the memories arrived in.
        let mut sorted: Vec<&str> = pairs
            .iter()
            .flat_map(|pair| [pair.from.as_str(), pair.to.as_str()])
            .collect();
        sorted.sort_unstable();
        sorted.dedup();
        let at = |name: &str| sorted.binary_search(&name).ok();
        let joins: Vec<(usize, usize)> = pairs
            .iter()
            .filter_map(|pair| Some((at(&pair.from)?, at(&pair.to)?)))
            .collect();
        let first = Layout::of(sorted.len(), &joins);
        // Then in reading order, so a service's place among the names is its
        // place in the drawing and the keys move through the picture.
        let order = first.order();
        let names: Vec<String> = order.iter().map(|&i| sorted[i].to_string()).collect();
        let place = |name: &str| names.iter().position(|n| n == name);
        let joins: Vec<(usize, usize)> = pairs
            .iter()
            .filter_map(|pair| Some((place(&pair.from)?, place(&pair.to)?)))
            .collect();
        let layout = Layout::of(names.len(), &joins);
        Services {
            edges,
            pairs,
            names,
            layout,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The place of the service called `name` among [`Self::names`].
    #[cfg(test)]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// The place of the service the most pairs point at: where the page opens,
    /// since that is the service whose blast radius has the most in it. The
    /// first of them in reading order, when several are level.
    pub fn most_depended_on(&self) -> usize {
        let pointed_at = |name: &String| self.pairs.iter().filter(|p| &p.to == name).count();
        self.names
            .iter()
            .enumerate()
            .max_by_key(|(place, name)| (pointed_at(name), std::cmp::Reverse(*place)))
            .map_or(0, |(place, _)| place)
    }

    /// What a change to `origin` reaches (`Dependents`), or what it needs
    /// (`Dependencies`), within `hops`: each service by the strongest path to
    /// it, strongest first.
    pub fn reach(
        &self,
        origin: &str,
        direction: Direction,
        hops: u32,
        now: DateTime<Utc>,
    ) -> Vec<Reached> {
        depgraph::blast_radius(&self.edges, origin, direction, hops, WALKED_FROM, now)
    }

    /// The pairs `service` is on either end of, strongest first.
    pub fn pairs_of<'a>(&'a self, service: &'a str) -> impl Iterator<Item = &'a Hop> {
        self.pairs
            .iter()
            .filter(move |pair| pair.from == service || pair.to == service)
    }
}

/// The source a pair is drawn as: its strongest claim's, which a pair's
/// claims are sorted to put first.
pub fn drawn_as(pair: &Hop) -> Source {
    pair.edges
        .first()
        .map_or(Source::Claimed, |edge| edge.source)
}

/// A service as a list names it: a repository's slug by its last part.
pub fn short(name: &str) -> &str {
    name.rsplit('/')
        .next()
        .filter(|last| !last.is_empty())
        .unwrap_or(name)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use antumbra_core::depgraph::Claim;
    use antumbra_core::{MemoryId, MemoryNetwork, TenantId};

    /// A dependency claim as the server stores one.
    pub(crate) fn claim(from: &str, to: &str, source: Source, detail: &str) -> Memory {
        let tenant = TenantId::new("ws:demo");
        let claim = Claim::new(from, to, source, detail);
        let mut m = Memory::new(
            MemoryId::new(claim.memory_id(&tenant)),
            tenant,
            MemoryNetwork::World,
            claim.content(),
            source.confidence(),
            Utc::now(),
        );
        m.evidence = claim.evidence(None);
        m
    }

    /// The website's example: eight services, declared, observed and claimed.
    pub(crate) fn ledger() -> Vec<Memory> {
        let d = |from: &str, to: &str| claim(from, to, Source::Declared, "Cargo.toml names it");
        vec![
            d(
                "github.com/acme/payments-webhook",
                "github.com/acme/ledger-core",
            ),
            d(
                "github.com/acme/payouts-worker",
                "github.com/acme/ledger-core",
            ),
            d("github.com/acme/ledger-api", "github.com/acme/ledger-core"),
            d("github.com/acme/billing-ui", "github.com/acme/ledger-api"),
            d("github.com/acme/ledger-core", "github.com/acme/ledger-db"),
            claim(
                "github.com/acme/reports-job",
                "github.com/acme/ledger-api",
                Source::Observed,
                "seen calling it",
            ),
            claim(
                "github.com/acme/ledger-api",
                "github.com/acme/auth-service",
                Source::Claimed,
                "said in review",
            ),
        ]
    }

    #[test]
    fn the_services_are_read_from_the_claims_in_reading_order() -> anyhow::Result<()> {
        let services = Services::read(&ledger(), Utc::now());
        assert_eq!(services.names.len(), 8);
        assert_eq!(services.pairs.len(), 7);
        // A dependent is drawn before what it depends on.
        for pair in &services.pairs {
            let (Some(from), Some(to)) =
                (services.index_of(&pair.from), services.index_of(&pair.to))
            else {
                anyhow::bail!("a pair names a service that is not listed: {pair:?}");
            };
            assert!(
                services.layout.layer_of(from) < services.layout.layer_of(to),
                "{} before {}",
                pair.from,
                pair.to
            );
        }
        // The names are the layout's own order, so a name's place is its
        // place in the drawing.
        let order: Vec<usize> = (0..services.names.len()).collect();
        assert_eq!(services.layout.order(), order);
        // And the same whatever order the memories came in.
        let mut reversed = ledger();
        reversed.reverse();
        assert_eq!(Services::read(&reversed, Utc::now()).names, services.names);
        Ok(())
    }

    #[test]
    fn a_change_reaches_what_depends_on_it_by_the_strongest_path() -> anyhow::Result<()> {
        let now = Utc::now();
        let services = Services::read(&ledger(), now);
        let core = "github.com/acme/ledger-core";
        let near: Vec<String> = services
            .reach(core, Direction::Dependents, 1, now)
            .into_iter()
            .map(|r| short(&r.service).to_string())
            .collect();
        assert_eq!(near, ["ledger-api", "payments-webhook", "payouts-worker"]);
        let far = services.reach(core, Direction::Dependents, 3, now);
        let reached: Vec<(&str, u32)> = far.iter().map(|r| (short(&r.service), r.depth)).collect();
        assert!(reached.contains(&("billing-ui", 2)), "{reached:?}");
        assert!(reached.contains(&("reports-job", 2)), "{reached:?}");
        // The other way: what it needs.
        let needs: Vec<String> = services
            .reach(core, Direction::Dependencies, 3, now)
            .into_iter()
            .map(|r| short(&r.service).to_string())
            .collect();
        assert_eq!(needs, ["ledger-db"]);
        // A claim alone is under the floor a walk follows.
        let api = "github.com/acme/ledger-api";
        let needs: Vec<String> = services
            .reach(api, Direction::Dependencies, 1, now)
            .into_iter()
            .map(|r| short(&r.service).to_string())
            .collect();
        assert_eq!(needs, ["ledger-core"], "auth-service is only claimed");
        Ok(())
    }

    #[test]
    fn a_pair_is_drawn_as_its_strongest_claim() -> anyhow::Result<()> {
        let mut memories = ledger();
        memories.push(claim(
            "github.com/acme/reports-job",
            "github.com/acme/ledger-api",
            Source::Declared,
            "package.json names it",
        ));
        let services = Services::read(&memories, Utc::now());
        let Some(pair) = services
            .pairs_of("github.com/acme/reports-job")
            .find(|pair| pair.to.ends_with("ledger-api"))
        else {
            anyhow::bail!("the pair is gone");
        };
        assert_eq!(pair.edges.len(), 2);
        assert_eq!(drawn_as(pair), Source::Declared);
        assert_eq!(short("github.com/acme/ledger-api"), "ledger-api");
        assert_eq!(short("everywhere"), "everywhere");
        assert!(Services::read(&[], Utc::now()).is_empty());
        Ok(())
    }
}
