//! A layered layout for a small directed graph: which column each node goes
//! in and in what order down it, so its edges can be drawn left to right
//! without running through the nodes between.
//!
//! An edge runs from a node to one in a later layer. One that skips layers is
//! carried through each layer between in a lane, a slot that holds no node,
//! and every edge into one node shares that node's lanes: ten services that
//! depend on one library are ten short lines into one trunk, not ten long
//! lines across the picture.
//!
//! This is the arithmetic only. What a node is called, how a line is styled
//! and where on the screen a slot falls are the drawing's (`ui::netdraw`).

/// One place in a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slot {
    /// A node, by its index.
    Node(usize),
    /// A lane carrying edges on to the node of this index.
    Lane(usize),
}

/// Where each node of a graph goes: its layer, and each layer's slots in
/// order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Layout {
    /// Each layer's slots, first to last. Layer 0 holds the nodes no edge
    /// points at.
    pub layers: Vec<Vec<Slot>>,
    /// Each node's layer.
    layer_of: Vec<usize>,
}

/// How many times the order is swept, down the layers and back.
const SWEEPS: usize = 4;

impl Layout {
    /// Lay out `nodes` nodes joined by `edges`, each `(from, to)`. An edge
    /// that names a node there is not, or joins a node to itself, is left
    /// out.
    pub fn of(nodes: usize, edges: &[(usize, usize)]) -> Self {
        let edges: Vec<(usize, usize)> = edges
            .iter()
            .copied()
            .filter(|&(from, to)| from != to && from < nodes && to < nodes)
            .collect();
        let layer_of = layers_of(nodes, &edges);
        let depth = layer_of.iter().max().map_or(0, |deepest| deepest + 1);
        let mut layers: Vec<Vec<Slot>> = vec![Vec::new(); depth];
        for (node, &layer) in layer_of.iter().enumerate() {
            layers[layer].push(Slot::Node(node));
        }
        // A lane in every layer an edge skips, one for each node skipped to.
        for &(from, to) in &edges {
            for layer in layers
                .iter_mut()
                .take(layer_of[to])
                .skip(layer_of[from] + 1)
            {
                if !layer.contains(&Slot::Lane(to)) {
                    layer.push(Slot::Lane(to));
                }
            }
        }
        let mut layout = Layout { layers, layer_of };
        for sweep in 0..SWEEPS {
            layout.sweep(&edges, sweep % 2 == 0);
        }
        layout
    }

    /// The layer `node` is in.
    pub fn layer_of(&self, node: usize) -> Option<usize> {
        self.layer_of.get(node).copied()
    }

    /// Where `slot` is down `layer`: its place, from 0.
    pub fn place(&self, layer: usize, slot: Slot) -> Option<usize> {
        self.layers.get(layer)?.iter().position(|s| *s == slot)
    }

    /// The slots the edge from `from` to `to` passes, as `(layer, slot)`: its
    /// two nodes, and between them the lanes of the layers it skips. An edge
    /// that points backward, as one of a cycle does, has no lanes.
    pub fn path(&self, from: usize, to: usize) -> Vec<(usize, Slot)> {
        let (Some(first), Some(last)) = (self.layer_of(from), self.layer_of(to)) else {
            return Vec::new();
        };
        let mut path = vec![(first, Slot::Node(from))];
        if last > first {
            path.extend((first + 1..last).map(|layer| (layer, Slot::Lane(to))));
        }
        path.push((last, Slot::Node(to)));
        path
    }

    /// The nodes in reading order: layer by layer, down each.
    pub fn order(&self) -> Vec<usize> {
        self.layers
            .iter()
            .flatten()
            .filter_map(|slot| match slot {
                Slot::Node(node) => Some(*node),
                Slot::Lane(_) => None,
            })
            .collect()
    }

    /// Reorder each layer by where its slots' neighbors sit in the layer
    /// before it (`down`) or after it, which is what uncrosses most edges. A
    /// slot with no neighbor there keeps its place.
    fn sweep(&mut self, edges: &[(usize, usize)], down: bool) {
        let depth = self.layers.len();
        let layers: Vec<usize> = if down {
            (1..depth).collect()
        } else {
            (0..depth.saturating_sub(1)).rev().collect()
        };
        for layer in layers {
            let beside = if down { layer - 1 } else { layer + 1 };
            let here = self.layers[layer].clone();
            let mut keyed: Vec<(f64, Slot)> = here
                .iter()
                .enumerate()
                .map(|(place, slot)| {
                    let near = self.neighbors(edges, layer, *slot, down);
                    let places: Vec<f64> = near
                        .iter()
                        .filter_map(|n| self.place(beside, *n))
                        .map(|p| share(p, self.layers[beside].len()))
                        .collect();
                    let key = if places.is_empty() {
                        share(place, here.len())
                    } else {
                        places.iter().sum::<f64>() / places.len() as f64
                    };
                    (key, *slot)
                })
                .collect();
            // Stable, so slots with one key keep the order they had.
            keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
            self.layers[layer] = keyed.into_iter().map(|(_, slot)| slot).collect();
        }
    }

    /// The slots `slot`, in `layer`, is joined to in the layer before
    /// (`before`) or the layer after.
    fn neighbors(
        &self,
        edges: &[(usize, usize)],
        layer: usize,
        slot: Slot,
        before: bool,
    ) -> Vec<Slot> {
        let at = |node: usize| self.layer_of[node];
        let mut near = Vec::new();
        for &(from, to) in edges {
            let passes = match slot {
                Slot::Node(node) if before => to == node && at(from) < layer,
                Slot::Node(node) => from == node && at(to) > layer,
                Slot::Lane(target) => to == target && at(from) < layer && at(to) > layer,
            };
            if !passes {
                continue;
            }
            let other = if before {
                if at(from) + 1 == layer {
                    Slot::Node(from)
                } else {
                    Slot::Lane(to)
                }
            } else if at(to) == layer + 1 {
                Slot::Node(to)
            } else {
                Slot::Lane(to)
            };
            if !near.contains(&other) {
                near.push(other);
            }
        }
        near
    }
}

/// A place in a layer of `len` slots as a share of the way down it, so that
/// layers of different lengths compare.
fn share(place: usize, len: usize) -> f64 {
    (place as f64 + 0.5) / len.max(1) as f64
}

/// Each node's layer: the longest run of edges that ends at it. Every edge is
/// relaxed once a node at most, which settles a graph without a cycle. One
/// with a cycle would never settle, so no layer is pushed past the last one a
/// graph of this many nodes can have, and an edge of the cycle is left
/// pointing backward.
fn layers_of(nodes: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut layer = vec![0usize; nodes];
    for _ in 0..nodes {
        let mut moved = false;
        for &(from, to) in edges {
            let next = layer[from] + 1;
            if layer[to] < next && next < nodes {
                layer[to] = next;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    layer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_is_a_layer_a_node() -> anyhow::Result<()> {
        let layout = Layout::of(3, &[(0, 1), (1, 2)]);
        assert_eq!(
            layout.layers,
            vec![
                vec![Slot::Node(0)],
                vec![Slot::Node(1)],
                vec![Slot::Node(2)]
            ]
        );
        assert_eq!(layout.order(), vec![0, 1, 2]);
        assert_eq!(layout.layer_of(2), Some(2));
        Ok(())
    }

    #[test]
    fn an_edge_that_skips_a_layer_is_carried_through_it_in_a_lane() -> anyhow::Result<()> {
        // 0 -> 1 -> 2, and 0 -> 2 straight.
        let layout = Layout::of(3, &[(0, 1), (1, 2), (0, 2)]);
        assert!(layout.layers[1].contains(&Slot::Node(1)));
        assert!(layout.layers[1].contains(&Slot::Lane(2)));
        assert_eq!(
            layout.path(0, 2),
            vec![(0, Slot::Node(0)), (1, Slot::Lane(2)), (2, Slot::Node(2))]
        );
        // A short edge passes no lane.
        assert_eq!(
            layout.path(0, 1),
            vec![(0, Slot::Node(0)), (1, Slot::Node(1))]
        );
        Ok(())
    }

    #[test]
    fn every_edge_into_one_node_shares_that_nodes_lanes() -> anyhow::Result<()> {
        // 3 is two layers on from 0, 1 and 2 by way of 4, and each of them
        // also points at it straight: three long edges, one lane.
        let edges = [(0, 4), (4, 3), (0, 3), (1, 3), (2, 3)];
        let layout = Layout::of(5, &edges);
        let lanes = layout.layers[1]
            .iter()
            .filter(|slot| matches!(slot, Slot::Lane(3)))
            .count();
        assert_eq!(lanes, 1, "{:?}", layout.layers);
        assert_eq!(layout.layer_of(3), Some(2));
        Ok(())
    }

    #[test]
    fn a_layer_is_ordered_by_where_its_neighbors_are() -> anyhow::Result<()> {
        // 0 and 1 above; 2 hangs from 1 and 3 from 0. Left as numbered the
        // two edges cross; ordered by their neighbors they do not.
        let layout = Layout::of(4, &[(1, 2), (0, 3)]);
        let above: Vec<usize> = layout.layers[0]
            .iter()
            .filter_map(|s| match s {
                Slot::Node(n) => Some(*n),
                Slot::Lane(_) => None,
            })
            .collect();
        let below: Vec<usize> = layout.layers[1]
            .iter()
            .filter_map(|s| match s {
                Slot::Node(n) => Some(*n),
                Slot::Lane(_) => None,
            })
            .collect();
        let first_above = above[0];
        let hangs_from_first = if first_above == 0 { 3 } else { 2 };
        assert_eq!(below[0], hangs_from_first, "{above:?} over {below:?}");
        Ok(())
    }

    #[test]
    fn a_cycle_settles_and_one_of_its_edges_points_backward() -> anyhow::Result<()> {
        let layout = Layout::of(3, &[(0, 1), (1, 2), (2, 0)]);
        // Every node is placed once, and no layer is past the last possible.
        let mut order = layout.order();
        order.sort_unstable();
        assert_eq!(order, vec![0, 1, 2]);
        assert!(layout.layers.len() <= 3);
        // The backward edge is its two ends and nothing between.
        let backward = [(0, 1), (1, 2), (2, 0)]
            .into_iter()
            .find(|&(from, to)| layout.layer_of(to) <= layout.layer_of(from));
        let Some((from, to)) = backward else {
            anyhow::bail!("a cycle has an edge that points backward: {layout:?}");
        };
        assert_eq!(layout.path(from, to).len(), 2);
        Ok(())
    }

    #[test]
    fn an_edge_to_nowhere_or_to_itself_is_left_out() -> anyhow::Result<()> {
        let layout = Layout::of(2, &[(0, 0), (0, 7), (0, 1)]);
        assert_eq!(layout.layers.len(), 2);
        assert_eq!(Layout::of(0, &[]), Layout::default());
        Ok(())
    }
}
