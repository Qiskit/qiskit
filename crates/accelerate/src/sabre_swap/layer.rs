// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ahash;
use indexmap::IndexMap;
use ndarray::prelude::*;
use rustworkx_core::petgraph::prelude::*;

use crate::nlayout::{NLayout, VirtualQubit};

/// A container for the current non-routable parts of the front layer.  This only ever holds
/// two-qubit gates; the only reason a 0q- or 1q operation can be unroutable is because it has an
/// unsatisfied 2q predecessor, which disqualifies it from being in the front layer.
pub struct FrontLayer {
    /// Map of the (index to the) node to the qubits it acts on.
    nodes: IndexMap<NodeIndex, [VirtualQubit; 2], ahash::RandomState>,
    /// Map of each qubit to the node that acts on it and the other qubit that node acts on, if this
    /// qubit is active (otherwise `None`).
    qubits: Vec<Option<(NodeIndex, VirtualQubit)>>,
}

impl FrontLayer {
    pub fn new(num_qubits: u32) -> Self {
        FrontLayer {
            // This is the maximum capacity of the front layer, since each qubit must be one of a
            // pair, and can only have one gate in the layer.
            nodes: IndexMap::with_capacity_and_hasher(
                num_qubits as usize / 2,
                ahash::RandomState::default(),
            ),
            qubits: vec![None; num_qubits as usize],
        }
    }

    /// Add a node into the front layer, with the two qubits it operates on.
    pub fn insert(&mut self, index: NodeIndex, qubits: [VirtualQubit; 2]) {
        let [a, b] = qubits;
        self.qubits[a.index()] = Some((index, b));
        self.qubits[b.index()] = Some((index, a));
        self.nodes.insert(index, qubits);
    }

    /// Remove a node from the front layer.
    pub fn remove(&mut self, index: &NodeIndex) {
        let [q0, q1] = self.nodes.remove(index).unwrap();
        self.qubits[q0.index()] = None;
        self.qubits[q1.index()] = None;
    }

    /// Query whether a qubit has an active node.
    #[inline]
    pub fn is_active(&self, qubit: VirtualQubit) -> bool {
        self.qubits[qubit.index()].is_some()
    }

    /// Calculate the score _difference_ caused by this swap, compared to not making the swap.
    #[inline]
    pub fn score(&self, swap: [VirtualQubit; 2], layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        // At most there can be two affected gates in the front layer (one on each qubit in the
        // swap), since any gate whose closest path passes through the swapped qubit link has its
        // "virtual-qubit path" order changed, but not the total weight.  In theory, we should
        // never consider the same gate in both `if let` branches, because if we did, the gate would
        // already be routable.  It doesn't matter, though, because the two distances would be
        // equal anyway, so not affect the score.
        let [a, b] = swap;
        let mut total = 0.0;
        if let Some((_, c)) = self.qubits[a.index()] {
            let p_c = c.to_phys(layout);
            total += dist[[b.to_phys(layout).index(), p_c.index()]]
                - dist[[a.to_phys(layout).index(), p_c.index()]]
        }
        if let Some((_, c)) = self.qubits[b.index()] {
            let p_c = c.to_phys(layout);
            total += dist[[a.to_phys(layout).index(), p_c.index()]]
                - dist[[b.to_phys(layout).index(), p_c.index()]]
        }
        total / self.nodes.len() as f64
    }

    /// Calculate the total absolute of the current front layer on the given layer.
    pub fn total_score(&self, layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.iter()
            .map(|(_, &[a, b])| dist[[a.to_phys(layout).index(), b.to_phys(layout).index()]])
            .sum::<f64>()
            / self.nodes.len() as f64
    }

    /// Populate a of nodes that would be routable if the given swap was applied to a layout.  This
    /// mutates `routable` to avoid heap allocations in the main logic loop.
    pub fn routable_after(
        &self,
        routable: &mut Vec<NodeIndex>,
        swap: &[VirtualQubit; 2],
        layout: &NLayout,
        coupling: &DiGraph<(), ()>,
    ) {
        let [a, b] = *swap;
        if let Some((node, c)) = self.qubits[a.index()] {
            if coupling.contains_edge(
                NodeIndex::new(b.to_phys(layout).index()),
                NodeIndex::new(c.to_phys(layout).index()),
            ) {
                routable.push(node);
            }
        }
        if let Some((node, c)) = self.qubits[b.index()] {
            if coupling.contains_edge(
                NodeIndex::new(a.to_phys(layout).index()),
                NodeIndex::new(c.to_phys(layout).index()),
            ) {
                routable.push(node);
            }
        }
    }

    /// True if there are no nodes in the current layer.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterator over the nodes and the pair of qubits they act on.
    pub fn iter(&self) -> impl Iterator<Item = (&NodeIndex, &[VirtualQubit; 2])> {
        self.nodes.iter()
    }

    /// Iterator over the nodes.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &NodeIndex> {
        self.nodes.keys()
    }

    /// Iterator over the qubits that have active nodes on them.
    pub fn iter_active(&self) -> impl Iterator<Item = &VirtualQubit> {
        self.nodes.values().flatten()
    }
}

/// This is largely similar to the `FrontLayer` struct but can have more than one node on each active
/// qubit.  This does not have `remove` method (and its data structures aren't optimised for fast
/// removal), since the extended set is built from scratch each time a new gate is routed.
pub struct ExtendedSet {
    nodes: IndexMap<NodeIndex, [VirtualQubit; 2], ahash::RandomState>,
    qubits: Vec<Vec<VirtualQubit>>,
}

impl ExtendedSet {
    pub fn new(num_qubits: u32, max_size: usize) -> Self {
        ExtendedSet {
            nodes: IndexMap::with_capacity_and_hasher(max_size, ahash::RandomState::default()),
            qubits: vec![Vec::new(); num_qubits as usize],
        }
    }

    /// Add a node and its active qubits to the extended set.
    pub fn insert(&mut self, index: NodeIndex, qubits: &[VirtualQubit; 2]) -> bool {
        let [a, b] = *qubits;
        if self.nodes.insert(index, *qubits).is_none() {
            self.qubits[a.index()].push(b);
            self.qubits[b.index()].push(a);
            true
        } else {
            false
        }
    }

    /// Calculate the score of applying the given swap, relative to not applying it.
    pub fn score(&self, swap: [VirtualQubit; 2], layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let [a, b] = swap;
        let p_a = a.to_phys(layout);
        let p_b = b.to_phys(layout);
        let mut total = 0.0;
        for other in self.qubits[a.index()].iter() {
            // If the other qubit is also active then the score won't have changed, but since the
            // distance is absolute, we'd double count rather than ignore if we didn't skip it.
            if *other == b {
                continue;
            }
            let p_other = other.to_phys(layout);
            total += dist[[p_b.index(), p_other.index()]] - dist[[p_a.index(), p_other.index()]];
        }
        for other in self.qubits[b.index()].iter() {
            if *other == a {
                continue;
            }
            let p_other = other.to_phys(layout);
            total += dist[[p_a.index(), p_other.index()]] - dist[[p_b.index(), p_other.index()]];
        }
        total / self.nodes.len() as f64
    }

    /// Calculate the total absolute score of this set of nodes over the given layout.
    pub fn total_score(&self, layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.nodes
            .values()
            .map(|&[a, b]| dist[[a.to_phys(layout).index(), b.to_phys(layout).index()]])
            .sum::<f64>()
            / self.nodes.len() as f64
    }

    /// Clear all nodes from the extended set.
    pub fn clear(&mut self) {
        for &[a, b] in self.nodes.values() {
            self.qubits[a.index()].clear();
            self.qubits[b.index()].clear();
        }
        self.nodes.clear()
    }

    /// Number of nodes in the set.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}
