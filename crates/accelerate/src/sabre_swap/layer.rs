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
use hashbrown::HashMap;
use indexmap::IndexMap;
use ndarray::prelude::*;
use rustworkx_core::petgraph::prelude::*;

use crate::nlayout::NLayout;

/// A container for the current non-routable parts of the front layer.  This only ever holds
/// two-qubit gates; the only reason a 0q- or 1q operation can be unroutable is because it has an
/// unsatisfied 2q predecessor, which disqualifies it from being in the front layer.
pub struct FrontLayer {
    /// Map of the (index to the) node to the qubits it acts on.
    nodes: IndexMap<NodeIndex, [usize; 2], ahash::RandomState>,
    /// Map of each qubit to the node that acts on it and the other qubit that node acts on, if this
    /// qubit is active (otherwise `None`).
    qubits: Vec<Option<(NodeIndex, usize)>>,
}

impl FrontLayer {
    pub fn new(num_qubits: usize) -> Self {
        FrontLayer {
            // This is the maximum capacity of the front layer, since each qubit must be one of a
            // pair, and can only have one gate in the layer.
            nodes: IndexMap::with_capacity_and_hasher(
                num_qubits / 2,
                ahash::RandomState::default(),
            ),
            qubits: vec![None; num_qubits],
        }
    }

    /// Add a node into the front layer, with the two qubits it operates on.
    pub fn insert(&mut self, index: NodeIndex, qubits: [usize; 2]) {
        let [a, b] = qubits;
        self.qubits[a] = Some((index, b));
        self.qubits[b] = Some((index, a));
        self.nodes.insert(index, qubits);
    }

    /// Remove a node from the front layer.
    pub fn remove(&mut self, index: &NodeIndex) {
        let [q0, q1] = self.nodes.remove(index).unwrap();
        self.qubits[q0] = None;
        self.qubits[q1] = None;
    }

    /// Query whether a qubit has an active node.
    #[inline]
    pub fn is_active(&self, qubit: usize) -> bool {
        self.qubits[qubit].is_some()
    }

    /// Calculate the score _difference_ caused by this swap, compared to not making the swap.
    #[inline]
    pub fn score(&self, swap: [usize; 2], layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
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
        if let Some((_, c)) = self.qubits[a] {
            let p_c = layout.logic_to_phys[c];
            total += dist[[layout.logic_to_phys[b], p_c]] - dist[[layout.logic_to_phys[a], p_c]]
        }
        if let Some((_, c)) = self.qubits[b] {
            let p_c = layout.logic_to_phys[c];
            total += dist[[layout.logic_to_phys[a], p_c]] - dist[[layout.logic_to_phys[b], p_c]]
        }
        total / self.nodes.len() as f64
    }

    /// Calculate the total absolute of the current front layer on the given layer.
    pub fn total_score(&self, layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.iter()
            .map(|(_, &[l_a, l_b])| dist[[layout.logic_to_phys[l_a], layout.logic_to_phys[l_b]]])
            .sum::<f64>()
            / self.nodes.len() as f64
    }

    /// Populate a of nodes that would be routable if the given swap was applied to a layout.  This
    /// mutates `routable` to avoid heap allocations in the main logic loop.
    pub fn routable_after(
        &self,
        routable: &mut Vec<NodeIndex>,
        swap: &[usize; 2],
        layout: &NLayout,
        coupling: &DiGraph<(), ()>,
    ) {
        let [a, b] = *swap;
        if let Some((node, c)) = self.qubits[a] {
            if coupling.contains_edge(
                NodeIndex::new(layout.logic_to_phys[b]),
                NodeIndex::new(layout.logic_to_phys[c]),
            ) {
                routable.push(node);
            }
        }
        if let Some((node, c)) = self.qubits[b] {
            if coupling.contains_edge(
                NodeIndex::new(layout.logic_to_phys[a]),
                NodeIndex::new(layout.logic_to_phys[c]),
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
    pub fn iter(&self) -> impl Iterator<Item = (&NodeIndex, &[usize; 2])> {
        self.nodes.iter()
    }

    /// Iterator over the nodes.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &NodeIndex> {
        self.nodes.keys()
    }

    /// Iterator over the qubits that have active nodes on them.
    pub fn iter_active(&self) -> impl Iterator<Item = &usize> {
        self.nodes.values().flatten()
    }
}

/// This is largely similar to the `FrontLayer` struct, but does not need to track the insertion
/// order of the nodes, and can have more than one node on each active qubit.  This does not have a
/// `remove` method (and its data structures aren't optimised for fast removal), since the extended
/// set is built from scratch each time a new gate is routed.
pub struct ExtendedSet {
    nodes: HashMap<NodeIndex, [usize; 2]>,
    qubits: Vec<Vec<usize>>,
}

impl ExtendedSet {
    pub fn new(num_qubits: usize, max_size: usize) -> Self {
        ExtendedSet {
            nodes: HashMap::with_capacity(max_size),
            qubits: vec![Vec::new(); num_qubits],
        }
    }

    /// Add a node and its active qubits to the extended set.
    pub fn insert(&mut self, index: NodeIndex, qubits: &[usize; 2]) -> bool {
        let [a, b] = *qubits;
        if self.nodes.insert(index, *qubits).is_none() {
            self.qubits[a].push(b);
            self.qubits[b].push(a);
            true
        } else {
            false
        }
    }

    /// Calculate the score of applying the given swap, relative to not applying it.
    pub fn score(&self, swap: [usize; 2], layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let [l_a, l_b] = swap;
        let p_a = layout.logic_to_phys[l_a];
        let p_b = layout.logic_to_phys[l_b];
        let mut total = 0.0;
        for &l_other in self.qubits[l_a].iter() {
            // If the other qubit is also active then the score won't have changed, but since the
            // distance is absolute, we'd double count rather than ignore if we didn't skip it.
            if l_other == l_b {
                continue;
            }
            let p_other = layout.logic_to_phys[l_other];
            total += dist[[p_b, p_other]] - dist[[p_a, p_other]];
        }
        for &l_other in self.qubits[l_b].iter() {
            if l_other == l_a {
                continue;
            }
            let p_other = layout.logic_to_phys[l_other];
            total += dist[[p_a, p_other]] - dist[[p_b, p_other]];
        }
        total / self.nodes.len() as f64
    }

    /// Calculate the total absolute score of this set of nodes over the given layout.
    pub fn total_score(&self, layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.nodes
            .iter()
            .map(|(_, &[l_a, l_b])| dist[[layout.logic_to_phys[l_a], layout.logic_to_phys[l_b]]])
            .sum::<f64>()
            / self.nodes.len() as f64
    }

    /// Clear all nodes from the extended set.
    pub fn clear(&mut self) {
        for &[a, b] in self.nodes.values() {
            self.qubits[a].clear();
            self.qubits[b].clear();
        }
        self.nodes.clear()
    }

    /// Number of nodes in the set.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}
