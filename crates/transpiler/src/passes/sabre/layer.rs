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

use indexmap::IndexMap;
use ndarray::prelude::*;
use rustworkx_core::petgraph::prelude::*;

use qiskit_circuit::PhysicalQubit;

use super::vec_map::VecMap;

/// A container for the current non-routable parts of the front layer.  This only ever holds
/// two-qubit gates; the only reason a 0q- or 1q operation can be unroutable is because it has an
/// unsatisfied 2q predecessor, which disqualifies it from being in the front layer.
///
/// It would be more algorithmically natural for this struct to work in terms of virtual qubits,
/// because then a swap insertion would not change the data contained.  However, for each swap we
/// insert, we score tens or hundreds, yet the subsequent update only affects two qubits.  This
/// makes it more efficient to do everything in terms of physical qubits, so the conversion between
/// physical and virtual qubits via the layout happens once per inserted swap and on layer
/// extension, not for every swap trialled.
#[derive(Clone, Debug)]
pub struct Layer {
    /// Map of the (index to the) node to the qubits it acts on.
    nodes: IndexMap<NodeIndex, [PhysicalQubit; 2], ::ahash::RandomState>,
    /// Map of each qubit to the node that acts on it and the other qubit that node acts on, if this
    /// qubit is active (otherwise `None`).
    qubits: VecMap<PhysicalQubit, Option<(NodeIndex, PhysicalQubit)>>,
}

impl Layer {
    pub fn new(num_qubits: u32) -> Self {
        Layer {
            // This is the maximum capacity of the front layer, since each qubit must be one of a
            // pair, and can only have one gate in the layer.
            nodes: IndexMap::with_capacity_and_hasher(
                num_qubits as usize / 2,
                ::ahash::RandomState::default(),
            ),
            qubits: vec![None; num_qubits as usize].into(),
        }
    }

    /// Number of gates currently stored in the layer.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// View onto the mapping between qubits and their `(node, other_qubit)` pair.  Index `i`
    /// corresponds to physical qubit `i`.
    pub fn qubits(&self) -> &VecMap<PhysicalQubit, Option<(NodeIndex, PhysicalQubit)>> {
        &self.qubits
    }

    /// Add a node into the front layer, with the two qubits it operates on.
    pub fn insert(&mut self, index: NodeIndex, qubits: [PhysicalQubit; 2]) {
        let [a, b] = qubits;
        self.qubits[a] = Some((index, b));
        self.qubits[b] = Some((index, a));
        self.nodes.insert(index, qubits);
    }

    /// Remove a node from the front layer.
    pub fn remove(&mut self, index: &NodeIndex) {
        // The actual order in the indexmap doesn't matter as long as it's reproducible.
        // Swap-remove is more efficient than a full shift-remove.
        let [a, b] = self
            .nodes
            .swap_remove(index)
            .expect("Tried removing index that does not exist.");
        self.qubits[a] = None;
        self.qubits[b] = None;
    }

    /// Remove all nodes from the layer.
    pub fn clear(&mut self) {
        for (_, [a, b]) in self.nodes.drain(..) {
            self.qubits[a] = None;
            self.qubits[b] = None;
        }
    }

    /// Query whether a qubit has an active node.
    #[inline]
    pub fn is_active(&self, qubit: PhysicalQubit) -> bool {
        self.qubits[qubit].is_some()
    }

    /// Calculate the score _difference_ caused by this swap, compared to not making the swap.
    #[inline(always)]
    pub fn score(&self, swap: [PhysicalQubit; 2], dist: &ArrayView2<f64>) -> f64 {
        // At most there can be two affected gates in the front layer (one on each qubit in the
        // swap), since any gate whose closest path passes through the swapped qubit link has its
        // "virtual-qubit path" order changed, but not the total weight.  In theory, we should
        // never consider the same gate in both `if let` branches, because if we did, the gate would
        // already be routable.  It doesn't matter, though, because the two distances would be
        // equal anyway, so not affect the score.
        let [a, b] = swap;
        let mut total = 0.0;
        if let Some((_, c)) = self.qubits[a] {
            if c == b {
                return 0.0;
            }
            total += dist[[b.index(), c.index()]] - dist[[a.index(), c.index()]]
        }
        if let Some((_, c)) = self.qubits[b] {
            total += dist[[a.index(), c.index()]] - dist[[b.index(), c.index()]]
        }
        total
    }

    /// Calculate the total absolute of the current front layer on the given layer.
    pub fn total_score(&self, dist: &ArrayView2<f64>) -> f64 {
        self.iter()
            .map(|(_, &[a, b])| dist[[a.index(), b.index()]])
            .sum::<f64>()
    }

    /// Apply a physical swap to the current layout data structure.
    pub fn apply_swap(&mut self, swap: [PhysicalQubit; 2]) {
        let [a, b] = swap;
        match (self.qubits[a], self.qubits[b]) {
            (Some((index1, _)), Some((index2, _))) if index1 == index2 => {
                let entry = self.nodes.get_mut(&index1).unwrap();
                *entry = [entry[1], entry[0]];
                return;
            }
            _ => {}
        }
        if let Some((index, c)) = self.qubits[a] {
            self.qubits[c] = Some((index, b));
            let entry = self.nodes.get_mut(&index).unwrap();
            *entry = if *entry == [a, c] { [b, c] } else { [c, b] };
        }
        if let Some((index, c)) = self.qubits[b] {
            self.qubits[c] = Some((index, a));
            let entry = self.nodes.get_mut(&index).unwrap();
            *entry = if *entry == [b, c] { [a, c] } else { [c, a] };
        }
        self.qubits.swap(a, b);
    }

    /// True if there are no nodes in the current layer.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterator over the nodes and the pair of qubits they act on.
    pub fn iter(&self) -> impl Iterator<Item = (&NodeIndex, &[PhysicalQubit; 2])> {
        self.nodes.iter()
    }

    /// Iterator over the nodes.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &NodeIndex> {
        self.nodes.keys()
    }

    /// Iterator over the qubits that have active nodes on them.
    pub fn iter_active(&self) -> impl Iterator<Item = &PhysicalQubit> {
        self.nodes.values().flatten()
    }
}
