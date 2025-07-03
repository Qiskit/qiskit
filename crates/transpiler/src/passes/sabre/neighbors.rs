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

use qiskit_circuit::PhysicalQubit;
use rustworkx_core::petgraph::visit::*;

/// A hash-free fixed-size sparse adjacency-list representation of the neighbors of a node.
///
/// This is similar in spirit to a graph representation where the qubits are the nodes and
/// undirected edges are the links, but all the edges are explicitly duplicated in order to improve
/// their locality.  The storage mechanism is flat vector, with its partition points indicated by a
/// separate vector.  It's similar to petgraph's `Csr` format, but has fixed data locality for us.
///
/// All interaction is done using the [Index] impl by [PhysicalQubit]; this returns a sorted slice
/// containing the neighbours.  Looking whether a node _is_ a neighbour is done by iterating through
/// the slice (it's allowable to binary search, but in practice the degree is likely sufficiently
/// small that a linear search is faster).
#[derive(Clone, Debug)]
pub struct Neighbors {
    neighbors: Vec<PhysicalQubit>,
    partition: Vec<usize>,
}
impl Neighbors {
    /// Construct the neighbor adjacency table from a coupling graph.
    ///
    /// The [PhysicalQubit] instances in the resulting [Neighbors] refer to the indices of the
    /// incoming [G::NodeId]s of the input graph.
    #[inline]
    pub fn from_coupling<G>(coupling: G) -> Self
    where
        G: Copy + NodeCompactIndexable + IntoNeighbors,
    {
        Self::from_coupling_with_maps(
            coupling,
            coupling.node_count(),
            |i| Some(PhysicalQubit::new(coupling.to_index(i) as u32)),
            |q| coupling.from_index(q.index()),
        )
    }
    /// Construct the neighbor adjacency table from a coupling graph subset.
    ///
    /// The [PhysicalQubit] instances in the resulting [Neighbors] refer to the indices into the
    /// `subset`.  Bear in mind that these might not actually be the original [PhysicalQubit] ids.
    ///
    /// # Panics
    ///
    /// If `subset` contains duplicates.
    #[inline]
    pub fn from_coupling_subset<G>(coupling: G, subset: &[G::NodeId]) -> Self
    where
        G: NodeIndexable + IntoNeighbors,
    {
        Self::from_coupling_subset_with_map(coupling, subset, |id| *id)
    }

    /// Construct the neighbor adjacency table from a coupling graph subset, where the subset is
    /// known in terms of a type that maps into the node indices of the coupling graph.
    ///
    /// The [PhysicalQubit] instances in the resulting [Neighbors] refer to the indices into the
    /// `subset`.  Bear in mind that these might not actually be the original [PhysicalQubit] ids.
    ///
    /// # Panics
    ///
    /// If `subset` contains duplicates after mapping.
    pub fn from_coupling_subset_with_map<G, T>(
        coupling: G,
        subset: &[T],
        map_fn: impl Fn(&T) -> G::NodeId,
    ) -> Self
    where
        G: NodeIndexable + IntoNeighbors,
    {
        let max = PhysicalQubit::new(u32::MAX);
        let mut compact = vec![
            max;
            subset
                .iter()
                // `+ 1` because we're calculating the length.
                .map(|i| coupling.to_index(map_fn(i)) + 1)
                .max()
                .unwrap_or_default()
        ];
        for (q, id) in subset.iter().enumerate() {
            let index = coupling.to_index(map_fn(id));
            if compact[index] != max {
                panic!("duplicate qubit in subset");
            }
            compact[index] = PhysicalQubit::new(q as u32);
        }
        Self::from_coupling_with_maps(
            coupling,
            subset.len(),
            |i| {
                compact
                    .get(coupling.to_index(i))
                    .copied()
                    .filter(|q| *q != max)
            },
            |q| map_fn(&subset[q.index()]),
        )
    }

    // This shouldn't be public interface; `compact_fn` and `expand_fn` are tightly coupled to be
    // inverses of each other, and the support of `expand_fn` is required to be `0..num_qubits`.
    // This is trivially realisable for no-subsetting and subset-by-slice, though.
    fn from_coupling_with_maps<G>(
        coupling: G,
        num_qubits: usize,
        compact_fn: impl Fn(G::NodeId) -> Option<PhysicalQubit>,
        expand_fn: impl Fn(PhysicalQubit) -> G::NodeId,
    ) -> Self
    where
        G: IntoNeighbors,
    {
        let mut neighbors = Vec::new();
        // The `+ 1` is to store an initial zero.
        let mut partition = Vec::with_capacity(num_qubits + 1);
        partition.push(0);
        for qubit in 0..num_qubits {
            let qubit = PhysicalQubit::new(qubit as u32);
            let node = expand_fn(qubit);
            for neighbor in coupling.neighbors(node).filter_map(&compact_fn) {
                neighbors.push(neighbor);
            }
            partition.push(neighbors.len());
            // Sort per neighbour in the vague hope that branch predicition later will be more
            // reliable, or memory access patterns will be more predictable.
            neighbors[partition[partition.len() - 2]..partition[partition.len() - 1]].sort();
        }
        Self {
            neighbors,
            partition,
        }
    }

    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.partition.len() - 1
    }

    /// Are two qubits neighbors?
    ///
    /// Computes in average time linear to the number of neighbors of `left`.
    #[inline]
    pub fn contains_edge(&self, left: PhysicalQubit, right: PhysicalQubit) -> bool {
        // Strictly this would asymptotically scale better as a binary search, but in practice we
        // don't expect the number of neighbours to be large enough to overcome the branching costs.
        self[left].contains(&right)
    }
}
impl std::ops::Index<PhysicalQubit> for Neighbors {
    type Output = [PhysicalQubit];

    #[inline]
    fn index(&self, index: PhysicalQubit) -> &Self::Output {
        let index = index.index();
        &self.neighbors[self.partition[index]..self.partition[index + 1]]
    }
}

/// Implementations of the various `petgraph` graph-visiting traits, which makes [Neighbors] (or
/// more specifically, `&Neighbors`) a directly usable graph object in most places we care about.
mod visit {
    use super::*;
    use qiskit_circuit::PhysicalQubit;

    use fixedbitset::FixedBitSet;
    use rustworkx_core::petgraph::prelude::*;
    use rustworkx_core::petgraph::visit;

    impl visit::GraphBase for Neighbors {
        type NodeId = PhysicalQubit;
        type EdgeId = usize;
    }
    impl visit::Data for Neighbors {
        type NodeWeight = ();
        type EdgeWeight = ();
    }
    impl visit::NodeIndexable for Neighbors {
        #[inline]
        fn node_bound(&self) -> usize {
            self.num_qubits()
        }
        fn to_index(&self, a: PhysicalQubit) -> usize {
            a.index()
        }
        fn from_index(&self, i: usize) -> PhysicalQubit {
            PhysicalQubit::new(i as u32)
        }
    }
    impl visit::NodeCompactIndexable for Neighbors {}
    impl visit::NodeCount for Neighbors {
        #[inline]
        fn node_count(&self) -> usize {
            self.num_qubits()
        }
    }
    impl visit::EdgeCount for Neighbors {
        #[inline]
        fn edge_count(&self) -> usize {
            self.neighbors.len()
        }
    }
    impl visit::Visitable for Neighbors {
        type Map = FixedBitSet;
        #[inline]
        fn visit_map(&self) -> FixedBitSet {
            FixedBitSet::with_capacity(self.num_qubits())
        }
        #[inline]
        fn reset_map(&self, map: &mut FixedBitSet) {
            map.clear();
            map.grow(self.num_qubits())
        }
    }

    impl<'a> visit::IntoNeighbors for &'a Neighbors {
        // On this one, we get a little help from `std`.
        type Neighbors = ::std::iter::Copied<::std::slice::Iter<'a, PhysicalQubit>>;
        fn neighbors(self, a: PhysicalQubit) -> Self::Neighbors {
            self[a].iter().copied()
        }
    }
    impl visit::IntoNeighborsDirected for &'_ Neighbors {
        // We're logically undirected.
        type NeighborsDirected = <Self as visit::IntoNeighbors>::Neighbors;
        #[inline]
        fn neighbors_directed(self, n: PhysicalQubit, _: Direction) -> Self::NeighborsDirected {
            <Self as visit::IntoNeighbors>::neighbors(self, n)
        }
    }

    impl visit::IntoNodeIdentifiers for &'_ Neighbors {
        type NodeIdentifiers = NodeIdentifiers;
        #[inline]
        fn node_identifiers(self) -> Self::NodeIdentifiers {
            NodeIdentifiers(0..self.num_qubits() as u32)
        }
    }
    pub struct NodeIdentifiers(::std::ops::Range<u32>);
    impl Iterator for NodeIdentifiers {
        type Item = PhysicalQubit;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().map(PhysicalQubit::new)
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.size_hint()
        }
    }
    impl ExactSizeIterator for NodeIdentifiers {}

    impl<'a> visit::IntoEdgeReferences for &'a Neighbors {
        type EdgeRef = EdgeRef;
        type EdgeReferences = EdgeReferences<'a>;
        #[inline]
        fn edge_references(self) -> Self::EdgeReferences {
            EdgeReferences {
                neighbors: self,
                source: PhysicalQubit::new(0),
                n: 0,
            }
        }
    }
    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    pub struct EdgeRef {
        nodes: [PhysicalQubit; 2],
        id: usize,
    }
    impl visit::EdgeRef for EdgeRef {
        type NodeId = PhysicalQubit;
        type EdgeId = usize;
        type Weight = ();
        #[inline]
        fn source(&self) -> PhysicalQubit {
            self.nodes[0]
        }
        #[inline]
        fn target(&self) -> PhysicalQubit {
            self.nodes[1]
        }
        #[inline]
        fn weight(&self) -> &() {
            &()
        }
        #[inline]
        fn id(&self) -> usize {
            self.id
        }
    }
    #[derive(Clone, Debug)]
    pub struct EdgeReferences<'a> {
        neighbors: &'a Neighbors,
        source: PhysicalQubit,
        // Counter of how far through the `source` slice we are.
        n: usize,
    }
    impl EdgeReferences<'_> {
        #[inline]
        fn id(&self) -> usize {
            self.neighbors.partition[self.source.index()] + self.n
        }
    }
    impl Iterator for EdgeReferences<'_> {
        type Item = EdgeRef;
        fn next(&mut self) -> Option<EdgeRef> {
            while self.source.index() < self.neighbors.num_qubits() {
                if let Some(other) = self.neighbors[self.source].get(self.n) {
                    let id = self.neighbors.partition[self.source.index()] + self.n;
                    self.n += 1;
                    return Some(EdgeRef {
                        nodes: [self.source, *other],
                        id,
                    });
                }
                self.source = PhysicalQubit::new(self.source.index() as u32 + 1);
                self.n = 0;
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let rem = self.neighbors.neighbors.len() - self.id();
            (rem, Some(rem))
        }
    }
    impl ExactSizeIterator for EdgeReferences<'_> {}

    impl<'a> visit::IntoEdges for &'a Neighbors {
        type Edges = Edges<'a>;
        #[inline]
        fn edges(self, a: PhysicalQubit) -> Self::Edges {
            Edges {
                source: a,
                neighbors: self[a].iter(),
                id: self.partition[a.index()],
            }
        }
    }
    pub struct Edges<'a> {
        source: PhysicalQubit,
        neighbors: ::std::slice::Iter<'a, PhysicalQubit>,
        id: usize,
    }
    impl Iterator for Edges<'_> {
        type Item = EdgeRef;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let target = self.neighbors.next()?;
            let id = self.id;
            self.id += 1;
            Some(EdgeRef {
                nodes: [self.source, *target],
                id,
            })
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.neighbors.size_hint()
        }
    }
    impl ExactSizeIterator for Edges<'_> {}
}
