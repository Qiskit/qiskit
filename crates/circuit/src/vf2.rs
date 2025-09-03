// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// This module was originally forked into Rustworkx from petgraph's isomorphism module @ v0.5.0
// to handle Rustworkx PyDiGraph inputs instead of petgraph's generic Graph.  It was modified
// substantially in Rustworkx.  It was then forked from Rustworkx to Qiskit from
//
//    https://github.com/Qiskit/rustworkx/blob/9f0646e8886cfecc55e59b96532c6f7f798524c0/src/isomorphism/vf2.rs
//
// It has subsequently been near-completely written within Qiskit.

use std::cmp::{Ordering, Reverse};
use std::convert::Infallible;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::iter::Iterator;
use std::marker;
use std::ops::Deref;

use hashbrown::{hash_map::Entry, HashMap};
use indexmap::IndexMap;
use smallvec::SmallVec;

use rustworkx_core::petgraph::data::{Build, Create, DataMap};
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::visit::{
    Data, EdgeCount, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, IntoEdgesDirected,
    IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable,
};
use rustworkx_core::petgraph::{Direction, Incoming, Outgoing};

use rayon::slice::ParallelSliceMut;

pub trait NodeSorter<G>
where
    G: GraphBase<NodeId = NodeIndex> + DataMap + NodeCount + EdgeCount + IntoEdgeReferences,
    G::NodeWeight: Clone,
    G::EdgeWeight: Clone,
{
    type OutputGraph: GraphBase<NodeId = NodeIndex>
        + Create
        + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>;

    fn sort(&self, _: G) -> Vec<NodeIndex>;

    fn reorder(&self, graph: G) -> (Self::OutputGraph, HashMap<NodeIndex, NodeIndex>) {
        let order = self.sort(graph);

        let mut new_graph =
            Self::OutputGraph::with_capacity(graph.node_count(), graph.edge_count());
        let mut id_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(graph.node_count());
        for node_index in order {
            let node_data = graph.node_weight(node_index).unwrap();
            let new_index = new_graph.add_node(node_data.clone());
            id_map.insert(node_index, new_index);
        }
        for edge in graph.edge_references() {
            let edge_w = edge.weight();
            let p_index = id_map[&edge.source()];
            let c_index = id_map[&edge.target()];
            new_graph.add_edge(p_index, c_index, edge_w.clone());
        }
        (
            new_graph,
            id_map.into_iter().map(|(old, new)| (new, old)).collect(),
        )
    }
}

/// Sort nodes based on node ids.
pub struct DefaultIdSorter;
impl<G> NodeSorter<G> for DefaultIdSorter
where
    G: Deref
        + GraphBase<NodeId = NodeIndex>
        + DataMap
        + NodeCount
        + EdgeCount
        + IntoEdgeReferences
        + IntoNodeIdentifiers,
    G::Target: GraphBase<NodeId = NodeIndex>
        + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>
        + Create,
    G::NodeWeight: Clone,
    G::EdgeWeight: Clone,
{
    type OutputGraph = G::Target;
    fn sort(&self, graph: G) -> Vec<NodeIndex> {
        graph.node_identifiers().collect()
    }
}

/// Sort nodes based on VF2++ heuristic.
pub struct Vf2ppSorter;
impl<G> NodeSorter<G> for Vf2ppSorter
where
    G: Deref
        + GraphProp
        + GraphBase<NodeId = NodeIndex>
        + DataMap
        + NodeCount
        + NodeIndexable
        + EdgeCount
        + IntoNodeIdentifiers
        + IntoEdgesDirected,
    G::Target: GraphBase<NodeId = NodeIndex>
        + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>
        + Create,
    G::NodeWeight: Clone,
    G::EdgeWeight: Clone,
{
    type OutputGraph = G::Target;
    fn sort(&self, graph: G) -> Vec<NodeIndex> {
        let n = graph.node_bound();

        let dout: Vec<usize> = (0..n)
            .map(|idx| {
                graph
                    .neighbors_directed(graph.from_index(idx), Outgoing)
                    .count()
            })
            .collect();

        let mut din: Vec<usize> = vec![0; n];
        if graph.is_directed() {
            din = (0..n)
                .map(|idx| {
                    graph
                        .neighbors_directed(graph.from_index(idx), Incoming)
                        .count()
                })
                .collect();
        }

        let mut conn_in: Vec<usize> = vec![0; n];
        let mut conn_out: Vec<usize> = vec![0; n];

        let mut order: Vec<NodeIndex> = Vec::with_capacity(n);

        // Process BFS level
        let mut process = |mut vd: Vec<usize>| -> Vec<usize> {
            // repeatedly bring largest element in front.
            for i in 0..vd.len() {
                let (index, &item) = vd[i..]
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &node)| {
                        (
                            conn_in[node],
                            dout[node],
                            conn_out[node],
                            din[node],
                            Reverse(node),
                        )
                    })
                    .unwrap();

                vd.swap(i, i + index);
                order.push(NodeIndex::new(item));

                for neigh in graph.neighbors_directed(graph.from_index(item), Outgoing) {
                    conn_in[graph.to_index(neigh)] += 1;
                }

                if graph.is_directed() {
                    for neigh in graph.neighbors_directed(graph.from_index(item), Incoming) {
                        conn_out[graph.to_index(neigh)] += 1;
                    }
                }
            }
            vd
        };

        let mut seen: Vec<bool> = vec![false; n];

        // Create BFS Tree from root and process each level.
        let mut bfs_tree = |root: usize| {
            if seen[root] {
                return;
            }

            let mut next_level: Vec<usize> = Vec::new();

            seen[root] = true;
            next_level.push(root);
            while !next_level.is_empty() {
                let this_level = next_level;
                let this_level = process(this_level);

                next_level = Vec::new();
                for bfs_node in this_level {
                    for neighbor in graph.neighbors_directed(graph.from_index(bfs_node), Outgoing) {
                        let neigh = graph.to_index(neighbor);
                        if !seen[neigh] {
                            seen[neigh] = true;
                            next_level.push(neigh);
                        }
                    }
                }
            }
        };

        let mut sorted_nodes: Vec<usize> =
            graph.node_identifiers().map(|node| node.index()).collect();
        sorted_nodes.par_sort_by_key(|&node| (dout[node], din[node], Reverse(node)));
        sorted_nodes.reverse();

        for node in sorted_nodes {
            bfs_tree(node);
        }

        order
    }
}

#[derive(Debug)]
struct Vf2State<G> {
    graph: G,
    reorder: HashMap<NodeIndex, NodeIndex>,
    /// The current mapping from indices in this graph to indices in the other graph.  If a node is
    /// not yet mapped, the other index is stored as `NodeIndex::end`.
    mapping: Vec<NodeIndex>,
    /// Mapping from node index to the generation at which a node was first added to the mapping
    /// that had an outbound edge to that index.  This can be used to find new candidate nodes to
    /// add to the mapping; you typically want your next node to be one that has edges linking it to
    /// the existing mapping (but isn't yet _in_ the mapping).
    out: Vec<usize>,
    /// Same as `out`, except we're tracking the nodes that have an edge from them _into_ the
    /// mapping.  This isn't used if the graph is undirected, since it'd just duplicate `out`.
    ins: Vec<usize>,
    /// The number of non-zero entries in `out`.
    out_size: usize,
    /// The number of non-zero entries in `in`.  This is always zero for undirected graphs.
    ins_size: usize,
    /// The edge multiplicity of a given node pair.  If the graph is directed, the keys are
    /// `(source, target)`.  If the graph is undirected, the keys are always in sorted order, and
    /// the multiplicity includes both "directions" of the edge.
    adjacency_matrix: HashMap<(NodeIndex, NodeIndex), usize>,
    /// Is this a multigraph?
    multigraph: bool,
    /// The number of nodes in currently in the mapping.
    generation: usize,
}

impl<G> Vf2State<G>
where
    G: GraphBase<NodeId = NodeIndex> + GraphProp + NodeCount + EdgeCount,
    for<'a> &'a G:
        GraphBase<NodeId = NodeIndex> + GraphProp + NodeCount + EdgeCount + IntoEdgesDirected,
{
    pub fn new(graph: G, reorder: HashMap<NodeIndex, NodeIndex>) -> Self {
        let c0 = graph.node_count();
        let is_directed = graph.is_directed();
        let mut adjacency_matrix = HashMap::with_capacity(graph.edge_count());
        let mut multigraph = false;
        for edge in graph.edge_references() {
            let item = if graph.is_directed() || edge.source() <= edge.target() {
                (edge.source(), edge.target())
            } else {
                (edge.target(), edge.source())
            };
            match adjacency_matrix.entry(item) {
                Entry::Vacant(entry) => {
                    entry.insert(1);
                }
                Entry::Occupied(mut entry) => {
                    multigraph = true;
                    *entry.get_mut() += 1;
                }
            }
        }
        Vf2State {
            graph,
            reorder,
            mapping: vec![NodeIndex::end(); c0],
            out: vec![0; c0],
            ins: if is_directed { vec![0; c0] } else { vec![] },
            out_size: 0,
            ins_size: 0,
            adjacency_matrix,
            multigraph,
            generation: 0,
        }
    }

    /// Find the mapping (in the other graph) of the `target` of a local edge.
    ///
    /// If the edge is a self-loop, return the mapping of `source`, if provided.
    #[inline]
    pub fn map_target(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        mapped_source: Option<NodeIndex>,
    ) -> Option<NodeIndex> {
        if source == target {
            mapped_source
        } else {
            let other = self.mapping[target.index()];
            (other != NodeIndex::end()).then_some(other)
        }
    }

    /// Is every node in the graph mapped?
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.generation == self.mapping.len()
    }

    /// Add a new entry into the mapping.
    pub fn push_mapping(&mut self, ours: NodeIndex, theirs: NodeIndex) {
        self.generation += 1;
        debug_assert_eq!(self.mapping[ours.index()], NodeIndex::end());
        self.mapping[ours.index()] = theirs;
        // Mark any nodes that are newly neighbors of the set of mapped nodes.  To be _newly_ a
        // neighbor, it must not already be a neighbor.
        for ix in self.graph.neighbors(ours) {
            if self.out[ix.index()] == 0 {
                self.out[ix.index()] = self.generation;
                self.out_size += 1;
            }
        }
        if self.graph.is_directed() {
            for ix in self.graph.neighbors_directed(ours, Incoming) {
                if self.ins[ix.index()] == 0 {
                    self.ins[ix.index()] = self.generation;
                    self.ins_size += 1;
                }
            }
        }
    }

    /// Undo the mapping of node `ours`.  The node `ours` must be the last one given to
    /// `push_mapping` for this to make sense.
    pub fn pop_mapping(&mut self, ours: NodeIndex) {
        // Any neighbors of ours that became neighbors of the mapping at our generation are now no
        // longer neighbors of the mapping, since all the nodes that got added to the mapping after
        // us are already popped.
        for ix in self.graph.neighbors(ours) {
            if self.out[ix.index()] == self.generation {
                self.out[ix.index()] = 0;
                self.out_size -= 1;
            }
        }
        if self.graph.is_directed() {
            for ix in self.graph.neighbors_directed(ours, Incoming) {
                if self.ins[ix.index()] == self.generation {
                    self.ins[ix.index()] = 0;
                    self.ins_size -= 1;
                }
            }
        }
        self.mapping[ours.index()] = NodeIndex::end();
        self.generation -= 1;
    }

    /// Get the next unmapped node in the priority queue from a specific list whose index is at
    /// least `start`.
    fn next_unmapped_from(&self, start: usize, list: Option<OpenList>) -> Option<NodeIndex> {
        let unmapped = NodeIndex::end();
        let filter = |(offset, &generation)| -> Option<NodeIndex> {
            let index = start + offset;
            ((generation > 0usize) && self.mapping[index] == unmapped)
                .then_some(NodeIndex::new(index))
        };
        match list {
            Some(OpenList::Out) => self.out[start..]
                .iter()
                .enumerate()
                .filter_map(filter)
                .next(),
            Some(OpenList::In) => {
                if self.graph.is_directed() {
                    self.ins[start..]
                        .iter()
                        .enumerate()
                        .filter_map(filter)
                        .next()
                } else {
                    None
                }
            }
            None => self.mapping[start..]
                .iter()
                .enumerate()
                .filter_map(|(offset, &theirs)| {
                    (theirs == unmapped).then_some(NodeIndex::new(start + offset))
                })
                .next(),
        }
    }

    /// Get the first unmapped node in a given list (or the set of all nodes), if any.
    #[inline]
    pub fn first_unmapped(&self, list: Option<OpenList>) -> Option<NodeIndex> {
        self.next_unmapped_from(0, list)
    }
    /// Get the next unmapped node in a given list (or set of all nodes) that comes after `node` in
    /// the priority queue.
    #[inline]
    pub fn next_unmapped_after(
        &self,
        node: NodeIndex,
        list: Option<OpenList>,
    ) -> Option<NodeIndex> {
        self.next_unmapped_from(node.index() + 1, list)
    }

    /// Number of edges from `source` to `target` (including the reverse, if the graph is
    /// undirected).
    ///
    /// If you already have an edge reference and want to know its multiplicity, use
    /// [edge_multiplicity_of], which is optimised in the case of non-multigraphs.
    #[inline]
    fn edge_multiplicity(&self, source: NodeIndex, target: NodeIndex) -> usize {
        let item = if self.graph.is_directed() || source <= target {
            (source, target)
        } else {
            (target, source)
        };
        *self.adjacency_matrix.get(&item).unwrap_or(&0)
    }

    /// What is the multiplicity of the given edge reference?
    ///
    /// This is optimised to avoid hash-map lookups in the case of a non-multigraph (since the
    /// answer is simply always 1; you've already proved you've got the edge, so it can't be zero).
    #[inline]
    fn edge_multiplicity_of(&self, edge: <&G as IntoEdgeReferences>::EdgeRef) -> usize {
        if self.multigraph {
            self.edge_multiplicity(edge.source(), edge.target())
        } else {
            1
        }
    }
}

#[derive(Debug)]
pub enum IsIsomorphicError<NME, EME> {
    NodeMatcher(NME),
    EdgeMatcher(EME),
}
impl<NME: Error, EME: Error> Display for IsIsomorphicError<NME, EME> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            IsIsomorphicError::NodeMatcher(e) => write!(f, "Node match callback failed with: {e}"),
            IsIsomorphicError::EdgeMatcher(e) => write!(f, "Edge match callback failed with: {e}"),
        }
    }
}
impl<NME: Error, EME: Error> Error for IsIsomorphicError<NME, EME> {}

// The `Clone` impl _could_ be spelled `combine(&self, &Self::id())`, but why bother?
pub trait Vf2Score: Clone {
    /// Combine two scores together.  This is typically addition.
    ///
    /// The combination of two scores must not be less than either, as measured by `cmp`.
    fn combine(left: &Self, right: &Self) -> Self;
    /// The identity value of the `combine` operation.
    fn id() -> Self;
    /// Compare two values.
    fn cmp(left: &Self, right: &Self) -> Ordering;
}

impl Vf2Score for () {
    #[inline]
    fn combine(_: &(), _: &()) {}
    #[inline]
    fn id() {}
    #[inline]
    fn cmp(_: &(), _: &()) -> Ordering {
        Ordering::Equal
    }
}
macro_rules! impl_vf2_score_int {
    ($($t:ty)*) => ($(
        impl Vf2Score for $t {
            #[inline]
            fn combine(left: &Self, right: &Self) -> Self {
                *left + *right
            }
            #[inline]
            fn id() -> Self {
                0
            }
            #[inline]
            fn cmp(left: &Self, right: &Self) -> Ordering {
                left.cmp(right)
            }
        }
    )*)
}
impl_vf2_score_int! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }
macro_rules! impl_vf2_score_float {
    ($($t:ty)*) => ($(
        impl Vf2Score for $t {
            #[inline]
            fn combine(left: &Self, right: &Self) -> Self {
                *left + *right
            }
            #[inline]
            fn id() -> Self {
                0.0
            }
            #[inline]
            fn cmp(left: &Self, right: &Self) -> Ordering {
                left.partial_cmp(right).expect("float scores must not return nan")
            }
        }
    )*)
}
impl_vf2_score_float! { f32 f64 }

/// Semantic matching for VF2.
///
/// Both scoring functions must return non-negative floating-point numbers or an error.
/// Returning a negative floating-point number will cause undefined behavior in the
/// algorithm.
///
/// The scoring functions do not have access to the general graph structures; they are only
/// permitted to access the weights.  This is deliberate; the scores cannot be suitably
/// combined and pruned if they are anything other than completely local.
pub trait Vf2Scorer<G0: GraphBase, G1: GraphBase> {
    type Score: Vf2Score;
    type NodeError: Error;
    type EdgeError: Error;

    const NODE_ENABLED: bool;
    const EDGE_ENABLED: bool;

    fn score_node(
        &self,
        needle_graph: &G0,
        haystack_graph: &G1,
        needle_node: G0::NodeId,
        haystack_node: G1::NodeId,
    ) -> Result<Option<Self::Score>, Self::NodeError>;

    fn score_edge(
        &self,
        needle_graph: &G0,
        haystack_graph: &G1,
        needle_edge: G0::EdgeId,
        haystack_edge: G1::EdgeId,
    ) -> Result<Option<Self::Score>, Self::EdgeError>;
}

pub struct NoSemanticMatch;
impl<G0: GraphBase, G1: GraphBase> Vf2Scorer<G0, G1> for (NoSemanticMatch, NoSemanticMatch) {
    type Score = ();
    type NodeError = Infallible;
    type EdgeError = Infallible;
    const NODE_ENABLED: bool = false;
    const EDGE_ENABLED: bool = false;

    #[inline]
    fn score_node(
        &self,
        _: &G0,
        _: &G1,
        _: G0::NodeId,
        _: G1::NodeId,
    ) -> Result<Option<()>, Infallible> {
        Ok(Some(()))
    }
    #[inline]
    fn score_edge(
        &self,
        _: &G0,
        _: &G1,
        _: G0::EdgeId,
        _: G1::EdgeId,
    ) -> Result<Option<()>, Infallible> {
        Ok(Some(()))
    }
}

impl<NS, S, E, G0, G1> Vf2Scorer<G0, G1> for (NS, NoSemanticMatch)
where
    NS: Fn(&G0::NodeWeight, &G1::NodeWeight) -> Result<Option<S>, E>,
    S: Vf2Score,
    E: Error,
    G0: GraphBase + DataMap,
    G1: GraphBase + DataMap,
{
    type Score = S;
    type NodeError = E;
    type EdgeError = Infallible;

    const NODE_ENABLED: bool = true;
    const EDGE_ENABLED: bool = false;

    fn score_node(
        &self,
        needle_graph: &G0,
        haystack_graph: &G1,
        needle_node: G0::NodeId,
        haystack_node: G1::NodeId,
    ) -> Result<Option<Self::Score>, Self::NodeError> {
        let Some(needle_weight) = needle_graph.node_weight(needle_node) else {
            return Ok(None);
        };
        let Some(haystack_weight) = haystack_graph.node_weight(haystack_node) else {
            return Ok(None);
        };
        self.0(needle_weight, haystack_weight)
    }

    fn score_edge(
        &self,
        _needle_graph: &G0,
        _haystack_graph: &G1,
        _needle_edge: G0::EdgeId,
        _haystack_edge: G1::EdgeId,
    ) -> Result<Option<Self::Score>, Self::EdgeError> {
        Ok(Some(S::id()))
    }
}

impl<ES, S, E, G0, G1> Vf2Scorer<G0, G1> for (NoSemanticMatch, ES)
where
    ES: Fn(&G0::EdgeWeight, &G1::EdgeWeight) -> Result<Option<S>, E>,
    S: Vf2Score,
    E: Error,
    G0: GraphBase + DataMap,
    G1: GraphBase + DataMap,
{
    type Score = S;
    type NodeError = Infallible;
    type EdgeError = E;

    const NODE_ENABLED: bool = false;
    const EDGE_ENABLED: bool = true;

    fn score_node(
        &self,
        _needle_graph: &G0,
        _haystack_graph: &G1,
        _needle_node: G0::NodeId,
        _haystack_node: G1::NodeId,
    ) -> Result<Option<Self::Score>, Self::NodeError> {
        Ok(Some(S::id()))
    }

    fn score_edge(
        &self,
        needle_graph: &G0,
        haystack_graph: &G1,
        needle_edge: G0::EdgeId,
        haystack_edge: G1::EdgeId,
    ) -> Result<Option<Self::Score>, Self::EdgeError> {
        let Some(needle_weight) = needle_graph.edge_weight(needle_edge) else {
            return Ok(None);
        };
        let Some(haystack_weight) = haystack_graph.edge_weight(haystack_edge) else {
            return Ok(None);
        };
        self.1(needle_weight, haystack_weight)
    }
}

impl<NS, ES, S, NE, EE, G0, G1> Vf2Scorer<G0, G1> for (NS, ES)
where
    NS: Fn(&G0::NodeWeight, &G1::NodeWeight) -> Result<Option<S>, NE>,
    ES: Fn(&G0::EdgeWeight, &G1::EdgeWeight) -> Result<Option<S>, EE>,
    S: Vf2Score,
    NE: Error,
    EE: Error,
    G0: GraphBase + DataMap,
    G1: GraphBase + DataMap,
{
    type Score = S;
    type NodeError = NE;
    type EdgeError = EE;

    const NODE_ENABLED: bool = true;
    const EDGE_ENABLED: bool = true;

    fn score_node(
        &self,
        needle_graph: &G0,
        haystack_graph: &G1,
        needle_node: G0::NodeId,
        haystack_node: G1::NodeId,
    ) -> Result<Option<Self::Score>, Self::NodeError> {
        let Some(needle_weight) = needle_graph.node_weight(needle_node) else {
            return Ok(None);
        };
        let Some(haystack_weight) = haystack_graph.node_weight(haystack_node) else {
            return Ok(None);
        };
        self.0(needle_weight, haystack_weight)
    }

    fn score_edge(
        &self,
        needle_graph: &G0,
        haystack_graph: &G1,
        needle_edge: G0::EdgeId,
        haystack_edge: G1::EdgeId,
    ) -> Result<Option<Self::Score>, Self::EdgeError> {
        let Some(needle_weight) = needle_graph.edge_weight(needle_edge) else {
            return Ok(None);
        };
        let Some(haystack_weight) = haystack_graph.edge_weight(haystack_edge) else {
            return Ok(None);
        };
        self.1(needle_weight, haystack_weight)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Problem {
    /// Exact isomorphism between the two graphs.
    Exact,
    /// The left graph is isomorphic to an induced subgraph of the right graph.
    InducedSubgraph,
    /// The left graph is isomorphic to any subgraph of the right graph.  This differs from the
    /// "induced subgraph" case in that there can be extra unmatched edges in the right graph.
    ///
    /// The match/score functions will not scan through every possible matching of edges in the case
    /// of a multigraph; the chosen edge matching is arbitrary and may not optimise the score.
    Subgraph,
}

/// [Graph] Return `true` if the graphs `g0` and `g1` are (sub) graph isomorphic.
///
/// Using the VF2 algorithm, examining both syntactic and semantic
/// graph isomorphism (graph structure and matching node and edge weights).
///
/// The graphs should not be multigraphs.
///
/// `scorer` is a 2-tuple of a node-scoring function and an edge-scoring function.  Either or both
/// can be replaced by `NoSemanticMatch` to disable the semantics of nodes and/or edges.  The node
/// scoring function has the signature.
pub fn is_isomorphic<G0, G1, S>(
    g0: &G0,
    g1: &G1,
    scorer: S,
    id_order: bool,
    problem: Problem,
    call_limit: Option<usize>,
) -> Result<bool, IsIsomorphicError<S::NodeError, S::EdgeError>>
where
    G0: GraphProp + GraphBase<NodeId = NodeIndex> + DataMap + Create + NodeCount + EdgeCount,
    for<'a> &'a G0: GraphBase<NodeId = G0::NodeId, EdgeId = G0::EdgeId>
        + Data<NodeWeight = G0::NodeWeight, EdgeWeight = G0::EdgeWeight>
        + NodeIndexable
        + IntoEdgesDirected
        + IntoNodeIdentifiers,
    G0::NodeWeight: Clone,
    G0::EdgeWeight: Clone,
    G1: GraphProp + GraphBase<NodeId = NodeIndex> + DataMap + Create + NodeCount + EdgeCount,
    for<'a> &'a G1: GraphBase<NodeId = G1::NodeId, EdgeId = G1::EdgeId>
        + Data<NodeWeight = G1::NodeWeight, EdgeWeight = G1::EdgeWeight>
        + NodeIndexable
        + IntoEdgesDirected
        + IntoNodeIdentifiers,
    G1::NodeWeight: Clone,
    G1::EdgeWeight: Clone,
    S: Vf2Scorer<G0, G1>,
{
    Vf2Algorithm::new(g0, g1, scorer, id_order, problem, call_limit)
        .next()
        .map(|res| res.map(|_| true))
        .unwrap_or(Ok(false))
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum OpenList {
    Out,
    In,
}

#[derive(Debug)]
enum Frame<N: Copy, T> {
    ChooseNextHaystack {
        nodes: [N; 2],
        open_list: Option<OpenList>,
        prev_score: T,
    },
    ChooseNextNeedle,
}

struct Vf2Semantic<G0, G1, S>
where
    G0: GraphBase,
    G1: GraphBase,
    S: Vf2Scorer<G0, G1>,
{
    scorer: S,
    cur: S::Score,
    limit: Option<S::Score>,
    _marker: marker::PhantomData<(G0, G1)>,
}

/// An iterator which uses the VF2(++) algorithm to produce isomorphic matches
/// between two graphs, examining both syntactic and semantic graph isomorphism
/// (graph structure and matching node and edge weights).
///
/// The graphs should not be multigraphs.
pub struct Vf2Algorithm<G0, G1, S>
where
    G0: GraphBase + Data,
    G1: GraphBase + Data,
    S: Vf2Scorer<G0, G1>,
{
    st: (Vf2State<G0>, Vf2State<G1>),
    semantic: Vf2Semantic<G0, G1, S>,
    problem: Problem,
    stack: Vec<Frame<NodeIndex, S::Score>>,
    remaining_calls: Option<usize>,
}

impl<G0, G1, S> Vf2Algorithm<G0, G1, S>
where
    G0: GraphProp + GraphBase<NodeId = NodeIndex> + DataMap + Create + NodeCount + EdgeCount,
    for<'a> &'a G0: GraphBase<NodeId = G0::NodeId, EdgeId = G0::EdgeId>
        + Data<NodeWeight = G0::NodeWeight, EdgeWeight = G0::EdgeWeight>
        + NodeIndexable
        + IntoEdgesDirected
        + IntoNodeIdentifiers,
    G0::NodeWeight: Clone,
    G0::EdgeWeight: Clone,
    G1: GraphProp + GraphBase<NodeId = NodeIndex> + DataMap + Create + NodeCount + EdgeCount,
    for<'a> &'a G1: GraphBase<NodeId = G1::NodeId, EdgeId = G1::EdgeId>
        + Data<NodeWeight = G1::NodeWeight, EdgeWeight = G1::EdgeWeight>
        + NodeIndexable
        + IntoEdgesDirected
        + IntoNodeIdentifiers,
    G1::NodeWeight: Clone,
    G1::EdgeWeight: Clone,
    S: Vf2Scorer<G0, G1>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        g0: &G0,
        g1: &G1,
        scorer: S,
        id_order: bool,
        problem: Problem,
        call_limit: Option<usize>,
    ) -> Self {
        let (g0, node_map_g0) = if id_order {
            DefaultIdSorter.reorder(g0)
        } else {
            Vf2ppSorter.reorder(g0)
        };

        let (g1, node_map_g1) = if id_order {
            DefaultIdSorter.reorder(g1)
        } else {
            Vf2ppSorter.reorder(g1)
        };

        // The stack typically will need to grow to account to store a "haystack" frame for each
        // node in the needle graph in a success path (and it's shorter in the failure path).
        let mut stack = Vec::with_capacity(g0.node_count());
        stack.push(Frame::ChooseNextNeedle);
        let st = (
            Vf2State::new(g0, node_map_g0),
            Vf2State::new(g1, node_map_g1),
        );
        Vf2Algorithm {
            st,
            semantic: Vf2Semantic {
                scorer,
                cur: S::Score::id(),
                limit: None,
                _marker: marker::PhantomData,
            },
            problem,
            stack,
            remaining_calls: call_limit,
        }
    }

    /// Apply a score limiter to any calls to the VF2 algorithm.
    ///
    /// Only isomorphisms of the given `Problem` with scores less than or equal to the best seen
    /// score (starting from the limit) will be returned.
    pub fn with_score_limit(mut self, limit: S::Score) -> Self {
        self.semantic.limit = Some(limit);
        self
    }

    fn mapping(&self) -> IndexMap<NodeIndex, NodeIndex, ::ahash::RandomState> {
        self.st
            .0
            .mapping
            .iter()
            .enumerate()
            .map(|(needle, haystack)| {
                debug_assert!(*haystack != NodeIndex::end());
                (
                    self.st.0.reorder[&NodeIndex::new(needle)],
                    self.st.1.reorder[haystack],
                )
            })
            .collect()
    }

    /// Find the unmapped candidates with the highest priority from both graphs.
    ///
    /// This returns the two candidates and the tracking list of neighbours to the existing mapping
    /// that they were found in. `None` implies that there were no matching pairs in the lists of
    /// nodes that are outgoing or incoming neighbors of the map, and therefore we're starting a
    /// search for a disjoint component.  In order for that to be performant, we'd expect the
    /// feasiblity checks to have realised that there's edges that will never be satisfied already.
    fn next_candidates(&mut self) -> Option<(NodeIndex, NodeIndex, Option<OpenList>)> {
        [Some(OpenList::Out), Some(OpenList::In), None]
            .into_iter()
            .filter_map(|list| {
                Some((
                    self.st.0.first_unmapped(list)?,
                    self.st.1.first_unmapped(list)?,
                    list,
                ))
            })
            .next()
    }

    /// Remove this pair of nodes from the mapping, and revert the total score to the given value.
    ///
    /// The pair of nodes must be on the top of the stack of pushes.
    fn pop_state(&mut self, nodes: [NodeIndex; 2], prev_score: S::Score) {
        // Restore state.
        self.st.0.pop_mapping(nodes[0]);
        self.st.1.pop_mapping(nodes[1]);
        self.semantic.cur = prev_score;
    }

    /// Add a new pair of nodes to the mapping, and set the total score, and return the score from
    /// before the nodes were pushed.
    fn push_state(&mut self, nodes: [NodeIndex; 2], new_score: S::Score) -> S::Score {
        // Add mapping nx <-> mx to the state
        self.st.0.push_mapping(nodes[0], nodes[1]);
        self.st.1.push_mapping(nodes[1], nodes[0]);
        ::std::mem::replace(&mut self.semantic.cur, new_score)
    }

    #[inline]
    fn directed(&self) -> bool {
        self.st.0.graph.is_directed()
    }

    /// If we added this pair of nodes to the mapping, would it still be feasible to reach a full
    /// solution in the future, and if so, what woould the new score of the partial mapping be?
    ///
    /// The `Result` is related to the semantic node- and edge-matching in the scoring.  The inner
    /// `Option` is `Some(new_score)` if the pair is feasible, and `None` if it is not.
    #[allow(clippy::type_complexity)]
    fn is_feasible(
        &self,
        nodes: [NodeIndex; 2],
    ) -> Result<Option<S::Score>, IsIsomorphicError<S::NodeError, S::EdgeError>> {
        // This is the core correctness component; are the nodes of this mapping and all the edges
        // that would be newly mapped consistent with a solution to the problem?  This must be
        // precise; we _must_ cut if the mapping would not be consistent.
        let Some(score) = self.is_consistent(nodes)? else {
            return Ok(None);
        };

        // Everything else in this function is lookahead feasibility tests.  It's important for
        // correctness that we never report a mapping as unfeasible if actually it does lead to a
        // valid solution.  If in doubt, say it's feasible; we attempt to eagerly cut as a
        // performance optimisation only.

        if !self.unmapped_existing_neighbors_feasible(nodes, OpenList::Out, Outgoing)
            || !self.unmapped_new_neighbors_feasible(nodes, Outgoing)
        {
            return Ok(None);
        }
        if self.directed()
            && (!self.unmapped_existing_neighbors_feasible(nodes, OpenList::In, Incoming)
                || !self.unmapped_existing_neighbors_feasible(nodes, OpenList::Out, Incoming)
                || !self.unmapped_existing_neighbors_feasible(nodes, OpenList::In, Outgoing)
                || !self.unmapped_new_neighbors_feasible(nodes, Incoming))
        {
            return Ok(None);
        }
        Ok(Some(score))
    }

    #[allow(clippy::type_complexity)]
    fn is_consistent(
        &self,
        nodes: [NodeIndex; 2],
    ) -> Result<Option<S::Score>, IsIsomorphicError<S::NodeError, S::EdgeError>> {
        // Are the semantics of these two nodes consistent?
        let node_score = if S::NODE_ENABLED {
            match self
                .semantic
                .scorer
                .score_node(&self.st.0.graph, &self.st.1.graph, nodes[0], nodes[1])
                .map_err(IsIsomorphicError::NodeMatcher)?
            {
                Some(score) => score,
                None => return Ok(None),
            }
        } else {
            // If there's no node-matching enabled, then the two are always semantically feasible.
            // The only reason they might fail is if the newly mapped connecting edges aren't.
            S::Score::id()
        };

        // Are the semantics of the edges that would become fully mapped consistent?
        let edge_score = if S::EDGE_ENABLED {
            let Some(outgoing_score) = self
                .mapped_edges_semantic_match(nodes, Outgoing)
                .map_err(IsIsomorphicError::EdgeMatcher)?
            else {
                return Ok(None);
            };
            if self.directed() {
                let Some(incoming_score) = self
                    .mapped_edges_semantic_match(nodes, Incoming)
                    .map_err(IsIsomorphicError::EdgeMatcher)?
                else {
                    return Ok(None);
                };
                S::Score::combine(&outgoing_score, &incoming_score)
            } else {
                outgoing_score
            }
        } else {
            if !self.mapped_edges_counts_match(nodes, Outgoing)
                || (self.directed() && !self.mapped_edges_counts_match(nodes, Incoming))
            {
                return Ok(None);
            }
            S::Score::id()
        };

        let new_score = S::Score::combine(
            &S::Score::combine(&self.semantic.cur, &node_score),
            &edge_score,
        );
        match self.semantic.limit.as_ref() {
            Some(limit) => {
                // We're not consistent if the score  breaks the limit.
                Ok((S::Score::cmp(&new_score, limit) != Ordering::Greater).then_some(new_score))
            }
            None => Ok(Some(new_score)),
        }
    }

    /// Are the directed edges of these two paired nodes consistent with all nodes already in the
    /// mapping?
    ///
    /// This is the more expensive semantic check, where we have to run pairwise through the edge
    /// lists to match them up against each other.  If there's no semantic checking enabled, use
    /// `mapped_edges_counts_match`, which short-circuits based on the pre-calculated adjacency
    /// matrices.
    fn mapped_edges_semantic_match(
        &self,
        nodes: [NodeIndex; 2],
        direction: Direction,
    ) -> Result<Option<S::Score>, S::EdgeError> {
        let needle = &self.st.0;
        let haystack = &self.st.1;
        // We only handle self loops in the `Outgoing` direction to avoid double-counting the score.
        let handle_self = direction == Direction::Outgoing;

        macro_rules! neighbor {
            ($edge:expr) => {
                match direction {
                    Direction::Outgoing => $edge.target(),
                    Direction::Incoming => $edge.source(),
                }
            };
        }

        if !needle.multigraph {
            let mut score = S::Score::id();
            let mapped_source = handle_self.then_some(nodes[1]);
            for needle_edge in needle.graph.edges_directed(nodes[0], direction) {
                let Some(haystack_neighbor) =
                    needle.map_target(nodes[0], neighbor!(needle_edge), mapped_source)
                else {
                    continue;
                };
                let haystack_multiplicity = match direction {
                    Direction::Outgoing => haystack.edge_multiplicity(nodes[1], haystack_neighbor),
                    Direction::Incoming => haystack.edge_multiplicity(haystack_neighbor, nodes[1]),
                };
                match self.problem {
                    Problem::Exact | Problem::InducedSubgraph => {
                        if haystack_multiplicity != 1 {
                            return Ok(None);
                        }
                    }
                    Problem::Subgraph => {
                        if haystack_multiplicity == 0 {
                            return Ok(None);
                        }
                    }
                }
                let Some(edge_score) = haystack
                    .graph
                    .edges_directed(nodes[1], direction)
                    .filter_map(|haystack_edge| {
                        if (neighbor!(haystack_edge) == haystack_neighbor) {
                            self.semantic
                                .scorer
                                .score_edge(
                                    &needle.graph,
                                    &haystack.graph,
                                    needle_edge.id(),
                                    haystack_edge.id(),
                                )
                                .transpose()
                        } else {
                            None
                        }
                    })
                    .next()
                    .transpose()?
                else {
                    return Ok(None);
                };
                score = S::Score::combine(&score, &edge_score);
            }
            return Ok(Some(score));
        }

        let mut needle_edges = HashMap::<NodeIndex, SmallVec<[G0::EdgeId; 4]>>::new();
        let mapped_source = handle_self.then_some(nodes[1]);
        for edge in needle.graph.edges_directed(nodes[0], direction) {
            let Some(haystack_neighbor) =
                needle.map_target(nodes[0], neighbor!(edge), mapped_source)
            else {
                continue;
            };
            needle_edges
                .entry(haystack_neighbor)
                .or_insert_with(|| SmallVec::with_capacity(needle.edge_multiplicity_of(edge)))
                .push(edge.id());
        }
        let mut haystack_edges = HashMap::<NodeIndex, SmallVec<[G1::EdgeId; 4]>>::new();
        let mapped_source = handle_self.then_some(nodes[0]);
        for edge in haystack.graph.edges_directed(nodes[1], direction) {
            let Some(needle_neighbor) =
                haystack.map_target(nodes[1], neighbor!(edge), mapped_source)
            else {
                continue;
            };
            haystack_edges
                .entry(needle_neighbor)
                .or_insert_with(|| SmallVec::with_capacity(haystack.edge_multiplicity_of(edge)))
                .push(edge.id());
        }

        // In all problems, every edge between two mapped nodes in the needle must map to an edge
        // between the two paired nodes in the haystack.  In the `Exact` and `InducedSubgraph`
        // problems, we also need the reciprocal condition; there must be no unmapped edges on the
        // haystack side.
        let mut score = S::Score::id();
        for (haystack_neighbor, needle_edges) in needle_edges {
            let needle_neighbor = haystack.mapping[haystack_neighbor.index()];
            let Some(mut haystack_edges) = haystack_edges.remove(&needle_neighbor) else {
                // Actually this shouldn't ever trigger if we already checked that the _number_ of
                // edges is consistent.
                return Ok(None);
            };
            let match_edge = |haystack_edges: &[G1::EdgeId],
                              needle_edge: G0::EdgeId|
             -> Result<Option<(usize, S::Score)>, S::EdgeError> {
                for (pos, haystack_edge) in haystack_edges.iter().enumerate() {
                    if let Some(score) = self.semantic.scorer.score_edge(
                        &needle.graph,
                        &haystack.graph,
                        needle_edge,
                        *haystack_edge,
                    )? {
                        return Ok(Some((pos, score)));
                    }
                }
                Ok(None)
            };
            for needle_edge in needle_edges {
                let Some((pos, edge_score)) = match_edge(haystack_edges.as_slice(), needle_edge)?
                else {
                    return Ok(None);
                };
                score = S::Score::combine(&score, &edge_score);
                haystack_edges.swap_remove(pos);
            }
            if !haystack_edges.is_empty()
                && (self.problem == Problem::Exact || self.problem == Problem::InducedSubgraph)
            {
                return Ok(None);
            }
        }
        Ok(Some(score))
    }

    fn mapped_edges_counts_match(&self, nodes: [NodeIndex; 2], direction: Direction) -> bool {
        let needle = &self.st.0;
        let haystack = &self.st.1;

        macro_rules! order {
            ($node:expr, $other:expr) => {
                match direction {
                    Direction::Outgoing => ($node, $other),
                    Direction::Incoming => ($other, $node),
                }
            };
        }

        let mapped_self = (direction == Direction::Outgoing).then_some(nodes[1]);
        for needle_neighbor in needle.graph.neighbors_directed(nodes[0], direction) {
            let Some(haystack_neighbor) = needle.map_target(nodes[0], needle_neighbor, mapped_self)
            else {
                continue;
            };
            let (needle_source, needle_target) = order!(nodes[0], needle_neighbor);
            let (haystack_source, haystack_target) = order!(nodes[1], haystack_neighbor);
            let needle_multiplicity = match needle.multigraph {
                true => needle.edge_multiplicity(needle_source, needle_target),
                false => 1,
            };
            let haystack_multiplicity =
                haystack.edge_multiplicity(haystack_source, haystack_target);
            match self.problem {
                Problem::Exact | Problem::InducedSubgraph => {
                    if needle_multiplicity != haystack_multiplicity {
                        return false;
                    }
                }
                Problem::Subgraph => {
                    if needle_multiplicity > haystack_multiplicity {
                        return false;
                    }
                }
            }
        }
        if let Problem::Subgraph = self.problem {
            // It's not a problem if there extra edges between mapped nodes in the haystack if we're
            // looking for a non-induced subgraph, but for everything else we have to check.
            return true;
        }

        let mapped_self = (direction == Direction::Outgoing).then_some(nodes[0]);
        for haystack_neighbor in haystack.graph.neighbors_directed(nodes[1], direction) {
            let Some(needle_neighbor) =
                haystack.map_target(nodes[1], haystack_neighbor, mapped_self)
            else {
                continue;
            };
            let (needle_source, needle_target) = order!(nodes[0], needle_neighbor);
            let (haystack_source, haystack_target) = order!(nodes[1], haystack_neighbor);
            let needle_multiplicity = needle.edge_multiplicity(needle_source, needle_target);
            let haystack_multiplicity = match haystack.multigraph {
                true => haystack.edge_multiplicity(haystack_source, haystack_target),
                false => 1,
            };
            if needle_multiplicity != haystack_multiplicity {
                return false;
            }
        }
        true
    }

    /// Do the nodes in this mapping have a compatible number of directed neighbors that are
    /// unmapped, and already directed neighbors of the partial mapping?
    fn unmapped_existing_neighbors_feasible(
        &self,
        nodes: [NodeIndex; 2],
        list: OpenList,
        direction: Direction,
    ) -> bool {
        let needle = &self.st.0;
        let haystack = &self.st.1;

        #[inline]
        fn filter<G>(node: NodeIndex, state: &Vf2State<G>, list: OpenList) -> bool {
            let index = node.index();
            let externals = match list {
                OpenList::Out => state.out.as_slice(),
                OpenList::In => state.ins.as_slice(),
            };
            externals[index] > 0 && state.mapping[index] == NodeIndex::end()
        }

        let needle_neighbors = needle
            .graph
            .neighbors_directed(nodes[0], direction)
            .filter(|node| filter(*node, needle, list))
            .count();
        let haystack_neighbors = haystack
            .graph
            .neighbors_directed(nodes[1], direction)
            .filter(|node| filter(*node, haystack, list))
            .count();

        match self.problem {
            Problem::Exact => needle_neighbors == haystack_neighbors,
            Problem::InducedSubgraph | Problem::Subgraph => needle_neighbors <= haystack_neighbors,
        }
    }

    /// Would this extension of the mapping add a compatible number of new neighbors to the entire
    /// mapping?
    fn unmapped_new_neighbors_feasible(&self, nodes: [NodeIndex; 2], direction: Direction) -> bool {
        if let Problem::Subgraph = self.problem {
            // If we're looking for a non-induced subgraph, the counts here don't mean anything.
            return true;
        }

        let needle = &self.st.0;
        let haystack = &self.st.1;

        #[inline]
        fn filter<G>(node: NodeIndex, state: &Vf2State<G>) -> bool {
            let index = node.index();
            state.out[index] == 0 && (state.ins.is_empty() || state.ins[index] == 0)
        }

        let needle_neighbors = needle
            .graph
            .neighbors_directed(nodes[0], direction)
            .filter(|node| filter(*node, needle))
            .count();
        let haystack_neighbors = haystack
            .graph
            .neighbors_directed(nodes[1], direction)
            .filter(|node| filter(*node, haystack))
            .count();

        match self.problem {
            Problem::Exact => needle_neighbors == haystack_neighbors,
            Problem::InducedSubgraph => needle_neighbors <= haystack_neighbors,
            Problem::Subgraph => unreachable!("already handled by early return"),
        }
    }

    /// Are the cardinalities of the sets of neighbors of the current mapping state still feasible
    /// for a complete solution?
    ///
    /// This is a cheap lookahead heuristic; if the nodes currently mapped in the needle graph have
    /// more neighbors than the nodes currently mapped in the haystack graph, there can't possibly
    /// be a solution, regardless of the problem.  If we're not looking for a subgraph, then those
    /// two sets of objects have to map exactly.
    fn neighbor_counts_feasible(&self) -> bool {
        let needle_outs = self.st.0.out_size;
        let needle_ins = self.st.0.ins_size;
        let haystack_outs = self.st.1.out_size;
        let haystack_ins = self.st.1.ins_size;

        match self.problem {
            Problem::Exact => (needle_outs == haystack_outs) && (needle_ins == haystack_ins),
            Problem::InducedSubgraph | Problem::Subgraph => {
                (needle_outs <= haystack_outs) && (needle_ins <= haystack_ins)
            }
        }
    }

    /// Increase the call count of the mapper.  Returns `None` if we're already exhausted.
    #[inline]
    fn try_add_call(&mut self) -> Option<()> {
        self.remaining_calls = self.remaining_calls.map(|rem| rem.saturating_sub(1));
        match self.remaining_calls {
            Some(0) => None,
            Some(_) | None => Some(()),
        }
    }
}

impl<G0, G1, S> Iterator for Vf2Algorithm<G0, G1, S>
where
    G0: GraphProp + GraphBase<NodeId = NodeIndex> + DataMap + Create + NodeCount + EdgeCount,
    for<'a> &'a G0: GraphBase<NodeId = G0::NodeId, EdgeId = G0::EdgeId>
        + Data<NodeWeight = G0::NodeWeight, EdgeWeight = G0::EdgeWeight>
        + NodeIndexable
        + IntoEdgesDirected
        + IntoNodeIdentifiers,
    G0::NodeWeight: Clone,
    G0::EdgeWeight: Clone,
    G1: GraphProp + GraphBase<NodeId = NodeIndex> + DataMap + Create + NodeCount + EdgeCount,
    for<'a> &'a G1: GraphBase<NodeId = G1::NodeId, EdgeId = G1::EdgeId>
        + Data<NodeWeight = G1::NodeWeight, EdgeWeight = G1::EdgeWeight>
        + NodeIndexable
        + IntoEdgesDirected
        + IntoNodeIdentifiers,
    G1::NodeWeight: Clone,
    G1::EdgeWeight: Clone,
    S: Vf2Scorer<G0, G1>,
{
    type Item = Result<
        (
            IndexMap<NodeIndex, NodeIndex, ::ahash::RandomState>,
            S::Score,
        ),
        IsIsomorphicError<S::NodeError, S::EdgeError>,
    >;

    /// Return Some(mapping) if isomorphism is decided, else None.
    fn next(&mut self) -> Option<Self::Item> {
        // Fast-path short circuits.
        let (g0, g1) = (&self.st.0.graph, &self.st.1.graph);
        match self.problem {
            Problem::Exact => {
                if g0.node_count() != g1.node_count() || g0.edge_count() != g1.edge_count() {
                    return None;
                }
            }
            Problem::InducedSubgraph | Problem::Subgraph => {
                if g0.node_count() > g1.node_count() || g0.edge_count() > g1.edge_count() {
                    return None;
                }
            }
        }

        // Main logic loop.
        //
        // The overall strategy is a nested loop, where the "outer" loop is over unmapped nodes in
        // the "needle" graph, and the "inner" loop is over unmapped nodes in the "haystack" graph.
        // In both cases, the ordering of the nodes was set earlier, during the initial setup of
        // `self`, depending on the base ordering heuristic.  The "stack" is to save our position
        // in the loop iteration if/when we cede control back to the calling function.
        loop {
            let (mut nodes, open_list) = match self.stack.pop()? {
                Frame::ChooseNextHaystack {
                    nodes,
                    open_list,
                    prev_score,
                } => {
                    self.pop_state(nodes, prev_score);
                    if let Some(haystack) = self.st.1.next_unmapped_after(nodes[1], open_list) {
                        ([nodes[0], haystack], open_list)
                    } else {
                        continue;
                    }
                }
                Frame::ChooseNextNeedle => {
                    if let Some((needle, haystack, open_list)) = self.next_candidates() {
                        ([needle, haystack], open_list)
                    } else if self.st.0.is_complete() {
                        // This only triggers if the needle graph is empty, and we're after a
                        // subgraph problem (since we'll already have exited if we're after an exact
                        // match and the numbers of nodes didn't match.
                        return Some(Ok((self.mapping(), self.semantic.cur.clone())));
                    } else {
                        continue;
                    }
                }
            };
            // At this point, we've got a pair of a needle node and a haystack node to try.  We loop
            // through the available haystack nodes until we find one that's feasible, at which
            // point we save our state in the haystack loop and move to finding the match to the
            // next needle node.  If none are feasible, we've finished this stack frame.
            loop {
                let feasible = match self.is_feasible(nodes) {
                    Ok(f) => f,
                    Err(e) => return Some(Err(e)),
                };
                if let Some(new_score) = feasible {
                    let prev_score = self.push_state(nodes, new_score);
                    if self.neighbor_counts_feasible() {
                        self.try_add_call()?;
                        self.stack.push(Frame::ChooseNextHaystack {
                            nodes,
                            open_list,
                            prev_score,
                        });
                        if self.st.0.is_complete() {
                            if let Some(limit) = self.semantic.limit.as_mut() {
                                *limit = self.semantic.cur.clone();
                            }
                            return Some(Ok((self.mapping(), self.semantic.cur.clone())));
                        }
                        self.stack.push(Frame::ChooseNextNeedle);
                        break;
                    }
                    self.pop_state(nodes, prev_score);
                }
                if let Some(nx) = self.st.1.next_unmapped_after(nodes[1], open_list) {
                    nodes[1] = nx;
                } else {
                    break;
                }
            }
        }
    }
}
