// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::data_tree::{DataTree, PathEntry};
use crate::program_node::ProgramNode;
use crate::tensor::{Tensor, TensorType};
use hashbrown::HashMap;
use rustworkx_core::dag_algo::lexicographical_topological_sort;
use rustworkx_core::petgraph::graph::{DiGraph, NodeIndex};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Local owned-path types and helpers (formerly in data_tree)
// Used by Port::path, error types, and NodeView's leaf path cache.
// ---------------------------------------------------------------------------

/// An owned variant of [`PathEntry`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OwnedPathEntry {
    /// A positional index into a branch's children.
    Index(usize),
    /// A named key into a branch's children.
    Key(String),
}

/// A path through a [`DataTree`] where all string keys are owned.
pub type OwnedPath = Vec<OwnedPathEntry>;

impl OwnedPathEntry {
    /// View this entry as a borrowed [`PathEntry`].
    pub fn as_borrowed(&self) -> PathEntry<'_> {
        match self {
            Self::Index(i) => PathEntry::Index(*i),
            Self::Key(k) => PathEntry::Key(k.as_str()),
        }
    }
}

impl<'a> From<&'a OwnedPathEntry> for PathEntry<'a> {
    fn from(entry: &'a OwnedPathEntry) -> Self {
        entry.as_borrowed()
    }
}

impl From<&str> for OwnedPathEntry {
    fn from(s: &str) -> Self {
        Self::Key(s.into())
    }
}

impl From<String> for OwnedPathEntry {
    fn from(s: String) -> Self {
        Self::Key(s)
    }
}

impl From<usize> for OwnedPathEntry {
    fn from(i: usize) -> Self {
        Self::Index(i)
    }
}

impl std::fmt::Display for OwnedPathEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Key(k) => f.write_str(k),
            Self::Index(i) => write!(f, "{i}"),
        }
    }
}

/// Render a path slice as a dotted string (`"x.0.creg"`); empty path renders as `""`.
///
/// Returns an `impl Display` to avoid intermediate allocation.
pub fn format_path(path: &[OwnedPathEntry]) -> impl std::fmt::Display + '_ {
    struct DottedPath<'a>(&'a [OwnedPathEntry]);
    impl std::fmt::Display for DottedPath<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            let mut first = true;
            for entry in self.0 {
                if !first {
                    f.write_str(".")?;
                }
                write!(f, "{entry}")?;
                first = false;
            }
            Ok(())
        }
    }
    DottedPath(path)
}

/// Walk `tree`, yielding each leaf's owned path. Free-function port of the
/// former `DataTree::iter_owned_paths` method.
pub fn iter_owned_paths<T>(tree: &DataTree<T>) -> impl Iterator<Item = (OwnedPath, &T)> {
    let mut out = Vec::new();
    iter_owned_paths_inner(tree, &mut Vec::new(), &mut out);
    out.into_iter()
}

fn iter_owned_paths_inner<'a, T>(
    tree: &'a DataTree<T>,
    prefix: &mut Vec<OwnedPathEntry>,
    out: &mut Vec<(Vec<OwnedPathEntry>, &'a T)>,
) {
    match tree {
        DataTree::Leaf(value) => out.push((prefix.clone(), value)),
        DataTree::Branch(_) => {
            for (i, (key, child)) in tree.iter_children().enumerate() {
                let entry = match key {
                    Some(k) => OwnedPathEntry::Key(k.to_string()),
                    None => OwnedPathEntry::Index(i),
                };
                prefix.push(entry);
                iter_owned_paths_inner(child, prefix, out);
                prefix.pop();
            }
        }
    }
}

/// Look up a node in `tree` by an owned path. Free-function port of the
/// former `DataTree::get_by_owned_path` method.
pub fn get_by_owned_path<'a, T>(
    tree: &'a DataTree<T>,
    path: &[OwnedPathEntry],
) -> Option<&'a DataTree<T>> {
    let borrowed: Vec<PathEntry<'_>> = path.iter().map(OwnedPathEntry::as_borrowed).collect();
    if borrowed.is_empty() {
        return Some(tree);
    }
    tree.get_by_path(&borrowed)
}

/// Resolve a path against `tree`, replacing each index with its name when
/// present. Returns `None` if the path doesn't reach a leaf in `tree`.
pub fn normalize_path<T>(
    tree: &DataTree<T>,
    path: &[OwnedPathEntry],
) -> Option<Vec<OwnedPathEntry>> {
    let mut normalized = Vec::with_capacity(path.len());
    let mut current = tree;
    for entry in path {
        let DataTree::Branch(_) = current else {
            return None;
        };
        // Build (key, index) listing of children at the current level via
        // public iter_children (yields (Option<&str>, &DataTree<T>) in
        // positional order).
        let children: Vec<(Option<&str>, &DataTree<T>)> = current.iter_children().collect();
        let (next, norm_entry) = match entry {
            OwnedPathEntry::Index(i) => {
                let (maybe_key, child) = children.get(*i).copied()?;
                let norm = match maybe_key {
                    Some(k) => OwnedPathEntry::Key(k.to_string()),
                    None => OwnedPathEntry::Index(*i),
                };
                (child, norm)
            }
            OwnedPathEntry::Key(k) => {
                let (_, child) = children
                    .iter()
                    .find(|(maybe_key, _)| *maybe_key == Some(k.as_str()))
                    .copied()?;
                (child, OwnedPathEntry::Key(k.clone()))
            }
        };
        current = next;
        normalized.push(norm_entry);
    }
    matches!(current, DataTree::Leaf(_)).then_some(normalized)
}

// ---------------------------------------------------------------------------
// Port
// ---------------------------------------------------------------------------

/// A port on a specific node in a [`QuantumProgram`].
#[derive(Clone, Debug)]
pub struct Port {
    /// The label of the node this port belongs to.
    pub label: String,
    /// Path through the node's input or output [`DataTree`] to the port leaf.
    pub path: OwnedPath,
}

impl Port {
    /// Construct a new [`Port`].
    pub fn new(label: impl Into<String>, path: OwnedPath) -> Self {
        Self {
            label: label.into(),
            path,
        }
    }
}

/// Which side (input or output) of a node a port is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PortSide {
    Input,
    Output,
}

// ---------------------------------------------------------------------------
// QuantumProgramError
// ---------------------------------------------------------------------------

/// Errors that can occur when building or querying a [`QuantumProgram`].
#[derive(Debug, Error)]
pub enum QuantumProgramError {
    #[error("a node with label {0:?} already exists")]
    DuplicateLabel(String),

    #[error("no node with label {0:?} exists")]
    UnknownLabel(String),

    #[error(
        "path {} is not a valid leaf port on node {label:?}",
        format_path(path)
    )]
    InvalidPort { label: String, path: OwnedPath },

    #[error("port {} on node {label:?} is already occupied", format_path(path))]
    PortAlreadyConnected { label: String, path: OwnedPath },

    #[error("an I/O key {0:?} is already declared")]
    DuplicateIOKey(String),
}

// ---------------------------------------------------------------------------
// QuantumProgramCallError
// ---------------------------------------------------------------------------

/// Boxes the error of some node whose call fails during [`QuantumProgram::call_flat`].
pub type BoxedNodeError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Errors that can occur while running [`QuantumProgram::call_flat`].
#[derive(Debug, Error)]
pub enum QuantumProgramCallError {
    /// The graph contains a cycle.
    #[error("quantum program graph contains a cycle")]
    Cycle,

    /// A program input was not supplied at call time.
    #[error("missing program input {key:?}")]
    MissingProgramInput { key: String },

    /// An inner node returned an error from its `call_flat` implementation.
    #[error("calling node {label:?}")]
    NodeCall {
        label: String,
        #[source]
        source: BoxedNodeError,
    },

    /// A node's `call_flat` returned a vector whose length didn't match the
    /// leaf count of its `output_types()`.
    #[error("node {label:?} returned {actual} outputs, expected {expected}")]
    NodeOutputArityMismatch {
        label: String,
        expected: usize,
        actual: usize,
    },

    /// A node's input port was neither wired by an edge nor declared as a
    /// program input — execution can't supply a value for it.
    #[error("node {label:?} has an unwired input at {}", format_path(path))]
    UnwiredInput { label: String, path: OwnedPath },
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Index of a leaf in a node's input or output schema, in DFS leaf order.
type LeafIdx = usize;

struct EdgeData {
    from_leaf: LeafIdx,
    to_leaf: LeafIdx,
}

/// Per-node bookkeeping computed once at `add_node` time. Caches the flat
/// DFS leaf views so the executor can address ports by integer index.
struct NodeView {
    node: DynNode,
    input_leaf_paths: Vec<OwnedPath>,
    output_leaf_paths: Vec<OwnedPath>,
}

impl NodeView {
    fn leaf_paths(&self, side: PortSide) -> &[OwnedPath] {
        match side {
            PortSide::Input => &self.input_leaf_paths,
            PortSide::Output => &self.output_leaf_paths,
        }
    }

    fn types(&self, side: PortSide) -> &DataTree<TensorType> {
        match side {
            PortSide::Input => self.node.input_types(),
            PortSide::Output => self.node.output_types(),
        }
    }
}

/// Graph-storage wrapper for a [`ProgramNode`]: erases the node's specific
/// `CallError` to [`BoxedNodeError`] so heterogeneous nodes can live in a single
/// `DiGraph`.
struct DynNode(Box<dyn ProgramNode<CallError = BoxedNodeError>>);

impl DynNode {
    fn new<N>(node: N) -> Self
    where
        N: ProgramNode + 'static,
        N::CallError: std::error::Error + Send + Sync + 'static,
    {
        struct Adapter<N>(N);
        impl<N> ProgramNode for Adapter<N>
        where
            N: ProgramNode,
            N::CallError: std::error::Error + Send + Sync + 'static,
        {
            type CallError = BoxedNodeError;
            fn name(&self) -> &str {
                self.0.name()
            }
            fn namespace(&self) -> &str {
                self.0.namespace()
            }
            fn input_types(&self) -> &DataTree<TensorType> {
                self.0.input_types()
            }
            fn output_types(&self) -> &DataTree<TensorType> {
                self.0.output_types()
            }
            fn implements_call(&self) -> bool {
                self.0.implements_call()
            }
            fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, BoxedNodeError> {
                self.0
                    .call_flat(args)
                    .map_err(|e| Box::new(e) as BoxedNodeError)
            }
        }
        DynNode(Box::new(Adapter(node)))
    }
}

impl ProgramNode for DynNode {
    type CallError = BoxedNodeError;
    fn name(&self) -> &str {
        self.0.name()
    }
    fn namespace(&self) -> &str {
        self.0.namespace()
    }
    fn input_types(&self) -> &DataTree<TensorType> {
        self.0.input_types()
    }
    fn output_types(&self) -> &DataTree<TensorType> {
        self.0.output_types()
    }
    fn implements_call(&self) -> bool {
        self.0.implements_call()
    }
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, BoxedNodeError> {
        self.0.call_flat(args)
    }
}

/// Where a leaf coming out of a node should be sent. Indices are local to a
/// single `call_flat` invocation: `target_topo_idx` is the position in the
/// topological order; `output_idx` is the position in `self.outputs`.
enum ConsumerKind {
    Edge {
        target_topo_idx: usize,
        target_leaf: LeafIdx,
    },
    ProgramOutput {
        output_idx: usize,
    },
}

// ---------------------------------------------------------------------------
// QuantumProgram
// ---------------------------------------------------------------------------

/// A quantum program represented as a directed graph of [`ProgramNode`]s.
///
/// Each node is identified by a unique string label; directed edges connect
/// output ports of one node to input ports of another. The program's own
/// `input_types`/`output_types` is declared explicitly via
/// [`set_input`](Self::set_input)/[`set_output`](Self::set_output) — it is
/// not inferred from unconnected ports.
pub struct QuantumProgram {
    graph: DiGraph<NodeView, EdgeData>,
    label_to_node: HashMap<String, NodeIndex>,
    occupied_input_ports: HashMap<NodeIndex, Vec<bool>>,
    inputs: Vec<(String, NodeIndex, LeafIdx)>,
    outputs: Vec<(String, NodeIndex, LeafIdx)>,
    input_types: DataTree<TensorType>,
    output_types: DataTree<TensorType>,
}

impl Default for QuantumProgram {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumProgram {
    /// Construct a new, empty [`QuantumProgram`].
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            label_to_node: HashMap::new(),
            occupied_input_ports: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_types: DataTree::new(),
            output_types: DataTree::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------

    /// Add a node to the program. `label` must be unique.
    ///
    /// Per [`ProgramNode`]'s schema-stability invariant, the node's IO trees
    /// are snapshotted here and treated as frozen.
    pub fn add_node<N>(
        &mut self,
        label: impl Into<String>,
        node: N,
    ) -> Result<(), QuantumProgramError>
    where
        N: ProgramNode + 'static,
        N::CallError: std::error::Error + Send + Sync + 'static,
    {
        let label = label.into();
        if self.label_to_node.contains_key(&label) {
            return Err(QuantumProgramError::DuplicateLabel(label));
        }
        let dyn_node = DynNode::new(node);
        let input_leaf_paths: Vec<OwnedPath> = iter_owned_paths(dyn_node.input_types())
            .map(|(p, _)| p)
            .collect();
        let output_leaf_paths: Vec<OwnedPath> = iter_owned_paths(dyn_node.output_types())
            .map(|(p, _)| p)
            .collect();
        let n_inputs = input_leaf_paths.len();
        let view = NodeView {
            node: dyn_node,
            input_leaf_paths,
            output_leaf_paths,
        };
        let idx = self.graph.add_node(view);
        self.label_to_node.insert(label, idx);
        self.occupied_input_ports.insert(idx, vec![false; n_inputs]);
        Ok(())
    }

    /// Add a directed edge from one node's output port to another's input port.
    ///
    /// Both ports must exist and resolve to a leaf. The destination port must
    /// not already be occupied (sources may fan out).
    pub fn add_edge(&mut self, from: Port, to: Port) -> Result<(), QuantumProgramError> {
        let from_idx = self.lookup_label(&from.label)?;
        let to_idx = self.lookup_label(&to.label)?;
        let from_leaf = self.resolve_leaf_or_err(from_idx, &from, PortSide::Output)?;
        let to_leaf = self.resolve_leaf_or_err(to_idx, &to, PortSide::Input)?;

        if self.occupied_input_ports[&to_idx][to_leaf] {
            return Err(QuantumProgramError::PortAlreadyConnected {
                label: to.label,
                path: self.graph[to_idx].input_leaf_paths[to_leaf].clone(),
            });
        }

        self.occupied_input_ports.get_mut(&to_idx).unwrap()[to_leaf] = true;
        self.graph
            .add_edge(from_idx, to_idx, EdgeData { from_leaf, to_leaf });
        Ok(())
    }

    /// Declare an input port of this program under `key`.
    pub fn set_input(&mut self, key: &str, port: Port) -> Result<(), QuantumProgramError> {
        if self.inputs.iter().any(|(k, _, _)| k == key) {
            return Err(QuantumProgramError::DuplicateIOKey(key.to_string()));
        }
        let idx = self.lookup_label(&port.label)?;
        let leaf = self.resolve_leaf_or_err(idx, &port, PortSide::Input)?;
        if self.occupied_input_ports[&idx][leaf] {
            return Err(QuantumProgramError::PortAlreadyConnected {
                label: port.label,
                path: self.graph[idx].input_leaf_paths[leaf].clone(),
            });
        }
        self.occupied_input_ports.get_mut(&idx).unwrap()[leaf] = true;
        self.inputs.push((key.to_string(), idx, leaf));
        self.rebuild_io_types();
        Ok(())
    }

    /// Declare an output port of this program under `key`. Multiple outputs
    /// may bind to the same source port (fan-out is supported).
    pub fn set_output(&mut self, key: &str, port: Port) -> Result<(), QuantumProgramError> {
        if self.outputs.iter().any(|(k, _, _)| k == key) {
            return Err(QuantumProgramError::DuplicateIOKey(key.to_string()));
        }
        let idx = self.lookup_label(&port.label)?;
        let leaf = self.resolve_leaf_or_err(idx, &port, PortSide::Output)?;
        self.outputs.push((key.to_string(), idx, leaf));
        self.rebuild_io_types();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Graph queries
    // -----------------------------------------------------------------------

    /// Return a reference to the node with the given label, or `None`.
    pub fn get_node(&self, label: &str) -> Option<&dyn ProgramNode<CallError = BoxedNodeError>> {
        self.label_to_node
            .get(label)
            .map(|&idx| &*self.graph[idx].node.0)
    }

    /// Return `true` if a node with the given label exists.
    pub fn has_node(&self, label: &str) -> bool {
        self.label_to_node.contains_key(label)
    }

    /// Iterate over all nodes as `(label, node)` pairs.
    pub fn iter_nodes(
        &self,
    ) -> impl Iterator<Item = (&str, &dyn ProgramNode<CallError = BoxedNodeError>)> {
        self.label_to_node
            .iter()
            .map(|(label, &idx)| (label.as_str(), &*self.graph[idx].node.0))
    }

    /// Iterate over all edges as `(from_port, to_port)` pairs.
    pub fn iter_edges(&self) -> impl Iterator<Item = (Port, Port)> + '_ {
        let idx_to_label: HashMap<NodeIndex, &str> = self
            .label_to_node
            .iter()
            .map(|(k, &v)| (v, k.as_str()))
            .collect();

        self.graph.edge_references().map(move |e| {
            use rustworkx_core::petgraph::visit::EdgeRef;
            let w = e.weight();
            let from = Port {
                label: idx_to_label[&e.source()].to_string(),
                path: self.graph[e.source()].output_leaf_paths[w.from_leaf].clone(),
            };
            let to = Port {
                label: idx_to_label[&e.target()].to_string(),
                path: self.graph[e.target()].input_leaf_paths[w.to_leaf].clone(),
            };
            (from, to)
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn lookup_label(&self, label: &str) -> Result<NodeIndex, QuantumProgramError> {
        self.label_to_node
            .get(label)
            .copied()
            .ok_or_else(|| QuantumProgramError::UnknownLabel(label.to_string()))
    }

    /// Resolve a user-supplied port path to its leaf index on `node_idx`'s
    /// input or output schema.
    fn resolve_leaf(
        &self,
        node_idx: NodeIndex,
        path: &[OwnedPathEntry],
        side: PortSide,
    ) -> Option<LeafIdx> {
        let view = &self.graph[node_idx];
        let normalized = normalize_path(view.types(side), path)?;
        view.leaf_paths(side).iter().position(|p| p == &normalized)
    }

    fn resolve_leaf_or_err(
        &self,
        node_idx: NodeIndex,
        port: &Port,
        side: PortSide,
    ) -> Result<LeafIdx, QuantumProgramError> {
        self.resolve_leaf(node_idx, &port.path, side)
            .ok_or_else(|| QuantumProgramError::InvalidPort {
                label: port.label.clone(),
                path: port.path.clone(),
            })
    }

    /// Rebuild `input_types` and `output_types` from declared I/O.
    fn rebuild_io_types(&mut self) {
        self.input_types = self.collect_declared_types(&self.inputs, PortSide::Input);
        self.output_types = self.collect_declared_types(&self.outputs, PortSide::Output);
    }

    fn collect_declared_types(
        &self,
        declared: &[(String, NodeIndex, LeafIdx)],
        side: PortSide,
    ) -> DataTree<TensorType> {
        let mut out = DataTree::new();
        for (key, idx, leaf) in declared {
            let path = &self.graph[*idx].leaf_paths(side)[*leaf];
            if let Some(DataTree::Leaf(tt)) = get_by_owned_path(self.graph[*idx].types(side), path)
            {
                out.insert_leaf(key, tt.clone());
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// ProgramNode impl
// ---------------------------------------------------------------------------

impl ProgramNode for QuantumProgram {
    type CallError = QuantumProgramCallError;

    fn name(&self) -> &'static str {
        "quantum_program"
    }

    fn namespace(&self) -> &'static str {
        "qiskit"
    }

    fn input_types(&self) -> &DataTree<TensorType> {
        &self.input_types
    }

    fn output_types(&self) -> &DataTree<TensorType> {
        &self.output_types
    }

    fn implements_call(&self) -> bool {
        true
    }

    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, QuantumProgramCallError> {
        use rustworkx_core::petgraph::visit::{EdgeRef, NodeIndexable};

        // The topological ordering, which determines call order, doubles as our local
        // dense indexing: each node's position in `topo_order` is its `topo_idx`.
        let topo_order: Vec<NodeIndex> = lexicographical_topological_sort(
            &self.graph,
            |n: NodeIndex| Ok::<usize, std::convert::Infallible>(n.index()),
            false,
            None,
        )
        .map_err(|_| QuantumProgramCallError::Cycle)?;

        // Map from NodeIndex → topo_idx, formatted as a list to avoid using a hash map.
        let mut node_idx_to_topo_idx: Vec<Option<usize>> = vec![None; self.graph.node_bound()];
        for (topo_idx, &node_idx) in topo_order.iter().enumerate() {
            node_idx_to_topo_idx[node_idx.index()] = Some(topo_idx);
        }

        // Per-node consumer table: for example, `consumers[topo_idx][2]` is the list of
        // where the 2nd output of the node corresponding to topo_idx should be sent.
        let mut consumers: Vec<Vec<Vec<ConsumerKind>>> = topo_order
            .iter()
            .map(|&idx| {
                let n_outputs = self.graph[idx].output_leaf_paths.len();
                (0..n_outputs).map(|_| Vec::new()).collect()
            })
            .collect();
        for edge_ref in self.graph.edge_references() {
            let edge_data = edge_ref.weight();
            let src_topo = node_idx_to_topo_idx[edge_ref.source().index()].unwrap();
            let target_topo_idx = node_idx_to_topo_idx[edge_ref.target().index()].unwrap();
            consumers[src_topo][edge_data.from_leaf].push(ConsumerKind::Edge {
                target_topo_idx,
                target_leaf: edge_data.to_leaf,
            });
        }
        for (output_idx, (_, source_idx, source_leaf)) in self.outputs.iter().enumerate() {
            let src_topo = node_idx_to_topo_idx[source_idx.index()].unwrap();
            consumers[src_topo][*source_leaf].push(ConsumerKind::ProgramOutput { output_idx });
        }

        // Per-node input slot buffer: filled as nodes produce outputs, emptied as they are consumed.
        // This buffer is initially populated with the inputs to this quantum program itself.
        // The indexing goes as `input_buffer[topo_idx][input_slot_idx]`.
        let mut input_buffer: Vec<Vec<Option<Tensor>>> = topo_order
            .iter()
            .map(|&idx| vec![None; self.graph[idx].input_leaf_paths.len()])
            .collect();
        for (i, (key, target_idx, target_leaf)) in self.inputs.iter().enumerate() {
            let tensor = args
                .get(i)
                .cloned()
                .ok_or_else(|| QuantumProgramCallError::MissingProgramInput { key: key.clone() })?;
            let target_topo = node_idx_to_topo_idx[target_idx.index()].unwrap();
            input_buffer[target_topo][*target_leaf] = Some(tensor);
        }

        // Final program outputs, indexed by position in `self.outputs`.
        let mut output_tensors: Vec<Option<Tensor>> =
            (0..self.outputs.len()).map(|_| None).collect();

        // Finally, we can loop through each node and call it.
        for (topo_idx, &node_idx) in topo_order.iter().enumerate() {
            let node_consumers = std::mem::take(&mut consumers[topo_idx]);
            if node_consumers.iter().all(|c| c.is_empty()) {
                // This node's output has no consumers, we might as well not call it.
                continue;
            }

            let view = &self.graph[node_idx];
            let label_for = || {
                self.label_to_node
                    .iter()
                    .find(|(_, i)| **i == node_idx)
                    .map(|(k, _)| k.clone())
                    .unwrap_or_default()
            };

            // Yoink all of this node's inputs out of the buffer. An empty slot
            // means a port was neither wired by an edge nor declared as a
            // program input — surface that as an error rather than panicking.
            let buf = std::mem::take(&mut input_buffer[topo_idx]);
            let flat_args: Vec<Tensor> = buf
                .into_iter()
                .enumerate()
                .map(|(leaf_idx, slot)| {
                    slot.ok_or_else(|| QuantumProgramCallError::UnwiredInput {
                        label: label_for(),
                        path: view.input_leaf_paths[leaf_idx].clone(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let outputs = view.node.call_flat(&flat_args).map_err(|source| {
                QuantumProgramCallError::NodeCall {
                    label: label_for(),
                    source,
                }
            })?;

            if outputs.len() != view.output_leaf_paths.len() {
                return Err(QuantumProgramCallError::NodeOutputArityMismatch {
                    label: label_for(),
                    expected: view.output_leaf_paths.len(),
                    actual: outputs.len(),
                });
            }

            // Clone for all consumers but the last; move into the last.
            for (out_leaf, tensor) in outputs.into_iter().enumerate() {
                let kinds = &node_consumers[out_leaf];
                if kinds.is_empty() {
                    continue;
                }
                let mut iter = kinds.iter();
                let last_kind = iter.next_back().unwrap();
                for kind in iter {
                    dispatch(kind, tensor.clone(), &mut input_buffer, &mut output_tensors);
                }
                dispatch(last_kind, tensor, &mut input_buffer, &mut output_tensors);
            }
        }

        Ok(self
            .outputs
            .iter()
            .enumerate()
            .map(|(i, (key, _, _))| {
                output_tensors[i]
                    .take()
                    .unwrap_or_else(|| unreachable!("program output {key:?} was not produced"))
            })
            .collect())
    }
}

/// Send `tensor` to its consumer (downstream input slot or program output).
fn dispatch(
    kind: &ConsumerKind,
    tensor: Tensor,
    input_buffer: &mut [Vec<Option<Tensor>>],
    output_tensors: &mut [Option<Tensor>],
) {
    match kind {
        ConsumerKind::Edge {
            target_topo_idx,
            target_leaf,
        } => {
            input_buffer[*target_topo_idx][*target_leaf] = Some(tensor);
        }
        ConsumerKind::ProgramOutput { output_idx } => {
            output_tensors[*output_idx] = Some(tensor);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_node::MissingCallError;
    use crate::store::Store;
    use crate::tensor::{DTypeLike, Tensor, TensorType};
    use std::sync::OnceLock;

    fn make_store(val: f64) -> Store {
        Store::new(DataTree::new_leaf(Tensor::from([val])))
    }

    /// A test node with two named inputs ("x", "y") and one leaf output.
    /// Doesn't implement call.
    struct BinaryTestNode;

    impl ProgramNode for BinaryTestNode {
        type CallError = MissingCallError;
        fn name(&self) -> &'static str {
            "binary_test"
        }
        fn namespace(&self) -> &'static str {
            "test"
        }
        fn input_types(&self) -> &DataTree<TensorType> {
            static LOCK: OnceLock<DataTree<TensorType>> = OnceLock::new();
            LOCK.get_or_init(|| {
                let mut t = DataTree::new();
                t.insert_leaf(
                    "x",
                    TensorType {
                        dtype: DTypeLike::Var("x".into()),
                        shape: vec![],
                        broadcastable: true,
                    },
                );
                t.insert_leaf(
                    "y",
                    TensorType {
                        dtype: DTypeLike::Var("y".into()),
                        shape: vec![],
                        broadcastable: true,
                    },
                );
                t
            })
        }
        fn output_types(&self) -> &DataTree<TensorType> {
            static LOCK: OnceLock<DataTree<TensorType>> = OnceLock::new();
            LOCK.get_or_init(|| {
                DataTree::new_leaf(TensorType {
                    dtype: DTypeLike::Var("out".into()),
                    shape: vec![],
                    broadcastable: false,
                })
            })
        }
        fn implements_call(&self) -> bool {
            false
        }
        fn call_flat(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, MissingCallError> {
            Err(MissingCallError::new(self.full_name()))
        }
    }

    #[test]
    fn test_add_node_and_has_node() {
        let mut prog = QuantumProgram::new();
        assert!(!prog.has_node("x"));
        prog.add_node("x", make_store(1.0)).unwrap();
        assert!(prog.has_node("x"));
    }

    #[test]
    fn test_add_node_duplicate_label() {
        let mut prog = QuantumProgram::new();
        prog.add_node("x", make_store(1.0)).unwrap();
        let err = prog.add_node("x", make_store(2.0)).unwrap_err();
        assert!(matches!(err, QuantumProgramError::DuplicateLabel(ref l) if l == "x"));
    }

    #[test]
    fn test_iter_nodes() {
        let mut prog = QuantumProgram::new();
        prog.add_node("a", make_store(1.0)).unwrap();
        prog.add_node("b", make_store(2.0)).unwrap();
        let mut labels: Vec<&str> = prog.iter_nodes().map(|(l, _)| l).collect();
        labels.sort_unstable();
        assert_eq!(labels, ["a", "b"]);
    }

    #[test]
    fn test_get_node() {
        let mut prog = QuantumProgram::new();
        assert!(prog.get_node("nope").is_none());
        prog.add_node("s", make_store(1.0)).unwrap();
        assert_eq!(prog.get_node("s").unwrap().name(), "store");
    }

    #[test]
    fn test_add_edge_basic() {
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(1.0)).unwrap();
        prog.add_node("op", BinaryTestNode).unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("op", vec!["x".into()]))
            .unwrap();

        let edges: Vec<_> = prog.iter_edges().collect();
        assert_eq!(edges.len(), 1);
        let (from, to) = &edges[0];
        assert_eq!(from.label, "s");
        assert_eq!(to.label, "op");
    }

    #[test]
    fn test_add_edge_unknown_node() {
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(1.0)).unwrap();
        let err = prog
            .add_edge(Port::new("s", vec![]), Port::new("missing", vec![]))
            .unwrap_err();
        assert!(matches!(err, QuantumProgramError::UnknownLabel(ref l) if l == "missing"));
    }

    #[test]
    fn test_add_edge_invalid_output_port() {
        let mut prog = QuantumProgram::new();
        prog.add_node("op", BinaryTestNode).unwrap();
        prog.add_node("op2", BinaryTestNode).unwrap();
        // BinaryTestNode's output is a single leaf (empty path); a bogus key fails.
        let err = prog
            .add_edge(
                Port::new("op", vec!["nonexistent".into()]),
                Port::new("op2", vec!["x".into()]),
            )
            .unwrap_err();
        assert!(matches!(err, QuantumProgramError::InvalidPort { .. }));
    }

    #[test]
    fn test_add_edge_port_already_connected() {
        let mut prog = QuantumProgram::new();
        prog.add_node("s1", make_store(1.0)).unwrap();
        prog.add_node("s2", make_store(2.0)).unwrap();
        prog.add_node("op", BinaryTestNode).unwrap();
        prog.add_edge(Port::new("s1", vec![]), Port::new("op", vec!["x".into()]))
            .unwrap();
        let err = prog
            .add_edge(Port::new("s2", vec![]), Port::new("op", vec!["x".into()]))
            .unwrap_err();
        assert!(matches!(
            err,
            QuantumProgramError::PortAlreadyConnected { .. }
        ));
    }

    #[test]
    fn test_set_input_output_and_types() {
        let mut prog = QuantumProgram::new();
        prog.add_node("op", BinaryTestNode).unwrap();
        prog.set_input("lhs", Port::new("op", vec!["x".into()]))
            .unwrap();
        prog.set_input("rhs", Port::new("op", vec!["y".into()]))
            .unwrap();
        prog.set_output("result", Port::new("op", vec![])).unwrap();

        assert!(prog.input_types().get_by_str_key("lhs").is_some());
        assert!(prog.input_types().get_by_str_key("rhs").is_some());
        assert!(prog.output_types().get_by_str_key("result").is_some());
    }

    #[test]
    fn test_set_input_duplicate_key() {
        let mut prog = QuantumProgram::new();
        prog.add_node("op", BinaryTestNode).unwrap();
        prog.set_input("x", Port::new("op", vec!["x".into()]))
            .unwrap();
        let err = prog
            .set_input("x", Port::new("op", vec!["y".into()]))
            .unwrap_err();
        assert!(matches!(err, QuantumProgramError::DuplicateIOKey(ref k) if k == "x"));
    }

    #[test]
    fn test_set_input_port_already_connected() {
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(1.0)).unwrap();
        prog.add_node("op", BinaryTestNode).unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("op", vec!["x".into()]))
            .unwrap();
        let err = prog
            .set_input("x", Port::new("op", vec!["x".into()]))
            .unwrap_err();
        assert!(matches!(
            err,
            QuantumProgramError::PortAlreadyConnected { .. }
        ));
    }

    #[test]
    fn test_fanout_same_output_to_two_inputs() {
        use crate::math_nodes::binary::Add;
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(1.0)).unwrap();
        prog.add_node("add", Add).unwrap();
        prog.set_output("total", Port::new("add", vec![])).unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("add", vec!["x".into()]))
            .unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("add", vec!["y".into()]))
            .unwrap();

        let result = prog.call_flat(&[]).unwrap();
        assert_eq!(result.len(), 1);
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected f64 leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[2.0]);
    }

    #[test]
    fn test_program_node_impl() {
        let prog = QuantumProgram::new();
        assert_eq!(prog.name(), "quantum_program");
        assert_eq!(prog.namespace(), "qiskit");
        assert_eq!(prog.full_name(), "qiskit.quantum_program");
        assert!(prog.implements_call());
    }

    #[test]
    fn test_call_store_add_pipeline() {
        // s1(3.0) and s2(5.0) feed into Add; result is exported.
        use crate::math_nodes::binary::Add;
        let mut prog = QuantumProgram::new();
        prog.add_node("s1", make_store(3.0)).unwrap();
        prog.add_node("s2", make_store(5.0)).unwrap();
        prog.add_node("add", Add).unwrap();
        prog.add_edge(Port::new("s1", vec![]), Port::new("add", vec!["x".into()]))
            .unwrap();
        prog.add_edge(Port::new("s2", vec![]), Port::new("add", vec!["y".into()]))
            .unwrap();
        prog.set_output("result", Port::new("add", vec![])).unwrap();

        let out = prog.call_flat(&[]).unwrap();
        let Tensor::F64(arr) = &out[0] else {
            panic!("expected f64 leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[8.0_f64]);
    }

    #[test]
    fn test_call_with_program_input() {
        // A Store provides one operand of Add; the other comes from a program input.
        use crate::math_nodes::binary::Add;
        let mut prog = QuantumProgram::new();
        prog.add_node("s", make_store(10.0)).unwrap();
        prog.add_node("add", Add).unwrap();
        prog.add_edge(Port::new("s", vec![]), Port::new("add", vec!["x".into()]))
            .unwrap();
        prog.set_input("delta", Port::new("add", vec!["y".into()]))
            .unwrap();
        prog.set_output("result", Port::new("add", vec![])).unwrap();

        let out = prog.call_flat(&[Tensor::from([4.0_f64])]).unwrap();
        let Tensor::F64(arr) = &out[0] else {
            panic!("expected f64 leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[14.0_f64]);
    }

    #[test]
    fn test_index_path_normalization() {
        // Address a Store output by Index(0) and confirm it resolves to the
        // same leaf as Key("a") — re-using either name on the same downstream
        // input should report it as already occupied.
        use crate::math_nodes::binary::Add;
        let mut data = DataTree::new();
        data.insert_leaf("a", Tensor::from([1.0_f64]));
        data.insert_leaf("b", Tensor::from([2.0_f64]));
        let mut prog = QuantumProgram::new();
        prog.add_node("s", Store::new(data)).unwrap();
        prog.add_node("add", Add).unwrap();
        prog.add_edge(
            Port::new("s", vec![OwnedPathEntry::Index(0)]),
            Port::new("add", vec!["x".into()]),
        )
        .unwrap();
        let err = prog
            .add_edge(
                Port::new("s", vec!["b".into()]),
                Port::new("add", vec!["x".into()]),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            QuantumProgramError::PortAlreadyConnected { .. }
        ));
    }
}
