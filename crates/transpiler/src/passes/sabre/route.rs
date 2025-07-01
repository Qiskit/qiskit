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

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::num::NonZero;

use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::Python;

use hashbrown::HashSet;
use indexmap::IndexMap;
use ndarray::Array2;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon_cond::CondIterator;
use rustworkx_core::dictmap::*;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::visit::{EdgeCount, EdgeRef};
use rustworkx_core::shortest_path::dijkstra;
use rustworkx_core::token_swapper::token_swapper;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeType, Wire};
use qiskit_circuit::nlayout::NLayout;
use qiskit_circuit::operations::{OperationRef, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{getenv_use_multiple_threads, imports, PhysicalQubit, Qubit, VirtualQubit};

use crate::target::{Target, TargetCouplingError};
use crate::TranspilerError;

use super::dag::{InteractionKind, SabreDAG};
use super::distance::distance_matrix;
use super::heuristic::{BasicHeuristic, DecayHeuristic, Heuristic, SetScaling};
use super::layer::Layer;
use super::neighbors::Neighbors;
use super::vec_map::VecMap;

/// Number of trials for control flow block swap epilogues.
const SWAP_EPILOGUE_TRIALS: usize = 4;

/// The number of control-flow blocks to take off the stack.
///
/// This funky struct is just a trick to get the Rust compiler to use the niche optimisation for
/// `RoutedItemKind`.  We actually store the number of blocks as the bitwise negation of the true
/// number, and disallow there being `u32::MAX` blocks.
///
/// At the time of writing (2025-06-11), Qiskit's Python-space model doesn't allow constructing any
/// control-flow operations with zero blocks, so we could use `NonZero<u32>` directly.  Technically,
/// though, a `switch` on a zero-bit register _could_ be valid and have zero blocks, so doing this
/// little trick makes us safe against that long-range assumption changing, for zero measureable
/// runtime cost.
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct ControlFlowBlockCount(NonZero<u32>);
impl ControlFlowBlockCount {
    pub fn get(self) -> u32 {
        !self.0.get()
    }
}
impl From<u32> for ControlFlowBlockCount {
    fn from(val: u32) -> Self {
        Self((!val).try_into().expect("cannot store u32::MAX blocks"))
    }
}

enum RoutedItemKind {
    Simple,
    /// How many blocks out of [RoutingResult::control_flow] we need to take.  This is stored
    /// out-of-band of the item kind because control-flow is expected to be very uncommon, and we
    /// don't want to needless increase the size of the enum.
    ControlFlow(ControlFlowBlockCount),
}
struct RoutedItem {
    initial_swaps: Option<Box<[[PhysicalQubit; 2]]>>,
    /// The corresponding node in the Sabre graph.
    node: NodeIndex,
    kind: RoutedItemKind,
}
impl RoutedItem {
    #[inline]
    pub fn initial_swaps(&self) -> &[[PhysicalQubit; 2]] {
        self.initial_swaps.as_deref().unwrap_or(&[])
    }
}

/// The final analysis of the Sabre routing algorithm.
///
/// This represents a total order of instructions to be applied, including swaps, to produce a
/// circuit that is fully routed.  This structure alone is insufficient; it contains references to a
/// [SabreDAG], which in turn contains references to a [DAGCircuit], and you need the initial
/// [NLayout] object to know where the virtual qubits in the input [DAGCircuit] should be mapped to
/// at the start of the circuit.  The [RoutingResult] object wraps up this object with the other
/// necessary components.
struct Order<'a> {
    order: Vec<RoutedItem>,
    final_swaps: Vec<[PhysicalQubit; 2]>,
    control_flow: Vec<RoutingResult<'a>>,
}
impl<'a> Order<'a> {
    /// Initialize an empty `Order` with suitable capacity for the given problem.
    #[inline]
    pub fn for_problem(problem: RoutingProblem<'a>) -> Self {
        Self {
            order: Vec::with_capacity(problem.sabre.dag.node_count()),
            final_swaps: Vec::new(),
            control_flow: Vec::new(),
        }
    }

    /// Count the number of swaps inserted at the top level (i.e. without recursing into
    /// control-flow operations).
    #[inline]
    pub fn swap_count(&self) -> usize {
        self.order
            .iter()
            .map(|item| item.initial_swaps().len())
            .sum()
    }
}

/// A complete result from the Sabre routing algorithm, including the initial problem and layout
/// that it searched from.
///
/// The [Order] is the calculated result from the analysis (and the [final_layout] field is a
/// derived quantity that we simply get for free at the end of the algorithm, so store here), and
/// this struct wraps it up with the problem description and initial layout necessary to fully
/// interpret it.
pub struct RoutingResult<'a> {
    problem: RoutingProblem<'a>,
    order: Order<'a>,
    /// The initial layout that the routing algorithm started from.
    pub initial_layout: NLayout,
    /// The layout after the routing algorithm had finished.  This can be rederived from [order] and
    /// [initial_layout], but we get it for free anyway.
    pub final_layout: NLayout,
}
impl RoutingResult<'_> {
    /// Count the number of swaps inserted at the top level (i.e. without recursing into
    /// control-flow operations).
    #[inline]
    pub fn swap_count(&self) -> usize {
        self.order.swap_count()
    }

    fn num_qubits(&self) -> usize {
        self.initial_layout.num_qubits()
    }

    /// Rebuild the physical circuit from the virtual DAG, using the natural width of the target of
    /// this component.
    ///
    /// This is the correct method to call if the [RoutingTarget] represented the entirety of the
    /// device.  If the device was subset (such as for disjoint handling), use [rebuild_onto] with
    /// suitable mappings back to the full-width [PhysicalQubit] instances instead.
    pub fn rebuild(&self) -> PyResult<DAGCircuit> {
        let num_swaps = self.order.swap_count();
        let dag = self.problem.dag.physical_empty_like_with_capacity(
            self.num_qubits(),
            self.problem.dag.num_ops() + num_swaps,
            self.problem.dag.dag().edge_count() + 2 * num_swaps,
        )?;
        self.rebuild_onto(dag, |q| q)
    }

    /// Construct the routed [DAGCircuit] from the result, placing the operations onto an existing
    /// [DAGCircuit]
    ///
    /// `dag` should be a physical circuit that is the full width of the device, and already contain
    /// all the classical data that is referenced by this circuit (clbits, classical registers,
    /// vars, stretches, global phase, metadata, etc).  The target [DAGCircuit] might have more
    /// qubits than the layouts in this [RoutingResult], for example if the device was subset for
    /// disjoint handling.  Use `map_fn` to map the "physical qubits" in this component to the
    /// actual physical qubits they refer to.
    pub fn rebuild_onto(
        &self,
        dag: DAGCircuit,
        map_fn: impl Fn(PhysicalQubit) -> PhysicalQubit,
    ) -> PyResult<DAGCircuit> {
        let apply_swap = |swap: &[PhysicalQubit; 2],
                          layout: &mut NLayout,
                          dag: &mut DAGCircuitBuilder|
         -> PyResult<NodeIndex> {
            layout.swap_physical(swap[0], swap[1]);
            let swap = PackedInstruction::from_standard_gate(
                StandardGate::Swap,
                None,
                dag.insert_qargs(&[Qubit(map_fn(swap[1]).0), Qubit(map_fn(swap[0]).0)]),
            );
            dag.push_back(swap)
        };
        // The size here is pretty arbitrary, providing it can fit at least 2q operations in.
        let mut apply_scratch = Vec::with_capacity(4);
        let mut apply_op = |inst: &PackedInstruction,
                            layout: &NLayout,
                            dag: &mut DAGCircuitBuilder|
         -> PyResult<NodeIndex> {
            apply_scratch.clear();
            for qubit in self.problem.dag.get_qargs(inst.qubits) {
                apply_scratch.push(Qubit(map_fn(VirtualQubit(qubit.0).to_phys(layout)).0));
            }
            let new_inst = PackedInstruction {
                qubits: dag.insert_qargs(&apply_scratch),
                ..inst.clone()
            };
            dag.push_back(new_inst)
        };

        let mut dag = dag.into_builder();
        let mut layout = self.initial_layout.clone();
        let mut blocks = self.order.control_flow.iter();
        for node in &self.problem.sabre.initial {
            let NodeType::Operation(inst) = &self.problem.dag[*node] else {
                panic!("Sabre DAG should only contain op nodes");
            };
            apply_op(inst, &layout, &mut dag)?;
        }
        for item in &self.order.order {
            for swap in item.initial_swaps() {
                apply_swap(swap, &mut layout, &mut dag)?;
            }
            // In theory, `indices` will always have at least one entry if you're rebuilding the
            // DAG from a Sabre result, because there wouldn't be a Sabre node without at least one
            // DAG node backing it.  That said, we _do_ allow construction of Sabre graphs that have
            // thrown away this information ([SabreDAG::only_interactions]), and there's still a
            // well-defined behaviour to take.
            let split = self.problem.sabre.dag[item.node].indices.split_first();
            let Some((head, rest)) = split else {
                continue;
            };
            let NodeType::Operation(inst) = &self.problem.dag[*head] else {
                panic!("Sabre DAG should only contain op nodes");
            };

            match item.kind {
                RoutedItemKind::Simple => apply_op(inst, &layout, &mut dag)?,
                RoutedItemKind::ControlFlow(num_blocks) => {
                    let blocks = blocks
                        .by_ref()
                        .take(num_blocks.get() as usize)
                        .map(|block| block.rebuild())
                        .collect::<Result<Vec<_>, _>>()?;
                    let explicit = self
                        .problem
                        .dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|q| VirtualQubit(q.index() as u32))
                        .collect::<HashSet<_>>();
                    // Collect lists of the qargs that will remain, and the idle qubits that need to
                    // be removed from the DAG, then remove the idle ones.
                    // TODO: this logic of collecting the remaining `qargs` in order is making an
                    // assumption that the later call to `DAGCircuit::remove_qubits` retains
                    // relative ordering of the remaining qubits, but the method doesn't formally
                    // commit to that.
                    let mut qargs = Vec::new();
                    let mut idle = Vec::new();
                    for qubit in 0..self.num_qubits() as u32 {
                        let phys = PhysicalQubit::new(qubit);
                        let virt = phys.to_virt(&layout);
                        let qubit = Qubit(qubit);
                        if explicit.contains(&virt)
                            || blocks
                                .iter()
                                .any(|dag| !dag.is_wire_idle(Wire::Qubit(qubit)))
                        {
                            qargs.push(Qubit(map_fn(phys).0));
                        } else {
                            idle.push(qubit);
                        }
                    }
                    let new_inst = Python::with_gil(|py| -> PyResult<_> {
                        // TODO: have to use Python-space `dag_to_circuit` because the Rust-space is
                        // only half the conversion (since it doesn't handle vars or stretches).
                        let dag_to_circuit = imports::DAG_TO_CIRCUIT.get_bound(py);
                        let blocks = blocks
                            .into_iter()
                            .map(|mut dag| {
                                dag.remove_qubits(idle.iter().copied())?;
                                dag_to_circuit.call1((dag, false))
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        let OperationRef::Instruction(py_inst) = inst.op.view() else {
                            panic!("control-flow nodes must be PyInstruction");
                        };
                        let new_node = py_inst
                            .instruction
                            .bind(py)
                            .call_method1("replace_blocks", (blocks,))?;
                        let op: OperationFromPython = new_node.extract()?;
                        Ok(PackedInstruction {
                            op: op.operation,
                            qubits: dag.insert_qargs(&qargs),
                            clbits: inst.clbits,
                            params: (!op.params.is_empty()).then(|| Box::new(op.params)),
                            label: op.label,
                            #[cfg(feature = "cache_pygates")]
                            py_op: new_node.unbind().into(),
                        })
                    })?;
                    dag.push_back(new_inst)?
                }
            };
            for node in rest {
                let NodeType::Operation(inst) = &self.problem.dag[*node] else {
                    panic!("sabre DAG should only contain op nodes");
                };
                apply_op(inst, &layout, &mut dag)?;
            }
        }
        for swap in &self.order.final_swaps {
            apply_swap(swap, &mut layout, &mut dag)?;
        }
        debug_assert_eq!(layout, self.final_layout);
        Ok(dag.build())
    }
}

/// A description of the QPU that we're routing to.
#[derive(Clone, Debug)]
pub struct RoutingTarget {
    pub neighbors: Neighbors,
    pub distance: Array2<f64>,
}
impl RoutingTarget {
    pub fn from_neighbors(neighbors: Neighbors) -> Self {
        Self {
            distance: distance_matrix(&neighbors, usize::MAX, f64::NAN),
            neighbors,
        }
    }

    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.neighbors.num_qubits()
    }
}

/// Python wrapper for the Rust-space Sabre target object.
///
/// Contains `None` when the target had all-to-all connectivity (in which case the two property
/// methods [coupling_list] and [distance_matrix] also return `None`).
#[pyclass]
#[pyo3(name = "RoutingTarget", module = "qiskit._accelerate.sabre")]
pub struct PyRoutingTarget(pub Option<RoutingTarget>);
#[pymethods]
impl PyRoutingTarget {
    #[staticmethod]
    fn from_target(target: &Target) -> PyResult<Self> {
        let coupling = match target.coupling_graph() {
            Ok(coupling) => coupling,
            Err(TargetCouplingError::AllToAll) => return Ok(Self(None)),
            Err(e @ TargetCouplingError::MultiQ) => {
                return Err(TranspilerError::new_err(e.to_string()))
            }
        };
        Ok(Self(Some(RoutingTarget::from_neighbors(
            Neighbors::from_coupling(&coupling),
        ))))
    }

    fn coupling_list(&self) -> Option<Vec<[PhysicalQubit; 2]>> {
        use rustworkx_core::petgraph::visit::IntoEdgeReferences;
        self.0.as_ref().map(|target| {
            target
                .neighbors
                .edge_references()
                .map(|edge| [edge.source(), edge.target()])
                .collect()
        })
    }

    fn distance_matrix<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.0.as_ref().map(|target| target.distance.to_pyarray(py))
    }
}

/// Helper record struct for a Sabre routing problem.
///
/// This is mostly just encapsulation to make the nested call sites less verbose.
#[derive(Clone, Copy, Debug)]
pub struct RoutingProblem<'a> {
    pub target: &'a RoutingTarget,
    pub sabre: &'a SabreDAG,
    pub dag: &'a DAGCircuit,
    pub heuristic: &'a Heuristic,
}
impl<'a> RoutingProblem<'a> {
    /// The same problem, but using a different [SabreDAG] representation.
    pub fn with_sabre(mut self, sabre: &'a SabreDAG) -> Self {
        self.sabre = sabre;
        self
    }
}

/// Long-term internal state of the Sabre routing algorithm.
///
/// This includes all the scratch space and tracking that we use over the course of many swap
/// insertions, but doesn't include ephemeral state that never needs to leave the main loop.  This
/// is mostly just a convenience, so we don't have to pass everything from function to function.
struct State {
    layout: NLayout,
    front_layer: Layer,
    lookahead_layers: Box<[Layer]>,
    /// How many predecessors still need to be satisfied for each node index before it is at the
    /// front of the topological iteration through the nodes as they're routed.
    required_predecessors: VecMap<NodeIndex, u32>,
    decay: VecMap<PhysicalQubit, f64>,
    /// Reusable allocated storage space for accumulating and scoring swaps.  This is owned as part
    /// of the general state to avoid reallocation costs.
    swap_scores: Vec<([PhysicalQubit; 2], f64)>,
    /// Reusable allocated storage space for tracking the current best swaps.  This is owned as
    /// part of the general state to avoid reallocation costs.
    best_swaps: Vec<[PhysicalQubit; 2]>,
    rng: Pcg64Mcg,
    seed: u64,
}

impl State {
    /// Apply a swap to the program-state structures (front layer, extended set and current
    /// layout).
    #[inline]
    fn apply_swap(&mut self, swap: [PhysicalQubit; 2]) {
        self.front_layer.apply_swap(swap);
        for layer in self.lookahead_layers.iter_mut() {
            layer.apply_swap(swap);
        }
        self.layout.swap_physical(swap[0], swap[1]);
    }

    /// Return the node, if any, that is on this qubit and is routable with the current layout.
    #[inline]
    fn routable_node_on_qubit(
        &self,
        problem: RoutingProblem,
        qubit: PhysicalQubit,
    ) -> Option<NodeIndex> {
        self.front_layer.qubits()[qubit].and_then(|(node, other)| {
            problem
                .target
                .neighbors
                .contains_edge(qubit, other)
                .then_some(node)
        })
    }

    /// Search forwards from [nodes], adding any that are reachable, routable and have no
    /// unsatisfied dependencies to the result.
    ///
    /// All nodes in [nodes] must:
    ///
    /// * have no unsatisfied predecessors
    /// * not be in the [FrontLayer]
    ///
    /// # Panics
    ///
    /// If [initial_swaps] is given, but no nodes can be routed.
    fn update_route<'a>(
        &mut self,
        problem: RoutingProblem<'a>,
        order: &mut Order<'a>,
        nodes: &[NodeIndex],
        mut initial_swaps: Option<Vec<[PhysicalQubit; 2]>>,
    ) {
        let mut to_visit = nodes.iter().copied().collect::<VecDeque<_>>();
        while let Some(node_id) = to_visit.pop_front() {
            let node = &problem.sabre.dag[node_id];
            let kind = match &node.kind {
                InteractionKind::Synchronize => RoutedItemKind::Simple,
                InteractionKind::TwoQ([a, b]) => {
                    let a = a.to_phys(&self.layout);
                    let b = b.to_phys(&self.layout);
                    if problem.target.neighbors.contains_edge(a, b) {
                        RoutedItemKind::Simple
                    } else {
                        self.front_layer.insert(node_id, [a, b]);
                        continue;
                    }
                }
                InteractionKind::ControlFlow(blocks) => {
                    let dag_node_id = *node.indices.first().expect(
                        "if control-flow interactions are included, so are original DAG indices",
                    );
                    let NodeType::Operation(inst) = &problem.dag[dag_node_id] else {
                        panic!("Sabre DAG should only contain op nodes");
                    };
                    // The control-flow blocks aren't full width, so their "virtual" qubits aren't
                    // numbered the same as the full circuit's.  We still need it to route _as if_
                    // it's fully expanded with ancillas, though.
                    let mut layout =
                        NLayout::generate_trivial_layout(problem.target.num_qubits() as u32);
                    for (inner, outer) in problem.dag.get_qargs(inst.qubits).iter().enumerate() {
                        // The virtual qubit _inside_ the DAG block is mapped to some meaningless
                        // physical qubit in our current layout...
                        let dummy = VirtualQubit::new(inner as u32).to_phys(&layout);
                        // ... and we want it to be mapped to the current physical qubit of the
                        // corresponding outer virtual qubit.  We don't care where the dummy goes.
                        let actual = VirtualQubit::new(outer.index() as u32).to_phys(&self.layout);
                        layout.swap_physical(dummy, actual);
                    }
                    order.control_flow.extend(blocks.iter().map(|(sabre, dag)| {
                        let problem = RoutingProblem {
                            sabre,
                            dag,
                            ..problem
                        };
                        self.route_control_flow_block(problem, &layout)
                    }));
                    RoutedItemKind::ControlFlow((blocks.len() as u32).into())
                }
            };
            order.order.push(RoutedItem {
                initial_swaps: initial_swaps.take().map(Vec::into_boxed_slice),
                node: node_id,
                kind,
            });
            for edge in problem
                .sabre
                .dag
                .edges_directed(node_id, Direction::Outgoing)
            {
                let successor = edge.target();
                self.required_predecessors[successor] -= 1;
                if self.required_predecessors[successor] == 0 {
                    to_visit.push_back(successor);
                }
            }
        }
        assert!(
            initial_swaps.is_none(),
            "if initial swaps are given, at least one node must be known to be routable"
        );
    }

    /// Inner worker to route a control-flow block.  Since control-flow blocks are routed to
    /// restore the layout at the end of themselves, and the recursive calls spawn their own
    /// tracking states, this does not affect our own state.
    fn route_control_flow_block<'a>(
        &self,
        problem: RoutingProblem<'a>,
        layout: &NLayout,
    ) -> RoutingResult<'a> {
        let mut result = swap_map_trial(problem, layout, self.seed);
        // For now, we always append a swap circuit that gets the inner block back to the
        // parent's layout.
        result.order.final_swaps = token_swapper(
            &problem.target.neighbors,
            // Map physical location in the final layout from the inner routing to the current
            // location in the outer routing.
            result
                .final_layout
                .iter_physical()
                .map(|(p, v)| (p, v.to_phys(layout)))
                .collect(),
            Some(SWAP_EPILOGUE_TRIALS),
            Some(self.seed),
            None,
        )
        .unwrap()
        .into_iter()
        .map(|(l, r)| {
            [
                PhysicalQubit::new(l.index() as u32),
                PhysicalQubit::new(r.index() as u32),
            ]
        })
        .collect();
        result.final_layout = layout.clone();
        result
    }

    /// Fill the given lookahead layers with the next nodes that would be reachable after the front
    /// layer (and themselves).  This uses `required_predecessors` as scratch space for efficiency,
    /// but returns it to the same state as the input on return.
    fn populate_extended_set(&mut self, problem: RoutingProblem) {
        let mut next_visit = self.front_layer.iter_nodes().copied().collect::<Vec<_>>();
        let mut to_visit = Vec::new();
        let mut decremented: IndexMap<NodeIndex, u32, ahash::RandomState> =
            IndexMap::with_hasher(ahash::RandomState::default());
        for layer in self.lookahead_layers.iter_mut() {
            for node in next_visit.drain(..) {
                for edge in problem.sabre.dag.edges_directed(node, Direction::Outgoing) {
                    let successor = edge.target();
                    *decremented.entry(successor).or_insert(0) += 1;
                    self.required_predecessors[successor] -= 1;
                    if self.required_predecessors[successor] == 0 {
                        to_visit.push(successor);
                    }
                }
            }

            let mut i = 0;
            while i < to_visit.len() {
                let node = to_visit[i];
                if let InteractionKind::TwoQ([a, b]) = &problem.sabre.dag[node].kind {
                    layer.insert(node, [self.layout[*a], self.layout[*b]]);
                    next_visit.push(node);
                } else {
                    for edge in problem.sabre.dag.edges_directed(node, Direction::Outgoing) {
                        let successor = edge.target();
                        *decremented.entry(successor).or_insert(0) += 1;
                        self.required_predecessors[successor] -= 1;
                        if self.required_predecessors[successor] == 0 {
                            to_visit.push(successor);
                        }
                    }
                }
                i += 1;
            }
            to_visit.clear();
        }
        for (node, amount) in decremented.iter() {
            self.required_predecessors[*node] += *amount;
        }
    }

    /// Add swaps to the current set that greedily bring the nearest node together.  This is a
    /// "release valve" mechanism; it ignores all the Sabre heuristics and forces progress, so we
    /// can't get permanently stuck.
    fn force_enable_closest_node(
        &mut self,
        problem: RoutingProblem,
        current_swaps: &mut Vec<[PhysicalQubit; 2]>,
    ) -> SmallVec<[NodeIndex; 2]> {
        let (&closest_node, &qubits) = {
            let dist = &problem.target.distance;
            self.front_layer
                .iter()
                .min_by(|(_, qubits_a), (_, qubits_b)| {
                    dist[[qubits_a[0].index(), qubits_a[1].index()]]
                        .partial_cmp(&dist[[qubits_b[0].index(), qubits_b[1].index()]])
                        .unwrap_or(Ordering::Equal)
                })
                .expect("front layer is never empty, except when routing is complete")
        };
        let shortest_path = {
            let mut shortest_paths: DictMap<PhysicalQubit, Vec<PhysicalQubit>> = DictMap::new();
            (dijkstra(
                &problem.target.neighbors,
                qubits[0],
                Some(qubits[1]),
                |_| Ok(1.),
                Some(&mut shortest_paths),
            ) as Result<Vec<_>, Infallible>)
                .expect("error is infallible");
            shortest_paths
                .swap_remove(&qubits[1])
                .expect("target is required to be connected")
        };
        // Insert greedy swaps along that shortest path, splitting them between moving the left side
        // and moving the right side to minimise the depth.  One side needs to move up to the split
        // point and the other can stop one short because the gate will be routable then.
        let split: usize = shortest_path.len() / 2;
        current_swaps.reserve(shortest_path.len() - 2);
        for i in 0..split {
            current_swaps.push([shortest_path[i], shortest_path[i + 1]]);
        }
        for i in 0..split - 1 {
            let end = shortest_path.len() - 1 - i;
            current_swaps.push([shortest_path[end], shortest_path[end - 1]]);
        }
        current_swaps.iter().for_each(|&swap| self.apply_swap(swap));

        // If we apply a single swap it could be that we route 2 nodes; that is a setup like
        //  A - B - A - B
        // and we swap the middle two qubits. This cannot happen if we apply 2 or more swaps.
        match current_swaps.as_slice() {
            [swap] => swap
                .iter()
                .filter_map(|q| self.routable_node_on_qubit(problem, *q))
                .collect(),
            _ => smallvec![closest_node],
        }
    }

    /// Return the swap of two virtual qubits that produces the best score of all possible swaps.
    fn choose_best_swap(&mut self, problem: RoutingProblem) -> [PhysicalQubit; 2] {
        // Obtain all candidate swaps from the front layer.  A coupling-map edge is a candidate
        // swap if it involves at least one active qubit (i.e. it must affect the "basic"
        // heuristic), and if it involves two active qubits, we choose the `swap[0] < swap[1]` form
        // to make a canonical choice.
        self.swap_scores.clear();
        for &phys in self.front_layer.iter_active() {
            for &neighbor in problem.target.neighbors[phys].iter() {
                if neighbor > phys || !self.front_layer.is_active(neighbor) {
                    self.swap_scores.push(([phys, neighbor], 0.0));
                }
            }
        }

        let dist = &problem.target.distance.view();
        let mut absolute_score = 0.0;

        if let Some(BasicHeuristic { weight, scale }) = problem.heuristic.basic {
            let weight = match scale {
                SetScaling::Constant => weight,
                SetScaling::Size => {
                    if self.front_layer.is_empty() {
                        0.0
                    } else {
                        weight / (self.front_layer.len() as f64)
                    }
                }
            };
            absolute_score += weight * self.front_layer.total_score(dist);
            for (swap, score) in self.swap_scores.iter_mut() {
                *score += weight * self.front_layer.score(*swap, dist);
            }
        }

        if let Some(lookahead) = &problem.heuristic.lookahead {
            for (layer, weight) in self.lookahead_layers.iter().zip(lookahead.weights()) {
                let weight = match lookahead.scale {
                    SetScaling::Constant => *weight,
                    SetScaling::Size => {
                        if layer.is_empty() {
                            0.0
                        } else {
                            *weight / (layer.len() as f64)
                        }
                    }
                };
                absolute_score += weight * layer.total_score(dist);
                for (swap, score) in self.swap_scores.iter_mut() {
                    *score += weight * layer.score(*swap, dist);
                }
            }
        }

        if let Some(DecayHeuristic { .. }) = problem.heuristic.decay {
            for (swap, score) in self.swap_scores.iter_mut() {
                *score = (absolute_score + *score) * self.decay[swap[0]].max(self.decay[swap[1]]);
            }
        }

        let mut min_score = f64::INFINITY;
        let epsilon = problem.heuristic.best_epsilon;
        for &(swap, score) in self.swap_scores.iter() {
            if score - min_score < -epsilon {
                min_score = score;
                self.best_swaps.clear();
                self.best_swaps.push(swap);
            } else if (score - min_score).abs() <= epsilon {
                self.best_swaps.push(swap);
            }
        }
        *self.best_swaps.choose(&mut self.rng).unwrap()
    }
}

/// Run Sabre swap on a circuit
///
/// Returns:
///     A two-tuple of the newly routed :class:`.DAGCircuit`, and the layout that maps virtual
///     qubits to their assigned physical qubits at the *end* of the circuit execution.
#[pyfunction]
#[pyo3(signature=(dag, target, heuristic, initial_layout, num_trials, seed=None, run_in_parallel=None))]
pub fn sabre_routing(
    dag: &DAGCircuit,
    target: &PyRoutingTarget,
    heuristic: &Heuristic,
    initial_layout: &NLayout,
    num_trials: usize,
    seed: Option<u64>,
    run_in_parallel: Option<bool>,
) -> PyResult<(DAGCircuit, NLayout)> {
    let Some(target) = target.0.as_ref() else {
        // All-to-all coupling.
        return Ok((dag.clone(), initial_layout.clone()));
    };
    let sabre = SabreDAG::from_dag(dag)?;
    let result = swap_map(
        RoutingProblem {
            target,
            sabre: &sabre,
            dag,
            heuristic,
        },
        initial_layout,
        seed,
        num_trials,
        run_in_parallel,
    );
    result.rebuild().map(|dag| (dag, result.final_layout))
}

/// Run (potentially in parallel) several trials of the Sabre routing algorithm on the given
/// problem and return the one with fewest swaps.
pub fn swap_map<'a>(
    problem: RoutingProblem<'a>,
    initial_layout: &'_ NLayout,
    seed: Option<u64>,
    num_trials: usize,
    run_in_parallel: Option<bool>,
) -> RoutingResult<'a> {
    let seeds = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_os_rng(),
    }
    .sample_iter(&rand::distr::StandardUniform)
    .take(num_trials)
    .collect::<Vec<_>>();

    CondIterator::new(
        seeds,
        num_trials > 1
            && run_in_parallel.unwrap_or_else(|| getenv_use_multiple_threads() && num_trials > 1),
    )
    .map(|seed| swap_map_trial(problem, initial_layout, seed))
    .enumerate()
    .min_by_key(|(index, result)| (result.order.swap_count(), *index))
    .map(|(_, result)| result)
    .expect("must have at least one trial")
}

/// Run a single trial of the Sabre routing algorithm.
pub fn swap_map_trial<'a>(
    problem: RoutingProblem<'a>,
    initial_layout: &NLayout,
    seed: u64,
) -> RoutingResult<'a> {
    let mut order = Order::for_problem(problem);

    let num_qubits: u32 = problem.target.num_qubits().try_into().unwrap();
    let mut required_predecessors = VecMap::from(vec![0; problem.sabre.dag.node_count()]);
    for edge in problem.sabre.dag.edge_references() {
        required_predecessors[edge.target()] += 1;
    }
    let mut state = State {
        front_layer: Layer::new(num_qubits),
        lookahead_layers: vec![
            Layer::new(num_qubits);
            problem
                .heuristic
                .lookahead
                .as_ref()
                .map(|x| x.num_layers() as usize)
                .unwrap_or(0)
        ]
        .into(),
        decay: vec![1.; num_qubits as usize].into(),
        required_predecessors,
        layout: initial_layout.clone(),
        swap_scores: Vec::with_capacity(problem.target.neighbors.edge_count() / 2),
        best_swaps: Vec::new(),
        rng: Pcg64Mcg::seed_from_u64(seed),
        seed,
    };
    state.update_route(problem, &mut order, &problem.sabre.first_layer, None);
    state.populate_extended_set(problem);

    let mut routable_nodes = Vec::<NodeIndex>::with_capacity(2);
    let mut num_search_steps = 0;
    // The front layer only becomes empty when all nodes have been routed.  At each iteration of
    // this loop, we route either one or two gates.
    while !state.front_layer.is_empty() {
        let mut current_swaps: Vec<[PhysicalQubit; 2]> = Vec::new();
        // Swap-mapping loop.  This is the main part of the algorithm, which we repeat until we
        // either successfully route a node, or exceed the maximum number of attempts.
        while routable_nodes.is_empty() && current_swaps.len() <= problem.heuristic.attempt_limit {
            let best_swap = state.choose_best_swap(problem);
            state.apply_swap(best_swap);
            current_swaps.push(best_swap);
            // These two nodes can't be the same; if they are, then the gate is on both qubits of
            // the swap, and so it would have been routable before we applied the swap too, i.e.
            // during the last iteration of the loop.
            if let Some(node) = state.routable_node_on_qubit(problem, best_swap[1]) {
                routable_nodes.push(node);
            }
            if let Some(node) = state.routable_node_on_qubit(problem, best_swap[0]) {
                routable_nodes.push(node);
            }
            if let Some(DecayHeuristic { increment, reset }) = problem.heuristic.decay {
                num_search_steps += 1;
                if num_search_steps >= reset {
                    state.decay.fill(1.);
                    num_search_steps = 0;
                } else {
                    state.decay[best_swap[0]] += increment;
                    state.decay[best_swap[1]] += increment;
                }
            }
        }
        if routable_nodes.is_empty() {
            // If we exceeded the max number of heuristic-chosen swaps without making progress,
            // unwind to the last progress point and greedily swap to bring a node together.
            // Efficiency doesn't matter much; this path never gets taken unless we're unlucky.
            current_swaps
                .drain(..)
                .rev()
                .for_each(|swap| state.apply_swap(swap));
            let force_routed = state.force_enable_closest_node(problem, &mut current_swaps);
            routable_nodes.extend(force_routed);
        }

        for node in &routable_nodes {
            state.front_layer.remove(node);
        }
        state.update_route(problem, &mut order, &routable_nodes, Some(current_swaps));
        // Ideally we'd know how to mutate the extended set directly, but since its limited size
        // easy to do better than just emptying it and rebuilding.
        state.lookahead_layers.iter_mut().for_each(Layer::clear);
        state.populate_extended_set(problem);

        if problem.heuristic.decay.is_some() {
            state.decay.fill(1.);
        }
        routable_nodes.clear();
    }
    RoutingResult {
        problem,
        order,
        initial_layout: initial_layout.clone(),
        final_layout: state.layout,
    }
}
