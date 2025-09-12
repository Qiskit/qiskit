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

use hashbrown::HashMap;
use indexmap::{IndexMap, IndexSet};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{create_exception, wrap_pyfunction};
use rayon::prelude::*;
use rustworkx_core::petgraph::prelude::*;
use std::convert::Infallible;
use std::time::Instant;

use qiskit_circuit::converters::circuit_to_dag;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{Operation, OperationRef, Param};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::vf2;

use super::error_map::ErrorMap;
use crate::target::{Qargs, QargsRef, Target};
use qiskit_circuit::nlayout::NLayout;
use qiskit_circuit::{PhysicalQubit, VirtualQubit};

const PARALLEL_THRESHOLD: usize = 50;

#[pyclass]
pub struct EdgeList {
    pub edge_list: Vec<([VirtualQubit; 2], i32)>,
}

#[pymethods]
impl EdgeList {
    #[new]
    pub fn new(edge_list: Vec<([VirtualQubit; 2], i32)>) -> Self {
        EdgeList { edge_list }
    }
}

create_exception!(qiskit, MultiQEncountered, pyo3::exceptions::PyException);

/// Build an average error map for a given target.
///
/// Returns `None` if there is a global 2q operation to avoid attempting to construct a meaningless
/// all-to-all connectivity graph.
fn build_average_error_map(target: &Target) -> Option<ErrorMap> {
    if let Ok(mut globals) = target.operations_for_qargs(QargsRef::Global) {
        if globals.any(|op| op.operation.num_qubits() == 2) {
            return None;
        }
    }
    let mut error_map = ErrorMap::new(Some(target.num_qargs()));
    let mut target_without_errors = true;
    for qargs in target.qargs()? {
        let Qargs::Concrete(qargs) = qargs else {
            // TODO: limitations in the Rust-space `Target` don't let us handle errors on
            // globally-defined 1q operations well, and we already handled the case of a 2q global,
            // so we can skip from here.
            continue;
        };
        let mut qarg_error: f64 = 0.;
        let mut count: usize = 0;
        for op in target
            .operation_names_for_qargs(QargsRef::Concrete(qargs))
            .expect("these qargs came from `target.qargs()`")
        {
            count += 1;
            // If the `target` has no error recorded for an operation, we treat it as errorless.
            qarg_error += target
                .get_error(op, qargs)
                .inspect(|_| target_without_errors = false)
                .unwrap_or(0.);
        }
        if count > 0 {
            let out_qargs = if qargs.len() == 1 {
                [qargs[0], qargs[0]]
            } else {
                [qargs[0], qargs[1]]
            };
            error_map
                .error_map
                .insert(out_qargs, qarg_error / count as f64);
        }
    }
    if target_without_errors {
        // Legacy back-up condition, which penalises high-degree nodes.  This was originally part of
        // VF2Layout to better support old IBM devices, like the bowties, which were unreliable at
        // reporting their error rates, but typically had significantly worse performance on the
        // high-degree nodes.
        let num_qubits = target.num_qubits.unwrap() as usize;
        // Using an undirected graph because we don't want to double-count neighbors in directed
        // graphs.  Most directed graphs likely have no reversed edges or are totally symmetric, but
        // this avoids making any assumptions.
        let mut coupling =
            Graph::<(), (), Undirected>::with_capacity(num_qubits, error_map.error_map.len());
        for _ in 0..num_qubits {
            coupling.add_node(());
        }
        for qargs in target.qargs()? {
            let QargsRef::Concrete(&[left, right]) = qargs.as_ref() else {
                continue;
            };
            coupling.update_edge(
                NodeIndex::new(left.index()),
                NodeIndex::new(right.index()),
                (),
            );
        }
        for index in coupling.node_indices() {
            let qubit = PhysicalQubit::new(index.index() as u32);
            let degree = coupling.neighbors(index).count();
            error_map
                .error_map
                .entry([qubit, qubit])
                .insert(degree as f64 / num_qubits as f64);
        }
        // We use the `target` here rather than the graph edges because we want the directionality.
        for qargs in target.qargs()? {
            let QargsRef::Concrete(&[left, right]) = qargs.as_ref() else {
                continue;
            };
            let avg =
                0.5 * (error_map.error_map[&[left, left]] + error_map.error_map[&[right, right]]);
            error_map.error_map.entry([left, right]).insert(avg);
        }
    }
    Some(error_map)
}

/// A full set of a virtual interaction graph, and any loose 1q qubits.
#[derive(Default, Debug, Clone)]
struct VirtualInteractions<T> {
    /// The graph of actual interactions.  Nodes each correspond to 1q operations on a single
    /// virtual qubit (the mapping `nodes` stores _which_ qubits), and edges between virtual qubits
    /// correspond to 2q operations.  Edges are always directed, even for when `strict_direction` is
    /// unset; we handle the fuzzy directional matching bu setting the edge weights of the coupling
    /// graph appropriately.
    graph: Graph<T, T>,
    /// Map of node index to the qubit it represents.  We could store this on the nodes themselves,
    /// but then all the scorers would need different weight types between the nodes and the edges.
    nodes: IndexSet<VirtualQubit>,
    /// The qubits that have only single-qubit operations on them, mapped to the interaction summary
    /// associated with them.  We iterate through this at the end, so need a consistent order.
    uncoupled: IndexMap<VirtualQubit, T>,
    /// The qubits that have no operations on them at all.
    idle: IndexSet<VirtualQubit>,
}
impl<T: Default> VirtualInteractions<T> {
    fn from_dag<W>(dag: &DAGCircuit, weighter: W) -> PyResult<Self>
    where
        W: Fn(&mut T, &PackedInstruction, usize),
    {
        let id_qubit_map = (0..dag.num_qubits())
            .map(|q| VirtualQubit(q as u32))
            .collect::<Vec<_>>();
        let mut out = Self::default();
        out.add_interactions_from(dag, &id_qubit_map, 1, &weighter)?;
        out.idle.extend(
            (0..dag.num_qubits() as u32)
                .map(VirtualQubit)
                .filter(|q| !(out.nodes.contains(q) || out.uncoupled.contains_key(q))),
        );
        Ok(out)
    }

    fn add_interactions_from<W>(
        &mut self,
        dag: &DAGCircuit,
        wire_map: &[VirtualQubit],
        repeats: usize,
        weighter: &W,
    ) -> PyResult<()>
    where
        W: Fn(&mut T, &PackedInstruction, usize),
    {
        for (_, inst) in dag.op_nodes(false) {
            let qubits = dag.get_qargs(inst.qubits);
            if inst.op.control_flow() {
                Python::attach(|py| -> PyResult<()> {
                    let OperationRef::Instruction(py_inst) = inst.op.view() else {
                        unreachable!("control-flow nodes are always PyInstructions");
                    };
                    let repeats = if py_inst.name() == "for_loop" {
                        let Param::Obj(indexset) = &inst.params_view()[0] else {
                            return Err(PyTypeError::new_err(
                                "unexpected object as for-loop indexset parameter",
                            ));
                        };
                        repeats * indexset.bind(py).len()?
                    } else {
                        repeats
                    };
                    let wire_map: Vec<_> = qubits.iter().map(|i| wire_map[i.index()]).collect();
                    let blocks = py_inst.instruction.bind(py).getattr("blocks")?;
                    for block in blocks.downcast::<PyTuple>()?.iter() {
                        self.add_interactions_from(
                            &circuit_to_dag(block.extract()?, false, None, None)?,
                            &wire_map,
                            repeats,
                            weighter,
                        )?;
                    }
                    Ok(())
                })?;
                continue;
            }
            match qubits {
                [] => (),
                [q] => {
                    let q = wire_map[q.index()];
                    if let Some(index) = self.nodes.get_index_of(&q) {
                        let weight = self
                            .graph
                            .node_weight_mut(NodeIndex::new(index))
                            .expect("node must be in graph if tracked in 'nodes'");
                        weighter(weight, inst, repeats);
                    } else {
                        let weight = self.uncoupled.entry(q).or_default();
                        weighter(weight, inst, repeats);
                    }
                }
                [q0, q1] => {
                    let q0 = wire_map[q0.index()];
                    let q1 = wire_map[q1.index()];
                    let node0 = self.ensure_1q_in_graph(q0);
                    let node1 = self.ensure_1q_in_graph(q1);
                    if let Some(edge) = self.graph.find_edge(node0, node1) {
                        let weight = self
                            .graph
                            .edge_weight_mut(edge)
                            .expect("this index came from a call to 'find_edge'");
                        weighter(weight, inst, repeats);
                    } else {
                        let mut weight = T::default();
                        weighter(&mut weight, inst, repeats);
                        self.graph.add_edge(node0, node1, weight);
                    }
                }
                _ => return Err(MultiQEncountered::new_err("")),
            }
        }
        Ok(())
    }

    fn ensure_1q_in_graph(&mut self, q: VirtualQubit) -> NodeIndex {
        if let Some(index) = self.nodes.get_index_of(&q) {
            return NodeIndex::new(index);
        }
        assert!(self.nodes.insert(q));
        self.graph
            .add_node(self.uncoupled.swap_remove(&q).unwrap_or_default())
    }
}

fn build_coupling_map(target: &Target, errors: &ErrorMap) -> Option<Graph<f64, f64>> {
    let neg_log_fidelity = |error: f64| {
        if error.is_nan() || error <= 0. {
            0.0
        } else if error >= 1. {
            f64::INFINITY
        } else {
            -((-error).ln_1p())
        }
    };
    let num_qubits = target.num_qubits.unwrap_or_default() as usize;
    if target.num_qargs() == 0 {
        return None;
    }
    let mut cm_graph =
        Graph::with_capacity(num_qubits, target.num_qargs().saturating_sub(num_qubits));
    for qubit in 0..num_qubits as u32 {
        let qubit = PhysicalQubit::new(qubit);
        cm_graph.add_node(neg_log_fidelity(
            *errors.error_map.get(&[qubit, qubit]).unwrap_or(&0.0),
        ));
    }
    for qargs in target.qargs()? {
        let QargsRef::Concrete(&[left, right]) = qargs.as_ref() else {
            // We just ignore globals; we're assuming that somewhere else has eagerly terminated in
            // the case of global 2q operations.  We also don't care about non-2q operations in this
            // loop (since we already did the single-qubit bits).
            continue;
        };
        let error = errors.error_map.get(&[left, right]).unwrap_or(&0.0);
        cm_graph.add_edge(
            NodeIndex::new(left.index()),
            NodeIndex::new(right.index()),
            neg_log_fidelity(*error),
        );
    }
    Some(cm_graph)
}

/// If an edge does not have a parallel but reversed counterpart, add one with the same weight.
fn loosen_directionality<S, T: Clone>(graph: &mut Graph<S, T>) {
    graph
        .edge_references()
        .filter(|edge| graph.find_edge(edge.target(), edge.source()).is_none())
        .map(|edge| (edge.target(), edge.source(), edge.weight().clone()))
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|(source, target, weight)| {
            graph.add_edge(source, target, weight);
        })
}

// This function assumes that there's a way of "sorting" the 1q interactions relative to each other,
// so it only really makes sense for the VF2 averaging, not VF2Post exact matching.
/// Assign the free isolated qubits to the physical qubits with the best error rates.
fn map_free_qubits(
    num_physical_qubits: usize,
    interactions: VirtualInteractions<usize>,
    mut partial_layout: HashMap<VirtualQubit, PhysicalQubit>,
    avg_error_map: &ErrorMap,
) -> Option<HashMap<VirtualQubit, PhysicalQubit>> {
    if num_physical_qubits
        < partial_layout.len() + interactions.uncoupled.len() + interactions.idle.len()
    {
        return None;
    }

    let normalize = |err: Option<&f64>| -> f64 {
        let err = err.copied().unwrap_or(f64::INFINITY);
        if err.is_nan() {
            0.0
        } else {
            err
        }
    };

    let mut free_physical = (0..num_physical_qubits)
        .map(|qubit| PhysicalQubit::new(qubit as u32))
        .collect::<IndexSet<_>>();
    partial_layout.values().for_each(|phys| {
        free_physical.swap_remove(phys);
    });
    free_physical.par_sort_by(|a, b| {
        let score_a = normalize(avg_error_map.error_map.get(&[*a, *a]));
        let score_b = normalize(avg_error_map.error_map.get(&[*b, *b]));
        score_a.partial_cmp(&score_b).expect("NaNs treated as zero")
    });

    let mut uncoupled_virtual = interactions.uncoupled.into_iter().collect::<Vec<_>>();
    uncoupled_virtual.par_sort_by_key(|(_, interactions)| *interactions);
    partial_layout.extend(
        uncoupled_virtual
            .into_iter()
            .rev() // We want the most used virtuals to get first pick.
            .map(|(virt, _)| virt)
            .chain(interactions.idle)
            .zip(free_physical),
    );
    Some(partial_layout)
}

#[pyfunction]
#[pyo3(signature = (dag, target, strict_direction=false, call_limit=None, time_limit=None, max_trials=None, avg_error_map=None))]
pub fn vf2_layout_pass(
    dag: &DAGCircuit,
    target: &Target,
    strict_direction: bool,
    call_limit: Option<usize>,
    time_limit: Option<f64>,
    max_trials: Option<isize>,
    avg_error_map: Option<ErrorMap>,
) -> PyResult<Option<HashMap<VirtualQubit, PhysicalQubit>>> {
    let add_interaction = |count: &mut usize, _: &PackedInstruction, repeats: usize| {
        *count += repeats;
    };
    let score = |count: &usize, err: &f64| -> Result<Option<f64>, Infallible> {
        Ok(Some(*err * (*count as f64)))
    };
    let Some(avg_error_map) = avg_error_map.or_else(|| build_average_error_map(target)) else {
        return Ok(None);
    };
    let Some(mut coupling_graph) = build_coupling_map(target, &avg_error_map) else {
        return Ok(None);
    };
    if !strict_direction {
        loosen_directionality(&mut coupling_graph);
    }
    let interactions = VirtualInteractions::from_dag(dag, add_interaction)?;
    let start_time = Instant::now();
    let mut times_up = false;
    let mut trials: usize = 0;
    let max_trials: usize = match max_trials {
        Some(max_trials) => max_trials.try_into().unwrap_or(0),
        None => {
            15 + interactions
                .graph
                .edge_count()
                .max(coupling_graph.edge_count())
        }
    };
    let time_limit = time_limit.unwrap_or(f64::INFINITY);
    let Some((mapping, _score)) = vf2::Vf2Algorithm::new(
        &interactions.graph,
        &coupling_graph,
        (score, score),
        false,
        vf2::Problem::Subgraph,
        call_limit,
    )
    .with_score_limit(f64::MAX)
    .take_while(|_| {
        if times_up {
            return false;
        }
        times_up = start_time.elapsed().as_secs_f64() >= time_limit;
        trials += 1;
        max_trials == 0 || trials <= max_trials
    })
    .map(|result| result.expect("error type is infallible"))
    // The iterator actually always returns in reverse-sorted order of its scores, but we take the
    // _first_ valid mapping if there are several with the same score for historical reasons.
    .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("score should never be NaN")) else {
        return Ok(None);
    };

    // Remap node indices back to virtual/physical qubits.
    let mapping = mapping
        .iter()
        .map(|(k, v)| {
            (
                interactions.nodes[k.index()],
                PhysicalQubit::new(v.index() as u32),
            )
        })
        .collect();
    Ok(map_free_qubits(
        coupling_graph.node_count(),
        interactions,
        mapping,
        &avg_error_map,
    ))
}

/// Score a given circuit with a layout applied
#[pyfunction]
#[pyo3(
    text_signature = "(bit_list, edge_list, error_matrix, layout, strict_direction, run_in_parallel, /)"
)]
pub fn score_layout(
    bit_list: PyReadonlyArray1<i32>,
    edge_list: &EdgeList,
    error_map: &ErrorMap,
    layout: &NLayout,
    strict_direction: bool,
    run_in_parallel: bool,
) -> PyResult<f64> {
    let bit_counts = bit_list.as_slice()?;
    let edge_filter_map = |(index_arr, gate_count): &([VirtualQubit; 2], i32)| -> Option<f64> {
        let mut error = error_map
            .error_map
            .get(&[index_arr[0].to_phys(layout), index_arr[1].to_phys(layout)]);
        if !strict_direction && error.is_none() {
            error = error_map
                .error_map
                .get(&[index_arr[1].to_phys(layout), index_arr[0].to_phys(layout)]);
        }
        error.map(|error| {
            if !error.is_nan() {
                (1. - error).powi(*gate_count)
            } else {
                1.
            }
        })
    };
    let bit_filter_map = |(v_bit_index, gate_counts): (usize, &i32)| -> Option<f64> {
        let p_bit = VirtualQubit::new(v_bit_index.try_into().unwrap()).to_phys(layout);
        let error = error_map.error_map.get(&[p_bit, p_bit]);

        error.map(|error| {
            if !error.is_nan() {
                (1. - error).powi(*gate_counts)
            } else {
                1.
            }
        })
    };

    let mut fidelity: f64 = if edge_list.edge_list.len() < PARALLEL_THRESHOLD || !run_in_parallel {
        edge_list
            .edge_list
            .iter()
            .filter_map(edge_filter_map)
            .product()
    } else {
        edge_list
            .edge_list
            .par_iter()
            .filter_map(edge_filter_map)
            .product()
    };
    fidelity *= if bit_list.len()? < PARALLEL_THRESHOLD || !run_in_parallel {
        bit_counts
            .iter()
            .enumerate()
            .filter_map(bit_filter_map)
            .product::<f64>()
    } else {
        bit_counts
            .par_iter()
            .enumerate()
            .filter_map(bit_filter_map)
            .product()
    };
    Ok(1. - fidelity)
}

pub fn vf2_layout_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(score_layout))?;
    m.add_wrapped(wrap_pyfunction!(vf2_layout_pass))?;
    m.add("MultiQEncountered", m.py().get_type::<MultiQEncountered>())?;
    m.add_class::<EdgeList>()?;
    Ok(())
}
