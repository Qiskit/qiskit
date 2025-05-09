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

use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{create_exception, wrap_pyfunction};
use rayon::prelude::*;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences, NodeRef};
use rustworkx_core::petgraph::EdgeType;
use std::cmp::Ordering;
use std::time::Instant;

use qiskit_circuit::converters::circuit_to_dag;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{Operation, OperationRef, Param};
use qiskit_circuit::rustworkx_core_vnext::isomorphism::vf2;
use qiskit_circuit::Qubit;

use super::error_map::ErrorMap;
use crate::target::{Qargs, Target};
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

fn build_average_error_map(target: &Target) -> ErrorMap {
    let qargs_count = target.qargs().unwrap().count();
    let mut error_map = ErrorMap::new(Some(qargs_count));
    for qargs in target.qargs().unwrap() {
        let mut qarg_error: f64 = 0.;
        let mut count: usize = 0;
        for op in target.operation_names_for_qargs(qargs).unwrap() {
            if let Some(error) = target.get_error(op, qargs) {
                count += 1;
                qarg_error += error;
            }
        }
        let Qargs::Concrete(qargs) = qargs else {
            continue;
        };
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
    error_map
}

struct InteractionGraphData<Ty: EdgeType> {
    im_graph: StableGraph<HashMap<String, usize>, HashMap<String, usize>, Ty>,
    reverse_im_graph_node_map: Vec<Option<Qubit>>,
    free_nodes: HashMap<NodeIndex, HashMap<String, usize>>,
}

fn generate_directed_interaction(dag: &DAGCircuit) -> PyResult<InteractionGraphData<Directed>> {
    let mut im_graph_node_map: Vec<Option<NodeIndex>> = vec![None; dag.num_qubits()];
    let mut reverse_im_graph_node_map: Vec<Option<Qubit>> = vec![None; dag.num_qubits()];
    let wire_map: Vec<Qubit> = (0..dag.num_qubits()).map(Qubit::new).collect();
    let weight = 1;
    let mut im_graph = StableDiGraph::with_capacity(dag.num_qubits(), dag.num_qubits());
    build_interaction_graph(
        dag,
        &wire_map,
        weight,
        &mut im_graph,
        &mut im_graph_node_map,
        &mut reverse_im_graph_node_map,
    )?;
    Ok(InteractionGraphData {
        im_graph,
        reverse_im_graph_node_map,
        free_nodes: HashMap::new(),
    })
}

fn generate_undirected_interaction(dag: &DAGCircuit) -> PyResult<InteractionGraphData<Undirected>> {
    let mut im_graph_node_map: Vec<Option<NodeIndex>> = vec![None; dag.num_qubits()];
    let mut reverse_im_graph_node_map: Vec<Option<Qubit>> = vec![None; dag.num_qubits()];
    let wire_map: Vec<Qubit> = (0..dag.num_qubits()).map(Qubit::new).collect();
    let weight = 1;
    let mut im_graph = StableUnGraph::with_capacity(dag.num_qubits(), dag.num_qubits());
    build_interaction_graph(
        dag,
        &wire_map,
        weight,
        &mut im_graph,
        &mut im_graph_node_map,
        &mut reverse_im_graph_node_map,
    )?;
    let mut free_nodes: HashMap<NodeIndex, HashMap<String, usize>> = HashMap::new();
    let indices = im_graph.node_indices().collect::<Vec<_>>();
    for index in indices {
        if im_graph.edges(index).next().is_none() {
            free_nodes.insert(index, im_graph.remove_node(index).unwrap());
        }
    }
    Ok(InteractionGraphData {
        im_graph,
        reverse_im_graph_node_map,
        free_nodes,
    })
}

fn build_interaction_graph<Ty: EdgeType>(
    dag: &DAGCircuit,
    wire_map: &[Qubit],
    weight: usize,
    im_graph: &mut StableGraph<HashMap<String, usize>, HashMap<String, usize>, Ty>,
    im_graph_node_map: &mut [Option<NodeIndex>],
    reverse_im_graph_node_map: &mut [Option<Qubit>],
) -> PyResult<()> {
    for (_index, inst) in dag.op_nodes(false) {
        if inst.op.control_flow() {
            Python::with_gil(|py| -> PyResult<_> {
                let inner_weight = if inst.op.name() == "for_loop" {
                    let Param::Obj(ref indexset) = inst.params_view()[0] else {
                        unreachable!("Invalid for loop definition");
                    };
                    indexset.bind(py).len().unwrap()
                } else {
                    weight
                };
                let OperationRef::Instruction(py_inst) = inst.op.view() else {
                    unreachable!("Control flow must be a python instruction");
                };
                let raw_blocks = py_inst.instruction.getattr(py, "blocks").unwrap();
                let blocks: &Bound<PyTuple> = raw_blocks.downcast_bound::<PyTuple>(py).unwrap();
                for block in blocks.iter() {
                    let mut inner_wire_map = vec![Qubit(u32::MAX); wire_map.len()];
                    let node_qargs = dag.get_qargs(inst.qubits);

                    for (outer, inner) in node_qargs.iter().zip(0..inst.op.num_qubits()) {
                        inner_wire_map[inner as usize] = wire_map[outer.index()]
                    }
                    let block_dag = circuit_to_dag(py, block.extract()?, false, None, None)?;
                    build_interaction_graph(
                        &block_dag,
                        &inner_wire_map,
                        inner_weight,
                        im_graph,
                        im_graph_node_map,
                        reverse_im_graph_node_map,
                    )?;
                }
                Ok(())
            })?;
            continue;
        }
        let len_args = inst.op.num_qubits();
        if len_args == 1 {
            let dag_qubits = dag.get_qargs(inst.qubits);
            let qargs = wire_map[dag_qubits[0].index()];
            match im_graph_node_map[qargs.index()] {
                None => {
                    let mut weights = HashMap::with_capacity(1);
                    weights.insert(inst.op.name().into(), weight);
                    let new_index = im_graph.add_node(weights);
                    im_graph_node_map[qargs.index()] = Some(new_index);
                    reverse_im_graph_node_map[new_index.index()] = Some(qargs);
                }
                Some(node_index) => {
                    let weights: &mut HashMap<String, usize> =
                        im_graph.node_weight_mut(node_index).unwrap();
                    weights
                        .entry(inst.op.name().into())
                        .and_modify(|gate_weight| *gate_weight += weight)
                        .or_insert(weight);
                }
            }
        } else if len_args == 2 {
            let dag_qubits = dag.get_qargs(inst.qubits);
            let qargs: [Qubit; 2] = [
                wire_map[dag_qubits[0].index()],
                wire_map[dag_qubits[1].index()],
            ];
            if im_graph_node_map[qargs[0].index()].is_none() {
                let weights = HashMap::new();
                let new_index = im_graph.add_node(weights);
                im_graph_node_map[qargs[0].index()] = Some(new_index);
                reverse_im_graph_node_map[new_index.index()] = Some(qargs[0]);
            }
            if im_graph_node_map[qargs[1].index()].is_none() {
                let weights = HashMap::new();
                let new_index = im_graph.add_node(weights);
                im_graph_node_map[qargs[1].index()] = Some(new_index);
                reverse_im_graph_node_map[new_index.index()] = Some(qargs[1]);
            }

            let edge_index = im_graph.find_edge(
                im_graph_node_map[qargs[0].index()].unwrap(),
                im_graph_node_map[qargs[1].index()].unwrap(),
            );
            match edge_index {
                Some(edge_index) => {
                    let weights: &mut HashMap<String, usize> =
                        im_graph.edge_weight_mut(edge_index).unwrap();
                    weights
                        .entry(inst.op.name().into())
                        .and_modify(|gate_weight| *gate_weight += weight)
                        .or_insert(weight);
                }
                None => {
                    let mut weights = HashMap::with_capacity(1);
                    weights.insert(inst.op.name().into(), weight);
                    im_graph.add_edge(
                        im_graph_node_map[qargs[0].index()].unwrap(),
                        im_graph_node_map[qargs[1].index()].unwrap(),
                        weights,
                    );
                }
            }
        }
        if len_args > 2 {
            return Err(MultiQEncountered::new_err(""));
        }
    }
    Ok(())
}

fn build_coupling_map<Ty: EdgeType>(
    target: &Target,
) -> Option<StableGraph<HashSet<String>, HashSet<String>, Ty>> {
    let num_qubits = target.num_qubits.unwrap_or_default();
    if target.num_qargs() == 0 {
        return None;
    }
    let mut cm_graph =
        StableGraph::with_capacity(num_qubits, target.num_qargs().saturating_sub(num_qubits));
    for _ in 0..num_qubits {
        cm_graph.add_node(HashSet::new());
    }
    for qarg in target.qargs()? {
        let Qargs::Concrete(qarg) = qarg else {
            continue;
        };
        if qarg.len() == 1 {
            let node_index = NodeIndex::new(qarg[0].index());
            let op_names = target.operation_names_for_qargs(qarg).unwrap();
            for name in op_names {
                cm_graph
                    .node_weight_mut(node_index)
                    .unwrap()
                    .insert(name.into());
            }
        } else if qarg.len() == 2 {
            let edge_index = cm_graph.find_edge(
                NodeIndex::new(qarg[0].index()),
                NodeIndex::new(qarg[1].index()),
            );
            let op_names = target.operation_names_for_qargs(qarg).unwrap();
            match edge_index {
                Some(edge_index) => {
                    let edge_weight: &mut HashSet<String> =
                        cm_graph.edge_weight_mut(edge_index).unwrap();
                    for name in op_names {
                        edge_weight.insert(name.into());
                    }
                }
                None => {
                    cm_graph.add_edge(
                        NodeIndex::new(qarg[0].index()),
                        NodeIndex::new(qarg[1].index()),
                        op_names.into_iter().map(|x| x.into()).collect(),
                    );
                }
            }
        }
    }
    Some(cm_graph)
}

fn mapping_to_layout<Ty: EdgeType>(
    dag: &DAGCircuit,
    mapping: IndexMap<usize, usize, ahash::RandomState>,
    data: &InteractionGraphData<Ty>,
) -> HashMap<VirtualQubit, PhysicalQubit> {
    let mut out_layout: HashMap<VirtualQubit, PhysicalQubit> =
        HashMap::with_capacity(dag.num_qubits());

    for (k, v) in mapping.iter() {
        out_layout.insert(
            VirtualQubit::new(data.reverse_im_graph_node_map[*v].unwrap().0),
            PhysicalQubit::new(*k as u32),
        );
    }
    out_layout
}

fn map_free_qubits(
    free_nodes: HashMap<NodeIndex, HashMap<String, usize>>,
    mut partial_layout: HashMap<VirtualQubit, PhysicalQubit>,
    reverse_im_graph_node_map: &[Option<Qubit>],
    avg_error_map: &ErrorMap,
    target: &Target,
) -> Option<HashMap<VirtualQubit, PhysicalQubit>> {
    if free_nodes.is_empty() {
        return Some(partial_layout);
    }
    let num_physical_qubits = target.num_qubits.unwrap() as u32;
    let mut free_qubits_set: HashSet<u32> = (0..num_physical_qubits).collect();
    for phys in partial_layout.values() {
        let qubit = phys.index() as u32;
        free_qubits_set.remove(&qubit);
    }
    let mut free_qubits: Vec<u32> = free_qubits_set.into_iter().collect();
    free_qubits.par_sort_by(|qubit_a, qubit_b| {
        let score_a = *avg_error_map
            .error_map
            .get(&[PhysicalQubit::new(*qubit_a), PhysicalQubit::new(*qubit_a)])
            .unwrap_or(&0.);
        let score_b = *avg_error_map
            .error_map
            .get(&[PhysicalQubit::new(*qubit_b), PhysicalQubit::new(*qubit_b)])
            .unwrap_or(&0.);
        // Reverse comparison so lower error rates are at the end of the vec.
        score_b.partial_cmp(&score_a).unwrap()
    });
    let mut free_indices: Vec<NodeIndex> = free_nodes.keys().copied().collect();
    free_indices.par_sort_by_key(|index| free_nodes[index].values().sum::<usize>());
    for im_index in free_indices {
        let selected_qubit = free_qubits.pop()?;
        partial_layout.insert(
            VirtualQubit(reverse_im_graph_node_map[im_index.index()].unwrap().0),
            PhysicalQubit::new(selected_qubit),
        );
    }
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
    max_trials: Option<usize>,
    avg_error_map: Option<ErrorMap>,
) -> PyResult<Option<HashMap<VirtualQubit, PhysicalQubit>>> {
    if strict_direction {
        let cm_graph: Option<StableDiGraph<_, _>> = build_coupling_map(target);
        if cm_graph.is_none() {
            return Ok(None);
        }
        let cm_graph = cm_graph.unwrap();
        let im_graph_data = generate_directed_interaction(dag)?;
        let mappings = vf2::Vf2Algorithm::new(
            &cm_graph,
            &im_graph_data.im_graph,
            vf2::NoSemanticMatch,
            vf2::NoSemanticMatch,
            false,
            Ordering::Greater,
            false,
            call_limit,
        );
        let mut trials: usize = 0;
        let start_time = Instant::now();
        let mut chosen_layout: Option<HashMap<VirtualQubit, PhysicalQubit>> = None;
        let mut chosen_layout_score = f64::MAX;
        let avg_error_map = avg_error_map.unwrap_or_else(|| build_average_error_map(target));
        for mapping in mappings {
            trials += 1;
            let mapping = mapping_to_layout(dag, mapping.unwrap(), &im_graph_data);
            if cm_graph.node_count() == im_graph_data.im_graph.node_count() {
                return Ok(Some(mapping));
            }
            let layout_score =
                score_layout_internal(&mapping, &avg_error_map, &im_graph_data, strict_direction)?;
            if layout_score == 0. {
                return Ok(Some(mapping));
            }
            if layout_score < chosen_layout_score {
                chosen_layout = Some(mapping);
                chosen_layout_score = layout_score;
            }

            if let Some(max_trials) = max_trials {
                if max_trials > 0 && trials >= max_trials {
                    break;
                }
            }
            if let Some(time_limit) = time_limit {
                let elapsed_time = start_time.elapsed().as_secs_f64();
                if elapsed_time >= time_limit {
                    break;
                }
            }
        }
        Ok(chosen_layout)
    } else {
        let cm_graph: Option<StableUnGraph<_, _>> = build_coupling_map(target);
        if cm_graph.is_none() {
            return Ok(None);
        }
        let cm_graph = cm_graph.unwrap();
        let im_graph_data = generate_undirected_interaction(dag)?;
        let avg_error_map = avg_error_map.unwrap_or_else(|| build_average_error_map(target));
        // If there are no virtual qubits in the interaction graph and we have free nodes
        // (virtual qubits with 1q operations but no 2q interactions) then we can skip vf2 and run
        // the free qubit mapping directly.
        if im_graph_data.im_graph.node_count() == 0 && !im_graph_data.free_nodes.is_empty() {
            return Ok(map_free_qubits(
                im_graph_data.free_nodes,
                HashMap::new(),
                &im_graph_data.reverse_im_graph_node_map,
                &avg_error_map,
                target,
            ));
        }
        let mappings = vf2::Vf2Algorithm::new(
            &cm_graph,
            &im_graph_data.im_graph,
            vf2::NoSemanticMatch,
            vf2::NoSemanticMatch,
            false,
            Ordering::Greater,
            false,
            call_limit,
        );
        let mut trials: usize = 0;
        let start_time = Instant::now();
        let mut chosen_layout: Option<HashMap<VirtualQubit, PhysicalQubit>> = None;
        let mut chosen_layout_score = f64::MAX;
        for mapping in mappings {
            trials += 1;
            let mapping = mapping_to_layout(dag, mapping.unwrap(), &im_graph_data);
            if cm_graph.node_count() == im_graph_data.im_graph.node_count() {
                return Ok(Some(mapping));
            }
            let layout_score =
                score_layout_internal(&mapping, &avg_error_map, &im_graph_data, strict_direction)?;
            if layout_score == 0. {
                return Ok(Some(mapping));
            }
            if layout_score < chosen_layout_score {
                chosen_layout = Some(mapping);
                chosen_layout_score = layout_score;
            }

            if let Some(max_trials) = max_trials {
                if max_trials > 0 && trials >= max_trials {
                    break;
                }
            }
            if let Some(time_limit) = time_limit {
                let elapsed_time = start_time.elapsed().as_secs_f64();
                if elapsed_time >= time_limit {
                    break;
                }
            }
        }
        if chosen_layout.is_none() {
            return Ok(None);
        }
        let chosen_layout = chosen_layout.unwrap();
        Ok(map_free_qubits(
            im_graph_data.free_nodes,
            chosen_layout,
            &im_graph_data.reverse_im_graph_node_map,
            &avg_error_map,
            target,
        ))
    }
}

fn score_layout_internal<Ty: EdgeType>(
    mapping: &HashMap<VirtualQubit, PhysicalQubit>,
    error_map: &ErrorMap,
    im_graph_data: &InteractionGraphData<Ty>,
    strict_direction: bool,
) -> PyResult<f64> {
    let edge_filter_map = |a: NodeIndex, b: NodeIndex, gate_count: usize| -> Option<f64> {
        let qubit_a = VirtualQubit(
            im_graph_data.reverse_im_graph_node_map[a.index()]
                .unwrap()
                .0,
        );
        let qubit_b = VirtualQubit(
            im_graph_data.reverse_im_graph_node_map[b.index()]
                .unwrap()
                .0,
        );

        let qargs = [mapping[&qubit_a], mapping[&qubit_b]];
        let mut error = error_map.error_map.get(&qargs);
        if !strict_direction && error.is_none() {
            error = error_map.error_map.get(&[qargs[1], qargs[0]]);
        }
        error.map(|error| {
            if !error.is_nan() {
                (1. - error).powi(gate_count as i32)
            } else {
                1.
            }
        })
    };

    let bit_filter_map = |v_bit_index: NodeIndex, gate_counts: usize| -> Option<f64> {
        let v_bit = VirtualQubit(
            im_graph_data.reverse_im_graph_node_map[v_bit_index.index()]
                .unwrap()
                .0,
        );
        let p_bit = mapping[&v_bit];
        let error = error_map.error_map.get(&[p_bit, p_bit]);

        error.map(|error| {
            if !error.is_nan() {
                (1. - error).powi(gate_counts as i32)
            } else {
                1.
            }
        })
    };

    let mut fidelity: f64 = im_graph_data
        .im_graph
        .edge_references()
        .filter_map(|edge| {
            edge_filter_map(edge.source(), edge.target(), edge.weight().values().sum())
        })
        .product();
    fidelity *= im_graph_data
        .im_graph
        .node_references()
        .filter_map(|node| bit_filter_map(node.id(), node.weight().values().sum()))
        .product::<f64>();
    Ok(1. - fidelity)
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
