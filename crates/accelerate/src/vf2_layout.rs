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
use pyo3::{create_exception, wrap_pyfunction};
use rayon::prelude::*;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::EdgeType;
use smallvec::smallvec;
use std::cmp::Ordering;
use std::time::Instant;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{Operation, Param};
use qiskit_circuit::rustworkx_core_vnext::isomorphism::vf2;
use qiskit_circuit::Qubit;

use crate::error_map::ErrorMap;
use crate::nlayout::{NLayout, PhysicalQubit, VirtualQubit};
use crate::target_transpiler::Target;

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

struct InteractionGraphData<Ty: EdgeType> {
    im_graph: StableGraph<HashMap<String, usize>, HashMap<String, usize>, Ty>,
    reverse_im_graph_node_map: Vec<Option<Qubit>>,
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
    Ok(InteractionGraphData {
        im_graph,
        reverse_im_graph_node_map,
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
                for block in inst.op.blocks() {
                    let mut inner_wire_map = vec![Qubit(u32::MAX); wire_map.len()];
                    let node_qargs = dag.get_qargs(inst.qubits);

                    for (outer, inner) in node_qargs.iter().zip(0..block.num_qubits()) {
                        inner_wire_map[inner] = wire_map[outer.index()]
                    }
                    let block_dag = DAGCircuit::from_circuit_data(py, block, false).unwrap();
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
    let mut cm_graph = StableGraph::with_capacity(num_qubits, target.num_qargs() - num_qubits);
    for _ in 0..num_qubits {
        cm_graph.add_node(HashSet::new());
    }
    let qargs = target.qargs();
    qargs.as_ref()?;
    for qarg in qargs.unwrap().flatten() {
        if qarg.len() == 1 {
            let node_index = NodeIndex::new(qarg[0].index());
            let op_names = target.operation_names_for_qargs(Some(qarg)).unwrap();
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
            let op_names = target.operation_names_for_qargs(Some(qarg)).unwrap();
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
) -> NLayout {
    let mut out_layout: Vec<PhysicalQubit> = vec![PhysicalQubit(u32::MAX); dag.num_qubits()];

    for (k, v) in mapping.iter() {
        out_layout[data.reverse_im_graph_node_map[*v].unwrap().index()] =
            PhysicalQubit::new(*k as u32);
    }
    NLayout::from_virtual_to_physical(out_layout).unwrap()
}

#[pyfunction]
#[pyo3(signature = (dag, target, strict_direction=false, call_limit=None, time_limit=None, max_trials=None))]
pub fn vf2_layout_pass(
    dag: &DAGCircuit,
    target: &Target,
    strict_direction: bool,
    call_limit: Option<usize>,
    time_limit: Option<f64>,
    max_trials: Option<usize>,
) -> PyResult<Option<NLayout>> {
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
        let mut chosen_layout: Option<IndexMap<usize, usize, ahash::RandomState>> = None;
        let mut chosen_layout_score = f64::MAX;
        for mapping in mappings {
            trials += 1;
            let mapping = mapping.unwrap();
            if cm_graph.node_count() == im_graph_data.im_graph.node_count() {
                return Ok(Some(mapping_to_layout(dag, mapping, &im_graph_data)));
            }
            let layout_score =
                score_layout_target(&mapping, target, &im_graph_data, strict_direction)?;
            if layout_score == 0. {
                return Ok(Some(mapping_to_layout(dag, mapping, &im_graph_data)));
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
        Ok(Some(mapping_to_layout(
            dag,
            chosen_layout.unwrap(),
            &im_graph_data,
        )))
    } else {
        let cm_graph: Option<StableUnGraph<_, _>> = build_coupling_map(target);
        if cm_graph.is_none() {
            return Ok(None);
        }
        let cm_graph = cm_graph.unwrap();
        let im_graph_data = generate_undirected_interaction(dag)?;
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
        let mut chosen_layout: Option<IndexMap<usize, usize, ahash::RandomState>> = None;
        let mut chosen_layout_score = f64::MAX;
        for mapping in mappings {
            trials += 1;
            let mapping = mapping.unwrap();
            if cm_graph.node_count() == im_graph_data.im_graph.node_count() {
                return Ok(Some(mapping_to_layout(dag, mapping, &im_graph_data)));
            }
            let layout_score =
                score_layout_target(&mapping, target, &im_graph_data, strict_direction)?;
            if layout_score == 0. {
                return Ok(Some(mapping_to_layout(dag, mapping, &im_graph_data)));
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
        Ok(Some(mapping_to_layout(
            dag,
            chosen_layout.unwrap(),
            &im_graph_data,
        )))
    }
}

fn score_layout_target<Ty: EdgeType>(
    mapping: &IndexMap<usize, usize, ahash::RandomState>,
    target: &Target,
    im_graph_data: &InteractionGraphData<Ty>,
    strict_direction: bool,
) -> PyResult<f64> {
    let edge_filter_map = |(a, b): (NodeIndex, NodeIndex)| -> Option<f64> {
        let qargs = smallvec![
            PhysicalQubit::new(mapping[a.index()] as u32),
            PhysicalQubit::new(mapping[b.index()] as u32)
        ];

        let ops = target.operation_names_for_qargs(Some(&qargs));
        let ops = match ops {
            Ok(ops) => ops,
            Err(_) => {
                if !strict_direction {
                    let qargs = smallvec![qargs[1], qargs[0]];
                    match target.operation_names_for_qargs(Some(&qargs)) {
                        Ok(ops) => ops,
                        Err(_) => {
                            return None;
                        }
                    }
                } else {
                    return None;
                }
            }
        };
        let error: f64 = ops
            .iter()
            .map(|name| target.get_error(name, &qargs).unwrap_or_default())
            .sum::<f64>()
            / ops.len() as f64;
        Some(if !error.is_nan() { 1. - error } else { 1. })
    };

    let bit_filter_map = |v_bit_index: NodeIndex| -> Option<f64> {
        let p_bit = smallvec![PhysicalQubit::new(mapping[v_bit_index.index()] as u32)];
        let ops = target.operation_names_for_qargs(Some(&p_bit));
        if ops.is_err() {
            return None;
        }
        let ops = ops.unwrap();
        let error: f64 = ops
            .iter()
            .map(|name| target.get_error(name, &p_bit).unwrap_or_default())
            .sum::<f64>()
            / ops.len() as f64;

        Some(if !error.is_nan() { 1. - error } else { 1. })
    };

    let mut fidelity: f64 = im_graph_data
        .im_graph
        .edge_indices()
        .map(|x| im_graph_data.im_graph.edge_endpoints(x).unwrap())
        .filter_map(edge_filter_map)
        .product();
    fidelity *= im_graph_data
        .im_graph
        .node_indices()
        .filter_map(bit_filter_map)
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

pub fn vf2_layout(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(score_layout))?;
    m.add_wrapped(wrap_pyfunction!(vf2_layout_pass))?;
    m.add("MultiQEncountered", m.py().get_type::<MultiQEncountered>())?;
    m.add_class::<EdgeList>()?;
    Ok(())
}
