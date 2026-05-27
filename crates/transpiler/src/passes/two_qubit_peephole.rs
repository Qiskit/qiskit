// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::cell::RefCell;
use std::sync::Mutex;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use nalgebra::U4;
use num_complex::Complex64;
use pyo3::Python;
use pyo3::intern;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::algo::toposort;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::visit::NodeIndexable;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PyOperationTypes, PythonOperation,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};

use super::unitary_synthesis::Direction2q;
use crate::passes::unitary_synthesis::{
    Approximation, QpuConstraint, TwoQSynthesisResult, fidelity_2q_sequence, synthesize_2q_matrix,
};
use crate::passes::{UnitarySynthesisConfig, UnitarySynthesisState};
use crate::target::Target;
use qiskit_circuit::PhysicalQubit;
use qiskit_synthesis::linalg::nalgebra_array_view;
use qiskit_synthesis::matrix::two_qubit::blocks_to_matrix;
use qiskit_synthesis::two_qubit_decompose::TwoQubitGateSequence;
use qiskit_util::getenv_use_multiple_threads;
use thread_local::ThreadLocal;

type MappingIterItem = Option<(TwoQSynthesisResult<f64>, [Qubit; 2])>;

#[pyclass(from_py_object)]
#[derive(Debug, Clone, Copy)]
pub enum HeuristicPriority {
    EstimatedFidelity,
    TwoQGate,
    TotalGate,
}

/// Scored used as the heuristic of the unitary synthesis output
///
/// This differs from from [`ComparisonScore`] since the unitary synthesis
/// scoring is trying to maximize so we use negative counts and i64. The
/// comparison score is minimizing the gate counts so it uses usize which is
/// the natural type of the counts.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum BestSynthesisHeuristicScore {
    GatePriority(i64, f64, i64),
    FidelityPriority(f64, i64, i64),
}

/// Score used to compare the original sequence to the best synthesis output
///
/// This differs from [`BestSynthesisHeuristicScore`] in the typing of the counts, usize is
/// used here because we are doing a minimum comparison for this comparison
/// while [`BestSynthesisHeuristicScore`] is doing a maxmimum comparison and
/// needs a negative gate count to work.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum ComparisonScore {
    GatePriority(usize, f64, usize),
    FidelityPriority(f64, usize, usize),
}

impl BestSynthesisHeuristicScore {
    fn get_fidelity(&self, heuristic: HeuristicPriority) -> f64 {
        match heuristic {
            HeuristicPriority::TwoQGate => {
                let BestSynthesisHeuristicScore::GatePriority(_twoq, fidelity, _total_gate) = self
                else {
                    panic!(
                        "Two qubit gate count heuristic priority must have a gate priority score"
                    );
                };
                *fidelity
            }
            HeuristicPriority::EstimatedFidelity => {
                let BestSynthesisHeuristicScore::FidelityPriority(fidelity, _twoq, _total_gate) =
                    self
                else {
                    panic!("fidelity heuristic priority must have a fidelity priority score");
                };
                *fidelity
            }
            HeuristicPriority::TotalGate => {
                let BestSynthesisHeuristicScore::GatePriority(_total_gate, fidelity, _twoq) = self
                else {
                    panic!("Total gate count heuristic priority must have a gate priority score");
                };
                *fidelity
            }
        }
    }

    fn get_two_qubit_gate_count(&self, heuristic: HeuristicPriority) -> usize {
        match heuristic {
            HeuristicPriority::TwoQGate => {
                let BestSynthesisHeuristicScore::GatePriority(twoq, _fidelity, _total_gate) = self
                else {
                    panic!(
                        "Two qubit gate count heuristic priority must have a gate priority score"
                    );
                };
                -twoq as usize
            }
            HeuristicPriority::EstimatedFidelity => {
                let BestSynthesisHeuristicScore::FidelityPriority(_fidelity, twoq, _total_gate) =
                    self
                else {
                    panic!("fidelity heuristic priority must have a fidelity priority score");
                };
                -twoq as usize
            }
            HeuristicPriority::TotalGate => {
                let BestSynthesisHeuristicScore::GatePriority(_total_gate, _fidelity, twoq) = self
                else {
                    panic!("Total gate count heuristic priority must have a gate priority score");
                };
                -twoq as usize
            }
        }
    }
}

/// A python entry-point to the pass function
///
/// This function explicitly releases the GIL prior to entering the parallel portion of the pass.
/// This is necessary because if there are any Python defined and owned objects in the circuit
/// the pass will need GIL access to interact with that object in parallel.
#[pyfunction(name = "two_qubit_unitary_peephole_optimize")]
pub fn py_two_qubit_unitary_peephole_optimize(
    py: Python,
    dag: &DAGCircuit,
    target: &Target,
    approximation_degree: Option<f64>,
    heuristic: HeuristicPriority,
) -> PyResult<Option<DAGCircuit>> {
    let result = py.detach(move || {
        two_qubit_unitary_peephole_optimize_analysis(dag, target, approximation_degree, heuristic)
    })?;
    let Some(result) = result else {
        return Ok(None);
    };
    two_qubit_unitary_peephole_optimize_apply(dag, result)
}

/// A non-python entry-point to the pass function.
///
/// This function is not safe in the context where Python owned objects are in the circuit.
/// It will hang/deadlock on the GIL if called in these contexts and should not be used if
/// there are Python owned objects in the circuit. If you're using this from python you should call
/// `py_two_qubit_unitary_peephole_optimize` instead.
pub fn two_qubit_unitary_peephole_optimize(
    dag: &DAGCircuit,
    target: &Target,
    approximation_degree: Option<f64>,
    heuristic: HeuristicPriority,
) -> PyResult<Option<DAGCircuit>> {
    let result =
        two_qubit_unitary_peephole_optimize_analysis(dag, target, approximation_degree, heuristic)?;
    let Some(result) = result else {
        return Ok(None);
    };
    two_qubit_unitary_peephole_optimize_apply(dag, result)
}

fn score_sequence(
    dir: &Direction2q,
    sequence: &TwoQubitGateSequence,
    constraint: &QpuConstraint,
    qargs: [PhysicalQubit; 2],
    heuristic: HeuristicPriority,
) -> BestSynthesisHeuristicScore {
    let fidelity = fidelity_2q_sequence(dir, sequence, constraint, qargs);
    // Make the gate counts negative because synthesize_2q_matrix picks the largest value
    // we want to minimize the gate counts.
    let gate_count = -(sequence.gates.len() as i64);
    let twoq_gate_count = -(sequence
        .gates
        .iter()
        .filter(|x| x.0.num_qubits() == 2)
        .count() as i64);
    match heuristic {
        HeuristicPriority::TwoQGate => {
            BestSynthesisHeuristicScore::GatePriority(twoq_gate_count, fidelity, gate_count)
        }
        HeuristicPriority::EstimatedFidelity => {
            BestSynthesisHeuristicScore::FidelityPriority(fidelity, twoq_gate_count, gate_count)
        }
        HeuristicPriority::TotalGate => {
            BestSynthesisHeuristicScore::GatePriority(gate_count, fidelity, twoq_gate_count)
        }
    }
}

struct PeepholeResult {
    run_mapping: Vec<MappingIterItem>,
    node_mapping: Vec<usize>,
}

fn two_qubit_unitary_peephole_optimize_analysis(
    dag: &DAGCircuit,
    target: &Target,
    approximation_degree: Option<f64>,
    heuristic: HeuristicPriority,
) -> PyResult<Option<PeepholeResult>> {
    let runs: Vec<Vec<NodeIndex>> = dag.collect_2q_runs().unwrap();
    if runs.is_empty() {
        return Ok(None);
    }
    let node_mapping: Vec<usize> = vec![usize::MAX; dag.dag().node_bound()];
    let locked_node_mapping = Mutex::new(node_mapping);
    let physical_qubits = (0..dag.num_qubits() as u32)
        .map(PhysicalQubit::new)
        .collect::<Vec<_>>();
    let approximation = Approximation::from_py_approximation_degree(approximation_degree);
    let unitary_synthesis_config = UnitarySynthesisConfig {
        approximation,
        ..Default::default()
    };
    let thread_local_states = ThreadLocal::new();
    let find_best_sequence = |run_index: usize,
                              node_indices: &[NodeIndex]|
     -> PyResult<MappingIterItem> {
        let q_virt = node_indices
            .iter()
            .find_map(|node_index| {
                let inst = dag.dag()[*node_index].unwrap_operation();
                let qubits = dag.get_qargs(inst.qubits);
                if qubits.len() == 2 {
                    Some([qubits[0], qubits[1]])
                } else {
                    None
                }
            })
            .expect("runs contain at least one 2q op");
        let q_phys = q_virt.map(|q| physical_qubits[q.index()]);
        let matrix = blocks_to_matrix(dag, node_indices, q_virt)?;
        let synthesis_state: &RefCell<UnitarySynthesisState> = thread_local_states
            .get_or(|| RefCell::new(UnitarySynthesisState::new(unitary_synthesis_config)));

        let scorer = |dir: &Direction2q,
                      sequence: &TwoQubitGateSequence,
                      constraint: &QpuConstraint,
                      qargs: [PhysicalQubit; 2]| {
            score_sequence(dir, sequence, constraint, qargs, heuristic)
        };

        let result = synthesize_2q_matrix(
            nalgebra_array_view::<Complex64, U4, U4>(matrix.as_view()).into(),
            q_phys,
            &mut synthesis_state.borrow_mut(),
            QpuConstraint::Target(target),
            scorer,
        )?;
        let Some(result) = result else {
            return Ok(None);
        };
        let mut original_fidelity: f64 = 1.;
        let mut original_2q_count: usize = 0;
        let original_total_count: usize = node_indices.len();
        let mut outside_target = false;
        for node_index in node_indices {
            let NodeType::Operation(ref inst) = dag.dag()[*node_index] else {
                unreachable!("All run nodes will be ops")
            };
            let qubits: &[_] = match dag.get_qargs(inst.qubits) {
                [q] => &[PhysicalQubit(q.0)],
                [q0, q1] => &[PhysicalQubit(q0.0), PhysicalQubit(q1.0)],
                _ => panic!("Runs should only contain 1q and 2q gates"),
            };
            if qubits.len() == 2 {
                original_2q_count += 1;
            }
            let name = inst.op.name();
            let gate_fidelity = match target.get_error(name, qubits) {
                Some(err) => 1. - err,
                None => {
                    // If error rate is None this can mean either the gate is not supported
                    // in the target or the gate is ideal. We need to do a second lookup
                    // to determine if the gate is supported, and if it isn't we don't need
                    // to finish scoring because we know we'll use the synthesis output
                    if !target.instruction_supported(name, qubits, inst.params_view(), true) {
                        outside_target = true;
                        break;
                    }
                    1.
                }
            };
            original_fidelity *= gate_fidelity;
        }
        let (new_score, original_score) = if !outside_target {
            let original_score = match heuristic {
                HeuristicPriority::EstimatedFidelity => ComparisonScore::FidelityPriority(
                    1. - original_fidelity,
                    original_2q_count,
                    original_total_count,
                ),
                HeuristicPriority::TwoQGate => ComparisonScore::GatePriority(
                    original_2q_count,
                    1. - original_fidelity,
                    original_total_count,
                ),
                HeuristicPriority::TotalGate => ComparisonScore::GatePriority(
                    original_total_count,
                    1. - original_fidelity,
                    original_2q_count,
                ),
            };
            let new_2q_count = result
                .score
                .map(|score| score.get_two_qubit_gate_count(heuristic))
                .unwrap_or_else(|| {
                    result
                        .sequence
                        .gates
                        .iter()
                        .filter(|x| x.0.num_qubits() == 2)
                        .count()
                });
            let new_gate_count = result.sequence.gates.len();
            let new_fidelity = 1.
                - result
                    .score
                    .map(|score| score.get_fidelity(heuristic))
                    .unwrap_or_else(|| {
                        fidelity_2q_sequence(
                            &result.dir,
                            &result.sequence,
                            &QpuConstraint::Target(target),
                            q_phys,
                        )
                    });
            let new_score = match heuristic {
                HeuristicPriority::EstimatedFidelity => {
                    ComparisonScore::FidelityPriority(new_fidelity, new_2q_count, new_gate_count)
                }
                HeuristicPriority::TwoQGate => {
                    ComparisonScore::GatePriority(new_2q_count, new_fidelity, new_gate_count)
                }
                HeuristicPriority::TotalGate => {
                    ComparisonScore::GatePriority(new_gate_count, new_fidelity, new_2q_count)
                }
            };
            (new_score, original_score)
        } else {
            // If we're outside the target we don't need to score since we're going
            // to make the substitution to correct the basis gates. So just set to
            // zeros they won't be read but are needed for the result object.
            match heuristic {
                HeuristicPriority::EstimatedFidelity => (
                    ComparisonScore::FidelityPriority(0., 0, 0),
                    ComparisonScore::FidelityPriority(0., 0, 0),
                ),
                _ => (
                    ComparisonScore::GatePriority(0, 0., 0),
                    ComparisonScore::GatePriority(0, 0., 0),
                ),
            }
        };
        // If we are not outside the target and the new score isn't any better just use the
        // original (this includes a tie).
        if !outside_target && new_score >= original_score {
            return Ok(None);
        }
        // This is done at the end of the map in some attempt to minimize
        // lock contention. If this were serial code it'd make more sense
        // to do this as part of the iteration building the 2q unitary that is
        // already iterating over the nodes. But since that happens at the start
        // there is a higher chance of contention.
        let mut node_mapping = locked_node_mapping.lock().unwrap();
        for node in node_indices {
            node_mapping[node.index()] = run_index;
        }
        let result = TwoQSynthesisResult {
            sequence: result.sequence,
            dir: result.dir,
            score: result.score.map(|score| score.get_fidelity(heuristic)),
        };
        Ok(Some((result, q_virt)))
    };

    let run_mapping: PyResult<Vec<MappingIterItem>> = if getenv_use_multiple_threads() {
        // Build a vec of all the best synthesized two qubit gate sequences from the collected runs.
        // This is done in parallel
        runs.into_par_iter()
            .enumerate()
            .map(|(index, sequence)| find_best_sequence(index, sequence.as_slice()))
            .collect()
    } else {
        runs.into_iter()
            .enumerate()
            .map(|(index, sequence)| find_best_sequence(index, sequence.as_slice()))
            .collect()
    };
    let run_mapping = run_mapping?;
    if run_mapping.iter().any(|run| run.is_some()) {
        Ok(Some(PeepholeResult {
            node_mapping: locked_node_mapping.into_inner().unwrap(),
            run_mapping,
        }))
    } else {
        Ok(None)
    }
}

/// This function runs the two qubit unitary peephole optimization pass
///
/// It returns None if there is no modifications/optimiations made to the input dag and the pass
/// function calling this should just return the input dag from the pass.
fn two_qubit_unitary_peephole_optimize_apply(
    dag: &DAGCircuit,
    result: PeepholeResult,
) -> PyResult<Option<DAGCircuit>> {
    // After we've computed all the sequences to execute now serially build up a new dag.
    let mut processed_runs: Vec<bool> = vec![false; result.run_mapping.len()];
    let out_dag = dag.copy_empty_like_with_same_capacity(VarsMode::Alike, BlocksMode::Keep)?;
    let mut out_dag_builder = out_dag.into_builder();
    for node in toposort(dag.dag(), None).expect("DAG has no cycles") {
        if !matches!(dag.dag()[node], NodeType::Operation(_)) {
            continue;
        }
        let run_index = result.node_mapping[node.index()];
        if run_index != usize::MAX {
            if processed_runs[run_index] {
                continue;
            }
            // If this is not a two qubit gate then there is a chance this will cause the
            // insertion to happen too early. We skip the nodes in a run until we encounter
            // a 2q gate which ensure the block is inserted into the correct location in the
            // circuit.
            if dag.dag()[node].unwrap_operation().op.num_qubits() != 2 {
                continue;
            }
            // A None is inserted into the run_mapping as the value for a run that we don't
            // substitute but was identified so we added an explicit None to preserve the
            // indexing with the vec. This shouldn't be possible to hit the else condition
            // since node mapping will never contain a value for a run_mapping index that
            // is set to None.
            let Some((result, qargs_virt)) = result.run_mapping[run_index].as_ref() else {
                unreachable!(
                    "node_mapping can't contain a value pointing to an unpopulated run in run_mapping"
                );
            };
            let order = result.dir.as_indices();
            let out_qargs = [qargs_virt[order[0] as usize], qargs_virt[order[1] as usize]];
            let qubit_keys = [
                out_dag_builder.insert_qargs(&[out_qargs[0]]),
                out_dag_builder.insert_qargs(&[out_qargs[1]]),
                out_dag_builder.insert_qargs(&[out_qargs[0], out_qargs[1]]),
                out_dag_builder.insert_qargs(&[out_qargs[1], out_qargs[0]]),
            ];
            for (gate, params, local_qubits) in &result.sequence.gates {
                let qubits = match local_qubits.as_slice() {
                    [0] => qubit_keys[0],
                    [1] => qubit_keys[1],
                    [0, 1] => qubit_keys[2],
                    [1, 0] => qubit_keys[3],
                    _ => panic!(
                        "internal logic error: decomposed sequence contained unexpected qargs"
                    ),
                };
                let op = match gate.view() {
                    OperationRef::StandardGate(gate) => PackedOperation::from(gate),
                    OperationRef::Gate(py_gate) => Python::attach(|py| -> PyResult<_> {
                        let gate = py_gate.py_copy(py)?;
                        gate.instruction
                            .setattr(py, intern!(py, "params"), params)?;
                        Ok(PackedOperation::from(Box::new(PyOperationTypes::Gate(
                            gate,
                        ))))
                    })?,
                    _ => {
                        panic!("internal logic error: decomposed sequence contains a non-gate")
                    }
                };
                let params = (!params.is_empty()).then(|| {
                    Box::new(Parameters::Params(
                        params.iter().copied().map(Param::Float).collect(),
                    ))
                });
                out_dag_builder.push_back(PackedInstruction {
                    op,
                    qubits,
                    clbits: Default::default(),
                    params,
                    label: None,
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                })?;
            }
            out_dag_builder.add_global_phase(&Param::Float(result.sequence.global_phase()))?;
            processed_runs[run_index] = true;
        } else {
            let NodeType::Operation(ref instr) = dag.dag()[node] else {
                unreachable!("Must be an op node")
            };
            out_dag_builder.push_back(instr.clone())?;
        }
    }
    Ok(Some(out_dag_builder.build()))
}

pub fn two_qubit_peephole_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_two_qubit_unitary_peephole_optimize))?;
    m.add_class::<HeuristicPriority>()?;
    Ok(())
}
