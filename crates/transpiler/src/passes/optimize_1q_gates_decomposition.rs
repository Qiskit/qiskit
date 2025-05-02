// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashSet;
use num_complex::Complex64;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use ndarray::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{Operation, Param};

use crate::target::Target;
use qiskit_accelerate::euler_one_qubit_decomposer::{
    unitary_to_gate_sequence_inner, EulerBasis, EulerBasisSet, OneQubitGateSequence, EULER_BASES,
    EULER_BASIS_NAMES,
};
use qiskit_circuit::PhysicalQubit;

fn compute_error_term_from_target(gate: &str, target: &Target, qubit: PhysicalQubit) -> f64 {
    1. - target.get_error(gate, &[qubit]).unwrap_or(0.)
}

fn compute_error_from_target_one_qubit_sequence(
    circuit: &OneQubitGateSequence,
    qubit: PhysicalQubit,
    target: Option<&Target>,
) -> (f64, usize) {
    match target {
        Some(target) => {
            let num_gates = circuit.gates.len();
            let gate_fidelities: f64 = circuit
                .gates
                .iter()
                .map(|gate| compute_error_term_from_target(gate.0.name(), target, qubit))
                .product();
            (1. - gate_fidelities, num_gates)
        }
        None => (circuit.gates.len() as f64, circuit.gates.len()),
    }
}

#[pyfunction]
#[pyo3(name = "optimize_1q_gates_decomposition", signature = (dag, *, target=None, basis_gates=None, global_decomposers=None))]
pub fn run_optimize_1q_gates_decomposition(
    dag: &mut DAGCircuit,
    target: Option<&Target>,
    basis_gates: Option<HashSet<String>>,
    global_decomposers: Option<Vec<String>>,
) -> PyResult<()> {
    let runs: Vec<Vec<NodeIndex>> = dag.collect_1q_runs().unwrap().collect();
    let dag_qubits = dag.num_qubits();
    let mut target_basis_per_qubit: Vec<EulerBasisSet> = vec![EulerBasisSet::new(); dag_qubits];
    let mut basis_gates_per_qubit: Vec<Option<HashSet<&str>>> = vec![None; dag_qubits];
    for raw_run in runs {
        let mut error = match target {
            Some(_) => 1.,
            None => raw_run.len() as f64,
        };
        let qubit: PhysicalQubit = if let NodeType::Operation(inst) = &dag[raw_run[0]] {
            PhysicalQubit::new(dag.get_qargs(inst.qubits)[0].0)
        } else {
            unreachable!("nodes in runs will always be op nodes")
        };
        if basis_gates_per_qubit[qubit.index()].is_none() {
            let basis_gates = match target {
                Some(target) => Some(target.operation_names_for_qargs(&[qubit]).unwrap()),
                None => {
                    let basis = basis_gates.as_ref();
                    basis.map(|basis| basis.iter().map(|x| x.as_str()).collect())
                }
            };
            basis_gates_per_qubit[qubit.index()] = basis_gates;
        }
        let basis_gates = &basis_gates_per_qubit[qubit.index()].as_ref();

        let target_basis_set = &mut target_basis_per_qubit[qubit.index()];
        if !target_basis_set.initialized() {
            match target {
                Some(_target) => EULER_BASES
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, gates)| {
                        if !gates
                            .iter()
                            .all(|gate| basis_gates.as_ref().unwrap().contains(gate))
                        {
                            return None;
                        }
                        let basis = EULER_BASIS_NAMES[idx];
                        Some(basis)
                    })
                    .for_each(|basis| target_basis_set.add_basis(basis)),
                None => match &global_decomposers {
                    Some(bases) => bases
                        .iter()
                        .map(|basis| EulerBasis::__new__(basis).unwrap())
                        .for_each(|basis| target_basis_set.add_basis(basis)),
                    None => match basis_gates {
                        Some(gates) => EULER_BASES
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, basis_gates)| {
                                if !gates.iter().all(|gate| basis_gates.as_ref().contains(gate)) {
                                    return None;
                                }
                                let basis = EULER_BASIS_NAMES[idx];
                                Some(basis)
                            })
                            .for_each(|basis| target_basis_set.add_basis(basis)),
                        None => target_basis_set.support_all(),
                    },
                },
            };
            if target_basis_set.basis_supported(EulerBasis::U3)
                && target_basis_set.basis_supported(EulerBasis::U321)
            {
                target_basis_set.remove(EulerBasis::U3);
            }
            if target_basis_set.basis_supported(EulerBasis::ZSX)
                && target_basis_set.basis_supported(EulerBasis::ZSXX)
            {
                target_basis_set.remove(EulerBasis::ZSX);
            }
        }
        let target_basis_set = &target_basis_per_qubit[qubit.index()];
        let operator = raw_run
            .iter()
            .map(|node_index| {
                let node = &dag[*node_index];
                if let NodeType::Operation(inst) = node {
                    if let Some(target) = target {
                        error *= compute_error_term_from_target(inst.op.name(), target, qubit);
                    }
                    inst.op.matrix(inst.params_view()).unwrap()
                } else {
                    unreachable!("Can only have op nodes here")
                }
            })
            .fold(
                [
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
                ],
                |mut operator, node| {
                    matmul_1q(&mut operator, node);
                    operator
                },
            );

        let old_error = if target.is_some() {
            (1. - error, raw_run.len())
        } else {
            (error, raw_run.len())
        };
        let sequence = unitary_to_gate_sequence_inner(
            aview2(&operator),
            target_basis_set,
            qubit.index(),
            None,
            true,
            None,
        );
        let sequence = match sequence {
            Some(seq) => seq,
            None => continue,
        };
        let new_error = compute_error_from_target_one_qubit_sequence(&sequence, qubit, target);

        let mut outside_basis = false;
        if let Some(basis) = basis_gates {
            for node in &raw_run {
                if let NodeType::Operation(inst) = &dag[*node] {
                    if !basis.contains(inst.op.name()) {
                        outside_basis = true;
                        break;
                    }
                }
            }
        } else {
            outside_basis = false;
        }
        if outside_basis
            || new_error < old_error
            || new_error.0.abs() < 1e-9 && old_error.0.abs() >= 1e-9
        {
            for gate in sequence.gates {
                dag.insert_1q_on_incoming_qubit((gate.0, &gate.1), raw_run[0]);
            }
            dag.add_global_phase(&Param::Float(sequence.global_phase))?;
            dag.remove_1q_sequence(&raw_run);
        }
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn matmul_1q(operator: &mut [[Complex64; 2]; 2], other: Array2<Complex64>) {
    *operator = [
        [
            other[[0, 0]] * operator[0][0] + other[[0, 1]] * operator[1][0],
            other[[0, 0]] * operator[0][1] + other[[0, 1]] * operator[1][1],
        ],
        [
            other[[1, 0]] * operator[0][0] + other[[1, 1]] * operator[1][0],
            other[[1, 0]] * operator[0][1] + other[[1, 1]] * operator[1][1],
        ],
    ];
}

pub fn optimize_1q_gates_decomposition_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_optimize_1q_gates_decomposition))?;
    Ok(())
}
