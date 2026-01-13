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

use std::str::FromStr;

use hashbrown::HashSet;
use num_complex::Complex64;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use ndarray::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::getenv_use_multiple_threads;
use qiskit_circuit::operations::{Operation, OperationRef, Param};

use crate::target::{Target, TargetOperation};
use qiskit_circuit::instruction::Instruction;
use qiskit_circuit::{PhysicalQubit, gate_matrix};
use qiskit_synthesis::euler_one_qubit_decomposer::{
    EULER_BASES, EULER_BASIS_NAMES, EulerBasis, EulerBasisSet, OneQubitGateSequence,
    unitary_to_gate_sequence_inner,
};

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

#[pyclass(module = "qiskit._accelerate.optimize_1q_gates_decomposition")]
pub struct Optimize1qGatesDecompositionState {
    target_basis_per_qubit: Vec<EulerBasisSet>,
    basis_gates_per_qubit: Vec<Option<HashSet<String>>>,
    global: bool,
}

type Optimize1qGatesDecompositionStatePickle =
    (Vec<Vec<String>>, Vec<Option<HashSet<String>>>, bool);

#[pymethods]
impl Optimize1qGatesDecompositionState {
    fn __getstate__(&self) -> Optimize1qGatesDecompositionStatePickle {
        (
            self.target_basis_per_qubit
                .iter()
                .map(|set| set.get_bases().map(|x| x.as_str().to_string()).collect())
                .collect(),
            self.basis_gates_per_qubit.clone(),
            self.global,
        )
    }

    fn __setstate__(&mut self, state: Optimize1qGatesDecompositionStatePickle) {
        self.target_basis_per_qubit = state
            .0
            .into_iter()
            .map(|set| {
                let mut euler_set = EulerBasisSet::new();
                for basis_str in set {
                    euler_set.add_basis(EulerBasis::from_str(basis_str.as_str()).unwrap());
                }
                euler_set
            })
            .collect();
        self.basis_gates_per_qubit = state.1;
        self.global = state.2;
    }

    #[new]
    #[pyo3(signature = (target=None, basis_gates=None, global_decomposers=None))]
    pub fn new(
        target: Option<&Target>,
        basis_gates: Option<HashSet<String>>,
        global_decomposers: Option<Vec<String>>,
    ) -> PyResult<Self> {
        match target {
            Some(target) => {
                match target.num_qubits {
                    Some(num_qubits) => {
                        let mut target_basis_per_qubit: Vec<EulerBasisSet> =
                            vec![EulerBasisSet::new(); num_qubits as usize];
                        let mut basis_gates_per_qubit: Vec<Option<HashSet<String>>> =
                            vec![None; num_qubits as usize];
                        for (qubit, basis_gates) in basis_gates_per_qubit.iter_mut().enumerate() {
                            let basis = target
                                .operation_names_for_qargs(&[PhysicalQubit::new(qubit as u32)])?
                                .into_iter()
                                .map(|x| x.to_string())
                                .filter(|gate_name| {
                                    let target_op = target.operation_from_name(gate_name).unwrap();
                                    let TargetOperation::Normal(gate) = target_op else {
                                        return false;
                                    };
                                    if let OperationRef::StandardGate(_) = gate.operation.view() {
                                        // For standard gates check that the target entry accepts any
                                        // params and if so then we can use the gate in the pass
                                        // else filter the operation since arbitrary angles are not
                                        // supported
                                        gate.params_view()
                                            .iter()
                                            .all(|x| matches!(x, Param::ParameterExpression(_)))
                                    } else {
                                        // For all other gates pass it through
                                        true
                                    }
                                })
                                .collect();
                            *basis_gates = Some(basis);
                        }
                        target_basis_per_qubit.iter_mut().enumerate().for_each(
                            |(qubit, target_basis_set)| {
                                EULER_BASES
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(idx, gates)| {
                                        let basis_gates = basis_gates_per_qubit[qubit].as_ref();
                                        if !gates.iter().all(|gate| {
                                            basis_gates
                                                .expect(
                                                    "the target path always provides a hash set",
                                                )
                                                .contains(*gate)
                                        }) {
                                            return None;
                                        }
                                        let basis = EULER_BASIS_NAMES[idx];
                                        Some(basis)
                                    })
                                    .for_each(|basis| target_basis_set.add_basis(basis));
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
                            },
                        );
                        Ok(Self {
                            target_basis_per_qubit,
                            basis_gates_per_qubit,
                            global: false,
                        })
                    }
                    None => {
                        let basis_gates_per_qubit = vec![Some(
                            target
                                .operations()
                                .filter_map(|op| {
                                    if op.operation.num_qubits() == 1 {
                                        Some(op.operation.name().to_string())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<HashSet<_>>(),
                        )];
                        let mut target_basis_set = EulerBasisSet::new();
                        EULER_BASES
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, gates)| {
                                let basis_gates = basis_gates_per_qubit[0].as_ref();
                                if !gates.iter().all(|gate| {
                                    basis_gates
                                        .expect("the target path always provides a hash set")
                                        .contains(*gate)
                                }) {
                                    return None;
                                }
                                let basis = EULER_BASIS_NAMES[idx];
                                Some(basis)
                            })
                            .for_each(|basis| target_basis_set.add_basis(basis));
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
                        let target_basis_per_qubit = vec![target_basis_set];

                        Ok(Self {
                            target_basis_per_qubit,
                            basis_gates_per_qubit,
                            global: true,
                        })
                    }
                }
            }
            None => {
                let basis_gates_per_qubit = vec![basis_gates];
                let mut target_basis_set = EulerBasisSet::new();
                match &global_decomposers {
                    Some(bases) => {
                        for basis in bases.iter() {
                            target_basis_set.add_basis(EulerBasis::__new__(basis)?)
                        }
                    }
                    None => match basis_gates_per_qubit[0] {
                        Some(ref gates) => EULER_BASES
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, basis_gates)| {
                                if !gates
                                    .iter()
                                    .all(|gate| basis_gates.as_ref().contains(&gate.as_str()))
                                {
                                    return None;
                                }
                                let basis = EULER_BASIS_NAMES[idx];
                                Some(basis)
                            })
                            .for_each(|basis| target_basis_set.add_basis(basis)),
                        None => target_basis_set.support_all(),
                    },
                }
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

                let target_basis_per_qubit = vec![target_basis_set];
                Ok(Self {
                    target_basis_per_qubit,
                    basis_gates_per_qubit,
                    global: true,
                })
            }
        }
    }
}

#[pyfunction]
#[pyo3(name = "optimize_1q_gates_decomposition", signature = (dag, state, *, target=None))]
pub fn run_optimize_1q_gates_decomposition(
    dag: &mut DAGCircuit,
    state: &Optimize1qGatesDecompositionState,
    target: Option<&Target>,
) -> PyResult<()> {
    let runs: Vec<Vec<NodeIndex>> = dag.collect_1q_runs().unwrap().collect();
    let process_run = |raw_run: &[NodeIndex], dag: &DAGCircuit| -> Option<OneQubitGateSequence> {
        let mut error = match target {
            Some(_) => 1.,
            None => raw_run.len() as f64,
        };
        let qubit: PhysicalQubit = if let NodeType::Operation(inst) = &dag[raw_run[0]] {
            PhysicalQubit::new(dag.get_qargs(inst.qubits)[0].0)
        } else {
            unreachable!("nodes in runs will always be op nodes")
        };
        let basis_gates = if state.global {
            state.basis_gates_per_qubit[0].as_ref()
        } else {
            state.basis_gates_per_qubit[qubit.index()].as_ref()
        };
        let target_basis_set = if state.global {
            &state.target_basis_per_qubit[0]
        } else {
            &state.target_basis_per_qubit[qubit.index()]
        };
        let operator = raw_run
            .iter()
            .map(|node_index| {
                let node = &dag[*node_index];
                if let NodeType::Operation(inst) = node {
                    if let Some(target) = target {
                        error *= compute_error_term_from_target(inst.op.name(), target, qubit);
                    }
                    inst.try_matrix_as_static_1q()
                        .expect("collect_1q_runs only collects gates that can produce a matrix")
                } else {
                    unreachable!("Can only have op nodes here")
                }
            })
            .fold(gate_matrix::ONE_QUBIT_IDENTITY, |mut operator, node| {
                matmul_1q_with_slice(&mut operator, &node);
                operator
            });

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
        let sequence = sequence?;
        let new_error = compute_error_from_target_one_qubit_sequence(&sequence, qubit, target);

        let mut outside_basis = false;
        if let Some(basis) = basis_gates {
            for node in raw_run {
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
            Some(sequence)
        } else {
            None
        }
    };
    if runs.len() > 100_000 && getenv_use_multiple_threads() {
        let sequences = runs
            .par_iter()
            .map(|raw_run| process_run(raw_run, dag))
            .collect::<Vec<_>>();
        runs.into_iter()
            .zip(sequences)
            .filter_map(|(raw_run, sequence)| sequence.map(|x| (raw_run, x)))
            .for_each(|(raw_run, sequence)| {
                for gate in sequence.gates {
                    dag.insert_1q_on_incoming_qubit((gate.0, &gate.1), raw_run[0]);
                }
                dag.add_global_phase(&Param::Float(sequence.global_phase))
                    .unwrap();
                dag.remove_1q_sequence(&raw_run);
            });
    } else {
        for raw_run in runs {
            let sequence = process_run(&raw_run, dag);
            if let Some(sequence) = sequence {
                for gate in sequence.gates {
                    dag.insert_1q_on_incoming_qubit((gate.0, &gate.1), raw_run[0]);
                }
                dag.add_global_phase(&Param::Float(sequence.global_phase))?;
                dag.remove_1q_sequence(&raw_run);
            }
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

/// Computes matrix product ``other * operator`` and stores the result within ``operator``.
#[inline(always)]
pub fn matmul_1q_with_slice(operator: &mut [[Complex64; 2]; 2], other: &[[Complex64; 2]; 2]) {
    *operator = [
        [
            other[0][0] * operator[0][0] + other[0][1] * operator[1][0],
            other[0][0] * operator[0][1] + other[0][1] * operator[1][1],
        ],
        [
            other[1][0] * operator[0][0] + other[1][1] * operator[1][0],
            other[1][0] * operator[0][1] + other[1][1] * operator[1][1],
        ],
    ];
}

pub fn optimize_1q_gates_decomposition_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_optimize_1q_gates_decomposition))?;
    m.add_class::<Optimize1qGatesDecompositionState>()?;
    Ok(())
}
