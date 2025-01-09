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

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::Python;

use nalgebra::UnitQuaternion;
use num_complex::Complex64;
use numpy::ndarray::linalg::kron;
use numpy::ndarray::{arr2, aview2, Array2, ArrayView2};
use numpy::PyReadonlyArray2;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::gate_matrix::TWO_QUBIT_IDENTITY;
use qiskit_circuit::imports::QI_OPERATOR;
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::Qubit;

use crate::qi::VersorGate;
use crate::QiskitError;

#[inline]
pub fn get_matrix_from_inst(py: Python, inst: &PackedInstruction) -> PyResult<Array2<Complex64>> {
    if let Some(mat) = inst.op.matrix(inst.params_view()) {
        Ok(mat)
    } else if inst.op.try_standard_gate().is_some() {
        Err(QiskitError::new_err(
            "Parameterized gates can't be consolidated",
        ))
    } else if let OperationRef::Gate(gate) = inst.op.view() {
        Ok(QI_OPERATOR
            .get_bound(py)
            .call1((gate.gate.clone_ref(py),))?
            .getattr(intern!(py, "data"))?
            .extract::<PyReadonlyArray2<Complex64>>()?
            .as_array()
            .to_owned())
    } else {
        Err(QiskitError::new_err(
            "Can't compute matrix of non-unitary op",
        ))
    }
}

/// Quaternion-based collect of two parallel runs of 1q gates.
#[derive(Clone, Debug)]
struct Separable1q {
    phase: f64,
    qubits: [UnitQuaternion<f64>; 2],
}
impl Separable1q {
    /// Construct an initial state from a single qubit operation.
    #[inline]
    fn from_qubit(n: usize, versor: VersorGate) -> Self {
        let mut qubits: [UnitQuaternion<f64>; 2] = Default::default();
        qubits[n] = versor.action;
        Self {
            phase: versor.phase,
            qubits,
        }
    }

    /// Apply a new gate to one of the two qubits.
    #[inline]
    fn apply_on_qubit(&mut self, n: usize, versor: &VersorGate) {
        self.phase += versor.phase;
        self.qubits[n] = versor.action * self.qubits[n];
    }

    fn matrix(&self) -> Array2<Complex64> {
        let q0 = VersorGate {
            phase: self.phase,
            action: self.qubits[0],
        };
        let q1 = VersorGate {
            phase: 0.,
            action: self.qubits[1],
        };
        kron(
            &aview2(&q1.matrix_contiguous()),
            &aview2(&q0.matrix_contiguous()),
        )
    }
}

/// Extract a versor representation of an arbitrary 1q DAG instruction.
fn versor_from_1q_gate(py: Python, inst: &PackedInstruction) -> PyResult<VersorGate> {
    let versor_result = if let Some(gate) = inst.standard_gate() {
        VersorGate::from_standard(gate, inst.params_view())
    } else {
        VersorGate::from_ndarray(&get_matrix_from_inst(py, inst)?.view(), 1e-12)
    };
    versor_result.map_err(|err| QiskitError::new_err(err.to_string()))
}

/// Return the matrix Operator resulting from a block of Instructions.
///
/// # Panics
///
/// If any node in `op_list` is not a 1q or 2q gate.
pub fn blocks_to_matrix(
    py: Python,
    dag: &DAGCircuit,
    op_list: &[NodeIndex],
    block_index_map: [Qubit; 2],
) -> PyResult<Array2<Complex64>> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Qarg {
        Q0 = 0,
        Q1 = 1,
        Q01,
        Q10,
    }
    let qarg_lookup = |qargs| {
        let interned = dag.get_qargs(qargs);
        if interned.len() == 1 {
            if interned[0] == block_index_map[0] {
                Qarg::Q0
            } else {
                Qarg::Q1
            }
        } else if interned.len() == 2 {
            if interned[0] == block_index_map[0] {
                Qarg::Q01
            } else {
                Qarg::Q10
            }
        } else {
            panic!("not a one- or two-qubit gate");
        }
    };

    let mut qubits_1q: Option<Separable1q> = None;
    let mut output_matrix: Option<Array2<Complex64>> = None;
    for node in op_list {
        let inst = dag.dag()[*node].unwrap_operation();
        let qarg = qarg_lookup(inst.qubits);
        match qarg {
            Qarg::Q0 | Qarg::Q1 => {
                let versor = versor_from_1q_gate(py, inst)?;
                match qubits_1q.as_mut() {
                    Some(sep) => sep.apply_on_qubit(qarg as usize, &versor),
                    None => qubits_1q = Some(Separable1q::from_qubit(qarg as usize, versor)),
                };
            }
            Qarg::Q01 | Qarg::Q10 => {
                let mut matrix = get_matrix_from_inst(py, inst)?;
                if qarg == Qarg::Q10 {
                    matrix = change_basis(matrix.view());
                }
                if let Some(sep) = qubits_1q.take() {
                    matrix = matrix.dot(&sep.matrix());
                }
                output_matrix = if let Some(state) = output_matrix {
                    Some(matrix.dot(&state))
                } else {
                    Some(matrix)
                };
            }
        }
    }
    match (qubits_1q.as_ref().map(Separable1q::matrix), output_matrix) {
        (Some(sep), Some(state)) => Ok(sep.dot(&state)),
        (None, Some(state)) => Ok(state),
        (Some(sep), None) => Ok(sep),
        // This shouldn't actually ever trigger, because we expect blocks to be non-empty, but it's
        // trivial to handle anyway.
        (None, None) => Ok(arr2(&TWO_QUBIT_IDENTITY)),
    }
}

/// Switches the order of qubits in a two qubit operation.
#[inline]
pub fn change_basis(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
    let mut trans_matrix: Array2<Complex64> = matrix.reversed_axes().to_owned();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix = trans_matrix.reversed_axes();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix
}
