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

use num_complex::Complex64;
use numpy::ndarray::linalg::kron;
use numpy::ndarray::{aview2, Array2, ArrayView2};
use numpy::PyReadonlyArray2;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::gate_matrix::ONE_QUBIT_IDENTITY;
use qiskit_circuit::imports::QI_OPERATOR;
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::Qubit;

use crate::QiskitError;

#[inline]
pub fn get_matrix_from_inst<'py>(
    py: Python<'py>,
    inst: &'py PackedInstruction,
) -> PyResult<Array2<Complex64>> {
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

/// Return the matrix Operator resulting from a block of Instructions.
pub fn blocks_to_matrix(
    py: Python,
    dag: &DAGCircuit,
    op_list: &[NodeIndex],
    block_index_map: [Qubit; 2],
) -> PyResult<Array2<Complex64>> {
    let map_bits = |bit: &Qubit| -> u8 {
        if *bit == block_index_map[0] {
            0
        } else {
            1
        }
    };
    let identity = aview2(&ONE_QUBIT_IDENTITY);
    let first_node = dag.dag()[op_list[0]].unwrap_operation();
    let input_matrix = get_matrix_from_inst(py, first_node)?;
    let mut matrix: Array2<Complex64> = match dag
        .get_qargs(first_node.qubits)
        .iter()
        .map(map_bits)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [0] => kron(&identity, &input_matrix),
        [1] => kron(&input_matrix, &identity),
        [0, 1] => input_matrix,
        [1, 0] => change_basis(input_matrix.view()),
        [] => Array2::eye(4),
        _ => unreachable!(),
    };
    for node in op_list.iter().skip(1) {
        let inst = dag.dag()[*node].unwrap_operation();
        let op_matrix = get_matrix_from_inst(py, inst)?;

        let result = match dag
            .get_qargs(inst.qubits)
            .iter()
            .map(map_bits)
            .collect::<Vec<_>>()
            .as_slice()
        {
            [0] => Some(kron(&identity, &op_matrix)),
            [1] => Some(kron(&op_matrix, &identity)),
            [1, 0] => Some(change_basis(op_matrix.view())),
            [] => Some(Array2::eye(4)),
            _ => None,
        };
        matrix = match result {
            Some(result) => result.dot(&matrix),
            None => op_matrix.dot(&matrix),
        };
    }
    Ok(matrix)
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
