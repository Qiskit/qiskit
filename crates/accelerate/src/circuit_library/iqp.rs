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

use std::f64::consts::PI;

use ndarray::{Array2, ArrayView2};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use qiskit_circuit::{
    circuit_data::CircuitData,
    operations::{Param, StandardGate},
    Qubit,
};
use rand::Rng;
use smallvec::{smallvec, SmallVec};

use crate::CircuitError;

const PI2: f64 = PI / 2.0;
const PI8: f64 = PI / 8.0;

fn iqp(
    interactions: ArrayView2<i64>,
) -> impl Iterator<Item = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)> + '_ {
    let num_qubits = interactions.ncols();

    // The initial and final Hadamard layer.
    let h_layer =
        (0..num_qubits).map(|i| (StandardGate::HGate, smallvec![], smallvec![Qubit(i as u32)]));

    // The circuit interactions are powers of the CSGate, which is implemented by calling
    // the CPhaseGate with angles of Pi/2 times the power. The gate powers are given by the
    // upper triangular part of the symmetric ``interactions`` matrix.
    let connections = (0..num_qubits).flat_map(move |i| {
        (i + 1..num_qubits)
            .map(move |j| (j, interactions[(i, j)]))
            .filter(move |(_, value)| value % 4 != 0)
            .map(move |(j, value)| {
                (
                    StandardGate::CPhaseGate,
                    smallvec![Param::Float(PI2 * value as f64)],
                    smallvec![Qubit(i as u32), Qubit(j as u32)],
                )
            })
    });

    // The layer of T gates. Again we use the PhaseGate, now with powers of Pi/8. The powers
    // are given by the diagonal of the ``interactions`` matrix.
    let shifts = (0..num_qubits)
        .map(move |i| interactions[(i, i)])
        .enumerate()
        .filter(|(_, value)| value % 8 != 0)
        .map(|(i, value)| {
            (
                StandardGate::PhaseGate,
                smallvec![Param::Float(PI8 * value as f64)],
                smallvec![Qubit(i as u32)],
            )
        });

    h_layer
        .clone()
        .chain(connections)
        .chain(shifts)
        .chain(h_layer)
}

/// This generates a random symmetric integer matrix with values in [0,7].
fn generate_random_interactions(num_qubits: u32) -> Array2<i64> {
    // We first generate all unique values, which can be thought of generating the upper
    // triangular matrix. This contains N(N+1)/2 unique random values, stored as
    //    [v_0 v_1 ... v_{N-1}]
    //    [    v_N ... v_{2N-2}]
    //             ...
    //    [            v_{-1}]
    let num_qubits = num_qubits as usize;
    let num_values = num_qubits * (num_qubits + 1) / 2;
    let values: Vec<i64> = (0..num_values)
        .map(|_| rand::thread_rng().gen_range(0..8) as i64)
        .collect();

    // We then build the matrix of dimension NxN by reading from the ``values`` vector.
    // * since the matrix is symmetric, we treat (i, j) and (j, i) equally, which we implement
    //   here by sorting the indices before accessing the value vector
    // * in each row, the offset with which we access ``values`` changes: in row K, we need to
    //   start reading from index: \sum_{k=0}^K N-k
    Array2::from_shape_fn((num_qubits, num_qubits), |(i, j)| {
        let (low, high) = if i < j { (i, j) } else { (j, i) };
        let offset = low * (2 * num_qubits - low + 1) / 2;
        values[offset + high - low]
    })
}

/// Returns true if the input matrix is symmetric, otherwise false.
fn check_symmetric(matrix: &ArrayView2<i64>) -> bool {
    let nrows = matrix.nrows();

    if matrix.ncols() != nrows {
        return false;
    }

    for i in 0..nrows {
        for j in i + 1..nrows {
            if matrix[(i, j)] != matrix[(j, i)] {
                return false;
            }
        }
    }

    true
}

/// Implement an Instantaneous Quantum Polynomial time (IQP) circuit.
///
/// This class of circuits is conjectured to be classically hard to simulate,
/// forming a generalization of the Boson sampling problem. See Ref. [1] for
/// more details.
///
/// Args:
///     num_qubits: The number of qubits in the IQP circuit.
///     interactions: If provided, this is a symmetric square matrix of width ``num_qubits``,
///         determining the operations in the IQP circuit. The diagonal represents the power
///         of single-qubit T gates and the upper triangular part the power of CS gates
///         in between qubit pairs. If None, a random interactions matrix will be sampled.
///
/// References:
///
///     [1] M. J. Bremner et al. Average-case complexity versus approximate simulation of
///     commuting quantum computations, Phys. Rev. Lett. 117, 080501 (2016).
///     `arXiv:1504.07999 <https://arxiv.org/abs/1504.07999>`_
#[pyfunction]
#[pyo3(signature = (num_qubits, interactions=None))]
pub fn py_iqp(
    py: Python,
    num_qubits: u32,
    interactions: Option<PyReadonlyArray2<i64>>,
) -> PyResult<CircuitData> {
    let array = match interactions {
        Some(matrix) => {
            let view = matrix.as_array();
            if !check_symmetric(&view) {
                return Err(CircuitError::new_err("IQP matrix must be symmetric."));
            }
            view.to_owned()
        }
        None => generate_random_interactions(num_qubits),
    };
    let instructions = iqp(array.view());
    CircuitData::from_standard_gates(py, num_qubits, instructions, Param::Float(0.0))
}
