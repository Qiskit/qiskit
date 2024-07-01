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

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

use qiskit_accelerate::{
    convert_2q_block_matrix::convert_2q_block_matrix, dense_layout::dense_layout,
    error_map::error_map, euler_one_qubit_decomposer::euler_one_qubit_decomposer,
    isometry::isometry, nlayout::nlayout, optimize_1q_gates::optimize_1q_gates,
    pauli_exp_val::pauli_expval, results::results, sabre::sabre, sampled_exp_val::sampled_exp_val,
    sparse_pauli_op::sparse_pauli_op, stochastic_swap::stochastic_swap, synthesis::synthesis,
    two_qubit_decompose::two_qubit_decompose, uc_gate::uc_gate, utils::utils,
    vf2_layout::vf2_layout,
};

#[pymodule]
fn _accelerate(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(qiskit_circuit::circuit))?;
    m.add_wrapped(wrap_pymodule!(qiskit_qasm2::qasm2))?;
    m.add_wrapped(wrap_pymodule!(qiskit_qasm3::qasm3))?;
    m.add_wrapped(wrap_pymodule!(convert_2q_block_matrix))?;
    m.add_wrapped(wrap_pymodule!(dense_layout))?;
    m.add_wrapped(wrap_pymodule!(error_map))?;
    m.add_wrapped(wrap_pymodule!(euler_one_qubit_decomposer))?;
    m.add_wrapped(wrap_pymodule!(isometry))?;
    m.add_wrapped(wrap_pymodule!(nlayout))?;
    m.add_wrapped(wrap_pymodule!(optimize_1q_gates))?;
    m.add_wrapped(wrap_pymodule!(pauli_expval))?;
    m.add_wrapped(wrap_pymodule!(synthesis))?;
    m.add_wrapped(wrap_pymodule!(results))?;
    m.add_wrapped(wrap_pymodule!(sabre))?;
    m.add_wrapped(wrap_pymodule!(sampled_exp_val))?;
    m.add_wrapped(wrap_pymodule!(sparse_pauli_op))?;
    m.add_wrapped(wrap_pymodule!(stochastic_swap))?;
    m.add_wrapped(wrap_pymodule!(two_qubit_decompose))?;
    m.add_wrapped(wrap_pymodule!(uc_gate))?;
    m.add_wrapped(wrap_pymodule!(utils))?;
    m.add_wrapped(wrap_pymodule!(vf2_layout))?;
    Ok(())
}
