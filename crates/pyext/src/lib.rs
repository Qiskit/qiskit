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
pub use qiskit_cext::*;

#[inline(always)]
#[doc(hidden)]
fn add_submodule<F>(m: &Bound<PyModule>, constructor: F, name: &str) -> PyResult<()>
where
    F: FnOnce(&Bound<PyModule>) -> PyResult<()>,
{
    let new_mod = PyModule::new(m.py(), name)?;
    constructor(&new_mod)?;
    m.add_submodule(&new_mod)
}

// Formatting is off here so every module import can just be a single line in sorted order, to help
// avoid merge conflicts as modules are added.
#[rustfmt::skip]
#[pymodule]
fn _accelerate(m: &Bound<PyModule>) -> PyResult<()> {
    add_submodule(m, ::qiskit_transpiler::passes::barrier_before_final_measurements_mod, "barrier_before_final_measurement")?;
    add_submodule(m, ::qiskit_transpiler::passes::basis_translator_mod, "basis_translator")?;
    add_submodule(m, ::qiskit_transpiler::passes::check_map_mod, "check_map")?;
    add_submodule(m, ::qiskit_transpiler::circuit_duration::compute_duration, "circuit_duration")?;
    add_submodule(m, ::qiskit_accelerate::circuit_library::circuit_library, "circuit_library")?;
    add_submodule(m, ::qiskit_transpiler::passes::commutation_analysis_mod, "commutation_analysis")?;
    add_submodule(m, ::qiskit_transpiler::passes::commutation_cancellation_mod, "commutation_cancellation")?;
    add_submodule(m, ::qiskit_transpiler::commutation_checker::commutation_checker, "commutation_checker")?;
    add_submodule(m, ::qiskit_transpiler::passes::consolidate_blocks_mod, "consolidate_blocks")?;
    add_submodule(m, ::qiskit_accelerate::cos_sin_decomp::cos_sin_decomp, "cos_sin_decomp")?;
    add_submodule(m, ::qiskit_transpiler::passes::dense_layout_mod, "dense_layout")?;
    add_submodule(m, ::qiskit_transpiler::equivalence::equivalence, "equivalence")?;
    add_submodule(m, ::qiskit_transpiler::passes::error_map_mod, "error_map")?;
    add_submodule(m, ::qiskit_transpiler::passes::elide_permutations_mod, "elide_permutations")?;
    add_submodule(m, ::qiskit_transpiler::passes::disjoint_utils_mod, "disjoint_utils")?;
    add_submodule(m, ::qiskit_accelerate::euler_one_qubit_decomposer::euler_one_qubit_decomposer, "euler_one_qubit_decomposer")?;
    add_submodule(m, ::qiskit_transpiler::passes::filter_op_nodes_mod, "filter_op_nodes")?;
    add_submodule(m, ::qiskit_transpiler::passes::gate_direction_mod, "gate_direction")?;
    add_submodule(m, ::qiskit_transpiler::passes::gates_in_basis_mod, "gates_in_basis")?;
    add_submodule(m, ::qiskit_transpiler::passes::inverse_cancellation_mod, "inverse_cancellation")?;
    add_submodule(m, ::qiskit_accelerate::isometry::isometry, "isometry")?;
    add_submodule(m, ::qiskit_circuit::nlayout::nlayout, "nlayout")?;
    add_submodule(m, ::qiskit_accelerate::optimize_1q_gates::optimize_1q_gates, "optimize_1q_gates")?;
    add_submodule(m, ::qiskit_transpiler::passes::optimize_1q_gates_decomposition_mod, "optimize_1q_gates_decomposition")?;
    add_submodule(m, ::qiskit_accelerate::pauli_exp_val::pauli_expval, "pauli_expval")?;
    add_submodule(m, ::qiskit_transpiler::passes::high_level_synthesis_mod, "high_level_synthesis")?;
    add_submodule(m, ::qiskit_transpiler::passes::remove_diagonal_gates_before_measure_mod, "remove_diagonal_gates_before_measure")?;
    add_submodule(m, ::qiskit_transpiler::passes::remove_identity_equiv_mod, "remove_identity_equiv")?;
    add_submodule(m, ::qiskit_accelerate::results::results, "results")?;
    add_submodule(m, ::qiskit_transpiler::passes::sabre::sabre, "sabre")?;
    add_submodule(m, ::qiskit_accelerate::sampled_exp_val::sampled_exp_val, "sampled_exp_val")?;
    add_submodule(m, ::qiskit_accelerate::sparse_observable::sparse_observable, "sparse_observable")?;
    add_submodule(m, ::qiskit_accelerate::sparse_pauli_op::sparse_pauli_op, "sparse_pauli_op")?;
    add_submodule(m, ::qiskit_transpiler::passes::split_2q_unitaries_mod, "split_2q_unitaries")?;
    add_submodule(m, ::qiskit_transpiler::passes::star_prerouting_mod, "star_prerouting")?;
    add_submodule(m, ::qiskit_accelerate::synthesis::synthesis, "synthesis")?;
    add_submodule(m, ::qiskit_transpiler::target::target, "target")?;
    add_submodule(m, ::qiskit_transpiler::twirling::twirling, "twirling")?;
    add_submodule(m, ::qiskit_accelerate::two_qubit_decompose::two_qubit_decompose, "two_qubit_decompose")?;
    add_submodule(m, ::qiskit_transpiler::passes::unitary_synthesis_mod, "unitary_synthesis")?;
    add_submodule(m, ::qiskit_accelerate::uc_gate::uc_gate, "uc_gate")?;
    add_submodule(m, ::qiskit_transpiler::passes::vf2_layout_mod, "vf2_layout")?;
    add_submodule(m, ::qiskit_circuit::circuit, "circuit")?;
    add_submodule(m, ::qiskit_circuit::converters::converters, "converters")?;
    add_submodule(m, ::qiskit_qasm2::qasm2, "qasm2")?;
    add_submodule(m, ::qiskit_qasm3::qasm3, "qasm3")?;
    Ok(())
}
