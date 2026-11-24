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

use pyo3::prelude::*;

mod capi;
mod passmanager;

// Formatting is off here so every module import can just be a single line in sorted order, to help
// avoid merge conflicts as modules are added.
#[rustfmt::skip]
#[pymodule]
mod _accelerate {
    #[pymodule_export] use crate::capi::capi_mod;
    #[pymodule_export] use crate::passmanager::passmanager_mod;
    #[pymodule_export] use ::qiskit_mir::py::mir_mod;
    #[pymodule_export] use ::qiskit_mir_passes::py::mir_passes_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::alap_schedule_analysis_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::asap_schedule_analysis_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::apply_layout_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::barrier_before_final_measurements_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::basis_translator_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::check_map_mod;
    #[pymodule_export] use ::qiskit_accelerate::circuit_duration::compute_duration;
    #[pymodule_export] use ::qiskit_circuit_library::circuit_library;
    #[pymodule_export] use ::qiskit_transpiler::passes::commutation_analysis_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::commutation_cancellation_mod;
    #[pymodule_export] use ::qiskit_transpiler::commutation_checker::commutation_checker;
    #[pymodule_export] use ::qiskit_transpiler::passes::commutative_optimization_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::consolidate_blocks_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::constrained_reschedule_mod;
    #[pymodule_export] use ::qiskit_synthesis::linalg::cos_sin_decomp::cos_sin_decomp;
    #[pymodule_export] use ::qiskit_transpiler::passes::dense_layout_mod;
    #[pymodule_export] use ::qiskit_transpiler::equivalence::equivalence;
    #[pymodule_export] use ::qiskit_transpiler::passes::error_map_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::elide_permutations_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::litinski_transformation_mod;
    #[pymodule_export] use ::qiskit_synthesis::euler_one_qubit_decomposer::euler_one_qubit_decomposer;
    #[pymodule_export] use ::qiskit_transpiler::passes::disjoint_utils_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::filter_op_nodes_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::gate_direction_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::gates_in_basis_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::instruction_duration_check_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::inverse_cancellation_mod;
    #[pymodule_export] use ::qiskit_accelerate::isometry::isometry;
    #[pymodule_export] use ::qiskit_circuit::nlayout::nlayout;
    #[pymodule_export] use ::qiskit_circuit::standard_gate::standard_generators::standard_generators;
    #[pymodule_export] use ::qiskit_accelerate::optimize_1q_gates::optimize_1q_gates;
    #[pymodule_export] use ::qiskit_transpiler::passes::optimize_1q_gates_decomposition_mod;
    #[pymodule_export] use ::qiskit_accelerate::pauli_exp_val::pauli_expval;
    #[pymodule_export] use ::qiskit_quantum_info::pauli_lindblad_map::pauli_lindblad_map;
    #[pymodule_export] use ::qiskit_transpiler::passes::high_level_synthesis_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::remove_diagonal_gates_before_measure_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::remove_identity_equiv_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::two_qubit_peephole_mod;
    #[pymodule_export] use ::qiskit_accelerate::results::results;
    #[pymodule_export] use ::qiskit_transpiler::passes::sabre::sabre;
    #[pymodule_export] use ::qiskit_accelerate::sampled_exp_val::sampled_exp_val;
    #[pymodule_export] use ::qiskit_quantum_info::sparse_observable::sparse_observable;
    #[pymodule_export] use ::qiskit_quantum_info::sparse_pauli_op::sparse_pauli_op;
    #[pymodule_export] use ::qiskit_transpiler::passes::scheduling_mod;
    #[pymodule_export] use ::qiskit_synthesis::matrix::sim::unitary_sim;
    #[pymodule_export] use ::qiskit_transpiler::passes::split_2q_unitaries_mod;
    #[pymodule_export] use ::qiskit_synthesis::synthesis;
    #[pymodule_export] use ::qiskit_transpiler::target::target;
    #[pymodule_export] use ::qiskit_accelerate::twirling::twirling;
    #[pymodule_export] use ::qiskit_synthesis::two_qubit_decompose::two_qubit_decompose;
    #[pymodule_export] use ::qiskit_synthesis::pauli_products::pauli_products_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::unitary_synthesis_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::unroll_3q_or_more_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::vf2_layout_mod;
    #[pymodule_export] use ::qiskit_circuit::circuit;
    #[pymodule_export] use ::qiskit_circuit::converters::converters;
    #[pymodule_export] use ::qiskit_qasm2::qasm2;
    #[pymodule_export] use ::qiskit_qasm3::qasm3;
    #[pymodule_export] use ::qiskit_qpy::qpy;
    #[pymodule_export] use ::qiskit_synthesis::qsd::qsd_mod;
    #[pymodule_export] use ::qiskit_synthesis::ross_selinger::ross_selinger_mod;
    #[pymodule_export] use ::qiskit_transpiler::angle_bound_registry::angle_bound_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::wrap_angles_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::optimize_clifford_t_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::substitute_pi4_rotations_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::synthesize_rz_rotations_mod;
    #[pymodule_export] use ::qiskit_transpiler::passes::convert_to_pauli_rotations_mod;
}
