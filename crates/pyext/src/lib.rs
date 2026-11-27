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

#[pymodule]
mod _accelerate {
    #[pymodule_export]
    use crate::{capi::capi_mod, passmanager::passmanager_mod};
    #[pymodule_export]
    use ::qiskit_accelerate::{
        circuit_duration::compute_duration, isometry::isometry,
        optimize_1q_gates::optimize_1q_gates, pauli_exp_val::pauli_expval, results::results,
        sampled_exp_val::sampled_exp_val, twirling::twirling,
    };
    #[pymodule_export]
    use ::qiskit_circuit::{
        circuit, converters::converters, nlayout::nlayout,
        standard_gate::standard_generators::standard_generators,
    };
    #[pymodule_export]
    use ::qiskit_circuit_library::circuit_library;
    #[pymodule_export]
    use ::qiskit_mir::py::mir_mod;
    #[pymodule_export]
    use ::qiskit_mir_passes::py::mir_passes_mod;
    #[pymodule_export]
    use ::qiskit_qasm2::qasm2;
    #[pymodule_export]
    use ::qiskit_qasm3::qasm3;
    #[pymodule_export]
    use ::qiskit_qpy::qpy;
    #[pymodule_export]
    use ::qiskit_quantum_info::{
        pauli_lindblad_map::pauli_lindblad_map, sparse_observable::sparse_observable,
        sparse_pauli_op::sparse_pauli_op,
    };
    #[pymodule_export]
    use ::qiskit_synthesis::{
        euler_one_qubit_decomposer::euler_one_qubit_decomposer,
        linalg::cos_sin_decomp::cos_sin_decomp, matrix::sim::unitary_sim,
        pauli_products::pauli_products_mod, qsd::qsd_mod, ross_selinger::ross_selinger_mod,
        synthesis, two_qubit_decompose::two_qubit_decompose,
    };
    #[pymodule_export]
    use ::qiskit_transpiler::{
        angle_bound_registry::angle_bound_mod,
        commutation_checker::commutation_checker,
        equivalence::equivalence,
        passes::{
            alap_schedule_analysis_mod, apply_layout_mod, asap_schedule_analysis_mod,
            barrier_before_final_measurements_mod, basis_translator_mod, check_map_mod,
            commutation_analysis_mod, commutation_cancellation_mod, commutative_optimization_mod,
            consolidate_blocks_mod, constrained_reschedule_mod, convert_to_pauli_rotations_mod,
            dense_layout_mod, disjoint_utils_mod, elide_permutations_mod, error_map_mod,
            filter_op_nodes_mod, gate_direction_mod, gates_in_basis_mod, high_level_synthesis_mod,
            instruction_duration_check_mod, inverse_cancellation_mod, litinski_transformation_mod,
            optimize_1q_gates_decomposition_mod, optimize_clifford_t_mod,
            remove_diagonal_gates_before_measure_mod, remove_identity_equiv_mod, sabre::sabre,
            scheduling_mod, split_2q_unitaries_mod, substitute_pi4_rotations_mod,
            synthesize_rz_rotations_mod, two_qubit_peephole_mod, unitary_synthesis_mod,
            unroll_3q_or_more_mod, vf2_layout_mod, wrap_angles_mod,
        },
        target::target,
    };
}
