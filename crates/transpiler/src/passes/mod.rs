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

//! Module for transpiler pass implementations
//!
//! This module contains function definitions for transpiler passes
//! these fall into two categories per pass. The standalone transpiler
//! pass functions are {verb}_pass_name where verb is typically run. These
//! function are a standalone pass functions that operated directly on a
//! [DAGCircuit]. The second class are the functions that end in _mod. These
//! are Python submodule functions designed to be exported to the qiskit-pyext
//! crate. These are public to be passed to qiskit-pyext and are only used
//! for building Python submodules.

mod barrier_before_final_measurement;
mod basis_translator;
mod check_map;
mod commutation_analysis;
mod commutation_cancellation;
mod consolidate_blocks;
mod dense_layout;
mod disjoint_layout;
mod elide_permutations;
mod filter_op_nodes;
mod gate_direction;
mod gates_in_basis;
mod high_level_synthesis;
mod inverse_cancellation;
mod optimize_1q_gates_decomposition;
mod remove_diagonal_gates_before_measure;
mod remove_identity_equiv;
pub mod sabre;
mod split_2q_unitaries;
mod star_prerouting;
mod unitary_synthesis;
mod vf2;

pub use barrier_before_final_measurement::{
    barrier_before_final_measurements_mod, run_barrier_before_final_measurements,
};
pub use basis_translator::{basis_translator_mod, run_basis_translator};
pub use check_map::{check_map_mod, run_check_map};
pub use commutation_analysis::{analyze_commutations, commutation_analysis_mod};
pub use commutation_cancellation::{cancel_commutations, commutation_cancellation_mod};
pub use consolidate_blocks::{consolidate_blocks_mod, run_consolidate_blocks, DecomposerType};
pub use dense_layout::{best_subset, dense_layout_mod};
pub use disjoint_layout::{combine_barriers, disjoint_utils_mod, distribute_components};
pub use elide_permutations::{elide_permutations_mod, run_elide_permutations};
pub use filter_op_nodes::{filter_labeled_op, filter_op_nodes_mod};
pub use gate_direction::{
    check_direction_coupling_map, check_direction_target, fix_direction_coupling_map,
    fix_direction_target, gate_direction_mod,
};
pub use gates_in_basis::{gates_in_basis_mod, gates_missing_from_basis, gates_missing_from_target};
pub use high_level_synthesis::{
    high_level_synthesis_mod, run_high_level_synthesis, HighLevelSynthesisData,
};
pub use inverse_cancellation::{inverse_cancellation_mod, run_inverse_cancellation};
pub use optimize_1q_gates_decomposition::{
    optimize_1q_gates_decomposition_mod, run_optimize_1q_gates_decomposition,
};
pub use remove_diagonal_gates_before_measure::{
    remove_diagonal_gates_before_measure_mod, run_remove_diagonal_before_measure,
};
pub use remove_identity_equiv::{remove_identity_equiv_mod, run_remove_identity_equiv};
pub use split_2q_unitaries::{run_split_2q_unitaries, split_2q_unitaries_mod};
pub use star_prerouting::{star_preroute, star_prerouting_mod};
pub use unitary_synthesis::{run_unitary_synthesis, unitary_synthesis_mod};
pub use vf2::{error_map_mod, score_layout, vf2_layout_mod, vf2_layout_pass, ErrorMap};
