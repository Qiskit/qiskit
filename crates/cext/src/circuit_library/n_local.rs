// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::pointers::const_ptr_as_ref;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit_library::blocks::{Block, Entanglement as CoreEntanglement};
use qiskit_circuit_library::entanglement::get_entanglement_from_str;
use qiskit_circuit_library::multi_local::n_local;
use qiskit_circuit_library::parameter_ledger::ParameterLedgerBuilder;
use std::ffi::{CStr, c_char};

// Opaque structs for FFI compatibility
pub struct QubitConnection(Vec<u32>);
pub struct BlockQubitConnection(Vec<QubitConnection>);
pub struct LayerEntanglement(Vec<BlockQubitConnection>);
pub struct Entanglement(Vec<LayerEntanglement>);

#[repr(C)]
pub enum EntanglementStrategy {
    Full,
    Linear,
    ReverseLinear,
    Sca,
    Circular,
    Pairwise,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_n_local(
    rotation_blocks: *const StandardGate,
    rotation_blocks_size: usize,
    entanglement_blocks: *const StandardGate,
    entanglement_blocks_size: usize,
    entanglement: *mut Entanglement,
    num_qubits: u32,
    reps: usize,
    parameter_prefix: *const c_char,
    insert_barriers: bool,
    skip_final_rotation_layer: bool,
) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let parameter_prefix = unsafe {
        CStr::from_ptr(parameter_prefix)
            .to_str()
            .expect("Invalid UTF-8 character")
            .to_string()
    };

    // SAFETY: per documentation, `rotation_blocks` is aligned and and points to a valid
    // aligned block of memory valid for `num_rotation_blocks` writes of `Block`s
    let rotation_blocks: Vec<Block> =
        unsafe { ::std::slice::from_raw_parts(rotation_blocks, rotation_blocks_size) }
            .iter()
            .map(|gate| Block::from_standard_gate(*gate))
            .collect();

    // SAFETY: per documentation, `entanglement_blocks` is aligned and and points to a valid
    // aligned block of memory valid for `num_entanglement_blocks` writes of `Block`s
    let entanglement_blocks: Vec<Block> =
        unsafe { ::std::slice::from_raw_parts(entanglement_blocks, entanglement_blocks_size) }
            .iter()
            .map(|gate| Block::from_standard_gate(*gate))
            .collect();

    // SAFETY: per documentation, `entanglement` points to valid data.
    let entanglement = CoreEntanglement {
        entanglement_vec: unsafe {
            const_ptr_as_ref(entanglement)
                .0
                .iter()
                .map(|layer| {
                    layer
                        .0
                        .iter()
                        .map(|block_entanglement| {
                            block_entanglement
                                .0
                                .iter()
                                .map(|connection| connection.0.clone())
                                .collect()
                        })
                        .collect()
                })
                .collect()
        },
    };

    match n_local(
        ParameterLedgerBuilder,
        num_qubits,
        &rotation_blocks.iter().collect::<Vec<&Block>>(),
        &entanglement_blocks.iter().collect::<Vec<&Block>>(),
        &entanglement,
        reps as usize,
        insert_barriers,
        &parameter_prefix,
        skip_final_rotation_layer,
        false,
    ) {
        Ok(circuit) => Box::into_raw(Box::new(circuit)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// @ingroup QkCircuitLibrary
/// Generate an entanglement following the provided strategies.
///
/// @param num_qubits The number of qubits of the circuit
/// @param reps Specifies how often the rotation blocks and entanglement blocks are repeated.
/// @param entanglement_strategy List of strings describing an entanglement strategy for each layer.
/// @param entanglement_strategy_size Length of the entanglement strategy list provided
/// @param entanglement_blocks The blocks used in the entanglement layers.
/// @param entanglement_blocks_size Length of the list of entanglement blocks provided
///
///
///
/// # Safety
///
/// Behavior is undefined ``entanglement_strategy`` is not a valid, non-null pointer to a ``QkEntanglementStrategy`` array.
///
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_get_entanglement_with_multiple_strategy(
    num_qubits: u32,
    reps: usize,
    entanglement_strategy: *const EntanglementStrategy,
    entanglement_strategy_size: usize,
    entanglement_blocks: *const StandardGate,
    entanglement_blocks_size: usize,
) -> *mut Entanglement {
    if entanglement_strategy_size != entanglement_blocks_size {
        panic!(
            "Number of block-entanglements {:?} must match number of entanglement blocks {:?}",
            entanglement_strategy_size, entanglement_blocks_size
        )
    }

    // SAFETY: per documentation, `entanglement_blocks` is aligned and and points to a valid
    // aligned block of memory valid for `entanglement_blocks_size` writes of `StandardGate`s
    let entanglement_blocks: Vec<&StandardGate> = unsafe {
        ::std::slice::from_raw_parts(entanglement_blocks, entanglement_blocks_size)
            .iter()
            .collect()
    };

    // SAFETY: per documentation, `entanglement_strategy` is aligned and and points to a valid
    // aligned block of memory valid for `entanglement_strategy_size` writes of `EntanglementStrategy`s
    let entanglement_strategy: Vec<&str> = unsafe {
        ::std::slice::from_raw_parts(entanglement_strategy, entanglement_strategy_size)
            .iter()
            .map(|strategy| get_entanglement_strategy(strategy))
            .collect()
    };

    Box::into_raw(Box::new(get_entanglement_with_strategy(
        num_qubits,
        &entanglement_blocks,
        &entanglement_strategy,
        reps,
    )))
}

/// @ingroup QkCircuitLibrary
/// Generate an entanglement following the provided strategies.
///
/// @param num_qubits The number of qubits of the circuit
/// @param reps Specifies how often the rotation blocks and entanglement blocks are repeated.
/// @param entanglement_strategy The entanglement strategy to apply for each layer.
/// @param entanglement_blocks The blocks used in the entanglement layers.
/// @param entanglement_blocks_size Length of the list of entanglement blocks provided.
///
///
///
/// # Safety
///
/// Behavior is undefined ``entanglement_blocks`` is not a valid, non-null pointer to a ``QkGate`` array.
///
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_get_entanglement_with_strategy(
    num_qubits: u32,
    reps: usize,
    entanglement_strategy: EntanglementStrategy,
    entanglement_blocks: *const StandardGate,
    entanglement_blocks_size: usize,
) -> *mut Entanglement {
    // SAFETY: per documentation, `entanglement_blocks` is aligned and and points to a valid
    // aligned block of memory valid for `entanglement_blocks_size` writes of `StandardGate`s
    let entanglement_blocks: Vec<&StandardGate> = unsafe {
        ::std::slice::from_raw_parts(entanglement_blocks, entanglement_blocks_size)
            .iter()
            .collect()
    };

    let entanglement_strategy: Vec<&str> = (0..reps)
        .map(|_| get_entanglement_strategy(&entanglement_strategy))
        .collect();

    Box::into_raw(Box::new(get_entanglement_with_strategy(
        num_qubits,
        &entanglement_blocks,
        &entanglement_strategy,
        reps,
    )))
}

fn get_entanglement_with_strategy(
    num_qubits: u32,
    entanglement_blocks: &[&StandardGate],
    entanglement_strategy: &[&str],
    reps: usize,
) -> Entanglement {
    Entanglement(
        (0..reps)
            .zip(entanglement_strategy)
            .map(|(layer, strategy)| {
                get_layer_entanglement_with_strategy(
                    num_qubits,
                    &entanglement_blocks,
                    strategy,
                    layer,
                )
            })
            .collect(),
    )
}

fn get_layer_entanglement_with_strategy(
    num_qubits: u32,
    entanglement_blocks: &[&StandardGate],
    entanglement_strategy: &str,
    layer_index: usize,
) -> LayerEntanglement {
    LayerEntanglement(
        entanglement_blocks
            .iter()
            .map(|gate| {
                get_block_qubit_connections_with_strategy(
                    num_qubits,
                    gate,
                    entanglement_strategy,
                    (entanglement_strategy == "sca").then_some(layer_index),
                )
            })
            .collect(),
    )
}

fn get_block_qubit_connections_with_strategy(
    num_qubits: u32,
    gate: &StandardGate,
    entanglement_strategy: &str,
    offset: Option<usize>,
) -> BlockQubitConnection {
    BlockQubitConnection(
        get_entanglement_from_str(
            num_qubits,
            gate.get_num_qubits(),
            entanglement_strategy,
            if offset.is_some() { offset.unwrap() } else { 0 },
        )
        .unwrap()
        .map(|connections| QubitConnection(connections))
        .collect(),
    )
}

fn get_entanglement_strategy(entanglement_strategy: &EntanglementStrategy) -> &'static str {
    match entanglement_strategy {
        EntanglementStrategy::Full => "full",
        EntanglementStrategy::Linear => "linear",
        EntanglementStrategy::ReverseLinear => "reverse_linear",
        EntanglementStrategy::Sca => "sca",
        EntanglementStrategy::Circular => "circular",
        EntanglementStrategy::Pairwise => "pairwise",
    }
}
