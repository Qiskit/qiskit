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
/// QubitConnection: Indices that the multi-qubit gate acts on
#[derive(Debug, Clone)]
pub struct QubitConnection(Vec<u32>);

/// BlockQubitConnection: Entanglement for single block
pub struct BlockQubitConnection(Vec<QubitConnection>);

/// LayerEntanglement: Entanglements for all blocks in the layer
pub struct LayerEntanglement(Vec<BlockQubitConnection>);
/// Entanglement: Entanglement for every layer
pub struct Entanglement(Vec<LayerEntanglement>);

/// Enum representing all the possible strategies for n_local entanglement blocks
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
    let parameter_prefix = if parameter_prefix.is_null() {
        "θ".to_string()
    } else {
        // SAFETY: Per documentation, the pointer is non-null and aligned.
        unsafe {
            CStr::from_ptr(parameter_prefix)
                .to_str()
                .expect("Invalid UTF-8 character")
                .to_string()
        }
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
        reps,
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
/// Obtain the layers count in the provided ``QkEntanglement``.
///
/// @param entanglement A pointer to the ``QkEntanglement`` object.
///
/// @return The layers count on the provided ``QkEntanglement``.
///
/// # Example
///
/// ```c
/// int reps = 1;
/// int num_qubits = 2;
/// QkGate entanglement_blocks[1] = {QkGate_CRX};
/// QkEntanglement *entanglement = qk_get_entanglement_with_strategy(
///     num_qubits, reps, QkEntanglementStrategy_Linear, entanglement_blocks, 1);
///
/// size_t layers_count = qk_get_entanglement_layers_quantity(entanglement);
///
/// qk_entanglement_free(entanglement);
/// ```
///
///
/// # Safety
///
/// Behavior is undefined if ``entanglement`` is not either null or a valid pointer to a ``QkEntanglement``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_get_entanglement_layers_quantity(
    entanglement: *const Entanglement,
) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let entanglement = unsafe { const_ptr_as_ref(entanglement) };
    entanglement.0.len()
}

/// @ingroup QkCircuitLibrary
/// Obtain the blocks count on a given layer in the provided ``QkEntanglement``.
///
/// @param entanglement A pointer to the ``QkEntanglement`` object.
/// @param layer_index The layer index where the blocks are.
///
/// @return The blocks count on the given layer.
///
/// # Example
///
/// ```c
/// int reps = 1;
/// int num_qubits = 2;
/// QkGate entanglement_blocks[1] = {QkGate_CRX};
/// QkEntanglement *entanglement = qk_get_entanglement_with_strategy(
///     num_qubits, reps, QkEntanglementStrategy_Linear, entanglement_blocks, 1);
///
/// size_t blocks_count = qk_get_entanglement_layer_blocks_quantity(entanglement, 0);
///
/// qk_entanglement_free(entanglement);
/// ```
///
///
/// # Safety
///
/// Behavior is undefined if ``entanglement`` is not either null or a valid pointer to a ``QkEntanglement``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_get_entanglement_layer_blocks_quantity(
    entanglement: *const Entanglement,
    layer_index: usize,
) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let entanglement = unsafe { const_ptr_as_ref(entanglement) };
    entanglement.0[layer_index].0.len()
}

/// @ingroup QkCircuitLibrary
/// Obtain the ``QkQubitConnection`` count on a given layer and block in the provided ``QkEntanglement``.
///
/// @param entanglement A pointer to the ``QkEntanglement`` object.
/// @param layer_index The layer index where the ``QkQubitConnection`` objects exist.
/// @param block_index The block index in the layer where the ``QkQubitConnection`` objects exist.
///
/// @return The qubit connections count on the given layer and block.
///
/// # Example
///
/// ```c
/// int reps = 1;
/// int num_qubits = 2;
/// QkGate entanglement_blocks[1] = {QkGate_CRX};
/// QkEntanglement *entanglement = qk_get_entanglement_with_strategy(
///     num_qubits, reps, QkEntanglementStrategy_Linear, entanglement_blocks, 1);
///
/// size_t connections_count = qk_get_entanglement_qubit_connections_quantity(entanglement, 0, 0);
///
/// qk_entanglement_free(entanglement);
/// ```
///
///
/// # Safety
///
/// Behavior is undefined if ``entanglement`` is not either null or a valid pointer to a ``QkEntanglement``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_get_entanglement_qubit_connections_quantity(
    entanglement: *const Entanglement,
    layer_index: usize,
    block_index: usize,
) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let entanglement = unsafe { const_ptr_as_ref(entanglement) };
    entanglement.0[layer_index].0[block_index].0.len()
}

/// @ingroup QkCircuitLibrary
/// Obtain the ``QkQubitConnection`` object at a given layer and block in the provided ``QkEntanglement``.
///
/// @param entanglement A pointer to the ``QkEntanglement`` object.
/// @param layer_index The layer index where the ``QkQubitConnection`` object exists.
/// @param block_index The block index in the layer where the ``QkQubitConnection`` object exists.
/// @param connection_index The ``QkQubitConnection`` object index where the multi-qubit gate acts on.
///
/// @return A pointer to the obtained ``QkQubitConnection``.
///
/// # Example
///
/// ```c
/// int reps = 1;
/// int num_qubits = 2;
/// QkGate entanglement_blocks[1] = {QkGate_CRX};
/// QkEntanglement *entanglement = qk_get_entanglement_with_strategy(
///     num_qubits, reps, QkEntanglementStrategy_Linear, entanglement_blocks, 1);
///
/// QkQubitConnection *connection = qk_get_entanglement_qubit_connections(entanglement, 0, 0, 1);
///
/// qk_entanglement_free(entanglement);
/// qk_qubit_connection_free(connection);
/// ```
///
///
/// # Safety
///
/// Behavior is undefined if ``entanglement`` is not either null or a valid pointer to a ``QkEntanglement``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_get_entanglement_qubit_connections(
    entanglement: *const Entanglement,
    layer_index: usize,
    block_index: usize,
    connection_index: usize,
) -> *mut QubitConnection {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let entanglement = unsafe { const_ptr_as_ref(entanglement) };
    Box::into_raw(Box::new(
        entanglement.0[layer_index].0[block_index].0[connection_index].clone(),
    ))
}

/// @ingroup QkCircuitLibrary
/// Construct a new ``QkQubitConnection`` representing the entanglement between qubits where a multi-qubit gate acts on.
///
/// @param num_qubits The size of the qubits connections array.
/// @param qubit_connections The qubits connections array.
///
/// @return A pointer to the created ``QkQubitConnection``.
///
/// # Example
///
/// ```c
/// QkQubitConnection *qconn = qk_qubit_connection_new((uint32_t[2]){0, 1}, 2);
/// ```
///
/// # Safety
///
/// Behavior is undefined ``qubit_connections`` is not a valid, non-null pointer to a ``uint32_t`` array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qubit_connection_new(
    num_qubits: usize,
    qubit_connections: *const u32,
) -> *mut QubitConnection {
    // SAFETY: per documentation, `qubit_connections` is aligned and and points to a valid
    // aligned block of memory valid for `qubit_connections_size` writes of `u32`s
    let qubits: Vec<u32> =
        unsafe { ::std::slice::from_raw_parts(qubit_connections, num_qubits).to_vec() };
    Box::into_raw(Box::new(QubitConnection(qubits)))
}

/// @ingroup QkCircuitLibrary
/// Compare two ``QkQubitConnection`` for equality.
///
/// @param c1 A pointer to the left hand side ``QkQubitConnection``.
/// @param c2 A pointer to the right hand side ``QkQubitConnection``.
///
/// @return ``true`` if the ``QkQubitConnection`` objects are equal, ``false`` otherwise.
///
/// # Example
///
/// ```c
/// QkQubitConnection *qconn = qk_qubit_connection_new((uint32_t[2]){0, 1}, 2);
/// QkQubitConnection *qconn2 = qk_qubit_connection_new((uint32_t[2]){0, 1}, 2);
///
/// bool equal = qk_qubit_connection_equal(qconn, qconn2);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``c1`` or ``c2`` is not a valid, non-null
/// pointer to a ``QkQubitConnection``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qubit_connection_equal(
    c1: *const QubitConnection,
    c2: *const QubitConnection,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let c1 = unsafe { const_ptr_as_ref(c1) };

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let c2 = unsafe { const_ptr_as_ref(c2) };

    c1.0.eq(&c2.0)
}

/// @ingroup QkCircuitLibrary
/// Generate an entanglement following the provided strategies.
///
/// @param num_qubits The number of qubits of the circuit
/// @param reps Specifies how often the entanglement blocks are repeated.
/// @param entanglement_strategy List of enum items describing an entanglement strategy for each layer.
/// @param entanglement_strategy_size Length of the entanglement strategy list provided
/// @param entanglement_blocks The blocks used in the entanglement layers.
/// @param entanglement_blocks_size Length of the list of entanglement blocks provided
///
/// @return A pointer to the created ``QkEntanglement``.
///
/// # Safety
///
/// Behavior is undefined ``entanglement_strategy`` is not a valid, non-null pointer to a ``QkEntanglementStrategy`` array.
/// Behavior is undefined ``entanglement_blocks`` is not a valid, non-null pointer to a ``StandardGate`` array.
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
/// @param reps Specifies how often the entanglement blocks are repeated.
/// @param entanglement_strategy The entanglement strategy to apply for each layer.
/// @param entanglement_blocks The blocks used in the entanglement layers.
/// @param entanglement_blocks_size Length of the list of entanglement blocks provided.
///
/// @return A pointer to the created ``QkEntanglement``.
///
/// # Safety
///
/// Behavior is undefined ``entanglement_blocks`` is not a valid, non-null pointer to a ``QkGate`` array.
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

/// @ingroup QkCircuitLibrary
/// Free the ``QkQubitConnection``.
///
/// @param qubit_connection A pointer to the ``QkQubitConnection`` to free.
///
/// # Example
///
/// ```c
/// QkQubitConnection *qconn = qk_qubit_connection_new((uint32_t[2]){0, 1}, 2);
/// qk_qubit_connection_free(qconn);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``qubit_connection`` is not either null or a valid pointer to a ``QubitConnection``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qubit_connection_free(qubit_connection: *mut QubitConnection) {
    if !qubit_connection.is_null() {
        if !qubit_connection.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(qubit_connection);
        }
    }
}

/// @ingroup QkCircuitLibrary
/// Free the ``QkEntanglement``.
///
/// @param entanglement A pointer to the ``QkEntanglement`` to free.
///
/// # Example
///
/// ```c
/// int reps = 1;
/// int num_qubits = 2;
/// QkGate entanglement_blocks[1] = {QkGate_CRX};
/// QkEntanglement *entanglement = qk_get_entanglement_with_strategy(
///     num_qubits, reps, QkEntanglementStrategy_Linear, entanglement_blocks, 1);
/// qk_entanglement_free(entanglement);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``entanglement`` is not either null or a valid pointer to a ``QkEntanglement``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_entanglement_free(entanglement: *mut Entanglement) {
    if !entanglement.is_null() {
        if !entanglement.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(entanglement);
        }
    }
}

fn get_entanglement_with_strategy(
    num_qubits: u32,
    entanglement_blocks: &[&StandardGate],
    entanglement_strategy: &[&str],
    reps: usize,
) -> Entanglement {
    Entanglement(
        (0..reps)
            .map(|layer| {
                get_layer_entanglement_with_strategy(
                    num_qubits,
                    entanglement_blocks,
                    entanglement_strategy,
                    layer,
                )
            })
            .collect(),
    )
}

fn get_layer_entanglement_with_strategy(
    num_qubits: u32,
    entanglement_blocks: &[&StandardGate],
    entanglement_strategy: &[&str],
    layer_index: usize,
) -> LayerEntanglement {
    LayerEntanglement(
        entanglement_blocks
            .iter()
            .zip(entanglement_strategy.iter())
            .map(|(gate, &strategy)| {
                get_block_qubit_connections_with_strategy(
                    num_qubits,
                    gate,
                    strategy,
                    (strategy == "sca").then_some(layer_index),
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
