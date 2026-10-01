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

use std::ffi::{CString, c_char};

use qiskit_circuit::{circuit_data::CircuitData, circuit_drawer::draw_circuit};

use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

/// The configuration options for the [`qk_circuit_draw`] function.
///
/// Create one with [`qk_circuit_drawer_config_new`], modify it with the setters below, and release
/// it with [`qk_circuit_drawer_config_free`].
pub struct CircuitDrawerConfig {
    /// If `true`, bundles classical registers into single wires.
    bundle_cregs: bool,
    /// If `true`, merges the bottom and top lines of adjacent wires.
    merge_wires: bool,
    /// Sets the line length for wrapping the rendered text. Use 0
    /// to auto-detect console width. Use `SIZE_MAX` to effectively skip
    /// wrapping altogether.
    fold: usize,
    /// Sets the number of characters to display for barrier labels. If
    /// this number is exceeded, the label is truncated at that number and
    /// '...' is appended. Use 0 to apply the default of 16 characters.
    barrier_label_len: usize,
}

impl Default for CircuitDrawerConfig {
    fn default() -> Self {
        Self {
            bundle_cregs: true,
            merge_wires: true,
            fold: 0,
            barrier_label_len: 0,
        }
    }
}

/// @ingroup QkCircuitDrawerConfig
/// Construct a new circuit-drawer configuration with the default options.
///
/// The defaults are:
///
/// * ``bundle_cregs = true``
/// * ``merge_wires = true``
/// * ``fold = 0`` (auto-detect the console width)
/// * ``barrier_label_len = 0`` (use the drawer default of 16 characters)
///
/// Call `qk_circuit_drawer_config_free` with the return value to release the memory when done.
///
/// @return A pointer to the created configuration.
///
/// # Example
/// ```c
/// QkCircuitDrawerConfig *config = qk_circuit_drawer_config_new();
/// qk_circuit_drawer_config_free(config);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn qk_circuit_drawer_config_new() -> *mut CircuitDrawerConfig {
    Box::into_raw(Box::new(CircuitDrawerConfig::default()))
}

/// @ingroup QkCircuitDrawerConfig
/// Free a circuit-drawer configuration.
///
/// @param config A pointer to the configuration to free.
///
/// # Example
/// ```c
/// QkCircuitDrawerConfig *config = qk_circuit_drawer_config_new();
/// qk_circuit_drawer_config_free(config);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``config`` is not either null or a valid pointer to a
/// :c:struct:`QkCircuitDrawerConfig`, or if this function is called more than once on the same
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_drawer_config_free(config: *mut CircuitDrawerConfig) {
    if !config.is_null() {
        if !config.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so it should be readable
        // by Box.
        unsafe {
            let _ = Box::from_raw(config);
        }
    }
}

/// @ingroup QkCircuitDrawerConfig
/// Set whether to bundle classical registers into single wires.
///
/// @param config A pointer to the configuration to update.
/// @param bundle_cregs If ``true``, each classical register is drawn as a single bundled wire
///     rather than one wire per bit.
///
/// # Example
/// ```c
/// QkCircuitDrawerConfig *config = qk_circuit_drawer_config_new();
/// qk_circuit_drawer_config_set_bundle_cregs(config, false);
/// qk_circuit_drawer_config_free(config);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``config`` is not a valid non-null pointer to a
/// :c:struct:`QkCircuitDrawerConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_drawer_config_set_bundle_cregs(
    config: *mut CircuitDrawerConfig,
    bundle_cregs: bool,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let config = unsafe { mut_ptr_as_ref(config) };
    config.bundle_cregs = bundle_cregs;
}

/// @ingroup QkCircuitDrawerConfig
/// Set whether to merge the bottom and top lines of adjacent wires.
///
/// @param config A pointer to the configuration to update.
/// @param merge_wires If ``true``, the bottom and top lines of adjacent wires are merged, giving
///     a more compact drawing.
///
/// # Example
/// ```c
/// QkCircuitDrawerConfig *config = qk_circuit_drawer_config_new();
/// qk_circuit_drawer_config_set_merge_wires(config, false);
/// qk_circuit_drawer_config_free(config);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``config`` is not a valid non-null pointer pointer to a
/// :c:struct:`QkCircuitDrawerConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_drawer_config_set_merge_wires(
    config: *mut CircuitDrawerConfig,
    merge_wires: bool,
) {
    // SAFETY: Per documentation, config is a valid pointer to QkCircuitDrawerConfig.
    let config = unsafe { mut_ptr_as_ref(config) };
    config.merge_wires = merge_wires;
}

/// @ingroup QkCircuitDrawerConfig
/// Set the line length used to wrap the rendered text.
///
/// @param config A pointer to the configuration to update.
/// @param fold The line length to wrap at.  Use 0 to auto-detect the console width, and
///     ``SIZE_MAX`` to effectively disable wrapping altogether.
///
/// # Example
/// ```c
/// QkCircuitDrawerConfig *config = qk_circuit_drawer_config_new();
/// qk_circuit_drawer_config_set_fold(config, 120);
/// qk_circuit_drawer_config_free(config);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``config`` is not a valid, non-null pointer to a
/// :c:struct:`QkCircuitDrawerConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_drawer_config_set_fold(
    config: *mut CircuitDrawerConfig,
    fold: usize,
) {
    // SAFETY: Per documentation, config is a valid pointer to QkCircuitDrawerConfig.
    let config = unsafe { mut_ptr_as_ref(config) };
    config.fold = fold;
}

/// @ingroup QkCircuitDrawerConfig
/// Set the number of characters to display for barrier labels.
///
/// @param config A pointer to the configuration to update.
/// @param barrier_label_len The maximum number of characters to show.  If a label is longer than
///     this, it is truncated and ``...`` is appended.  Use 0 to apply the drawer's default of 16
///     characters.
///
/// # Example
/// ```c
/// QkCircuitDrawerConfig *config = qk_circuit_drawer_config_new();
/// qk_circuit_drawer_config_set_barrier_label_len(config, 8);
/// qk_circuit_drawer_config_free(config);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``config`` is not a valid non-null pointer to a
/// :c:struct:`QkCircuitDrawerConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_drawer_config_set_barrier_label_len(
    config: *mut CircuitDrawerConfig,
    barrier_label_len: usize,
) {
    // SAFETY: Per documentation, config is a valid pointer to QkCircuitDrawerConfig.
    let config = unsafe { mut_ptr_as_ref(config) };
    config.barrier_label_len = barrier_label_len;
}

/// @ingroup QkCircuit
/// Draw the circuit as text.
///
/// @param circuit A pointer to the circuit to draw.
/// @param config A pointer to a :c:struct:`QkCircuitDrawerConfig`, or ``NULL`` to use the default
///     options.
///     See `qk_circuit_drawer_config_new` for what those defaults are.
///
/// @return A pointer to a null-terminated string containing the circuit representation.
///     You must use `qk_str_free` to release the allocated memory when done.
///
/// # Example
/// ```c
/// QkCircuit *circuit = qk_circuit_new(2, 1);
///
/// qk_circuit_gate(circuit, QkGate_H, (uint32_t[]){0}, NULL);
/// qk_circuit_gate(circuit, QkGate_CX, (uint32_t[]){0, 1}, NULL);
/// qk_circuit_measure(circuit, 0, 0);
/// qk_circuit_measure(circuit, 1, 0);
///
/// QkCircuitDrawerConfig *config = qk_circuit_drawer_config_new();
/// qk_circuit_drawer_config_set_bundle_cregs(config, false);
///
/// char *circ_str = qk_circuit_draw(circuit, config);
///
/// printf("%s", circ_str);
///
/// qk_str_free(circ_str);
/// qk_circuit_drawer_config_free(config);
/// qk_circuit_free(circuit);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``, or
/// if ``config`` is not ``NULL`` and not a valid pointer to a :c:struct:`QkCircuitDrawerConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_draw(
    circuit: *const CircuitData,
    config: *const CircuitDrawerConfig,
) -> *mut c_char {
    // SAFETY: Per documentation, circuit is a valid pointer to a QkCircuit object.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    // The temporary is lifetime-extended to the end of the enclosing block by the `let`.
    let config = if config.is_null() {
        &CircuitDrawerConfig::default()
    } else {
        // SAFETY: Per documentation, config is a valid pointer to QkCircuitDrawerConfig object.
        unsafe { const_ptr_as_ref(config) }
    };

    let circuit_str = draw_circuit(
        circuit,
        config.bundle_cregs,
        config.merge_wires,
        (config.fold != 0).then_some(config.fold),
        config.barrier_label_len,
    )
    .unwrap();

    CString::new(circuit_str).unwrap().into_raw()
}
