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

use std::ffi::{CStr, c_char};
use std::fs;

use qiskit_circuit::circuit_data::CircuitData;

use crate::exit_codes::ExitCode;
use crate::pointers::check_ptr;

/// Write one circuit to a QPY file.
///
/// The circuit is copied before serialization and remains owned by the caller.
///
/// @param circuit A valid, non-null circuit pointer.
/// @param filename A valid, non-null, nul-terminated UTF-8 path.
/// @param version The QPY format version. Rust QPY writing currently supports version 17 or later.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` for an invalid path, unsupported QPY data, or I/O failure.
///
/// # Safety
/// ``circuit`` must point to a valid ``QkCircuit`` and ``filename`` must point to a valid
/// nul-terminated string for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_file(
    circuit: *const CircuitData,
    filename: *const c_char,
    version: u8,
) -> ExitCode {
    if let Err(error) = check_ptr(circuit) {
        return error.into();
    }
    if filename.is_null() {
        return ExitCode::NullPointerError;
    }
    // SAFETY: upheld by the caller contract and checked non-null above.
    let Ok(filename) = unsafe { CStr::from_ptr(filename) }.to_str() else {
        return ExitCode::QpyError;
    };
    // SAFETY: upheld by the caller contract and checked for alignment/null above.
    let circuit = unsafe { &*circuit }.clone();
    match qiskit_qpy::native_dump_qpy(vec![circuit], version)
        .and_then(|payload| fs::write(filename, payload).map_err(Into::into))
    {
        Ok(()) => ExitCode::Success,
        Err(_) => ExitCode::QpyError,
    }
}

/// Load the first circuit from a QPY file.
///
/// @param filename A valid, non-null, nul-terminated UTF-8 path.
/// @param circuit Output location for the newly allocated circuit. It is unchanged on failure.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` if the file cannot be read, contains no circuits, or is not
/// supported QPY. The returned circuit must be released with ``qk_circuit_free``.
///
/// # Safety
/// ``filename`` must point to a valid nul-terminated string and ``circuit`` must be valid for one
/// pointer write for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_load_file(
    filename: *const c_char,
    circuit: *mut *mut CircuitData,
) -> ExitCode {
    if filename.is_null() {
        return ExitCode::NullPointerError;
    }
    if let Err(error) = check_ptr(circuit) {
        return error.into();
    }
    // SAFETY: upheld by the caller contract and checked non-null above.
    let Ok(filename) = unsafe { CStr::from_ptr(filename) }.to_str() else {
        return ExitCode::QpyError;
    };
    let result = fs::read(filename)
        .map_err(|_| ())
        .and_then(|payload| qiskit_qpy::native_load_qpy(&payload).map_err(|_| ()))
        .and_then(|mut circuits| {
            if circuits.is_empty() {
                Err(())
            } else {
                Ok(circuits.remove(0))
            }
        });
    match result {
        Ok(loaded) => {
            // SAFETY: the caller guarantees the output location is writable.
            unsafe { *circuit = Box::into_raw(Box::new(loaded)) };
            ExitCode::Success
        }
        Err(()) => ExitCode::QpyError,
    }
}
