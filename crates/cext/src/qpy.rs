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
use std::slice;

use qiskit_circuit::circuit_data::CircuitData;

use crate::exit_codes::ExitCode;
use crate::pointers::check_ptr;

fn dump(circuit: &CircuitData, version: u8) -> Result<Vec<u8>, ()> {
    #[cfg(feature = "python_binding")]
    let result =
        pyo3::Python::attach(|_| qiskit_qpy::native_dump_qpy(vec![circuit.clone()], version));
    #[cfg(not(feature = "python_binding"))]
    let result = qiskit_qpy::native_dump_qpy(vec![circuit.clone()], version);
    result.map(|payload| payload.to_vec()).map_err(|_| ())
}

fn load(payload: &[u8]) -> Result<CircuitData, ()> {
    qiskit_qpy::native_load_qpy(payload)
        .map_err(|_| ())
        .and_then(|mut circuits| {
            if circuits.is_empty() {
                Err(())
            } else {
                Ok(circuits.remove(0))
            }
        })
}

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
    let result = dump(unsafe { &*circuit }, version);
    match result.and_then(|payload| fs::write(filename, payload).map_err(|_| ())) {
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
        .and_then(|payload| load(&payload));
    match result {
        Ok(loaded) => {
            // SAFETY: the caller guarantees the output location is writable.
            unsafe { *circuit = Box::into_raw(Box::new(loaded)) };
            ExitCode::Success
        }
        Err(()) => ExitCode::QpyError,
    }
}

/// Serialize one circuit into a newly allocated QPY buffer.
///
/// @param circuit A valid, non-null circuit pointer.
/// @param buffer Output location for the newly allocated buffer. It is unchanged on failure.
/// @param size Output location for the buffer size in bytes. It is unchanged on failure.
/// @param version The QPY format version. Rust QPY writing currently supports version 17 or later.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` if serialization fails. The returned buffer must be
/// released with ``qk_qpy_free_buffer``.
///
/// # Safety
/// ``circuit`` must point to a valid ``QkCircuit``. ``buffer`` and ``size`` must each be valid
/// for one pointer-sized write for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_buffer(
    circuit: *const CircuitData,
    buffer: *mut *mut c_char,
    size: *mut usize,
    version: u8,
) -> ExitCode {
    if let Err(error) = check_ptr(circuit) {
        return error.into();
    }
    if let Err(error) = check_ptr(buffer) {
        return error.into();
    }
    if let Err(error) = check_ptr(size) {
        return error.into();
    }
    // SAFETY: upheld by the caller contract and checked for alignment/null above.
    match dump(unsafe { &*circuit }, version) {
        Ok(payload) => {
            let mut payload = payload.into_boxed_slice();
            let payload_size = payload.len();
            let payload_ptr = payload.as_mut_ptr().cast::<c_char>();
            std::mem::forget(payload);
            // SAFETY: the caller guarantees both output locations are writable.
            unsafe {
                buffer.write(payload_ptr);
                size.write(payload_size);
            }
            ExitCode::Success
        }
        Err(()) => ExitCode::QpyError,
    }
}

/// Load the first circuit from a QPY buffer.
///
/// @param buffer A valid buffer containing ``size`` bytes of QPY data.
/// @param size The size of ``buffer`` in bytes.
/// @param circuit Output location for the newly allocated circuit. It is unchanged on failure.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` if the buffer contains no circuits or is not supported QPY.
/// The returned circuit must be released with ``qk_circuit_free``.
///
/// # Safety
/// ``buffer`` must be valid for reads of ``size`` bytes and ``circuit`` must be valid for one
/// pointer write for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_load_buffer(
    buffer: *const c_char,
    size: usize,
    circuit: *mut *mut CircuitData,
) -> ExitCode {
    if buffer.is_null() {
        return ExitCode::NullPointerError;
    }
    if let Err(error) = check_ptr(circuit) {
        return error.into();
    }
    // SAFETY: upheld by the caller contract and checked non-null above.
    match load(unsafe { slice::from_raw_parts(buffer.cast::<u8>(), size) }) {
        Ok(loaded) => {
            // SAFETY: the caller guarantees the output location is writable.
            unsafe { circuit.write(Box::into_raw(Box::new(loaded))) };
            ExitCode::Success
        }
        Err(()) => ExitCode::QpyError,
    }
}

/// Free a buffer returned by ``qk_qpy_dump_buffer``.
///
/// @param buffer A buffer returned by ``qk_qpy_dump_buffer``, or null.
/// @param size The buffer size returned by ``qk_qpy_dump_buffer``.
///
/// # Safety
/// A non-null ``buffer`` must have been returned by ``qk_qpy_dump_buffer`` with exactly this
/// ``size`` and must not have been freed previously.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_free_buffer(buffer: *mut c_char, size: usize) {
    if !buffer.is_null() {
        // SAFETY: upheld by the caller contract; dump_buffer allocated this as a boxed slice.
        unsafe {
            drop(Box::from_raw(slice::from_raw_parts_mut(
                buffer.cast::<u8>(),
                size,
            )));
        }
    }
}
