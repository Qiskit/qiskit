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

fn dump(circuits: Vec<CircuitData>, version: Option<u8>) -> Result<Vec<u8>, ()> {
    #[cfg(feature = "python_binding")]
    // in order to clone a circuit which contains python native data, we need the GIL
    let result = pyo3::Python::attach(|_| qiskit_qpy::native_dump_qpy(circuits, version));
    #[cfg(not(feature = "python_binding"))]
    let result = qiskit_qpy::native_dump_qpy(circuits, version);
    result.map(|payload| payload.to_vec()).map_err(|_| ())
}

fn load(payload: &[u8]) -> Result<Vec<CircuitData>, ()> {
    qiskit_qpy::native_load_qpy(payload).map_err(|_| ())
}

unsafe fn clone_circuits(
    circuits: *const *const CircuitData,
    num_circuits: usize,
) -> Result<Vec<CircuitData>, ExitCode> {
    check_ptr(circuits).map_err(ExitCode::from)?;
    // SAFETY: upheld by the caller contract and checked non-null above.
    let circuits = unsafe { slice::from_raw_parts(circuits, num_circuits) };
    circuits
        .iter()
        .map(|&circuit| {
            check_ptr(circuit).map_err(ExitCode::from)?;
            // SAFETY: upheld by the caller contract and checked for alignment/null above.
            Ok(unsafe { &*circuit }.clone())
        })
        .collect()
}

unsafe fn dump_file_impl(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    filename: *const c_char,
    version: Option<u8>,
) -> ExitCode {
    // SAFETY: this function's caller upholds the circuit-array contract.
    let circuits = match unsafe { clone_circuits(circuits, num_circuits) } {
        Ok(circuits) => circuits,
        Err(error) => return error,
    };
    if filename.is_null() {
        return ExitCode::NullPointerError;
    }
    // SAFETY: upheld by the caller contract and checked non-null above.
    let Ok(filename) = unsafe { CStr::from_ptr(filename) }.to_str() else {
        return ExitCode::QpyError;
    };
    // SAFETY: upheld by the caller contract and checked for alignment/null above.
    let result = dump(circuits, version);
    match result.and_then(|payload| fs::write(filename, payload).map_err(|_| ())) {
        Ok(()) => ExitCode::Success,
        Err(_) => ExitCode::QpyError,
    }
}

unsafe fn dump_buffer_impl(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    buffer: *mut *mut u8,
    size: *mut usize,
    version: Option<u8>,
) -> ExitCode {
    // SAFETY: this function's caller upholds the circuit-array contract.
    let circuits = match unsafe { clone_circuits(circuits, num_circuits) } {
        Ok(circuits) => circuits,
        Err(error) => return error,
    };
    if let Err(error) = check_ptr(buffer) {
        return error.into();
    }
    if let Err(error) = check_ptr(size) {
        return error.into();
    }
    // SAFETY: upheld by the caller contract and checked for alignment/null above.
    match dump(circuits, version) {
        Ok(payload) => {
            let mut payload = payload.into_boxed_slice();
            let payload_size = payload.len();
            let payload_ptr = payload.as_mut_ptr();
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

/// @ingroup QkQpy
/// Write circuits to a QPY file.
///
/// The circuits are copied before serialization and remain owned by the caller.
///
/// @param circuits A valid, non-null pointer to an array of ``num_circuits`` circuit pointers.
/// @param num_circuits The number of circuit pointers in ``circuits``.
/// @param filename A valid, non-null, nul-terminated UTF-8 path.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` for an invalid path, unsupported QPY data, or I/O failure.
///
/// # Safety
/// Every element of ``circuits`` must point to a valid ``QkCircuit`` and ``filename`` must point
/// to a valid nul-terminated string for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_file(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    filename: *const c_char,
) -> ExitCode {
    // SAFETY: this function has the same pointer requirements as the implementation.
    unsafe { dump_file_impl(circuits, num_circuits, filename, None) }
}

/// @ingroup QkQpy
/// Write circuits to a QPY file using a specific format version.
///
/// @param circuits A valid, non-null pointer to an array of ``num_circuits`` circuit pointers.
/// @param num_circuits The number of circuit pointers in ``circuits``.
/// @param filename A valid, non-null, nul-terminated UTF-8 path.
/// @param version The QPY format version. Rust QPY writing currently supports version 17 or later.
/// @return The same exit codes as ``qk_qpy_dump_file``.
///
/// # Safety
/// The pointer requirements are the same as for ``qk_qpy_dump_file``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_file_with_version(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    filename: *const c_char,
    version: u8,
) -> ExitCode {
    // SAFETY: this function has the same pointer requirements as the implementation.
    unsafe { dump_file_impl(circuits, num_circuits, filename, Some(version)) }
}

/// @ingroup QkQpy
/// Load all circuits from a QPY file.
///
/// @param filename A valid, non-null, nul-terminated UTF-8 path.
/// @param circuits Output location for a newly allocated circuit-pointer array.
/// @param num_circuits Output location for the number of circuits in ``circuits``.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` if the file cannot be read or is not supported QPY.
/// The returned array and all its circuits must be released with
/// ``qk_qpy_free_circuits``.
///
/// # Safety
/// ``filename`` must point to a valid nul-terminated string; ``circuits`` and ``num_circuits`` must
/// be valid for one write for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_load_file(
    circuits: *mut *mut *mut CircuitData,
    num_circuits: *mut usize,
    filename: *const c_char,
) -> ExitCode {
    if filename.is_null() {
        return ExitCode::NullPointerError;
    }
    if let Err(error) = check_ptr(circuits) {
        return error.into();
    }
    if let Err(error) = check_ptr(num_circuits) {
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
            let mut loaded: Box<[*mut CircuitData]> = loaded
                .into_iter()
                .map(|circuit| Box::into_raw(Box::new(circuit)))
                .collect();
            let len = loaded.len();
            let ptr = loaded.as_mut_ptr();
            std::mem::forget(loaded);
            // SAFETY: the caller guarantees the output locations are writable.
            unsafe {
                circuits.write(ptr);
                num_circuits.write(len);
            }
            ExitCode::Success
        }
        Err(()) => ExitCode::QpyError,
    }
}

/// @ingroup QkQpy
/// Serialize circuits into a newly allocated QPY buffer.
///
/// @param circuits A valid, non-null pointer to an array of ``num_circuits`` circuit pointers.
/// @param num_circuits The number of circuit pointers in ``circuits``.
/// @param buffer Output location for the newly allocated buffer. It is unchanged on failure.
/// @param size Output location for the buffer size in bytes. It is unchanged on failure.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` if serialization fails. The returned buffer must be
/// released with ``qk_qpy_free_buffer``.
///
/// # Safety
/// Every element of ``circuits`` must point to a valid ``QkCircuit``. ``buffer`` and ``size`` must
/// each be valid for one pointer-sized write for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_buffer(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    buffer: *mut *mut u8,
    size: *mut usize,
) -> ExitCode {
    // SAFETY: this function has the same pointer requirements as the implementation.
    unsafe { dump_buffer_impl(circuits, num_circuits, buffer, size, None) }
}

/// @ingroup QkQpy
/// Serialize circuits into a newly allocated QPY buffer using a specific format version.
///
/// @param circuits A valid, non-null pointer to an array of ``num_circuits`` circuit pointers.
/// @param num_circuits The number of circuit pointers in ``circuits``.
/// @param buffer Output location for the newly allocated buffer. It is unchanged on failure.
/// @param size Output location for the buffer size in bytes. It is unchanged on failure.
/// @param version The QPY format version. Rust QPY writing currently supports version 17 or later.
/// @return The same exit codes as ``qk_qpy_dump_buffer``.
///
/// # Safety
/// The pointer requirements are the same as for ``qk_qpy_dump_buffer``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_buffer_with_version(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    buffer: *mut *mut u8,
    size: *mut usize,
    version: u8,
) -> ExitCode {
    // SAFETY: this function has the same pointer requirements as the implementation.
    unsafe { dump_buffer_impl(circuits, num_circuits, buffer, size, Some(version)) }
}

/// @ingroup QkQpy
/// Load all circuits from a QPY buffer.
///
/// @param buffer A valid buffer containing ``size`` bytes of QPY data.
/// @param size The size of ``buffer`` in bytes.
/// @param circuits Output location for a newly allocated circuit-pointer array.
/// @param num_circuits Output location for the number of circuits in ``circuits``.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` if the buffer is not supported QPY. The returned array and
/// all its circuits must be released with ``qk_qpy_free_circuits``.
///
/// # Safety
/// ``buffer`` must be valid for reads of ``size`` bytes; ``circuits`` and ``num_circuits`` must be
/// valid for one write for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_load_buffer(
    circuits: *mut *mut *mut CircuitData,
    num_circuits: *mut usize,
    buffer: *const u8,
    size: usize,
) -> ExitCode {
    if buffer.is_null() {
        return ExitCode::NullPointerError;
    }
    if let Err(error) = check_ptr(circuits) {
        return error.into();
    }
    if let Err(error) = check_ptr(num_circuits) {
        return error.into();
    }
    // SAFETY: upheld by the caller contract and checked non-null above.
    match load(unsafe { slice::from_raw_parts(buffer.cast::<u8>(), size) }) {
        Ok(loaded) => {
            let mut loaded: Box<[*mut CircuitData]> = loaded
                .into_iter()
                .map(|circuit| Box::into_raw(Box::new(circuit)))
                .collect();
            let len = loaded.len();
            let ptr = loaded.as_mut_ptr();
            std::mem::forget(loaded);
            // SAFETY: the caller guarantees the output locations are writable.
            unsafe {
                circuits.write(ptr);
                num_circuits.write(len);
            }
            ExitCode::Success
        }
        Err(()) => ExitCode::QpyError,
    }
}

/// @ingroup QkQpy
/// Free circuits returned by ``qk_qpy_load_file`` or ``qk_qpy_load_buffer``.
///
/// # Safety
/// ``circuits`` must be null, or have been returned by a QPY load function with exactly
/// ``num_circuits`` elements and not previously freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_free_circuits(
    circuits: *mut *mut CircuitData,
    num_circuits: usize,
) {
    if !circuits.is_null() {
        // SAFETY: upheld by the caller contract.
        let circuits =
            unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(circuits, num_circuits)) };
        for circuit in circuits.iter().copied() {
            if !circuit.is_null() {
                // SAFETY: each pointer was allocated by a QPY load function.
                unsafe { drop(Box::from_raw(circuit)) };
            }
        }
    }
}

/// @ingroup QkQpy
/// Free a buffer returned by ``qk_qpy_dump_buffer``.
///
/// @param buffer A buffer returned by ``qk_qpy_dump_buffer``, or null.
/// @param size The buffer size returned by ``qk_qpy_dump_buffer``.
///
/// # Safety
/// A non-null ``buffer`` must have been returned by ``qk_qpy_dump_buffer`` with exactly this
/// ``size`` and must not have been freed previously.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_free_buffer(buffer: *mut u8, size: usize) {
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
