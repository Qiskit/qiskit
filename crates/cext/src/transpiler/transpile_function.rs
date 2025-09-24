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
use std::ffi::c_char;
use std::ffi::CString;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_transpiler::target::Target;

use qiskit_transpiler::transpile;
use qiskit_transpiler::transpile_layout::TranspileLayout;

use crate::exit_codes::ExitCode;
use crate::pointers::const_ptr_as_ref;

/// The container result object from ``qk_transpile``
///
/// When the transpiler successfully compiles a quantum circuit for a given target it
/// returns the transpiled circuit and the layout. The ``qk_transpile`` function will
/// write pointers to the fields in this struct when it successfully executes, you can
/// initialize this struct with null pointers or leave them unset as the values are never
/// read by ``qk_transpile`` and only written to. After calling ``qk_transpile`` you are
/// responsible for calling ``qk_circuit_free`` and ``qk_transpile_layout_free`` on the
/// members of this struct.
#[repr(C)]
pub struct TranspileResult {
    circuit: *mut CircuitData,
    layout: *mut TranspileLayout,
}

/// The options for running the transpiler
#[repr(C)]
pub struct TranspileOptions {
    /// The optimization level to run the transpiler with. Valid values are 0, 1, 2, or 3.
    optimization_level: u8,
    /// The seed for the transpiler. If set to a negative number this means no seed will be
    /// set and the RNGs used in the transpiler will be seeded from system entropy.
    seed: i64,
    /// The approximation degree a heurstic dial where 1.0 means no approximation (up to numerical
    /// tolerance) and 0.0 means the maximum approximation. A `NAN` value indicates that
    /// approximation is allowed up to the reported error rate for an operation in the target.
    approximation_degree: f64,
}

impl Default for TranspileOptions {
    fn default() -> Self {
        TranspileOptions {
            optimization_level: 2,
            seed: -1,
            approximation_degree: 1.0,
        }
    }
}

/// @ingroup QkTranspiler
///
/// Generate transpiler options defaults
///
/// This function generates a QkTranspileOptions with the default settings
/// This currently is ``optimization_level`` 2, no seed, and no approximation.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_transpiler_default_options() -> TranspileOptions {
    TranspileOptions::default()
}

/// @ingroup QkTranspiler
/// Transpile a single circuit.
///
/// The Qiskit transpiler is a quantum circuit compiler that rewrites a given
/// input circuit to match the constraints of a QPU and optimizes the circuit
/// for execution. This function should only be used with circuits constructed
/// using Qiskit's C API. It makes assumptions on the circuit only using features exposed via C,
/// if you are in a mixed Python and C environment it is typically better to invoke the transpiler
/// via Python.
///
/// This function is multithreaded internally and will launch a thread pool
/// with threads equal to the number of CPUs reported by the operating system by default.
/// This will include logical cores on CPUs with simultaneous multithreading. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// @param circuit A pointer to the circuit to run the transpiler on.
/// @param target A pointer to the target to compile the circuit for.
/// @param options A pointer to an options object that defines user options. If this is a null
///   pointer the default values will be used. See ``qk_transpile_default_options``
///   for more details on the default values.
/// @param result A pointer to the memory location of the transpiler result. On a successful
///   execution (return code 0) the output of the transpiler will be written to the pointer. The
///   members of the result struct are owned by the caller and you are responsible for freeing
///   the members using the respective free functions.
/// @param error A pointer to a pointer with an nul terminated string with an error description.
///   If the transpiler fails a pointer to the string with the error description will be written
///   to this pointer. That pointer needs to be freed with ``qk_str_free```. This can be a null
///   pointer in which case the error will not be written out.
///
/// @returns The return code for the transpiler, ``QkExitCode_Success`` means success and all
///   other values indicate an error.
///
/// # Safety
///
/// Behavior is undefined if ``circuit``, ``target``, or ``result``, are not valid, non-null
/// pointers to a ``QkCircuit``, ``QkTarget``, or ``QkTranspileResult`` respectively.
/// ``options`` must be a valid pointer a to a ``QkTranspileOptions`` or ``NULL`.
/// ``error`` must be a valid pointer to a ``char`` pointer or ``NULL``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpile(
    qc: *const CircuitData,
    target: *const Target,
    options: *const TranspileOptions,
    result: *mut TranspileResult,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let qc = unsafe { const_ptr_as_ref(qc) };
    let target = unsafe { const_ptr_as_ref(target) };
    let options = if options.is_null() {
        &TranspileOptions::default()
    } else {
        // SAFETY: We checked the pointer is not null, then, per documentation, it is a valid
        // and aligned pointer.
        unsafe { const_ptr_as_ref(options) }
    };

    if !(0..=3u8).contains(&options.optimization_level) {
        panic!(
            "Invalid optimization level specified {}",
            options.optimization_level
        );
    }

    let seed = if options.seed < 0 {
        None
    } else {
        Some(options.seed as u64)
    };
    let approximation_degree = if options.approximation_degree.is_nan() {
        None
    } else {
        if !(0.0..=1.0).contains(&options.approximation_degree) {
            panic!(
                "Invalid value provided for approximation degree, only NAN or values between 0.0 and 1.0 inclusive are valid"
            );
        }
        Some(options.approximation_degree)
    };

    if let Some(target_qubits) = target.num_qubits {
        if target_qubits < qc.num_qubits() as u32 {
            if !error.is_null() {
                unsafe {
                    *error = CString::new(format!(
                        "Insufficient qubits in target: {}, the circuit uses {}",
                        target_qubits,
                        qc.num_qubits()
                    ))
                    .unwrap()
                    .into_raw();
                }
            }
            return ExitCode::TranspilerError;
        }
    }

    match transpile(
        qc,
        target,
        options.optimization_level.into(),
        approximation_degree,
        seed,
    ) {
        Ok(transpile_result) => {
            unsafe {
                *result = TranspileResult {
                    circuit: Box::into_raw(Box::new(transpile_result.0)),
                    layout: Box::into_raw(Box::new(transpile_result.1)),
                };
            }
            ExitCode::Success
        }
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    // Right now we return a backtrace of the error. This at least gives a hint as to
                    // which pass failed when we have rust errors normalized we can actually have error
                    // messages which are user facing. But most likely this will be a PyErr and panic
                    // when trying to extract the string.
                    *error = CString::new(format!(
                        "Transpilation failed with this backtrace: {}",
                        e.backtrace()
                    ))
                    .unwrap()
                    .into_raw();
                }
            }
            ExitCode::TranspilerError
        }
    }
}
