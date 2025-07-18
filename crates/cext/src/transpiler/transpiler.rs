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

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::target::Target;

use qiskit_transpiler::{transpile, TranspileLayout};

/// @ingroup QkTranspiler
/// The container result object from ``qk_transpile``
///
/// When the transpiler successfully compiles a quantum circuit for a given target it
/// returns the transpiled circuit and the layout.
#[repr(C)]
struct TranspileResult {
    circuit: *mut CircuitData,
    layout: *mut TranspileLayout,
}

/// The options for running the transpiler
#[repr(C)]
struct TranspileOptions {
    /// The optimization level to run the transpiler with
    optimization_level: i8,
    /// The seed for the transpiler
    seed: i64,
    /// The approximation degree a heurstic dial where 1.0 means no approximation (up to numerical
    /// tolerance) and 0.0 means the maximum approximation. A `NAN` value indicates
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
/// Transpile a single circuit that was constructed using the C API
///
/// The Qiskit transpiler is a quantum circuit compiler that rewrites a given
/// input circuit to match the constraints of a QPU and/or optimize the circuit
/// for execution
/// @param circuit A pointer to the circuit to run the transpiler on
/// @param target A pointer to the target to compile the circuit for
/// @params options A pointer to an options object that define user options if this is a null
///   pointer the default values will be used
/// @param result A pointer to the memory location of the transpiler result. On a successful
///   execution (return code 0) the output of the transpiler will be written to the pointer
/// @param error A pointer to a pointer with an nul terminated string with an error description.
///   If the transpiler fails a pointer to the string with the error description will be written
///   to this pointer. That pointer needs to be freed with `qk_str_free`.
///
/// @returns the return code for the transpiler, 0 means success and all other values indicate an
///   error
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpile(
    qc: *const CiruitData,
    target: *const Target,
    options: *const TranspileOptions,
    result: *mut TranspileResult,
    error: *mut *mut c_char,
) -> i32 {
    if ![0, 1, 2, 3].contains(options.optimization_level) {
        panic!("Invalid optimization level specified {optimization_level}");
    }
    let seed = if options.seed < 0 {
        None
    } else {
        seed = Some(seed as u64)
    };
    if approximation_degree.is_nan() {
        None
    } else {
        if !(0.0..=1.0).contains(&approximation_degree) {
            panic!("Invalid value provided for approximation degree, only NAN or values between 0.0 and 1.0 inclusive are valid");
        }
        Some(approximation_degree)
    };

    approximation_degree = Some(1.0);
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    match transpile(
        circuit,
        target,
        optimization_level,
        seed,
        approximation_degree,
    ) {
        Ok(result) => {
            *result = TranspileResult {
                circuit: Box::into_raw(Box::new(result.0)),
                layout: Box::into_raw(Box::new(TranspileLayout)),
            };
            0
        }
        Err(e) => {
            *error = CString::new(e).unwrap().into_raw();
            1
        }
    }
}
