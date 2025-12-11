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

use std::ffi::{CStr, CString, c_char};

use crate::exit_codes::ExitCode;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use nalgebra::{Matrix2, Matrix4};
use ndarray::{Array2, ArrayView2};
use num_complex::{Complex64, ComplexFloat};

use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::interner::Interner;
use qiskit_circuit::operations::{
    ArrayType, DelayUnit, Operation, Param, StandardGate, StandardInstruction, UnitaryGate,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, Clbit, Qubit, VarsMode};

use smallvec::smallvec;

/// @ingroup QkCircuit
/// Construct a new circuit with the given number of qubits and clbits.
///
/// @param num_qubits The number of qubits the circuit contains.
/// @param num_clbits The number of clbits the circuit contains.
///
/// @return A pointer to the created circuit.
///
/// # Example
///
///     QkCircuit *empty = qk_circuit_new(100, 100);
///
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_circuit_new(num_qubits: u32, num_clbits: u32) -> *mut CircuitData {
    let qubits = if num_qubits > 0 {
        Some(
            (0..num_qubits)
                .map(|_| ShareableQubit::new_anonymous())
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let clbits = if num_clbits > 0 {
        Some(
            (0..num_clbits)
                .map(|_| ShareableClbit::new_anonymous())
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let circuit = CircuitData::new(qubits, clbits, None, 0, (0.).into()).unwrap();
    Box::into_raw(Box::new(circuit))
}

/// @ingroup QkQuantumRegister
/// Construct a new owning quantum register with a given number of qubits and name
///
/// @param num_qubits The number of qubits to create the register for
/// @param name The name string for the created register. The name must be comprised of
/// valid UTF-8 characters.
///
/// @return A pointer to the created register
///
/// # Example
/// ```c
///     QkQuantumRegister *qr = qk_quantum_register_new(5, "five_qubits");
/// ```
///
/// # Safety
///
/// The `name` parameter must be a pointer to memory that contains a valid
/// nul terminator at the end of the string. It also must be valid for reads of
/// bytes up to and including the nul terminator.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_quantum_register_new(
    num_qubits: u32,
    name: *const c_char,
) -> *mut QuantumRegister {
    let name = unsafe {
        CStr::from_ptr(name)
            .to_str()
            .expect("Invalid UTF-8 character")
            .to_string()
    };
    // SAFETY: Per documentation the pointer for name is a valid CStr pointer
    let reg = QuantumRegister::new_owning(name, num_qubits);
    Box::into_raw(Box::new(reg))
}

/// @ingroup QkQuantumRegister
/// Free a quantum register.
///
/// @param reg A pointer to the register to free.
///
/// # Example
/// ```c
///     QkQuantumRegister *qr = qk_quantum_register_new(1024, "qreg");
///     qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``reg`` is not either null or a valid pointer to a
/// ``QkQuantumRegister``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_quantum_register_free(reg: *mut QuantumRegister) {
    if !reg.is_null() {
        if !reg.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(reg);
        }
    }
}

/// @ingroup QkClassicalRegister
/// Free a classical register.
///
/// @param reg A pointer to the register to free.
///
/// # Example
/// ```c
///     QkClassicalRegister *cr = qk_classical_register_new(1024, "creg");
///     qk_classical_register_free(cr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``reg`` is not either null or a valid pointer to a
/// ``QkClassicalRegister``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_classical_register_free(reg: *mut ClassicalRegister) {
    if !reg.is_null() {
        if !reg.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(reg);
        }
    }
}

/// @ingroup QkClassicalRegister
/// Construct a new owning classical register with a given number of clbits and name
///
/// @param num_clbits The number of clbits to create the register for
/// @param name The name string for the created register. The name must be comprised of
/// valid UTF-8 characters.
///
/// @return A pointer to the created register
///
/// # Example
/// ```c
///     QkClassicalRegister *cr = qk_classical_register_new(5, "five_qubits");
/// ```
///
/// # Safety
///
/// The `name` parameter must be a pointer to memory that contains a valid
/// nul terminator at the end of the string. It also must be valid for reads of
/// bytes up to and including the nul terminator.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_classical_register_new(
    num_clbits: u32,
    name: *const c_char,
) -> *mut ClassicalRegister {
    // SAFETY: Per documentation the pointer for name is a valid CStr pointer
    let name = unsafe {
        CStr::from_ptr(name)
            .to_str()
            .expect("Invalid UTF-8 character")
            .to_string()
    };
    let reg = ClassicalRegister::new_owning(name, num_clbits);
    Box::into_raw(Box::new(reg))
}

/// @ingroup QkCircuit
/// Add a quantum register to a given quantum circuit
///
/// @param circuit A pointer to the circuit.
/// @param reg A pointer to the quantum register
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_little_register");
///     qk_circuit_add_quantum_register(qc, qr);
///     qk_quantum_register_free(qr);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkQuantumRegister``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_add_quantum_register(
    circuit: *mut CircuitData,
    reg: *const QuantumRegister,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let qreg = unsafe { const_ptr_as_ref(reg) };

    circuit
        .add_qreg(qreg.clone(), true)
        .expect("Invalid register unable to be added to circuit");
}

/// @ingroup QkCircuit
/// Add a classical register to a given quantum circuit
///
/// @param circuit A pointer to the circuit.
/// @param reg A pointer to the classical register
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkClassicalRegister *cr = qk_classical_register_new(24, "my_big_register");
///     qk_circuit_add_classical_register(qc, cr);
///     qk_classical_register_free(cr);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkClassicalRegister``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_add_classical_register(
    circuit: *mut CircuitData,
    reg: *const ClassicalRegister,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let creg = unsafe { const_ptr_as_ref(reg) };

    circuit
        .add_creg(creg.clone(), true)
        .expect("Invalid register unable to be added to circuit");
}

/// @ingroup QkCircuit
/// Create a copy of a ``QkCircuit``.
///
/// @param circuit A pointer to the circuit to copy.
///
/// @return A new pointer to a copy of the input ``circuit``.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 100);
///     QkCircuit *copy = qk_circuit_copy(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_copy(circuit: *const CircuitData) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    Box::into_raw(Box::new(circuit.clone()))
}

/// @ingroup QkCircuit
/// Get the number of qubits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of qubits the circuit is defined on.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 100);
///     uint32_t num_qubits = qk_circuit_num_qubits(qc);  // num_qubits==100
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_num_qubits(circuit: *const CircuitData) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.num_qubits() as u32
}

/// @ingroup QkCircuit
/// Get the number of clbits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of qubits the circuit is defined on.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 50);
///     uint32_t num_clbits = qk_circuit_num_clbits(qc);  // num_clbits==50
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_num_clbits(circuit: *const CircuitData) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.num_clbits() as u32
}

/// @ingroup QkCircuit
/// Free the circuit.
///
/// @param circuit A pointer to the circuit to free.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 100);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not either null or a valid pointer to a
/// ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_free(circuit: *mut CircuitData) {
    if !circuit.is_null() {
        if !circuit.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(circuit);
        }
    }
}

/// @ingroup QkCircuit
/// Append a ``QkGate`` to the circuit.
///
/// @param circuit A pointer to the circuit to add the gate to.
/// @param gate The StandardGate to add to the circuit.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the gate on. This
///     can be a null pointer if there are no qubits for ``gate`` (e.g. ``QkGate_GlobalPhase``).
/// @param params The pointer to the array of ``double`` values to use for the gate parameters.
///     This can be a null pointer if there are no parameters for ``gate`` (e.g. ``QkGate_H``).
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
/// ```
///
/// # Safety
///
/// The ``qubits`` and ``params`` types are expected to be a pointer to an array of ``uint32_t``
/// and ``double`` respectively where the length is matching the expectations for the standard
/// gate. If the array is insufficiently long the behavior of this function is undefined as this
/// will read outside the bounds of the array. It can be a null pointer if there are no qubits
/// or params for a given gate. You can check ``qk_gate_num_qubits`` and ``qk_gate_num_params`` to
/// determine how many qubits and params are required for a given gate.
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_gate(
    circuit: *mut CircuitData,
    gate: StandardGate,
    qubits: *const u32,
    params: *const f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    // SAFETY: Per the documentation the qubits and params pointers are arrays of num_qubits()
    // and num_params() elements respectively.
    unsafe {
        let qargs: &[Qubit] = match gate.num_qubits() {
            0 => &[],
            1 => &[Qubit(*qubits.wrapping_add(0))],
            2 => &[
                Qubit(*qubits.wrapping_add(0)),
                Qubit(*qubits.wrapping_add(1)),
            ],
            3 => &[
                Qubit(*qubits.wrapping_add(0)),
                Qubit(*qubits.wrapping_add(1)),
                Qubit(*qubits.wrapping_add(2)),
            ],
            4 => &[
                Qubit(*qubits.wrapping_add(0)),
                Qubit(*qubits.wrapping_add(1)),
                Qubit(*qubits.wrapping_add(2)),
                Qubit(*qubits.wrapping_add(3)),
            ],
            // There are no ``QkGate``s > 4 qubits
            _ => unreachable!(),
        };
        let params: &[Param] = match gate.num_params() {
            0 => &[],
            1 => &[(*params.wrapping_add(0)).into()],
            2 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
            ],
            3 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
            ],
            4 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
                (*params.wrapping_add(3)).into(),
            ],
            // There are no ``QkGate``s that take > 4 params
            _ => unreachable!(),
        };
        circuit.push_standard_gate(gate, params, qargs).unwrap()
    }
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Get the number of qubits for a ``QkGate``.
///
/// @param gate The ``QkGate`` to get the number of qubits for.
///
/// @return The number of qubits the gate acts on.
///
/// # Example
/// ```c
///     uint32_t num_qubits = qk_gate_num_qubits(QkGate_CCX);
/// ```
///
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_gate_num_qubits(gate: StandardGate) -> u32 {
    gate.num_qubits()
}

/// @ingroup QkCircuit
/// Get the number of parameters for a ``QkGate``.
///
/// @param gate The ``QkGate`` to get the number of qubits for.
///
/// @return The number of parameters the gate has.
///
/// # Example
/// ```c
///     uint32_t num_params = qk_gate_num_params(QkGate_R);
/// ```
///
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_gate_num_params(gate: StandardGate) -> u32 {
    gate.num_params()
}

/// @ingroup QkCircuit
/// Append a measurement to the circuit
///
/// @param circuit A pointer to the circuit to add the measurement to
/// @param qubit The ``uint32_t`` for the qubit to measure
/// @param clbit The ``uint32_t`` for the clbit to store the measurement outcome in
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 1);
///     qk_circuit_measure(qc, 0, 0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_measure(
    circuit: *mut CircuitData,
    qubit: u32,
    clbit: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    circuit
        .push_packed_operation(
            PackedOperation::from_standard_instruction(StandardInstruction::Measure),
            None,
            &[Qubit(qubit)],
            &[Clbit(clbit)],
        )
        .unwrap();
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Append a reset to the circuit
///
/// @param circuit A pointer to the circuit to add the reset to
/// @param qubit The ``uint32_t`` for the qubit to reset
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     qk_circuit_reset(qc, 0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_reset(circuit: *mut CircuitData, qubit: u32) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    circuit
        .push_packed_operation(
            PackedOperation::from_standard_instruction(StandardInstruction::Reset),
            None,
            &[Qubit(qubit)],
            &[],
        )
        .unwrap();
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Append a barrier to the circuit.
///
/// @param circuit A pointer to the circuit to add the barrier to.
/// @param num_qubits The number of qubits wide the barrier is.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the barrier on.
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 1);
///     uint32_t qubits[5] = {0, 1, 2, 3, 4};
///     qk_circuit_barrier(qc, qubits, 5);
/// ```
///
/// # Safety
///
/// The length of the array ``qubits`` points to must be ``num_qubits``. If there is
/// a mismatch the behavior is undefined.
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_barrier(
    circuit: *mut CircuitData,
    qubits: *const u32,
    num_qubits: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qubits: Vec<Qubit> = unsafe {
        (0..num_qubits)
            .map(|idx| Qubit(*qubits.wrapping_add(idx as usize)))
            .collect()
    };
    circuit
        .push_packed_operation(
            PackedOperation::from_standard_instruction(StandardInstruction::Barrier(num_qubits)),
            None,
            &qubits,
            &[],
        )
        .unwrap();
    ExitCode::Success
}

/// An individual operation count represented by the operation name
/// and the number of instances in the circuit.
#[repr(C)]
pub struct OpCount {
    /// A nul terminated string representing the operation name
    name: *const c_char,
    /// The number of instances of this operation in the circuit
    count: usize,
}

/// An array of ``OpCount`` objects representing the total counts of all
/// the operation types in a circuit.
#[repr(C)]
pub struct OpCounts {
    /// A array of size ``len`` containing ``OpCount`` objects for each
    /// type of operation in the circuit
    data: *mut OpCount,
    /// The number of elements in ``data``
    len: usize,
}

#[inline]
fn conjugate(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(i, j)| {
        matrix[(j, i)].conj()
    })
}

/// Check an [ArrayType] represents a unitary matrix. Uses an element-wise check; if
/// any element in ``conjugate(matrix) * matrix`` differs from the identity by more than ``tol``
/// (in magnitude), the matrix is not considered unitary.
fn is_unitary(matrix: &ArrayType, tol: f64) -> bool {
    let not_unitary = match matrix {
        ArrayType::OneQ(mat) => (mat.adjoint() * mat - Matrix2::identity())
            .iter()
            .any(|val| val.abs() > tol),
        ArrayType::TwoQ(mat) => (mat.adjoint() * mat - Matrix4::identity())
            .iter()
            .any(|val| val.abs() > tol),
        ArrayType::NDArray(mat) => {
            let product = mat.dot(&conjugate(mat.view()));
            product.indexed_iter().any(|((row, col), value)| {
                if row == col {
                    (value - Complex64::ONE).abs() > tol
                } else {
                    value.abs() > tol
                }
            })
        }
    };
    !not_unitary // using double negation to use ``any`` (faster) instead of ``all``
}

/// Create a unitary matrix `ArrayType` from a pointer to a row-major contiguous matrix of the
/// correct dimensions.
///
/// If `tol` is `Some`, the unitary matrix is checked for tolerance against the given value.  If the
/// tolerance check fails, no array is returned.
///
/// The data is copied out of `matrix`.
///
/// # Safety
///
/// `matrix` must be aligned and valid for `4 ** num_qubits` reads.
pub(crate) unsafe fn unitary_from_pointer(
    matrix: *const Complex64,
    num_qubits: u32,
    tol: Option<f64>,
) -> Option<ArrayType> {
    let dim = 1 << num_qubits;
    // SAFETY: per documentation, `matrix` is aligned and valid for `4**num_qubits` reads.
    let raw = unsafe { ::std::slice::from_raw_parts(matrix, dim * dim) };
    let mat = match num_qubits {
        1 => ArrayType::OneQ(Matrix2::from_fn(|i, j| raw[i * dim + j])),
        2 => ArrayType::TwoQ(Matrix4::from_fn(|i, j| raw[i * dim + j])),
        _ => ArrayType::NDArray(Array2::from_shape_fn((dim, dim), |(i, j)| raw[i * dim + j])),
    };
    match tol {
        Some(tol) => is_unitary(&mat, tol).then_some(mat),
        None => Some(mat),
    }
}

/// @ingroup QkCircuit
/// Append an arbitrary unitary matrix to the circuit.
///
/// @param circuit A pointer to the circuit to append the unitary to.
/// @param matrix A pointer to the ``QkComplex64`` array representing the unitary matrix.
///     This must be a row-major, unitary matrix of dimension ``2 ^ num_qubits x 2 ^ num_qubits``.
///     More explicitly: the ``(i, j)``-th element is given by ``matrix[i * 2^n + j]``.
///     The contents of ``matrix`` are copied inside this function before being added to the circuit,
///     so caller keeps ownership of the original memory that ``matrix`` points to and can reuse it
///     after the call and the caller is responsible for freeing it.
/// @param qubits A pointer to array of qubit indices, of length ``num_qubits``.
/// @param num_qubits The number of qubits the unitary acts on.
/// @param check_input When true, the function verifies that the matrix is unitary.
///     If set to False the caller is responsible for ensuring the matrix is unitary, if
///     the matrix is not unitary this is undefined behavior and will result in a corrupt
///     circuit.
///
/// @return An exit code.
///
/// # Example
/// ```c
/// QkComplex64 c0 = {0, 0};  // 0+0i
/// QkComplex64 c1 = {1, 0};  // 1+0i
///
/// const uint32_t num_qubits = 1;
/// QkComplex64 unitary[2*2] = {c0, c1,  // row 0
///                             c1, c0}; // row 1
///
/// QkCircuit *circuit = qk_circuit_new(1, 0);  // 1 qubit circuit
/// uint32_t qubit[1] = {0};  // qubit to apply the unitary on
/// qk_circuit_unitary(circuit, unitary, qubit, num_qubits, true);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated:
///
/// * ``circuit`` is a valid, non-null pointer to a ``QkCircuit``
/// * ``matrix`` is an aligned pointer to ``4**num_qubits`` initialized ``QkComplex64`` values
/// * ``qubits`` is an aligned pointer to ``num_qubits`` initialized ``uint32_t`` values
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_unitary(
    circuit: *mut CircuitData,
    matrix: *const Complex64,
    qubits: *const u32,
    num_qubits: u32,
    check_input: bool,
) -> ExitCode {
    // SAFETY: Caller quarantees pointer validation, alignment
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let mat = unsafe { unitary_from_pointer(matrix, num_qubits, check_input.then_some(1e-12)) };
    let Some(mat) = mat else {
        return ExitCode::ExpectedUnitary;
    };
    let qubits = if num_qubits == 0 {
        // This handles the case of C passing us a null pointer for the qubits; Rust slices
        // can't be backed by the null pointer even when empty.
        &[]
    } else {
        // SAFETY: per documentation, `qubits` is aligned and valid for `num_qubits` reads.  Per
        // previous check, `num_qubits` is nonzero so `qubits` cannot be null.
        unsafe { ::std::slice::from_raw_parts(qubits as *const Qubit, num_qubits as usize) }
    };

    // Create PackedOperation -> push to circuit_data
    let u_gate = Box::new(UnitaryGate { array: mat });
    let op = PackedOperation::from_unitary(u_gate);
    circuit
        .push_packed_operation(op, None, qubits, &[])
        .unwrap();
    // Return success
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Return a list of string names for instructions in a circuit and their counts.
///
/// To properly free the memory allocated by the struct, you should call ``qk_opcounts_clear``.
/// Dropping the ``QkOpCounts`` struct without doing so will leave the stored array of ``QkOpCount``
/// allocated and produce a memory leak.
///
/// @param circuit A pointer to the circuit to get the counts for.
///
/// @return An ``QkOpCounts`` struct containing the circuit operation counts.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t qubits[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubits, NULL);
///     QkOpCounts counts = qk_circuit_count_ops(qc);
///     // .. once done
///     qk_opcounts_clear(&counts);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_count_ops(circuit: *const CircuitData) -> OpCounts {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let count_ops = circuit.count_ops();
    let output = {
        let vec: Vec<OpCount> = count_ops
            .into_iter()
            .map(|(name, count)| OpCount {
                name: CString::new(name).unwrap().into_raw(),
                count,
            })
            .collect();
        vec.into_boxed_slice()
    };
    let len = output.len();
    let data = Box::into_raw(output) as *mut OpCount;
    OpCounts { data, len }
}

/// @ingroup QkCircuit
/// Return the total number of instructions in the circuit.
///
/// @param circuit A pointer to the circuit to get the total number of instructions for.
///
/// @return The total number of instructions in the circuit.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
///     size_t num = qk_circuit_num_instructions(qc); // 1
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_num_instructions(circuit: *const CircuitData) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    circuit.__len__()
}

/// A circuit instruction representation.
///
/// This struct represents the data contained in an individual instruction in a ``QkCircuit``.
/// It is not a pointer to the underlying object, but contains a copy of the properties of the
/// instruction for inspection.
#[repr(C)]
pub struct CInstruction {
    /// The instruction name
    name: *mut c_char,
    /// A pointer to an array of qubit indices this instruction operates on.
    qubits: *mut u32,
    /// A pointer to an array of clbit indices this instruction operates on.
    clbits: *mut u32,
    /// A pointer to an array of parameter values for this instruction.
    params: *mut f64,
    /// The number of qubits for this instruction.
    num_qubits: u32,
    /// The number of clbits for this instruction.
    num_clbits: u32,
    /// The number of parameters for this instruction.
    num_params: u32,
}
impl CInstruction {
    /// Create a `CInstruction` that owns pointers to copies of the information in the given
    /// `PackedInstruction`.
    ///
    /// This must be cleared by a call to `qk_circuit_instruction_clear` to avoid leaking its
    /// allocations.
    ///
    /// Panics if the operation name contains a nul, or if the instruction has non-float parameters.
    pub(crate) fn from_packed_instruction_with_floats(
        packed: &PackedInstruction,
        qargs_interner: &Interner<[Qubit]>,
        cargs_interner: &Interner<[Clbit]>,
    ) -> Self {
        let name = CString::new(packed.op.name())
            .expect("names do not contain nul")
            .into_raw();
        let qargs = qargs_interner.get(packed.qubits);
        let cargs = cargs_interner.get(packed.clbits);
        let params = packed
            .params_view()
            .iter()
            .map(|p| match p {
                Param::Float(p) => Some(*p),
                _ => None,
            })
            .collect::<Option<Box<[f64]>>>()
            .expect("caller is responsible for ensuring all parameters are floats");
        Self {
            name,
            num_qubits: qargs.len() as u32,
            qubits: Box::leak(qargs.iter().map(|q| q.0).collect::<Box<[u32]>>()).as_mut_ptr(),
            num_clbits: cargs.len() as u32,
            clbits: Box::leak(cargs.iter().map(|c| c.0).collect::<Box<[u32]>>()).as_mut_ptr(),
            num_params: params.len() as u32,
            params: Box::leak(params).as_mut_ptr(),
        }
    }
}

/// @ingroup QkCircuit
/// Return the instruction details for an instruction in the circuit.
///
/// This function is used to get the instruction details for a given instruction in
/// the circuit.
///
/// This function allocates memory internally for the provided ``QkCircuitInstruction``
/// and thus you are responsible for calling ``qk_circuit_instruction_clear`` to
/// free it.
///
/// @param circuit A pointer to the circuit to get the instruction details for.
/// @param index The instruction index to get the instruction details of.
/// @param instruction A pointer to where to write out the ``QkCircuitInstruction``
///
///
/// # Example
/// ```c
///     QkCircuitInstruction inst;
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
///     qk_circuit_get_instruction(qc, 0, &inst);
///     qk_circuit_instruction_clear(&inst);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``. The
/// value for ``index`` must be less than the value returned by ``qk_circuit_num_instructions``
/// otherwise this function will panic. Behavior is undefined if ``instruction`` is not a valid,
/// non-null pointer to a memory allocation with sufficient space for a ``QkCircuitInstruction``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_get_instruction(
    circuit: *const CircuitData,
    index: usize,
    instruction: *mut CInstruction,
) {
    // SAFETY: Per documentation, `circuit` is a pointer to valid data.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let inst = CInstruction::from_packed_instruction_with_floats(
        &circuit.data()[index],
        circuit.qargs_interner(),
        circuit.cargs_interner(),
    );
    // SAFETY: per documentation, `instruction` is a pointer to a sufficient allocation.
    unsafe { instruction.write(inst) };
}

/// @ingroup QkCircuit
/// Clear the data in circuit instruction object.
///
/// This function doesn't free the allocation for the provided ``QkCircuitInstruction`` pointer, it
/// only frees the internal allocations for the data contained in the instruction. You are
/// responsible for allocating and freeing the actual allocation used to store a
/// ``QkCircuitInstruction``.
///
/// @param inst A pointer to the instruction to free.
///
/// # Example
/// ```c
///     QkCircuitInstruction *inst = malloc(sizeof(QkCircuitInstruction));
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t q0[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, q0, NULL);
///     qk_circuit_get_instruction(qc, 0, inst);
///     qk_circuit_instruction_clear(inst); // clear internal allocations
///     free(inst); // free struct
///     qk_circuit_free(qc); // free the circuit
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``inst`` is not a valid, non-null pointer to a ``QkCircuitInstruction``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_instruction_clear(inst: *mut CInstruction) {
    // SAFETY: Loading the data from pointers contained in a CInstruction. These should only be
    // created by rust code and are constructed from Vecs internally or CStrings.
    unsafe {
        let inst = mut_ptr_as_ref(inst);
        if inst.num_qubits > 0 && !inst.qubits.is_null() {
            let qubits = std::slice::from_raw_parts_mut(inst.qubits, inst.num_qubits as usize);
            let _: Box<[u32]> = Box::from_raw(qubits as *mut [u32]);
            inst.qubits = std::ptr::null_mut();
        }
        inst.num_qubits = 0;
        if inst.num_clbits > 0 && !inst.clbits.is_null() {
            let clbits = std::slice::from_raw_parts_mut(inst.clbits, inst.num_clbits as usize);
            let _: Box<[u32]> = Box::from_raw(clbits as *mut [u32]);
            inst.clbits = std::ptr::null_mut();
        }
        inst.num_clbits = 0;
        if inst.num_params > 0 && !inst.params.is_null() {
            let params = std::slice::from_raw_parts_mut(inst.params, inst.num_params as usize);
            let _ = Box::from_raw(params as *mut [f64]);
            inst.params = std::ptr::null_mut();
        }
        inst.num_params = 0;
        if !inst.name.is_null() {
            let _ = CString::from_raw(inst.name);
            inst.name = std::ptr::null_mut();
        }
    }
}

/// @ingroup QkCircuit
/// Clear the content in a circuit operation count list.
///
/// @param op_counts The returned op count list from ``qk_circuit_count_ops``.
///
/// # Safety
///
/// Behavior is undefined if ``op_counts`` is not the object returned by ``qk_circuit_count_ops``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_opcounts_clear(op_counts: *mut OpCounts) {
    // SAFETY: The user guarantees the input is a valid OpCounts pointer.
    let op_counts = unsafe { mut_ptr_as_ref(op_counts) };

    if op_counts.len > 0 && !op_counts.data.is_null() {
        // SAFETY: We load the box from a slice pointer created from
        // the raw parts from the OpCounts::data attribute.
        unsafe {
            let slice: Box<[OpCount]> = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                op_counts.data,
                op_counts.len,
            ));
            // free the allocated strings in each OpCount
            for count in slice.iter() {
                if !count.name.is_null() {
                    let _ = CString::from_raw(count.name as *mut c_char);
                }
            }
            // the variable vec goes out of bounds and is freed too
        }
    }
    op_counts.len = 0;
    op_counts.data = std::ptr::null_mut();
}

/// @ingroup QkCircuit
/// Convert to a Python-space ``QuantumCircuit``.
///
/// This function takes ownership of the pointer and gives it to Python. Using
/// the input ``circuit`` pointer after it's passed to this function is
/// undefined behavior. In particular, ``qk_circuit_free`` should not be called
/// on this pointer anymore.
///
/// @param circuit The C-space ``QkCircuit`` pointer.
///
/// @return A Python ``QuantumCircuit`` object.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to
/// a ``QkCircuit``
///
/// It is assumed that the thread currently executing this function holds the
/// Python GIL. This is required to create the Python object returned by this
/// function.
#[unsafe(no_mangle)]
#[cfg(feature = "python_binding")]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_to_python(
    circuit: *mut CircuitData,
) -> *mut ::pyo3::ffi::PyObject {
    // SAFETY: per documentation, `circuit` is a valid and owned `CircuitData`.
    let circuit = unsafe { Box::from_raw(mut_ptr_as_ref(circuit)) };
    // SAFETY: per documentation, we are attached to an interpreter.
    let py = unsafe { ::pyo3::Python::assume_attached() };
    circuit
        .into_py_quantum_circuit(py)
        .expect("Unable to create a Python circuit")
        .into_ptr()
}

/// @ingroup QkCircuit
///
/// Units for circuit delays.
#[repr(u8)]
pub enum CDelayUnit {
    /// Seconds.
    S = 0,
    /// Milliseconds.
    MS = 1,
    /// Microseconds.
    US = 2,
    /// Nanoseconds.
    NS = 3,
    /// Picoseconds.
    PS = 4,
}

impl From<CDelayUnit> for DelayUnit {
    fn from(value: CDelayUnit) -> Self {
        match value {
            CDelayUnit::S => DelayUnit::S,
            CDelayUnit::MS => DelayUnit::MS,
            CDelayUnit::US => DelayUnit::US,
            CDelayUnit::NS => DelayUnit::NS,
            CDelayUnit::PS => DelayUnit::PS,
        }
    }
}

/// @ingroup QkCircuit
/// Append a delay instruction to the circuit.
///
/// @param circuit A pointer to the circuit to add the delay to.
/// @param qubit The ``uint32_t`` index of the qubit to apply the delay to.
/// @param duration The duration of the delay.
/// @param unit An enum representing the unit of the duration.
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(1, 0);
///     qk_circuit_delay(qc, 0, 100.0, QkDelayUnit_NS);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_delay(
    circuit: *mut CircuitData,
    qubit: u32,
    duration: f64,
    unit: CDelayUnit,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };

    let delay_unit_variant = unit.into();

    let duration_param: Param = duration.into();
    let delay_instruction = StandardInstruction::Delay(delay_unit_variant);

    let params = Parameters::Params(smallvec![duration_param]);
    circuit
        .push_packed_operation(
            PackedOperation::from_standard_instruction(delay_instruction),
            Some(params),
            &[Qubit(qubit)],
            &[],
        )
        .unwrap();

    ExitCode::Success
}

/// @ingroup QkCircuit
/// Convert a given circuit to a DAG.
///
/// The new DAG is copied from the circuit; the original ``circuit`` reference is still owned by the
/// caller and still required to be freed with `qk_circuit_free`.  You must free the returned DAG
/// with ``qk_dag_free`` when done with it.
///
/// @param circuit A pointer to the circuit from which to create the DAG.
///
/// @return A pointer to the new DAG.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkQuantumRegister *qr = qk_quantum_register_new(3, "qr");
///     qk_circuit_add_quantum_register(qc, qr);
///     qk_quantum_register_free(qr);
///     
///     QkDag *dag = qk_circuit_to_dag(qc);
///     
///     qk_dag_free(dag);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.  
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_to_dag(circuit: *const CircuitData) -> *mut DAGCircuit {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    let dag = DAGCircuit::from_circuit_data(circuit, true, None, None, None, None)
        .expect("Error occurred while converting CircuitData to DAGCircuit");

    Box::into_raw(Box::new(dag))
}

/// @ingroup QkCircuit
///
/// The mode to copy the classical variables, for operations that create a new
/// circuit based on an existing one.
#[repr(u8)]
pub enum CVarsMode {
    /// Each variable has the same type it had in the input.
    Alike = 0,
    /// Each variable becomes a "capture".
    Captures = 1,
    /// Do not copy the variable data.
    Drop = 2,
}

impl From<CVarsMode> for VarsMode {
    fn from(value: CVarsMode) -> Self {
        match value {
            CVarsMode::Alike => VarsMode::Alike,
            CVarsMode::Captures => VarsMode::Captures,
            CVarsMode::Drop => VarsMode::Drop,
        }
    }
}

/// @ingroup QkCircuit
///
/// The mode to use to copy blocks in control-flow instructions, for operations that
/// create a new circuit based on an existing one.
#[repr(u8)]
pub enum CBlocksMode {
    /// Drop the blocks.
    Drop = 0,
    /// Keep the blocks.
    Keep = 1,
}

impl From<CBlocksMode> for BlocksMode {
    fn from(value: CBlocksMode) -> Self {
        match value {
            CBlocksMode::Drop => BlocksMode::Drop,
            CBlocksMode::Keep => BlocksMode::Keep,
        }
    }
}

/// @ingroup QkCircuit
/// Return a copy of self with the same structure but empty.
///
/// That structure includes:
/// * global phase
/// * all the qubits and clbits, including the registers.
///
/// @param circuit A pointer to the circuit to copy.
/// @param vars_mode The mode for handling classical variables.
/// @param blocks_mode The mode for handling blocks.
///
/// @return The pointer to the copied circuit.
///
/// # Example
/// ```c
/// QkCircuit *qc = qk_circuit_new(10, 10);
/// for (int i = 0; i < 10; i++) {
///     qk_circuit_measure(qc, i, i);
///     uint32_t qubits[1] = {i};
///     qk_circuit_gate(qc, QkGate_H, qubits, NULL);
/// }
///
/// // As the circuit does not contain any control-flow instructions,
/// // vars_mode and blocks_mode do not have any effect.
/// QkCircuit *copy = qk_circuit_copy_empty_like(qc, QkVarsMode_Alike, QkBlocksMode_Drop);
///
/// size_t num_copy_instructions = qk_circuit_num_instructions(copy); // 0
///
/// // do something with the copy
///
/// qk_circuit_free(qc);
/// qk_circuit_free(copy);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_copy_empty_like(
    circuit: *const CircuitData,
    vars_mode: CVarsMode,
    blocks_mode: CBlocksMode,
) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is to valid data.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let vars_mode = vars_mode.into();
    let blocks_mode = blocks_mode.into();

    let copied_circuit = circuit
        .copy_empty_like(vars_mode, blocks_mode)
        .expect("Failed to copy the circuit.");
    Box::into_raw(Box::new(copied_circuit))
}
