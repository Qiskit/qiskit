// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::ffi::{c_char, CString};

use crate::exit_codes::{CInputError, ExitCode};
use crate::pointers::{check_ptr, const_ptr_as_ref, mut_ptr_as_ref};
use num_complex::Complex64;

use qiskit_accelerate::sparse_observable::{BitTerm, SparseObservable, SparseTermView};

#[cfg(feature = "python_binding")]
use pyo3::ffi::PyObject;
#[cfg(feature = "python_binding")]
use pyo3::{Py, Python};
#[cfg(feature = "python_binding")]
use qiskit_accelerate::sparse_observable::PySparseObservable;

/// A term in a ``QkObs``.
///
/// This contains the coefficient (``coeff``), the number of qubits of the observable
/// (``num_qubits``) and pointers to the ``bit_terms`` and ``indices`` arrays, which have
/// length ``len``. It's the responsibility of the user that the data is coherent,
/// see also the below section on safety.
///
/// # Safety
///
/// * ``bit_terms`` must be a non-null, aligned pointer to ``len`` elements of type ``QkBitTerm``.
/// * ``indices`` must be a non-null, aligned pointer to ``len`` elements of type ``uint32_t``.
#[repr(C)]
pub struct CSparseTerm {
    /// The coefficient of the observable term.
    coeff: Complex64,
    /// Length of the ``bit_terms`` and ``indices`` arrays.
    len: usize,
    /// A non-null, aligned pointer to ``len`` elements of type ``QkBitTerm``.
    bit_terms: *mut BitTerm,
    /// A non-null, aligned pointer to ``len`` elements of type ``uint32_t``.
    indices: *mut u32,
    /// The number of qubits the observable term is defined on.
    num_qubits: u32,
}

impl TryFrom<&CSparseTerm> for SparseTermView<'_> {
    type Error = CInputError;

    fn try_from(value: &CSparseTerm) -> Result<Self, Self::Error> {
        check_ptr(value.bit_terms)?;
        check_ptr(value.indices)?;

        // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C
        // that the arrays have the appropriate length, which is documented as requirement in the
        // CSparseTerm class.
        let bit_terms = unsafe { ::std::slice::from_raw_parts(value.bit_terms, value.len) };
        let indices = unsafe { ::std::slice::from_raw_parts(value.indices, value.len) };

        Ok(SparseTermView {
            num_qubits: value.num_qubits,
            coeff: value.coeff,
            bit_terms,
            indices,
        })
    }
}

/// @ingroup QkObs
/// Construct the zero observable (without any terms).
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// # Example
///
///     QkObs *zero = qk_obs_zero(100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_zero(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::zero(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup QkObs
/// Construct the identity observable.
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// # Example
///
///     QkObs *identity = qk_obs_identity(100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_identity(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::identity(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup QkObs
/// Construct a new observable from raw data.
///
/// @param num_qubits The number of qubits the observable is defined on.
/// @param num_terms The number of terms.
/// @param num_bits The total number of non-identity bit terms.
/// @param coeffs A pointer to the first element of the coefficients array, which has length
///     ``num_terms``.
/// @param bit_terms A pointer to the first element of the bit terms array, which has length
///     ``num_bits``.
/// @param indices A pointer to the first element of the indices array, which has length
///     ``num_bits``. Note that, per term, these *must* be sorted incrementally.
/// @param boundaries A pointer to the first element of the boundaries array, which has length
///     ``num_terms + 1``.
///
/// @return If the input data was coherent and the construction successful, the result is a pointer
///     to the observable. Otherwise a null pointer is returned.
///
/// # Example
///
///     // define the raw data for the 100-qubit observable |01><01|_{0, 1} - |+-><+-|_{98, 99}
///     uint32_t num_qubits = 100;
///     uint64_t num_terms = 2;  // we have 2 terms: |01><01|, -1 * |+-><+-|
///     uint64_t num_bits = 4; // we have 4 non-identity bits: 0, 1, +, -
///
///     complex double coeffs[2] = {1, -1};
///     QkBitTerm bits[4] = {QkBitTerm_Zero, QkBitTerm_One, QkBitTerm_Plus, QkBitTerm_Minus};
///     uint32_t indices[4] = {0, 1, 98, 99};  // <-- e.g. {1, 0, 99, 98} would be invalid
///     size_t boundaries[3] = {0, 2, 4};
///
///     QkObs *obs = qk_obs_new(
///         num_qubits, num_terms, num_bits, coeffs, bits, indices, boundaries
///     );
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
///   * ``coeffs`` is a pointer to a ``complex double`` array of length ``num_terms``
///   * ``bit_terms`` is a pointer to an array of valid ``QkBitTerm`` elements of length ``num_bits``
///   * ``indices`` is a pointer to a ``uint32_t`` array of length ``num_bits``, which is
///     term-wise sorted in strict ascending order, and every element is smaller than ``num_qubits``
///   * ``boundaries`` is a pointer to a ``size_t`` array of length ``num_terms + 1``, which is
///     sorted in ascending order, the first element is 0 and the last element is
///     smaller than ``num_terms``
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_new(
    num_qubits: u32,
    num_terms: u64,
    num_bits: u64,
    coeffs: *mut Complex64,
    bit_terms: *mut BitTerm,
    indices: *mut u32,
    boundaries: *mut usize,
) -> *mut SparseObservable {
    let num_terms = num_terms as usize;
    let num_bits = num_bits as usize;

    check_ptr(coeffs).unwrap();
    check_ptr(bit_terms).unwrap();
    check_ptr(indices).unwrap();
    check_ptr(boundaries).unwrap();

    // SAFETY: At this point we know the pointers are non-null and aligned. We rely on C that
    // the pointers point to arrays of appropriate length, as specified in the function docs.
    let coeffs = unsafe { ::std::slice::from_raw_parts(coeffs, num_terms).to_vec() };
    let bit_terms = unsafe { ::std::slice::from_raw_parts(bit_terms, num_bits).to_vec() };
    let indices = unsafe { ::std::slice::from_raw_parts(indices, num_bits).to_vec() };
    let boundaries = unsafe { ::std::slice::from_raw_parts(boundaries, num_terms + 1).to_vec() };

    let result = SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries);
    match result {
        Ok(obs) => Box::into_raw(Box::new(obs)),
        Err(_) => ::std::ptr::null_mut(),
    }
}

/// @ingroup QkObs
/// Free the observable.
///
/// @param obs A pointer to the observable to free.
///
/// # Example
///
///     QkObs *obs = qk_obs_zero(100);
///     qk_obs_free(obs);
///
/// # Safety
///
/// Behavior is undefined if ``obs`` is not either null or a valid pointer to a ``QkObs``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_free(obs: *mut SparseObservable) {
    if !obs.is_null() {
        if !obs.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(obs);
        }
    }
}

/// @ingroup QkObs
/// Add a term to the observable.
///
/// @param obs A pointer to the observable.
/// @param cterm A pointer to the term to add.
///
/// @return An exit code. This is ``>0`` if the term is incoherent or adding the term fails.
///
/// # Example
///
///     uint32_t num_qubits = 100;
///     QkObs *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     QkObsTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///
///     int exit_code = qk_obs_add_term(obs, &term);
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated:
///
///   * ``obs`` is a valid, non-null pointer to a ``QkObs``
///   * ``cterm`` is a valid, non-null pointer to a ``QkObsTerm``
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_add_term(
    obs: *mut SparseObservable,
    cterm: *const CSparseTerm,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };
    let cterm = unsafe { const_ptr_as_ref(cterm) };

    let view = match cterm.try_into() {
        Ok(view) => view,
        Err(err) => return ExitCode::from(err),
    };

    match obs.add_term(view) {
        Ok(_) => ExitCode::Success,
        Err(err) => ExitCode::from(err),
    }
}

/// @ingroup QkObs
/// Get an observable term by reference.
///
/// A ``QkObsTerm`` contains pointers to the indices and bit terms in the term, which
/// can be used to modify the internal data of the observable. This can leave the observable
/// in an incoherent state and should be avoided, unless great care is taken. It is generally
/// safer to construct a new observable instead of attempting in-place modifications.
///
/// @param obs A pointer to the observable.
/// @param index The index of the term to get.
/// @param out A pointer to a ``QkObsTerm`` used to return the observable term.
///
/// @return An exit code.
///
/// # Example
///
///     QkObs *obs = qk_obs_identity(100);
///     QkObsTerm term;
///     int exit_code = qk_obs_term(obs, 0, &term);
///     // out-of-bounds indices return an error code
///     // int error = qk_obs_term(obs, 12, &term);
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated
/// * ``obs`` is a valid, non-null pointer to a ``QkObs``
/// * ``out`` is a valid, non-null pointer to a ``QkObsTerm``
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_term(
    obs: *mut SparseObservable,
    index: u64,
    out: *mut CSparseTerm,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let obs = unsafe { mut_ptr_as_ref(obs) };

    let index = index as usize;
    if index > obs.num_terms() {
        return ExitCode::IndexError;
    }

    out.len = obs.boundaries()[index + 1] - obs.boundaries()[index];
    out.coeff = obs.coeffs()[index];
    out.num_qubits = obs.num_qubits();

    let start = obs.boundaries()[index];
    out.bit_terms = &mut obs.bit_terms_mut()[start];
    // SAFETY: mutating the indices can leave the observable in an incoherent state.
    out.indices = &mut unsafe { obs.indices_mut() }[start];

    ExitCode::Success
}

/// @ingroup QkObs
/// Get the number of terms in the observable.
///
/// @param obs A pointer to the observable.
///
/// @return The number of terms in the observable.
///
/// # Example
///
///     QkObs *obs = qk_obs_identity(100);
///     size_t num_terms = qk_obs_num_terms(obs);  // num_terms==1
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_num_terms(obs: *const SparseObservable) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    obs.num_terms()
}

/// @ingroup QkObs
/// Get the number of qubits the observable is defined on.
///
/// @param obs A pointer to the observable.
///
/// @return The number of qubits the observable is defined on.
///
/// # Example
///
///     QkObs *obs = qk_obs_identity(100);
///     uint32_t num_qubits = qk_obs_num_qubits(obs);  // num_qubits==100
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_num_qubits(obs: *const SparseObservable) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    obs.num_qubits()
}

/// @ingroup QkObs
/// Get the number of bit terms/indices in the observable.
///
/// @param obs A pointer to the observable.
///
/// @return The number of terms in the observable.
///
/// # Example
///
///     QkObs *obs = qk_obs_identity(100);
///     size_t len = qk_obs_len(obs);  // len==0, as there are no non-trivial bit terms
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_len(obs: *const SparseObservable) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    obs.bit_terms().len()
}

/// @ingroup QkObs
/// Get a pointer to the coefficients.
///
/// This can be used to read and modify the observable's coefficients. The resulting
/// pointer is valid to read for ``qk_obs_num_terms(obs)`` elements of ``complex double``.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the coefficients.
///
/// # Example
///
///     QkObs *obs = qk_obs_identity(100);
///     size_t num_terms = qk_obs_num_terms(obs);
///     complex double *coeffs = qk_obs_coeffs(obs);
///
///     for (size_t i = 0; i < num_terms; i++) {
///         printf("%f + i%f\n", creal(coeffs[i]), cimag(coeffs[i]));
///     }
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_coeffs(obs: *mut SparseObservable) -> *mut Complex64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };

    obs.coeffs_mut().as_mut_ptr()
}

/// @ingroup QkObs
/// Get a pointer to the indices.
///
/// This can be used to read and modify the observable's indices. The resulting pointer is
/// valid to read for ``qk_obs_len(obs)`` elements of size ``uint32_t``.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the indices.
///
/// # Example
///
///     uint32_t num_qubits = 100;
///     QkObs *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     QkObsTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     size_T len = qk_obs_len(obs);
///     uint32_t *indices = qk_obs_indices(obs);
///
///     for (size_t i = 0; i < len; i++) {
///         printf("index %i: %i\n", i, indices[i]);
///     }
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_indices(obs: *mut SparseObservable) -> *mut u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };

    // SAFETY: Mutating the indices can leave the observable in an incoherent state.
    unsafe { obs.indices_mut() }.as_mut_ptr()
}

/// @ingroup QkObs
/// Get a pointer to the term boundaries.
///
/// This can be used to read and modify the observable's term boundaries. The resulting pointer is
/// valid to read for ``qk_obs_num_terms(obs) + 1`` elements of size ``size_t``.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the boundaries.
///
/// # Example
///
///     uint32_t num_qubits = 100;
///     QkObs *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     QkObsTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     size_t num_terms = qk_obs_num_terms(obs);
///     uint32_t *boundaries = qk_obs_boundaries(obs);
///
///     for (size_t i = 0; i < num_terms + 1; i++) {
///         printf("boundary %i: %i\n", i, boundaries[i]);
///     }
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_boundaries(obs: *mut SparseObservable) -> *mut usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };

    // SAFETY: Modifying the boundaries can leave the observable in an incoherent state. It is
    // the responsibility of the user that the data is coherent.
    unsafe { obs.boundaries_mut() }.as_mut_ptr()
}

/// @ingroup QkObs
/// Get a pointer to the bit terms.
///
/// This can be used to read and modify the observable's bit terms. The resulting pointer is
/// valid to read for ``qk_obs_len(obs)`` elements of size ``uint8_t``.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the bit terms.
///
/// # Example
///
///     uint32_t num_qubits = 100;
///     QkObs *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     QkObsTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     size_t len = qk_obs_len(obs);
///     QkBitTerm *bits = qk_obs_bit_terms(obs);
///
///     for (size_t i = 0; i < len; i++) {
///         printf("bit term %i: %i\n", i, bits[i]);
///     }
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``,
/// or if invalid valus are written into the resulting ``QkBitTerm`` pointer.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_bit_terms(obs: *mut SparseObservable) -> *mut BitTerm {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };

    obs.bit_terms_mut().as_mut_ptr()
}

/// @ingroup QkObs
/// Multiply the observable by a complex coefficient.
///
/// @param obs A pointer to the observable.
/// @param coeff The coefficient to multiply the observable with.
///
/// # Example
///
///     QkObs *obs = qk_obs_identity(100);
///     complex double coeff = 2;
///     QkObs *result = qk_obs_multiply(obs, &coeff);
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated
/// * ``obs`` is a valid, non-null pointer to a ``QkObs``
/// * ``coeff`` is a valid, non-null pointer to a ``complex double``
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_multiply(
    obs: *const SparseObservable,
    coeff: *const Complex64,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };
    let coeff = unsafe { const_ptr_as_ref(coeff) };

    let result = obs * (*coeff);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Add two observables.
///
/// @param left A pointer to the left observable.
/// @param right A pointer to the right observable.
///
/// @return A pointer to the result ``left + right``.
///
/// # Example
///
///     QkObs *left = qk_obs_identity(100);
///     QkObs *right = qk_obs_zero(100);
///     QkObs *result = qk_obs_add(left, right);
///
/// # Safety
///
/// Behavior is undefined if ``left`` or ``right`` are not valid, non-null pointers to
/// ``QkObs``\ s.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_add(
    left: *const SparseObservable,
    right: *const SparseObservable,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    let result = left + right;
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Compose (multiply) two observables.
///
/// @param first One observable.
/// @param second The other observable.
///
/// @return ``first.compose(second)`` which equals the observable ``result = second @ first``,
///     in terms of the matrix multiplication ``@``.
///
/// # Example
///
///     QkObs *first = qk_obs_zero(100);
///     QkObs *second = qk_obs_identity(100);
///     QkObs *result = qk_obs_compose(first, second);
///
/// # Safety
///
/// Behavior is undefined if ``first`` or ``second`` are not valid, non-null pointers to
/// ``QkObs``\ s.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_compose(
    first: *const SparseObservable,
    second: *const SparseObservable,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let first = unsafe { const_ptr_as_ref(first) };
    let second = unsafe { const_ptr_as_ref(second) };

    let result = first.compose(second);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Compose (multiply) two observables according to a custom qubit order.
///
/// Notably, this allows composing two observables of different size.
///
/// @param first One observable.
/// @param second The other observable. The number of qubits must match the length of ``qargs``.
/// @param qargs The qubit arguments specified which indices in ``first`` to associate with
///     the ones in ``second``.
///
/// @return ``first.compose(second)`` which equals the observable ``result = second @ first``,
///     in terms of the matrix multiplication ``@``.
///
/// # Example
///
///     QkObs *first = qk_obs_zero(100);
///     QkObs *second = qk_obs_identity(100);
///     QkObs *result = qk_obs_compose(first, second);
///
/// # Safety
///
/// To call this function safely
///
///   * ``first`` and ``second`` must be valid, non-null pointers to ``QkObs``\ s
///   * ``qargs`` must point to an array of ``uint32_t``, readable for ``qk_obs_num_qubits(second)``
///     elements (meaning the number of qubits in ``second``)
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_compose_map(
    first: *const SparseObservable,
    second: *const SparseObservable,
    qargs: *const u32,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let first = unsafe { const_ptr_as_ref(first) };
    let second = unsafe { const_ptr_as_ref(second) };

    let qargs = if qargs.is_null() {
        if second.num_qubits() != 0 {
            panic!("If qargs is null, then second must have 0 qubits.");
        }
        &[]
    } else {
        if !qargs.is_aligned() {
            panic!("qargs pointer is not aligned to u32");
        }
        // SAFETY: Per documentation, qargs is safe to read up to ``second.num_qubits()`` elements,
        // which is the maximal value of ``index`` here.
        unsafe { ::std::slice::from_raw_parts(qargs, second.num_qubits() as usize) }
    };

    let qargs_map = |index: u32| qargs[index as usize];

    let result = first.compose_map(second, qargs_map);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Calculate the canonical representation of the observable.
///
/// @param obs A pointer to the observable.
/// @param tol The tolerance below which coefficients are considered to be zero.
///
/// @return The canonical representation of the observable.
///
/// # Example
///
///     QkObs *iden = qk_obs_identity(100);
///     QkObs *two = qk_obs_add(iden, iden);
///
///     double tol = 1e-6;
///     QkObs *canonical = qk_obs_canonicalize(two);
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_canonicalize(
    obs: *const SparseObservable,
    tol: f64,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    let result = obs.canonicalize(tol);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Copy the observable.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to a copy of the observable.
///
/// # Example
///
///     QkObs *original = qk_obs_identity(100);
///     QkObs *copied = qk_obs_copy(original);
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_copy(obs: *const SparseObservable) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    let copied = obs.clone();
    Box::into_raw(Box::new(copied))
}

/// @ingroup QkObs
/// Compare two observables for equality.
///
/// Note that this does not compare mathematical equality, but data equality. This means
/// that two observables might represent the same observable but not compare as equal.
///
/// @param obs A pointer to one observable.
/// @param other A pointer to another observable.
///
/// @return ``true`` if the observables are equal, ``false`` otherwise.
///
/// # Example
///
///     QkObs *observable = qk_obs_identity(100);
///     QkObs *other = qk_obs_identity(100);
///     bool are_equal = qk_obs_equal(observable, other);
///
/// # Safety
///
/// Behavior is undefined if ``obs`` or ``other`` are not valid, non-null pointers to
/// ``QkObs``\ s.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_equal(
    obs: *const SparseObservable,
    other: *const SparseObservable,
) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };
    let other = unsafe { const_ptr_as_ref(other) };

    obs.eq(other)
}

/// @ingroup QkObs
/// Return a string representation of a ``QkObs``.
///
/// @param obs A pointer to the ``QkObs`` to get the string for.
///
/// @return A pointer to a nul-terminated char array of the string representation for ``obs``
///
/// # Example
///
///     QkObs *obs = qk_obs_identity(100);
///     char *string = qk_obs_str(obs);
///     qk_str_free(string);
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
///
/// The string must not be freed with the normal C free, you must use ``qk_str_free`` to
/// free the memory consumed by the String. Not calling ``qk_str_free`` will lead to a
/// memory leak.
///
/// Do not change the length of the string after it's returned (by writing a nul byte somewhere
/// inside the string or removing the final one), although values can be mutated.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_str(obs: *const SparseObservable) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };
    let string: String = format!("{:?}", obs);
    CString::new(string).unwrap().into_raw()
}

/// @ingroup QkObs
/// Free a string representation.
///
/// @param string A pointer to the returned string representation from ``qk_obs_str`` or
///     ``qk_obsterm_str``.
///
/// # Safety
///
/// Behavior is undefined if ``str`` is not a pointer returned by ``qk_obs_str`` or
/// ``qk_obsterm_str``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_str_free(string: *mut c_char) {
    unsafe {
        let _ = CString::from_raw(string);
    }
}

/// @ingroup QkObsTerm
/// Return a string representation of the sparse term.
///
/// @param term A pointer to the term.
///
/// @return The function exit code. This is ``>0`` if reading the term failed.
///
/// # Example
///
///     QkObs *obs = qk_obs_identity(100);
///     QkObsTerm term;
///     qk_obs_term(obs, 0, &term);
///     char *string = qk_obsterm_str(&term);
///     qk_str_free(string);
///
/// # Safety
///
/// Behavior is undefined ``term`` is not a valid, non-null pointer to a ``QkObsTerm``.
///
/// The string must not be freed with the normal C free, you must use ``qk_str_free`` to
/// free the memory consumed by the String. Not calling ``qk_str_free`` will lead to a
/// memory leak.
///
/// Do not change the length of the string after it's returned, although values can be mutated.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obsterm_str(term: *const CSparseTerm) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let term = unsafe { const_ptr_as_ref(term) };

    let view: SparseTermView = term.try_into().unwrap();
    let string: String = format!("{:?}", view);
    CString::new(string).unwrap().into_raw()
}

/// @ingroup QkBitTerm
/// Get the label for a bit term.
///
/// @param bit_term The bit term.
///
/// @return The label as ``uint8_t``, which can be cast to ``char`` to obtain the character.
///
/// # Example
///
///     QkBitTerm bit_term = QkBitTerm_Y;
///     // cast the uint8_t to char
///     char label = qk_bitterm_label(bit_term);
///
/// # Safety
///
/// The behavior is undefined if ``bit_term`` is not a valid ``uint8_t`` value of a ``QkBitTerm``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_bitterm_label(bit_term: BitTerm) -> u8 {
    // BitTerm is implemented as u8, which is calling convention compatible with C,
    // hence we can pass ``bit_term`` by value
    bit_term
        .py_label()
        .chars()
        .next()
        .expect("Label has exactly one character") as u8
}

/// @ingroup QkObs
/// Convert to a Python-space ``SparseObservable``.
///
/// @param obs The C-space ``QkObs`` pointer.
///
/// @return A Python object representing the ``SparseObservable``.
///
/// # Safety
///
/// Behavior is undefined if ``obs`` is not a valid, non-null pointer to a ``QkObs``.
///
/// It is assumed that the thread currently executing this function holds the
/// Python GIL this is required to create the Python object returned by this
/// function.
#[no_mangle]
#[cfg(feature = "python_binding")]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_to_python(obs: *const SparseObservable) -> *mut PyObject {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };
    let py_obs: PySparseObservable = obs.clone().into();

    // SAFETY: the C caller is required to hold the GIL.
    unsafe {
        let py = Python::assume_gil_acquired();
        Py::new(py, py_obs)
            .expect("Unable to create a Python object")
            .into_ptr()
    }
}
