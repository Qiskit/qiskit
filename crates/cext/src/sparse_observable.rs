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

use crate::exit_codes::{CInputError, ExitCode};
use num_complex::Complex64;
use qiskit_accelerate::sparse_observable::{BitTerm, SparseObservable, SparseTermView};

/// @ingroup QkSparseTerm
/// A term in a [SparseObservable].
///
/// This term does not own the data, but only contains pointers (and a size) to the
/// bit terms, indices and the coefficient.
#[repr(C)]
pub struct CSparseTerm {
    coeff: *mut Complex64,
    len: usize,
    bit_terms: *mut BitTerm,
    indices: *mut u32,
    num_qubits: u32,
}

impl TryFrom<&CSparseTerm> for SparseTermView<'_> {
    type Error = CInputError;

    fn try_from(value: &CSparseTerm) -> Result<Self, Self::Error> {
        if value.bit_terms.is_null() || value.indices.is_null() {
            return Err(CInputError::NullPointerError);
        }

        if !value.bit_terms.is_aligned() || !value.indices.is_aligned() {
            return Err(CInputError::AlignmentError);
        }

        let bit_terms = unsafe { ::std::slice::from_raw_parts(value.bit_terms, value.len) };
        let indices = unsafe { ::std::slice::from_raw_parts(value.indices, value.len) };

        Ok(SparseTermView {
            num_qubits: value.num_qubits,
            coeff: unsafe { *value.coeff },
            bit_terms,
            indices,
        })
    }
}

/// Check the pointer is not null and is aligned.
fn check_ptr<T>(ptr: *const T) {
    if ptr.is_null() {
        panic!("Unexpected null pointer.");
    };
    if !ptr.is_aligned() {
        panic!("Pointer is not properly aligned.");
    };
}

/// Check a const pointer is not null and aligned, then cast it to a reference.
///
/// # Safety
///
/// This function is unsafe since it is reading from a raw pointer. Even if the pointer
/// is not null and aligned, the memory could still be an invalid representation of the
/// target type ``T``.
unsafe fn const_ptr_as_ref<'a, T>(ptr: *const T) -> &'a T {
    check_ptr(ptr);
    let as_ref = unsafe { ptr.as_ref() };
    as_ref.unwrap() // we know the pointer is not null, hence we can safely unwrap
}

/// Check a const pointer is not null and aligned, then cast it to a reference.
///
/// # Safety
///
/// This function is unsafe since it is reading from a raw pointer. Even if the pointer
/// is not null and aligned, the memory could still be an invalid representation of the
/// target type ``T``.
unsafe fn mut_ptr_as_ref<'a, T>(ptr: *mut T) -> &'a mut T {
    check_ptr(ptr);
    let as_mut_ref = unsafe { ptr.as_mut() };
    as_mut_ref.unwrap() // we know the pointer is not null, hence we can safely unwrap
}

/// @ingroup QkSparseObservable
/// Construct the zero observable (without any terms).
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// # Example
///
///     QkSparseObservable *zero = qk_obs_zero(100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_zero(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::zero(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup QkSparseObservable
/// Construct the identity observable.
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// # Example
///
///     QkSparseObservable *identity = qk_obs_identity(100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_identity(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::identity(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup QkSparseObservable
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
///     QkSparseObservable *obs = qk_obs_new(
///         num_qubits, num_terms, num_bits, coeffs, bits, indices, boundaries
///     );
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
///   * ``coeffs`` is a pointer to a ``complex double`` array of length ``num_terms``
///   * ``bit_terms`` is a pointer to a ``QkBitTerm`` array of length ``num_bits``
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

/// @ingroup QkSparseObservable
/// Free the observable.
///
/// @param obs A pointer to the observable to free.
///
/// # Example
///
///     QkSparseObservable *obs = qk_obs_zero(100);
///     qk_obs_free(obs);
///
/// # Safety
///
/// This is unsafe as it is reading and freeing raw memory, which could lead to problems
/// e.g. if the same pointer is freed multiple times.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_free(obs: *mut SparseObservable) {
    if !obs.is_null() {
        unsafe {
            let _ = Box::from_raw(obs);
        }
    }
}

/// @ingroup QkSparseObservable
/// Add a term to the observable.
///
/// @param obs A pointer to the observable.
/// @param A pointer to the term to add.
///
/// @return An exit code. This is ``>0`` if the term is incoherent or adding the term fails.
///
/// # Example
///
///     uint32_t num_qubits = 100;
///     QkSparseObservable *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     QkSparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///
///     int exit_code = qk_obs_add_term(obs, &term);
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated:
///
///   * ``obs`` is a valid, non-null pointer to a ``QkSparseObservable``
///   * ``cterm`` is a valid, non-null pointer to a ``QkSparseTerm``
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_add_term(
    obs: *mut SparseObservable,
    cterm: *const CSparseTerm,
) -> ExitCode {
    // check pointers are valid and cast them to a reference
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

/// @ingroup QkSparseObservable
/// Get an observable term by reference. This can modify the underlying observable.
///
/// @param obs A pointer to the observable.
/// @param index The index of the term to get.
/// @param out A pointer to a [CSparseTerm] used to return the observable term.
///
/// @return An exit code.
///
/// # Example
///
///     QkSparseObservable *obs = qk_obs_identity(100);
///     QkSparseTerm term;
///     int exit_code = qk_obs_term(obs, 0, &term);
///     // out-of-bounds indices return an error code
///     // int error = qk_obs_term(obs, 12, &term);
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated
/// * ``obs`` is a valid, non-null pointer to a ``QkSparseObservable``
/// * ``out`` is a valid, non-null pointer to a ``QkSparseTerm``
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_term(
    obs: *mut SparseObservable,
    index: u64,
    out: *mut CSparseTerm,
) -> ExitCode {
    let out = unsafe { mut_ptr_as_ref(out) };
    let obs = unsafe { mut_ptr_as_ref(obs) };

    let index = index as usize;
    if index > obs.num_terms() {
        return ExitCode::IndexError;
    }

    out.len = obs.boundaries()[index + 1] - obs.boundaries()[index];
    out.coeff = &mut obs.coeffs_mut()[index];
    out.num_qubits = obs.num_qubits();

    let start = obs.boundaries()[index];
    out.bit_terms = &mut obs.bit_terms_mut()[start];
    out.indices = unsafe { &mut obs.indices_mut()[start] };

    ExitCode::Success
}

/// @ingroup QkSparseObservable
/// Get the number of terms in the observable.
///
/// @param obs A pointer to the observable.
///
/// @return The number of terms in the observable.
///
/// # Example
///
///     QkSparseObservable *obs = qk_obs_identity(100);
///     uint64_t num_terms = qk_obs_num_terms(obs);  // num_terms==1
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_num_terms(obs: *const SparseObservable) -> u64 {
    let obs = unsafe { const_ptr_as_ref(obs) };
    obs.num_terms() as u64
}

/// @ingroup QkSparseObservable
/// Get the number of qubits the observable is defined on.
///
/// @param obs A pointer to the observable.
///
/// @return The number of qubits the observable is defined on.
///
/// # Example
///
///     QkSparseObservable *obs = qk_obs_identity(100);
///     uint32_t num_qubits = qk_obs_num_qubits(obs);  // num_qubits==100
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_num_qubits(obs: *const SparseObservable) -> u32 {
    let obs = unsafe { const_ptr_as_ref(obs) };
    obs.num_qubits()
}

/// @ingroup QkSparseObservable
/// Get the number of bit terms/indices in the observable.
///
/// @param obs A pointer to the observable.
///
/// @return The number of terms in the observable.
///
/// # Example
///
///     QkSparseObservable *obs = qk_obs_identity(100);
///     uint64_t len = qk_obs_len(obs);  // len==0, as there are no non-trivial bit terms
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_len(obs: *const SparseObservable) -> u64 {
    let obs = unsafe { const_ptr_as_ref(obs) };
    obs.bit_terms().len() as u64
}

/// @ingroup QkSparseObservable
/// Get a pointer to the coefficients.
///
/// This can be used to read and modify the observable's coefficients.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the coefficients.
///
/// # Example
///     
///     QkSparseObservable *obs = qk_obs_identity(100);
///     uint64_t num_terms = qk_obs_num_terms(obs);
///     complex double *coeffs = qk_obs_coeffs(obs);
///
///     for (uint64_t i = 0; i < num_terms; i++) {
///         printf("%f + i%f\n", creal(coeffs[i]), cimag(coeffs[i]));
///     }
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_coeffs(obs: *mut SparseObservable) -> *mut Complex64 {
    let obs = unsafe { mut_ptr_as_ref(obs) };
    &mut obs.coeffs_mut()[0]
}

/// @ingroup QkSparseObservable
/// Get a pointer to the indices.
///
/// This can be used to read and modify the observable's indices.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the indices.
///
/// # Example
///
///     uint32_t num_qubits = 100;
///     QkSparseObservable *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     QkSparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     uint64_t len = qk_obs_len(obs);
///     uint32_t *indices = qk_obs_indices(obs);
///
///     for (uint64_t i = 0; i < len; i++) {
///         printf("index %i: %i\n", i, indices[i]);
///     }
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_indices(obs: *mut SparseObservable) -> *mut u32 {
    let obs = unsafe { mut_ptr_as_ref(obs) };

    // this is unsafe as we can no longer ensure the indices are within
    // range of the observable's number of qubits
    unsafe { &mut obs.indices_mut()[0] }
}

/// @ingroup QkSparseObservable
/// Get a pointer to the term boundaries.
///
/// This can be used to read and modify the observable's term boundaries.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the boundaries.
///
/// # Example
///
///     uint32_t num_qubits = 100;
///     QkSparseObservable *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     QkSparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     uint64_t len = qk_obs_len(obs);
///     uint32_t *boundaries = qk_obs_boundaries(obs);
///
///     for (uint64_t i = 0; i < len + 1; i++) {
///         printf("boundary %i: %i\n", i, boundaries[i]);
///     }
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_boundaries(obs: *mut SparseObservable) -> *mut usize {
    let obs = unsafe { mut_ptr_as_ref(obs) };

    // this is unsafe as modifying the boundaries can leave the observable
    // in an incoherent state
    unsafe { &mut obs.boundaries_mut()[0] }
}

/// @ingroup QkSparseObservable
/// Get a pointer to the bit terms.
///
/// This can be used to read and modify the observable's bit terms.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the bit terms.
///
/// # Example
///
///     uint32_t num_qubits = 100;
///     QkSparseObservable *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     QkSparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     uint64_t len = qk_obs_len(obs);
///     QkBitTerm *bits = qk_obs_bit_terms(obs);
///
///     for (uint64_t i = 0; i < len; i++) {
///         printf("bit term %i: %i\n", i, bits[i]);
///     }
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_bit_terms(obs: *mut SparseObservable) -> *mut BitTerm {
    let obs = unsafe { mut_ptr_as_ref(obs) };
    &mut obs.bit_terms_mut()[0]
}

/// @ingroup QkSparseObservable
/// Multiply the observable by a complex coefficient.
///
/// @param obs A pointer to the observable.
/// @param coeff The coefficient to multiply the observable with.
///
/// # Example
///
///     QkSparseObservable *obs = qk_obs_identity(100);
///     QkSparseObservable *result = qk_obs_multiply(obs, 2);
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated
/// * ``obs`` is a valid, non-null pointer to a ``QkSparseObservable``
/// * ``coeff`` is a valid, non-null pointer to a ``complex double``
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_multiply(
    obs: *const SparseObservable,
    coeff: *const Complex64,
) -> *mut SparseObservable {
    let obs = unsafe { const_ptr_as_ref(obs) };
    let coeff = unsafe { const_ptr_as_ref(coeff) };

    let result = obs * (*coeff);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkSparseObservable
/// Add two observables.
///
/// @param left A pointer to the left observable.
/// @param right A pointer to the right observable.
///
/// @return A pointer to the result ``left + right``.
///
/// # Example
///
///     QkSparseObservable *left = qk_obs_identity(100);
///     QkSparseObservable *right = qk_obs_zero(100);
///     QkSparseObservable *result = qk_obs_add(left, right);
///
/// # Safety
///
/// Behavior is undefined if ``left`` or ``right`` are not valid, non-null pointers to
/// ``QkSparseObservable``\ s.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_add(
    left: *const SparseObservable,
    right: *const SparseObservable,
) -> *mut SparseObservable {
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    let result = left + right;
    Box::into_raw(Box::new(result))
}

/// @ingroup QkSparseObservable
/// Calculate the canonical representation of the observable.
///
/// @param obs A pointer to the observable.
/// @param tol The tolerance below which coefficients are considered to be zero.
///
/// @return The canonical representation of the observable.
///
/// # Example
///
///     QkSparseObservable *iden = qk_obs_identity(100);
///     QkSparseObservable *two = qk_obs_add(iden, iden);
///
///     double tol = 1e-6;
///     QkSparseObservable *canonical = qk_obs_canonicalize(two);
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_canonicalize(
    obs: *const SparseObservable,
    tol: f64, // no optional arguments in C -- welcome to the ancient past
) -> *mut SparseObservable {
    let obs = unsafe { const_ptr_as_ref(obs) };

    let result = obs.canonicalize(tol);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkSparseObservable
/// Copy the observable.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to a copy of the observable.
///
/// # Example
///
///     QkSparseObservable *original = qk_obs_identity(100);
///     QkSparseObservable *copied = qk_obs_copy(original);
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_copy(obs: *const SparseObservable) -> *mut SparseObservable {
    let obs = unsafe { const_ptr_as_ref(obs) };

    let copied = obs.clone();
    Box::into_raw(Box::new(copied))
}

/// @ingroup QkSparseObservable
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
///     QkSparseObservable *observable = qk_obs_identity(100);
///     QkSparseObservable *other = qk_obs_identity(100);
///     bool are_equal = qk_obs_equal(observable, other);
///
/// # Safety
///
/// Behavior is undefined if ``obs`` or ``other`` are not valid, non-null pointers to
/// ``QkSparseObservable``\ s.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_equal(
    obs: *const SparseObservable,
    other: *const SparseObservable,
) -> bool {
    let obs = unsafe { const_ptr_as_ref(obs) };
    let other = unsafe { const_ptr_as_ref(other) };

    obs.eq(other)
}

/// @ingroup QkSparseObservable
/// Print the observable.
///
/// @param obs A pointer to the ``SparseObservable`` to print.
///
/// # Example
///
///     QkSparseObservable *obs = qk_obs_identity(100);
///     qk_obs_print(obs);
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkSparseObservable``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obs_print(obs: *const SparseObservable) {
    let obs = unsafe { const_ptr_as_ref(obs) };
    println!("{:?}", obs);
}

/// @ingroup QkSparseTerm
/// Print the sparse term.
///
/// @param term A pointer to the term.
///
/// @return The function exit code. This is ``>0`` if reading the term failed.
///
/// # Example
///
///     QkSparseObservable *obs = qk_obs_identity(100);
///     QkSparseTerm term;
///     qk_obs_term(obs, &term, 0);
///     int exit_code = qk_obsterm_print(&term);
///
/// # Safety
///
/// Behavior is undefined ``term`` is not a valid, non-null pointer to a ``QkSparseTerm``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_obsterm_print(term: *const CSparseTerm) -> ExitCode {
    let term = unsafe { const_ptr_as_ref(term) };
    let view: SparseTermView = match term.try_into() {
        Ok(view) => view,
        Err(err) => return ExitCode::from(err),
    };

    println!("{:?}", view);
    ExitCode::Success
}

/// @ingroup QkBitTerm
/// Get the label for a bit term.
///
/// @param bit The bit term.
///
/// @return The label as unsigned integer.
///
/// # Example
///     
///     QkBitTerm bit_term = QkBitTerm_Y;
///     // cast the uint32_t to char
///     char label = qk_bitterm_label(bit_term);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_bitterm_label(bit_term: BitTerm) -> u32 {
    // BitTerm is implemented as u8, which is calling convention compatible with C,
    // hence we can pass ``bit_term`` by value
    bit_term
        .py_label()
        .chars()
        .next()
        .expect("Label has exactly one character") as u32
}
