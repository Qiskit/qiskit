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

/// @ingroup SparseTerm
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

        // not stable in Rust 1.70 yet
        // if !value.bit_terms.is_aligned() || !value.indices.is_aligned() {
        //     return Err(CInputError::AlignmentError);
        // }

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

/// @ingroup SparseObservable
/// Construct the zero observable (without any terms).
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// Example:
///
///     SparseObservable *zero = qk_obs_zero(100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_zero(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::zero(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup SparseObservable
/// Construct the identity observable.
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// Example:
///
///     SparseObservable *identity = qk_obs_identity(100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_identity(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::identity(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup SparseObservable
/// Free the observable.
///
/// @param obs A pointer to the observable to free.
///
/// Example:
///
///     SparseObservable *obs = qk_obs_zero(100);
///     qk_obs_free(obs);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_free(obs: &mut SparseObservable) {
    unsafe {
        let _ = Box::from_raw(obs);
    }
}

/// @ingroup SparseObservable
/// Add a term to the observable.
///
/// @param obs A pointer to the observable.
/// @param A pointer to the term to add.
///
/// @return An exit code. This is ``>0`` if the term is incoherent or adding the term fails.
///
/// Example:
///
///     uint32_t num_qubits = 100;
///     SparseObservable *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     BitTerm bit_terms[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     SparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///
///     int exit_code = qk_obs_add_term(obs, &term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_add_term(obs: &mut SparseObservable, cterm: &CSparseTerm) -> ExitCode {
    let view = match cterm.try_into() {
        Ok(view) => view,
        Err(err) => return ExitCode::from(err),
    };

    match obs.add_term(view) {
        Ok(_) => ExitCode::Success,
        Err(err) => ExitCode::from(err),
    }
}

/// @ingroup SparseObservable
/// Get an observable term by reference. This can modify the underlying observable.
///
/// @param obs A pointer to the observable.
/// @param index The index of the term to get.
/// @param out A pointer to a [CSparseTerm] used to return the observable term.
///
/// @return An exit code.
///
/// Example:
///
///     SparseObservable *obs = qk_obs_identity(100);
///     SparseTerm term;
///     int exit_code = qk_obs_term(obs, 0, &term);
///     // out-of-bounds indices return an error code
///     // int error = qk_obs_term(obs, 12, &term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_term(
    obs: &mut SparseObservable,
    index: u64,
    out: &mut CSparseTerm,
) -> ExitCode {
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

/// @ingroup SparseObservable
/// Get the number of terms in the observable.
///
/// @param observable A pointer to the observable.
///
/// @return The number of terms in the observable.
///
/// Example:
///
///     SparseObservable *obs = qk_obs_identity(100);
///     uint64_t num_terms = qk_obs_num_terms(obs);  // num_terms==1
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_num_terms(observable: &SparseObservable) -> u64 {
    observable.num_terms() as u64
}

/// @ingroup SparseObservable
/// Get the number of qubits the observable is defined on.
///
/// @param observable A pointer to the observable.
///
/// @return The number of qubits the observable is defined on.
///
/// Example:
///
///     SparseObservable *obs = qk_obs_identity(100);
///     uint32_t num_qubits = qk_obs_num_qubits(obs);  // num_qubits==100
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_num_qubits(observable: &SparseObservable) -> u32 {
    observable.num_qubits()
}

/// @ingroup SparseObservable
/// Get the number of bit terms/indices in the observable.
///
/// @param observable A pointer to the observable.
///
/// @return The number of terms in the observable.
///
/// Example:
///
///     SparseObservable *obs = qk_obs_identity(100);
///     uint64_t len = qk_obs_len(obs);  // len==0, as there are no non-trivial bit terms
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_len(observable: &SparseObservable) -> u64 {
    observable.bit_terms().len() as u64
}

/// @ingroup SparseObservable
/// Get a pointer to the coefficients.
///
/// This can be used to read and modify the observable's coefficients.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the coefficients.
///
/// Example:
///     
///     SparseObservable *obs = qk_obs_identity(100);
///     uint64_t num_terms = qk_obs_num_terms(obs);
///     complex double *coeffs = qk_obs_coeffs(obs);
///
///     for (uint64_t i = 0; i < num_terms; i++) {
///         printf("%f + i%f\n", creal(coeffs[i]), cimag(coeffs[i]));
///     }
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_coeffs(obs: &mut SparseObservable) -> *mut Complex64 {
    &mut obs.coeffs_mut()[0]
}

/// @ingroup SparseObservable
/// Get a pointer to the indices.
///
/// This can be used to read and modify the observable's indices.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the indices.
///
/// Example:
///
///     uint32_t num_qubits = 100;
///     SparseObservable *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     BitTerm bit_terms[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     SparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     uint64_t len = qk_obs_len(obs);
///     uint32_t *indices = qk_obs_indices(obs);
///
///     for (uint64_t i = 0; i < len; i++) {
///         printf("index %i: %i\n", i, indices[i]);
///     }
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_indices(obs: &mut SparseObservable) -> *mut u32 {
    // this is unsafe as we can no longer ensure the indices are within
    // range of the observable's number of qubits
    unsafe { &mut obs.indices_mut()[0] }
}

/// @ingroup SparseObservable
/// Get a pointer to the term boundaries.
///
/// This can be used to read and modify the observable's term boundaries.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the boundaries.
///
/// Example:
///
///     uint32_t num_qubits = 100;
///     SparseObservable *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     BitTerm bit_terms[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     SparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     uint64_t len = qk_obs_len(obs);
///     uint32_t *boundaries = qk_obs_boundaries(obs);
///
///     for (uint64_t i = 0; i < len + 1; i++) {
///         printf("boundary %i: %i\n", i, boundaries[i]);
///     }
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_boundaries(obs: &mut SparseObservable) -> *mut usize {
    // this is unsafe as modifying the boundaries can leave the observable
    // in an incoherent state
    unsafe { &mut obs.boundaries_mut()[0] }
}

/// @ingroup SparseObservable
/// Get a pointer to the bit terms.
///
/// This can be used to read and modify the observable's bit terms.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the bit terms.
///
/// Example:
///
///     uint32_t num_qubits = 100;
///     SparseObservable *obs = qk_obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     BitTerm bit_terms[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     SparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     uint64_t len = qk_obs_len(obs);
///     BitTerm *bits = qk_obs_bit_terms(obs);
///
///     for (uint64_t i = 0; i < len; i++) {
///         printf("bit term %i: %i\n", i, bits[i]);
///     }
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_bit_terms(obs: &mut SparseObservable) -> *mut BitTerm {
    &mut obs.bit_terms_mut()[0]
}

/// @ingroup SparseObservable
/// Multiply the observable by a complex coefficient.
///
/// @param obs A pointer to the observable.
/// @param coeff The coefficient to multiply the observable with.
///
/// Example:
///
///     SparseObservable *obs = qk_obs_identity(100);
///     SparseObservable *result = qk_obs_multiply(obs, 2);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_multiply(
    obs: &SparseObservable,
    coeff: &Complex64,
) -> *mut SparseObservable {
    let result = obs * (*coeff);
    Box::into_raw(Box::new(result))
}

/// @ingroup SparseObservable
/// Add two observables.
///
/// @param left A pointer to the left observable.
/// @param right A pointer to the right observable.
///
/// @return A pointer to the result ``left + right``.
///
/// Example:
///
///     SparseObservable *left = qk_obs_identity(100);
///     SparseObservable *right = qk_obs_zero(100);
///     SparseObservable *result = qk_obs_add(left, right);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_add(
    left: &SparseObservable,
    right: &SparseObservable,
) -> *mut SparseObservable {
    let result = left + right;
    Box::into_raw(Box::new(result))
}

/// @ingroup SparseObservable
/// Calculate the canonical representation of the observable.
///
/// @param obs A pointer to the observable.
/// @param tol The tolerance below which coefficients are considered to be zero.
///
/// @return The canonical representation of the observable.
///
/// Example:
///
///     SparseObservable *iden = qk_obs_identity(100);
///     SparseObservable *two = qk_obs_add(iden, iden);
///
///     double tol = 1e-6;
///     SparseObservable *canonical = qk_obs_canonicalize(two);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_canonicalize(
    obs: &SparseObservable,
    tol: f64, // no optional arguments in C -- welcome to the ancient past
) -> *mut SparseObservable {
    let result = obs.canonicalize(tol);
    Box::into_raw(Box::new(result))
}

/// @ingroup SparseObservable
/// Copy the observable.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to a copy of the observable.
///
/// Example:
///
///     SparseObservable *original = qk_obs_identity(100);
///     SparseObservable *copied = qk_obs_copy(original);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_copy(obs: &SparseObservable) -> *mut SparseObservable {
    let copied = obs.clone();
    Box::into_raw(Box::new(copied))
}

/// @ingroup SparseObservable
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
/// Example:
///
///     SparseObservable *observable = qk_obs_identity(100);
///     SparseObservable *other = qk_obs_identity(100);
///     bool are_equal = qk_obs_equal(observable, other);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_equal(obs: &SparseObservable, other: &SparseObservable) -> bool {
    obs.eq(other)
}

/// @ingroup SparseObservable
/// Print the observable.
///
/// @param obs A pointer to the ``SparseObservable`` to print.
///
/// Example:
///
///     SparseObservable *obs = qk_obs_identity(100);
///     qk_obs_print(obs);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obs_print(obs: &SparseObservable) {
    println!("{:?}", obs);
}

/// @ingroup SparseTerm
/// Print the sparse term.
///
/// @param term A pointer to the term.
///
/// @return The function exit code. This is ``>0`` if reading the term failed.
///
/// Example:
///
///     SparseObservable *obs = qk_obs_identity(100);
///     SparseTerm term;
///     qk_obs_term(obs, &term, 0);
///     int exit_code = qk_obsterm_print(&term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_obsterm_print(view: &CSparseTerm) -> ExitCode {
    let rust_view: SparseTermView = match view.try_into() {
        Ok(view) => view,
        Err(err) => return ExitCode::from(err),
    };

    println!("{:?}", rust_view);
    ExitCode::Success
}

/// @ingroup BitTerm
/// Get the label for a bit term.
///
/// @param bit The bit term.
///
/// @return The label as unsigned integer.
///
/// Example:
///     
///     BitTerm bit_term = BitTerm_Y;
///     // cast the uint32_t to char
///     char label = qk_bitterm_label(&bit_term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_bitterm_label(bit_term: &BitTerm) -> u32 {
    bit_term
        .py_label()
        .chars()
        .next()
        .expect("Label has exactly one character") as u32
}
