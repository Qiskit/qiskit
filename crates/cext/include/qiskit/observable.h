// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef QISKIT__OBSERVABLE_H
#define QISKIT__OBSERVABLE_H

#ifdef QISKIT_C_PYTHON_INTERFACE
#include <Python.h>
#endif

#include "complex.h"
#include <stdbool.h>
#include <stdlib.h>

/// Named handle to the alphabet of single-qubit terms.
///
/// This is just the Rust-space representation.  We make a separate Python-space `enum.IntEnum` to
/// represent the same information, since we enforce strongly typed interactions in Rust, including
/// not allowing the stored values to be outside the valid `BitTerm`s, but doing so in Python would
/// make it very difficult to use the class efficiently with Numpy array views.  We attach this
/// sister class of `BitTerm` to `SparseObservable` as a scoped class.
///
/// # Representation
///
/// The `u8` representation and the exact numerical values of these are part of the public API.  The
/// low two bits are the symplectic Pauli representation of the required measurement basis with Z in
/// the Lsb0 and X in the Lsb1 (e.g. X and its eigenstate projectors all have their two low bits as
/// `0b10`).  The high two bits are `00` for the operator, `10` for the projector to the positive
/// eigenstate, and `01` for the projector to the negative eigenstate.
///
/// The `0b00_00` representation thus ends up being the natural representation of the `I` operator,
/// but this is never stored, and is not named in the enumeration.
///
/// This operator does not store phase terms of $-i$.  `BitTerm::Y` has `(1, 1)` as its `(z, x)`
/// representation, and represents exactly the Pauli Y operator; any additional phase is stored only
/// in a corresponding coefficient.
///
/// # Dev notes
///
/// This type is required to be `u8`, but it's a subtype of `u8` because not all `u8` are valid
/// `BitTerm`s.  For interop with Python space, we accept Numpy arrays of `u8` to represent this,
/// which we transmute into slices of `BitTerm`, after checking that all the values are correct (or
/// skipping the check if Python space promises that it upheld the checks).
///
/// We deliberately _don't_ impl `numpy::Element` for `BitTerm` (which would let us accept and
/// return `PyArray1<BitTerm>` at Python-space boundaries) so that it's clear when we're doing
/// the transmute, and we have to be explicit about the safety of that.
enum QkBitTerm
#ifdef __cplusplus
    : uint8_t
#endif // __cplusplus
{
    /// Pauli X operator.
    QkBitTerm_X = 2,
    /// Projector to the positive eigenstate of Pauli X.
    QkBitTerm_Plus = 10,
    /// Projector to the negative eigenstate of Pauli X.
    QkBitTerm_Minus = 6,
    /// Pauli Y operator.
    QkBitTerm_Y = 3,
    /// Projector to the positive eigenstate of Pauli Y.
    QkBitTerm_Right = 11,
    /// Projector to the negative eigenstate of Pauli Y.
    QkBitTerm_Left = 7,
    /// Pauli Z operator.
    QkBitTerm_Z = 1,
    /// Projector to the positive eigenstate of Pauli Z.
    QkBitTerm_Zero = 9,
    /// Projector to the negative eigenstate of Pauli Z.
    QkBitTerm_One = 5,
};
#ifndef __cplusplus
typedef uint8_t QkBitTerm;
#endif // __cplusplus

/// An observable over Pauli bases that stores its data in a qubit-sparse format.
///
/// See [PySparseObservable] for detailed docs.
typedef struct QkObs QkObs;

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
typedef struct {
    /// The coefficient of the observable term.
    QkComplex64 coeff;
    /// Length of the ``bit_terms`` and ``indices`` arrays.
    uintptr_t len;
    /// A non-null, aligned pointer to ``len`` elements of type ``QkBitTerm``.
    QkBitTerm *bit_terms;
    /// A non-null, aligned pointer to ``len`` elements of type ``uint32_t``.
    uint32_t *indices;
    /// The number of qubits the observable term is defined on.
    uint32_t num_qubits;
} QkObsTerm;

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
QkObs *qk_obs_zero(uint32_t num_qubits);

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
QkObs *qk_obs_identity(uint32_t num_qubits);

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
///   * ``bit_terms`` is a pointer to an array of valid ``QkBitTerm`` elements of length
///   ``num_bits``
///   * ``indices`` is a pointer to a ``uint32_t`` array of length ``num_bits``, which is
///     term-wise sorted in strict ascending order, and every element is smaller than ``num_qubits``
///   * ``boundaries`` is a pointer to a ``size_t`` array of length ``num_terms + 1``, which is
///     sorted in ascending order, the first element is 0 and the last element is
///     smaller than ``num_terms``
QkObs *qk_obs_new(uint32_t num_qubits, uint64_t num_terms, uint64_t num_bits, QkComplex64 *coeffs,
                  QkBitTerm *bit_terms, uint32_t *indices, uintptr_t *boundaries);

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
void qk_obs_free(QkObs *obs);

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
QkExitCode qk_obs_add_term(QkObs *obs, const QkObsTerm *cterm);

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
QkExitCode qk_obs_term(QkObs *obs, uint64_t index, QkObsTerm *out);

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
uintptr_t qk_obs_num_terms(const QkObs *obs);

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
uint32_t qk_obs_num_qubits(const QkObs *obs);

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
uintptr_t qk_obs_len(const QkObs *obs);

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
QkComplex64 *qk_obs_coeffs(QkObs *obs);

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
///     size_t len = qk_obs_len(obs);
///     uint32_t *indices = qk_obs_indices(obs);
///
///     for (size_t i = 0; i < len; i++) {
///         printf("index %i: %i\n", i, indices[i]);
///     }
///
///     qk_obs_free(obs);
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
uint32_t *qk_obs_indices(QkObs *obs);

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
uintptr_t *qk_obs_boundaries(QkObs *obs);

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
///     qk_obs_free(obs);
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``,
/// or if invalid valus are written into the resulting ``QkBitTerm`` pointer.
QkBitTerm *qk_obs_bit_terms(QkObs *obs);

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
QkObs *qk_obs_multiply(const QkObs *obs, const QkComplex64 *coeff);

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
QkObs *qk_obs_add(const QkObs *left, const QkObs *right);

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
QkObs *qk_obs_compose(const QkObs *first, const QkObs *second);

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
QkObs *qk_obs_compose_map(const QkObs *first, const QkObs *second, const uint32_t *qargs);

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
QkObs *qk_obs_canonicalize(const QkObs *obs, double tol);

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
QkObs *qk_obs_copy(const QkObs *obs);

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
bool qk_obs_equal(const QkObs *obs, const QkObs *other);

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
char *qk_obs_str(const QkObs *obs);

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
void qk_str_free(char *string);

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
char *qk_obsterm_str(const QkObsTerm *term);

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
uint8_t qk_bitterm_label(QkBitTerm bit_term);

#ifdef QISKIT_C_PYTHON_INTERFACE
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
PyObject *qk_obs_to_python(const QkObs *obs);
#endif

#endif // QISKIT__OBSERVABLE_H
