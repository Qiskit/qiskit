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
#include "exit.h"
#include <stdbool.h>
#include <stdlib.h>

/**
 * Named handle to the alphabet of single-qubit observable terms.
 *
 * The numeric structure of these is that they are all four-bit values of which the low two
 * bits are the (phase-less) symplectic representation of the Pauli operator related to the
 * object, where the low bit denotes a contribution by :math:`Z` and the second lowest a
 * contribution by :math:`X`, while the upper two bits are ``00`` for a Pauli operator, ``01``
 * for the negative-eigenstate projector, and ``10`` for the positive-eigenstate projector.
 *
 * The enum is stored as single byte, its elements are represented as unsigned 8-bit integer.
 *
 * .. warning::
 *
 *   Not all ``uint8_t`` values are valid bit terms. Passing invalid values is undefined behavior.
 *
 */
enum QkBitTerm
#ifdef __cplusplus
    : uint8_t
#endif // __cplusplus
{
    /* Pauli X operator. */
    QkBitTerm_X = 2,
    /* Projector to the positive eigenstate of Pauli X. */
    QkBitTerm_Plus = 10,
    /* Projector to the negative eigenstate of Pauli X. */
    QkBitTerm_Minus = 6,
    /* Pauli Y operator. */
    QkBitTerm_Y = 3,
    /* Projector to the positive eigenstate of Pauli Y. */
    QkBitTerm_Right = 11,
    /* Projector to the negative eigenstate of Pauli Y. */
    QkBitTerm_Left = 7,
    /* Pauli Z operator. */
    QkBitTerm_Z = 1,
    /* Projector to the positive eigenstate of Pauli Z. */
    QkBitTerm_Zero = 9,
    /* Projector to the negative eigenstate of Pauli Z. */
    QkBitTerm_One = 5,
};
#ifndef __cplusplus
typedef uint8_t QkBitTerm;
#endif // not __cplusplus

/**
 * An observable over Pauli bases that stores its data in a qubit-sparse format.
 *
 *
 * Mathematics
 * ===========
 *
 * This observable represents a sum over strings of the Pauli operators and Pauli-eigenstate
 * projectors, with each term weighted by some complex number.  That is, the full observable is
 *
 * .. math::
 *     \text{\texttt{QkObs}} = \sum_i c_i \bigotimes_n A^{(n)}_i
 *
 * for complex numbers :math:`c_i` and single-qubit operators acting on qubit :math:`n` from a
 * restricted alphabet :math:`A^{(n)}_i`.  The sum over :math:`i` is the sum of the individual
 * terms, and the tensor product produces the operator strings.
 * The alphabet of allowed single-qubit operators that the :math:`A^{(n)}_i` are drawn from is the
 * Pauli operators and the Pauli-eigenstate projection operators.  Explicitly, these are:
 *
 * .. _qkobs-alphabet:
 * .. table:: Alphabet of single-qubit terms used in ``QkObs``
 *
 *   +----------------------------------------+--------------------+----------------+
 *   | Operator                               | ``QkBitTerm``      | Numeric value  |
 *   +========================================+====================+================+
 *   | :math:`I` (identity)                   | Not stored.        | Not stored.    |
 *   +----------------------------------------+--------------------+----------------+
 *   | :math:`X` (Pauli X)                    | ``QkBitTerm_X``    | ``0b0010`` (2) |
 *   +----------------------------------------+--------------------+----------------+
 *   | :math:`Y` (Pauli Y)                    | ``QkBitTerm_Y``    | ``0b0011`` (3) |
 *   +----------------------------------------+--------------------+----------------+
 *   | :math:`Z` (Pauli Z)                    | ``QkBitTerm_Z``    | ``0b0001`` (1) |
 *   +----------------------------------------+--------------------+----------------+
 *   | :math:`\lvert+\rangle\langle+\rvert`   | ``QkBitTerm_Plus`` | ``0b1010`` (10)|
 *   | (projector to positive eigenstate of X)|                    |                |
 *   +----------------------------------------+--------------------+----------------+
 *   | :math:`\lvert-\rangle\langle-\rvert`   | ``QkBitTerm_Minus``| ``0b0110`` (6) |
 *   | (projector to negative eigenstate of X)|                    |                |
 *   +----------------------------------------+--------------------+----------------+
 *   | :math:`\lvert r\rangle\langle r\rvert` | ``QkBitTerm_Right``| ``0b1011`` (11)|
 *   | (projector to positive eigenstate of Y)|                    |                |
 *   +----------------------------------------+--------------------+----------------+
 *   | :math:`\lvert l\rangle\langle l\rvert` | ``QkBitTerm_Left`` | ``0b0111`` (7) |
 *   | (projector to negative eigenstate of Y)|                    |                |
 *   +----------------------------------------+--------------------+----------------+
 *   | :math:`\lvert0\rangle\langle0\rvert`   | ``QkBitTerm_Zero`` | ``0b1001`` (9) |
 *   | (projector to positive eigenstate of Z)|                    |                |
 *   +----------------------------------------+--------------------+----------------+
 *   | :math:`\lvert1\rangle\langle1\rvert`   | ``QkBitTerm_One``  | ``0b0101`` (5) |
 *   | (projector to negative eigenstate of Z)|                    |                |
 *   +----------------------------------------+--------------------+----------------+
 *
 * Due to allowing both the Paulis and their projectors, the allowed alphabet forms an overcomplete
 * basis of the operator space.  This means that there is not a unique summation to represent a
 * given observable. As a consequence, comparison requires additional care and using
 * ``qk_obs_canonicalize`` on two mathematically equivalent observables might not result in the same
 * representation.
 *
 * ``QkObs`` uses its particular overcomplete basis with the aim of making
 * "efficiency of measurement" equivalent to "efficiency of representation".  For example, the
 * observable :math:`{\lvert0\rangle\langle0\rvert}^{\otimes n}` can be efficiently measured on
 * hardware with simple :math:`Z` measurements, but can only be represented in terms of Paulis
 * as :math:`{(I + Z)}^{\otimes n}/2^n`, which requires :math:`2^n` stored terms. ``QkObs`` requires
 * only a single term to store this. The downside to this is that it is impractical to take an
 * arbitrary matrix and find the *best* ``QkObs`` representation.  You typically will want to
 * construct a ``QkObs`` directly, rather than trying to decompose into one.
 *
 *
 * Representation
 * ==============
 *
 * The internal representation of a ``QkObs`` stores only the non-identity qubit
 * operators.  This makes it significantly more efficient to represent observables such as
 * :math:`\sum_{n\in \text{qubits}} Z^{(n)}`; ``QkObs`` requires an amount of
 * memory linear in the total number of qubits.
 * The terms are stored compressed, similar in spirit to the compressed sparse row format of sparse
 * matrices.  In this analogy, the terms of the sum are the "rows", and the qubit terms are the
 * "columns", where an absent entry represents the identity rather than a zero.  More explicitly,
 * the representation is made up of four contiguous arrays:
 *
 * .. _qkobs-arrays:
 * .. table:: Data arrays used to represent ``QkObs``
 *
 *   =======================  ===========
 * ============================================================ Attribute accessible by  Length
 * Description
 *   =======================  ===========
 * ============================================================
 *   ``qk_obs_coeffs``        :math:`t`    The complex scalar multiplier for each term.
 *
 *   ``qk_obs_bit_terms``     :math:`s`    Each of the non-identity single-qubit terms for all of
 *                                         the operators, in order. These correspond to the
 *                                         non-identity :math:`A^{(n)}_i` in the sum description,
 *                                         where the entries are stored in order of increasing
 *                                         :math:`i` first, and in order of increasing :math:`n`
 *                                         within each term.
 *
 *   ``qk_obs_indices``       :math:`s`    The corresponding qubit (:math:`n`) for each of the
 *                                         bit terms. ``QkObs`` requires that this list is term-wise
 *                                         sorted, and algorithms can rely on this invariant being
 *                                         upheld.
 *
 *   ``qk_obs_boundaries``    :math:`t+1`  The indices that partition the bit terms and indices
 *                                         into complete terms.  For term number :math:`i`, its
 *                                         complex coefficient is stored at index ``i``, and its
 *                                         non-identity single-qubit operators and their
 * corresponding qubits are in the range ``[boundaries[i], boundaries[i+1])`` in the bit terms and
 * indices, respectively. The boundaries always have an explicit 0 as their first element.
 *   =======================  ===========
 * ============================================================
 *
 * The length parameter :math:`t` is the number of terms in the sum and can be queried using
 * ``qk_obs_num_terms``. The parameter :math:`s` is the total number of non-identity single-qubit
 * terms and can be queried using ``qk_obs_len``.
 *
 * As illustrative examples:
 *
 * * in the case of a zero operator, the boundaries are length 1 (a single 0) and all other
 *   vectors are empty.
 *
 * * in the case of a fully simplified identity operator, the boundaries are ``{0, 0}``,
 *   the coefficients have a single entry, and both the bit terms and indices are empty.
 *
 * * for the operator :math:`Z_2 Z_0 - X_3 Y_1`, the boundaries are ``{0, 2, 4}``,
 *   the coeffs are ``{1.0, -1.0}``, the bit terms are ``{QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Y,
 *   QkBitTerm_X}`` and the indices are ``{0, 2, 1, 3}``.  The operator might act on more than
 *   four qubits, depending on the the number of qubits (see ``qk_obs_num_qubits``). Note
 *   that the single-bit terms and indices are sorted into termwise sorted order.
 *
 * These cases are not special, they're fully consistent with the rules and should not need special
 * handling.
 *
 *
 * Canonical ordering
 * ------------------
 *
 * For any given mathematical observable, there are several ways of representing it with
 * ``QkObs``.  For example, the same set of single-bit terms and their corresponding indices might
 * appear multiple times in the observable.  Mathematically, this is equivalent to having only a
 * single term with all the coefficients summed.  Similarly, the terms of the sum in a ``QkObs``
 * can be in any order while representing the same observable, since addition is commutative
 * (although while floating-point addition is not associative, ``QkObs`` makes no guarantees about
 * the summation order).
 *
 * These two categories of representation degeneracy can cause the operator equality,
 * ``qk_obs_equal``, to claim that two observables are not equal, despite representating the same
 * object.  In these cases, it can be convenient to define some *canonical form*, which allows
 * observables to be compared structurally.
 * You can put a ``QkObs`` in canonical form by using the ``qk_obs_canonicalize`` function.
 * The precise ordering of terms in canonical ordering is not specified, and may change between
 * versions of Qiskit.  Within the same version of Qiskit, however, you can compare two observables
 * structurally by comparing their simplified forms.
 *
 * .. note::
 *
 *     If you wish to account for floating-point tolerance in the comparison, it is safest to use
 *     a recipe such as:
 *
 *     .. code-block:: c
 *
 *         bool equivalent(QkObs *left, QkObs *right, double tol) {
 *             // compare a canonicalized version of left - right to the zero observable
 *             QkObs *neg_right = qk_obs_mul(right, -1);
 *             QkObs *diff = qk_obs_add(left, neg_right);
 *             QkObs *canonical = qk_obs_canonicalize(diff, tol);
 *
 *             QkObs *zero = qk_obs_zero(qk_obs_num_qubits(left));
 *             bool equiv = qk_obs_equal(diff, zero);
 *             // free all temporary variables
 *             qk_obs_free(neg_right);
 *             qk_obs_free(diff);
 *             qk_obs_free(canonical);
 *             qk_obs_free(zero);
 *             return equiv;
 *         }
 *
 * .. note::
 *
 *     The canonical form produced by ``qk_obs_canonicalize`` alone will not universally detect all
 *     observables that are equivalent due to the over-complete basis alphabet.
 *
 *
 * Indexing
 * --------
 *
 * Individual observable sum terms in ``QkObs`` can be accessed via ``qk_obs_term`` and return
 * objects of type ``QkObsTerm``. These terms then contain fields with the coefficient of the term,
 * its bit terms, indices and the number of qubits it is defined on. Together with the information
 * of the number of terms, you can iterate over all observable terms as
 *
 * .. code-block:: c
 *
 *     size_t num_terms = qk_obs_num_terms(obs);  // obs is QkObs*
 *     for (size_t i = 0; i < num_terms; i++) {
 *         QkObsTerm term;  // allocate term on stack
 *         int exit = qk_obs_term(obs, i, &term);  // get the term (exit > 0 upon index errors)
 *         // do something with the term...
 *     }
 *
 * .. warning::
 *
 *     Populating a ``QkObsTerm`` via ``qk_obs_term`` will reference data of the original
 *     ``QkObs``. Modifying the bit terms or indices will change the observable and can leave
 *     it in an incoherent state.
 *
 *
 * Construction
 * ============
 *
 * ``QkObs`` can be constructed by initializing an empty observable (with ``qk_obs_zero``) and
 * iteratively adding terms (with ``qk_obs_add_term``). Alternatively, an observable can be
 * constructed from "raw" data (with ``qk_obs_new``) if all internal data is specified. This
 * requires care to ensure the data is coherent and results in a valid observable.
 *
 * .. _qkobs-constructors:
 * .. table:: Constructors
 *
 *   ===================  =========================================================================
 *   Function             Summary
 *   ===================  =========================================================================
 *   ``qk_obs_zero``      Construct an empty observable on a given number of qubits.
 *
 *   ``qk_obs_identity``  Construct the identity observable on a given number of qubits.
 *
 *   ``qk_obs_new``       Construct an observable from :ref:`the raw data arrays <qkobs-arrays>`.
 *   ===================  =========================================================================
 *
 *
 * Mathematical manipulation
 * =========================
 *
 * ``QkObs`` supports fundamental arithmetic operations in between observables or with scalars.
 * You can:
 *
 * * add two observables using ``qk_obs_add``
 *
 * * multiply by a complex number with ``qk_obs_multiply``
 *
 * * compose (multiply) two observables via ``qk_obs_compose`` and ``qk_obs_compose_map``
 */
typedef struct QkObs QkObs;

/**
 * A term in a ``QkObs``.
 *
 * This contains the coefficient (``coeff``), the number of qubits of the observable
 * (``num_qubits``) and pointers to the ``bit_terms`` and ``indices`` arrays, which have
 * length ``len``. It's the responsibility of the user that the data is coherent,
 * see also the below section on safety.
 *
 * # Safety
 *
 * * ``bit_terms`` must be a non-null, aligned pointer to ``len`` elements of type ``QkBitTerm``.
 * * ``indices`` must be a non-null, aligned pointer to ``len`` elements of type ``uint32_t``.
 */
typedef struct {
    /* The coefficient of the observable term. */
    QkComplex64 coeff;
    /* Length of the ``bit_terms`` and ``indices`` arrays. */
    uintptr_t len;
    /* A non - null, aligned pointer to ``len`` elements of type ``QkBitTerm``. */
    QkBitTerm *bit_terms;
    /* A non - null, aligned pointer to ``len`` elements of type ``uint32_t``. */
    uint32_t *indices;
    /* The number of qubits the observable term is defined on. */
    uint32_t num_qubits;
} QkObsTerm;

/**
 * @ingroup QkObsMethods
 * Construct the zero observable (without any terms).
 *
 * @param num_qubits The number of qubits the observable is defined on.
 *
 * @return A pointer to the created observable.
 *
 * # Example
 *
 *    QkObs *zero = qk_obs_zero(100);
 *
 */
QkObs *qk_obs_zero(uint32_t num_qubits);

/**
 * @ingroup QkObsMethods
 * Construct the identity observable.
 *
 * @param num_qubits The number of qubits the observable is defined on.
 *
 * @return A pointer to the created observable.
 *
 * # Example
 *
 *    QkObs *identity = qk_obs_identity(100);
 *
 */
QkObs *qk_obs_identity(uint32_t num_qubits);

/**
 * @ingroup QkObsMethods
 * Construct a new observable from raw data.
 *
 * @param num_qubits The number of qubits the observable is defined on.
 * @param num_terms The number of terms.
 * @param num_bits The total number of non-identity bit terms.
 * @param coeffs A pointer to the first element of the coefficients array, which has length
 *    ``num_terms``.
 * @param bit_terms A pointer to the first element of the bit terms array, which has length
 *    ``num_bits``.
 * @param indices A pointer to the first element of the indices array, which has length
 *    ``num_bits``. Note that, per term, these *must* be sorted incrementally.
 * @param boundaries A pointer to the first element of the boundaries array, which has length
 *    ``num_terms + 1``.
 *
 * @return If the input data was coherent and the construction successful, the result is a pointer
 *    to the observable. Otherwise a null pointer is returned.
 *
 * # Example
 *
 *    // define the raw data for the 100-qubit observable |01><01|_{0, 1} - |+-><+-|_{98, 99}
 *    uint32_t num_qubits = 100;
 *    uint64_t num_terms = 2;  // we have 2 terms: |01><01|, -1 * |+-><+-|
 *    uint64_t num_bits = 4; // we have 4 non-identity bits: 0, 1, +, -
 *
 *    complex double coeffs[2] = {1, -1};
 *    QkBitTerm bits[4] = {QkBitTerm_Zero, QkBitTerm_One, QkBitTerm_Plus, QkBitTerm_Minus};
 *    uint32_t indices[4] = {0, 1, 98, 99};  // <-- e.g. {1, 0, 99, 98} would be invalid
 *    size_t boundaries[3] = {0, 2, 4};
 *
 *    QkObs *obs = qk_obs_new(
 *        num_qubits, num_terms, num_bits, coeffs, bits, indices, boundaries
 *    );
 *
 * # Safety
 *
 * Behavior is undefined if any of the following conditions are violated:
 *
 *  * ``coeffs`` is a pointer to a ``complex double`` array of length ``num_terms``
 *  * ``bit_terms`` is a pointer to an array of valid ``QkBitTerm`` elements of length
 *  ``num_bits``
 *  * ``indices`` is a pointer to a ``uint32_t`` array of length ``num_bits``, which is
 *    term-wise sorted in strict ascending order, and every element is smaller than ``num_qubits``
 *  * ``boundaries`` is a pointer to a ``size_t`` array of length ``num_terms + 1``, which is
 *    sorted in ascending order, the first element is 0 and the last element is
 *    smaller than ``num_terms``
 */
QkObs *qk_obs_new(uint32_t num_qubits, uint64_t num_terms, uint64_t num_bits, QkComplex64 *coeffs,
                  QkBitTerm *bit_terms, uint32_t *indices, uintptr_t *boundaries);
/**
 * @ingroup QkObsMethods
 * Free the observable.
 *
 * @param obs A pointer to the observable to free.
 *
 * # Example
 *
 *    QkObs *obs = qk_obs_zero(100);
 *    qk_obs_free(obs);
 *
 * # Safety
 *
 * Behavior is undefined if ``obs`` is not either null or a valid pointer to a ``QkObs``.
 */
void qk_obs_free(QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Add a term to the observable.
 *
 * @param obs A pointer to the observable.
 * @param cterm A pointer to the term to add.
 *
 * @return An exit code. This is ``>0`` if the term is incoherent or adding the term fails.
 *
 * # Example
 *
 *    uint32_t num_qubits = 100;
 *    QkObs *obs = qk_obs_zero(num_qubits);
 *
 *    complex double coeff = 1;
 *    QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
 *    uint32_t indices[3] = {0, 1, 2};
 *    QkObsTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
 *
 *    int exit_code = qk_obs_add_term(obs, &term);
 *
 * # Safety
 *
 * Behavior is undefined if any of the following is violated:
 *
 *  * ``obs`` is a valid, non-null pointer to a ``QkObs``
 *  * ``cterm`` is a valid, non-null pointer to a ``QkObsTerm``
 */
QkExitCode qk_obs_add_term(QkObs *obs, const QkObsTerm *cterm);

/**
 * @ingroup QkObsMethods
 * Get an observable term by reference.
 *
 * A ``QkObsTerm`` contains pointers to the indices and bit terms in the term, which
 * can be used to modify the internal data of the observable. This can leave the observable
 * in an incoherent state and should be avoided, unless great care is taken. It is generally
 * safer to construct a new observable instead of attempting in-place modifications.
 *
 * @param obs A pointer to the observable.
 * @param index The index of the term to get.
 * @param out A pointer to a ``QkObsTerm`` used to return the observable term.
 *
 * @return An exit code.
 *
 * # Example
 *
 *     QkObs *obs = qk_obs_identity(100);
 *     QkObsTerm term;
 *     int exit_code = qk_obs_term(obs, 0, &term);
 *     // out-of-bounds indices return an error code
 *     // int error = qk_obs_term(obs, 12, &term);
 *
 * # Safety
 *
 * Behavior is undefined if any of the following is violated
 * * ``obs`` is a valid, non-null pointer to a ``QkObs``
 * * ``out`` is a valid, non-null pointer to a ``QkObsTerm``
 */
QkExitCode qk_obs_term(QkObs *obs, uint64_t index, QkObsTerm *out);

/**
 * @ingroup QkObsMethods
 * Get the number of terms in the observable.
 *
 * @param obs A pointer to the observable.
 *
 * @return The number of terms in the observable.
 *
 * # Example
 *
 *     QkObs *obs = qk_obs_identity(100);
 *     size_t num_terms = qk_obs_num_terms(obs);  // num_terms==1
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 */
uintptr_t qk_obs_num_terms(const QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Get the number of qubits the observable is defined on.
 *
 * @param obs A pointer to the observable.
 *
 * @return The number of qubits the observable is defined on.
 *
 * # Example
 *
 *     QkObs *obs = qk_obs_identity(100);
 *     uint32_t num_qubits = qk_obs_num_qubits(obs);  // num_qubits==100
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 */
uint32_t qk_obs_num_qubits(const QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Get the number of bit terms/indices in the observable.
 *
 * @param obs A pointer to the observable.
 *
 * @return The number of terms in the observable.
 *
 * # Example
 *
 *     QkObs *obs = qk_obs_identity(100);
 *     size_t len = qk_obs_len(obs);  // len==0, as there are no non-trivial bit terms
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 */
uintptr_t qk_obs_len(const QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Get a pointer to the coefficients.
 *
 * This can be used to read and modify the observable's coefficients. The resulting
 * pointer is valid to read for ``qk_obs_num_terms(obs)`` elements of ``complex double``.
 *
 * @param obs A pointer to the observable.
 *
 * @return A pointer to the coefficients.
 *
 * # Example
 *
 *     QkObs *obs = qk_obs_identity(100);
 *     size_t num_terms = qk_obs_num_terms(obs);
 *     complex double *coeffs = qk_obs_coeffs(obs);
 *
 *     for (size_t i = 0; i < num_terms; i++) {
 *         printf("%f + i%f\n", creal(coeffs[i]), cimag(coeffs[i]));
 *     }
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 */
QkComplex64 *qk_obs_coeffs(QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Get a pointer to the indices.
 *
 * This can be used to read and modify the observable's indices. The resulting pointer is
 * valid to read for ``qk_obs_len(obs)`` elements of size ``uint32_t``.
 *
 * @param obs A pointer to the observable.
 *
 * @return A pointer to the indices.
 *
 * # Example
 *
 *     uint32_t num_qubits = 100;
 *     QkObs *obs = qk_obs_zero(num_qubits);
 *
 *     complex double coeff = 1;
 *     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
 *     uint32_t indices[3] = {0, 1, 2};
 *     QkObsTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
 *     qk_obs_add_term(obs, &term);
 *
 *     size_t len = qk_obs_len(obs);
 *     uint32_t *indices = qk_obs_indices(obs);
 *
 *     for (size_t i = 0; i < len; i++) {
 *         printf("index %i: %i\n", i, indices[i]);
 *     }
 *
 *     qk_obs_free(obs);
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 */
uint32_t *qk_obs_indices(QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Get a pointer to the term boundaries.
 *
 * This can be used to read and modify the observable's term boundaries. The resulting pointer is
 * valid to read for ``qk_obs_num_terms(obs) + 1`` elements of size ``size_t``.
 *
 * @param obs A pointer to the observable.
 *
 * @return A pointer to the boundaries.
 *
 * # Example
 *
 *     uint32_t num_qubits = 100;
 *     QkObs *obs = qk_obs_zero(num_qubits);
 *
 *     complex double coeff = 1;
 *     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
 *     uint32_t indices[3] = {0, 1, 2};
 *     QkObsTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
 *     qk_obs_add_term(obs, &term);
 *
 *     size_t num_terms = qk_obs_num_terms(obs);
 *     uint32_t *boundaries = qk_obs_boundaries(obs);
 *
 *     for (size_t i = 0; i < num_terms + 1; i++) {
 *         printf("boundary %i: %i\n", i, boundaries[i]);
 *     }
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 */
uintptr_t *qk_obs_boundaries(QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Get a pointer to the bit terms.
 *
 * This can be used to read and modify the observable's bit terms. The resulting pointer is
 * valid to read for ``qk_obs_len(obs)`` elements of size ``uint8_t``.
 *
 * @param obs A pointer to the observable.
 *
 * @return A pointer to the bit terms.
 *
 * # Example
 *
 *     uint32_t num_qubits = 100;
 *     QkObs *obs = qk_obs_zero(num_qubits);
 *
 *     complex double coeff = 1;
 *     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
 *     uint32_t indices[3] = {0, 1, 2};
 *     QkObsTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
 *     qk_obs_add_term(obs, &term);
 *
 *     size_t len = qk_obs_len(obs);
 *     QkBitTerm *bits = qk_obs_bit_terms(obs);
 *
 *     for (size_t i = 0; i < len; i++) {
 *         printf("bit term %i: %i\n", i, bits[i]);
 *     }
 *
 *     qk_obs_free(obs);
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``,
 * or if invalid values are written into the resulting ``QkBitTerm`` pointer.
 */
QkBitTerm *qk_obs_bit_terms(QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Multiply the observable by a complex coefficient.
 *
 * @param obs A pointer to the observable.
 * @param coeff The coefficient to multiply the observable with.
 *
 * # Example
 *
 *     QkObs *obs = qk_obs_identity(100);
 *     complex double coeff = 2;
 *     QkObs *result = qk_obs_multiply(obs, &coeff);
 *
 * # Safety
 *
 * Behavior is undefined if any of the following is violated
 * * ``obs`` is a valid, non-null pointer to a ``QkObs``
 * * ``coeff`` is a valid, non-null pointer to a ``complex double``
 */
QkObs *qk_obs_multiply(const QkObs *obs, const QkComplex64 *coeff);

/**
 * @ingroup QkObsMethods
 * Add two observables.
 *
 * @param left A pointer to the left observable.
 * @param right A pointer to the right observable.
 *
 * @return A pointer to the result ``left + right``.
 *
 * # Example
 *
 *     QkObs *left = qk_obs_identity(100);
 *     QkObs *right = qk_obs_zero(100);
 *     QkObs *result = qk_obs_add(left, right);
 *
 * # Safety
 *
 * Behavior is undefined if ``left`` or ``right`` are not valid, non-null pointers to
 * ``QkObs``\ s.
 */
QkObs *qk_obs_add(const QkObs *left, const QkObs *right);

/**
 * @ingroup QkObsMethods
 * Compose (multiply) two observables.
 *
 * @param first One observable.
 * @param second The other observable.
 *
 * @return ``first.compose(second)`` which equals the observable ``result = second @ first``,
 *     in terms of the matrix multiplication ``@``.
 *
 * # Example
 *
 *     QkObs *first = qk_obs_zero(100);
 *     QkObs *second = qk_obs_identity(100);
 *     QkObs *result = qk_obs_compose(first, second);
 *
 * # Safety
 *
 * Behavior is undefined if ``first`` or ``second`` are not valid, non-null pointers to
 * ``QkObs``\ s.
 */
QkObs *qk_obs_compose(const QkObs *first, const QkObs *second);

/**
 * @ingroup QkObsMethods
 * Compose (multiply) two observables according to a custom qubit order.
 *
 * Notably, this allows composing two observables of different size.
 *
 * @param first One observable.
 * @param second The other observable. The number of qubits must match the length of ``qargs``.
 * @param qargs The qubit arguments specified which indices in ``first`` to associate with
 *     the ones in ``second``.
 *
 * @return ``first.compose(second)`` which equals the observable ``result = second @ first``,
 *     in terms of the matrix multiplication ``@``.
 *
 * # Example
 *
 *     QkObs *first = qk_obs_zero(100);
 *     QkObs *second = qk_obs_identity(100);
 *     QkObs *result = qk_obs_compose(first, second);
 *
 * # Safety
 *
 * To call this function safely
 *
 *   * ``first`` and ``second`` must be valid, non-null pointers to ``QkObs``\ s
 *   * ``qargs`` must point to an array of ``uint32_t``, readable for ``qk_obs_num_qubits(second)``
 *     elements (meaning the number of qubits in ``second``)
 */
QkObs *qk_obs_compose_map(const QkObs *first, const QkObs *second, const uint32_t *qargs);

/**
 * @ingroup QkObsMethods
 * Calculate the canonical representation of the observable.
 *
 * @param obs A pointer to the observable.
 * @param tol The tolerance below which coefficients are considered to be zero.
 *
 * @return The canonical representation of the observable.
 *
 * # Example
 *
 *     QkObs *iden = qk_obs_identity(100);
 *     QkObs *two = qk_obs_add(iden, iden);
 *
 *     double tol = 1e-6;
 *     QkObs *canonical = qk_obs_canonicalize(two);
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 */
QkObs *qk_obs_canonicalize(const QkObs *obs, double tol);

/**
 * @ingroup QkObsMethods
 * Copy the observable.
 *
 * @param obs A pointer to the observable.
 *
 * @return A pointer to a copy of the observable.
 *
 * # Example
 *
 *     QkObs *original = qk_obs_identity(100);
 *     QkObs *copied = qk_obs_copy(original);
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 */
QkObs *qk_obs_copy(const QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Compare two observables for equality.
 *
 * Note that this does not compare mathematical equality, but data equality. This means
 * that two observables might represent the same observable but not compare as equal.
 *
 * @param obs A pointer to one observable.
 * @param other A pointer to another observable.
 *
 * @return ``true`` if the observables are equal, ``false`` otherwise.
 *
 * # Example
 *
 *     QkObs *observable = qk_obs_identity(100);
 *     QkObs *other = qk_obs_identity(100);
 *     bool are_equal = qk_obs_equal(observable, other);
 *
 * # Safety
 *
 * Behavior is undefined if ``obs`` or ``other`` are not valid, non-null pointers to
 * ``QkObs``\ s.
 */
bool qk_obs_equal(const QkObs *obs, const QkObs *other);

/**
 * @ingroup QkObsMethods
 * Return a string representation of a ``QkObs``.
 *
 * @param obs A pointer to the ``QkObs`` to get the string for.
 *
 * @return A pointer to a nul-terminated char array of the string representation for ``obs``
 *
 * # Example
 *
 *     QkObs *obs = qk_obs_identity(100);
 *     char *string = qk_obs_str(obs);
 *     qk_str_free(string);
 *
 * # Safety
 *
 * Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 *
 * The string must not be freed with the normal C free, you must use ``qk_str_free`` to
 * free the memory consumed by the String. Not calling ``qk_str_free`` will lead to a
 * memory leak.
 *
 * Do not change the length of the string after it's returned (by writing a nul byte somewhere
 * inside the string or removing the final one), although values can be mutated.
 */
char *qk_obs_str(const QkObs *obs);

/**
 * @ingroup QkObsMethods
 * Free a string representation.
 *
 * @param string A pointer to the returned string representation from ``qk_obs_str`` or
 *     ``qk_obsterm_str``.
 *
 * # Safety
 *
 * Behavior is undefined if ``str`` is not a pointer returned by ``qk_obs_str`` or
 * ``qk_obsterm_str``.
 */
void qk_str_free(char *string);

/**
 * @ingroup QkObsMethodsTerm
 * Return a string representation of the sparse term.
 *
 * @param term A pointer to the term.
 *
 * @return The function exit code. This is ``>0`` if reading the term failed.
 *
 * # Example
 *
 *     QkObs *obs = qk_obs_identity(100);
 *     QkObsTerm term;
 *     qk_obs_term(obs, 0, &term);
 *     char *string = qk_obsterm_str(&term);
 *     qk_str_free(string);
 *
 * # Safety
 *
 * Behavior is undefined ``term`` is not a valid, non-null pointer to a ``QkObsTerm``.
 *
 * The string must not be freed with the normal C free, you must use ``qk_str_free`` to
 * free the memory consumed by the String. Not calling ``qk_str_free`` will lead to a
 * memory leak.
 *
 * Do not change the length of the string after it's returned, although values can be mutated.
 */
char *qk_obsterm_str(const QkObsTerm *term);

/**
 * @ingroup QkBitTermMethods
 * Get the label for a bit term.
 *
 * @param bit_term The bit term.
 *
 * @return The label as ``uint8_t``, which can be cast to ``char`` to obtain the character.
 *
 * # Example
 *
 *     QkBitTerm bit_term = QkBitTerm_Y;
 *     // cast the uint8_t to char
 *     char label = qk_bitterm_label(bit_term);
 *
 * # Safety
 *
 * The behavior is undefined if ``bit_term`` is not a valid ``uint8_t`` value of a ``QkBitTerm``.
 */
uint8_t qk_bitterm_label(QkBitTerm bit_term);

#ifdef QISKIT_C_PYTHON_INTERFACE
/**
 * @ingroup QkObsMethods
 * Convert to a Python-space ``SparseObservable``.
 *
 * @param obs The C-space ``QkObs`` pointer.
 *
 * @return A Python object representing the ``SparseObservable``.
 *
 * # Safety
 *
 * Behavior is undefined if ``obs`` is not a valid, non-null pointer to a ``QkObs``.
 *
 * It is assumed that the thread currently executing this function holds the
 * Python GIL this is required to create the Python object returned by this
 * function.
 */
PyObject *qk_obs_to_python(const QkObs *obs);
#endif // QISKIT_C_PYTHON_INTERFACE

#endif // QISKIT__OBSERVABLE_H
