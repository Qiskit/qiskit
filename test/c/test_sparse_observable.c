// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include <complex.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/**
 * Test the zero constructor.
 */
int test_zero(void) {
    QkObs *obs = qk_obs_zero(100);
    size_t num_terms = qk_obs_num_terms(obs);
    uint32_t num_qubits = qk_obs_num_qubits(obs);
    qk_obs_free(obs);

    return (num_terms != 0 || num_qubits != 100) ? EqualityError : Ok;
}

/**
 * Test the identity constructor.
 */
int test_identity(void) {
    QkObs *obs = qk_obs_identity(100);
    size_t num_terms = qk_obs_num_terms(obs);
    uint32_t num_qubits = qk_obs_num_qubits(obs);
    qk_obs_free(obs);

    return (num_terms != 1 || num_qubits != 100) ? EqualityError : Ok;
}

/**
 * Test copying an observable.
 */
int test_copy(void) {
    QkObs *obs = qk_obs_identity(100);
    QkObs *copied = qk_obs_copy(obs);

    bool are_equal = qk_obs_equal(obs, copied);

    qk_obs_free(obs);
    qk_obs_free(copied);

    return are_equal ? Ok : EqualityError;
}

/**
 * Test adding two observables.
 */
int test_add(void) {
    QkObs *left = qk_obs_identity(100);
    QkObs *right = qk_obs_identity(100);
    QkObs *obs = qk_obs_add(left, right);

    size_t num_terms = qk_obs_num_terms(obs);

    qk_obs_free(left);
    qk_obs_free(right);
    qk_obs_free(obs);

    return (num_terms != 2) ? EqualityError : Ok;
}

/**
 * Test composing two observables.
 */
int test_compose(void) {
    uint32_t num_qubits = 100;

    QkObs *op1 = qk_obs_zero(num_qubits);
    QkComplex64 coeff1 = make_complex_double(1.0, 0.0);
    QkBitTerm op1_bits[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t op1_indices[3] = {0, 1, 2};
    QkObsTerm term1 = {coeff1, 3, op1_bits, op1_indices, num_qubits};
    qk_obs_add_term(op1, &term1);

    QkObs *op2 = qk_obs_zero(num_qubits);
    QkComplex64 coeff2 = make_complex_double(2.0, 0.0);
    QkBitTerm op2_bits[3] = {QkBitTerm_Plus, QkBitTerm_X, QkBitTerm_Z};
    uint32_t op2_indices[3] = {0, 1, 3};
    QkObsTerm term2 = {coeff2, 3, op2_bits, op2_indices, num_qubits};
    qk_obs_add_term(op2, &term2);

    QkObs *result = qk_obs_compose(op1, op2);

    QkObs *expected = qk_obs_zero(num_qubits);
    QkComplex64 expected_coeff = make_complex_double(0.0, 2.0);
    QkBitTerm expected_bits[4] = {QkBitTerm_Plus, QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Z};
    uint32_t expected_indices[4] = {0, 1, 2, 3};
    QkObsTerm expected_term = {expected_coeff, 4, expected_bits, expected_indices, num_qubits};
    qk_obs_add_term(expected, &expected_term);

    bool is_equal = qk_obs_equal(expected, result);

    qk_obs_free(op1);
    qk_obs_free(op2);
    qk_obs_free(result);
    qk_obs_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

/**
 * Test composing two observables and specifying the qargs argument.
 */
int test_compose_map(void) {
    uint32_t num_qubits = 100;

    QkObs *op1 = qk_obs_zero(num_qubits);
    QkComplex64 coeff1 = make_complex_double(1.0, 0.0);
    QkBitTerm op1_bits[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t op1_indices[3] = {97, 98, 99};
    QkObsTerm term1 = {coeff1, 3, op1_bits, op1_indices, num_qubits};
    qk_obs_add_term(op1, &term1);

    QkObs *op2 = qk_obs_zero(2);
    QkComplex64 coeff2 = make_complex_double(2.0, 0.0);
    QkBitTerm op2_bits[3] = {QkBitTerm_Right, QkBitTerm_X};
    uint32_t op2_indices[3] = {0, 1};
    QkObsTerm term2 = {coeff2, 2, op2_bits, op2_indices, 2};
    qk_obs_add_term(op2, &term2);

    uint32_t qargs[2] = {98, 97}; // compose op2 onto these indices in op1

    QkObs *result = qk_obs_compose_map(op1, op2, qargs);

    QkObs *expected = qk_obs_zero(num_qubits);
    QkBitTerm expected_bits[2] = {QkBitTerm_Right, QkBitTerm_Z};
    uint32_t expected_indices[2] = {98, 99};
    QkObsTerm expected_term = {coeff2, 2, expected_bits, expected_indices, num_qubits};
    qk_obs_add_term(expected, &expected_term);

    bool is_equal = qk_obs_equal(expected, result);

    qk_obs_free(op1);
    qk_obs_free(op2);
    qk_obs_free(result);
    qk_obs_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

/**
 * Test composing an observables with a scalar observable.
 */
int test_compose_scalar(void) {
    uint32_t num_qubits = 100;

    QkObs *op = qk_obs_zero(num_qubits);
    QkComplex64 coeff = make_complex_double(1.0, 0.0);
    QkBitTerm bits[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t indices[3] = {97, 98, 99};
    QkObsTerm term = {coeff, 3, bits, indices, num_qubits};
    qk_obs_add_term(op, &term);

    QkObs *scalar = qk_obs_identity(0);
    QkComplex64 factor = make_complex_double(2.0, 0.0);
    QkObs *mult = qk_obs_multiply(scalar, &factor);
    uint32_t *qargs = NULL; // no value will be read (also MSVC doesn't allow qargs[0], so use this)

    QkObs *result = qk_obs_compose_map(op, mult, qargs);

    QkObs *expected = qk_obs_multiply(op, &factor);

    bool is_equal = qk_obs_equal(expected, result);

    qk_obs_free(op);
    qk_obs_free(scalar);
    qk_obs_free(mult);
    qk_obs_free(result);
    qk_obs_free(expected);

    if (!is_equal) {
        return EqualityError;
    }
    return Ok;
}

/**
 * Test multiplying an observable by a complex coefficient.
 */
int test_mult(void) {
    QkComplex64 coeffs[3] = {make_complex_double(2.0, 0.0), make_complex_double(0.0, 2.0),
                             make_complex_double(2.0, 2.0)};

    for (int i = 0; i < 3; i++) {
        QkObs *obs = qk_obs_identity(100);

        QkObs *result = qk_obs_multiply(obs, &coeffs[i]);

        // construct the expected observable: coeff * Id
        QkObs *expected = qk_obs_zero(100);
        QkBitTerm bit_terms[] = {};
        uint32_t indices[] = {};
        QkObsTerm term = {coeffs[i], 0, bit_terms, indices, 100};
        qk_obs_add_term(expected, &term);

        // perform the check
        bool is_equal = qk_obs_equal(expected, result);

        // deallocate before returning
        qk_obs_free(obs);
        qk_obs_free(result);
        qk_obs_free(expected);

        if (!is_equal) {
            return EqualityError;
        }
    }

    return Ok;
}

/**
 * Test bringing an observable into canonical form.
 */
int test_canonicalize(void) {
    QkObs *left = qk_obs_identity(100);
    QkObs *right = qk_obs_identity(100);
    QkObs *obs = qk_obs_add(left, right);

    double tol = 1e-5;
    QkObs *simplified = qk_obs_canonicalize(obs, tol);

    // construct the expected observable: 2 * Id
    QkObs *expected = qk_obs_zero(100);
    QkBitTerm bit_terms[] = {};
    uint32_t indices[] = {};
    QkComplex64 coeff = make_complex_double(2.0, 0.0);
    QkObsTerm term = {coeff, 0, bit_terms, indices, 100};
    qk_obs_add_term(expected, &term);

    bool is_equal = qk_obs_equal(expected, simplified);

    qk_obs_free(obs);
    qk_obs_free(right);
    qk_obs_free(left);
    qk_obs_free(simplified);
    qk_obs_free(expected);

    return is_equal ? Ok : EqualityError;
}

/**
 * Test getting the number of terms in an observable.
 */
int test_num_terms(void) {
    int result = Ok;
    size_t num_terms;

    QkObs *zero = qk_obs_zero(100);
    num_terms = qk_obs_num_terms(zero);
    if (num_terms != 0) {
        result = EqualityError;
    }
    qk_obs_free(zero);

    QkObs *iden = qk_obs_identity(100);
    num_terms = qk_obs_num_terms(iden);
    if (num_terms != 1) {
        result = EqualityError;
    }
    qk_obs_free(iden);

    return result;
}

/**
 * Test getting the number of qubits in an observable.
 */
int test_num_qubits(void) {
    int result = Ok;
    uint32_t num_qubits;

    QkObs *obs = qk_obs_zero(1);
    num_qubits = qk_obs_num_qubits(obs);
    if (num_qubits != 1) {
        result = EqualityError;
    }
    qk_obs_free(obs);

    QkObs *obs100 = qk_obs_zero(100);
    num_qubits = qk_obs_num_qubits(obs100);
    if (num_qubits != 100) {
        result = EqualityError;
    }
    qk_obs_free(obs100);

    return result;
}

/**
 * Test adding an individual term to an observable.
 */
int test_custom_build(void) {
    uint32_t num_qubits = 100;
    QkObs *obs = qk_obs_zero(num_qubits);

    QkComplex64 coeff = make_complex_double(1.0, 0.0);
    QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};
    QkObsTerm term = {coeff, 3, bit_terms, indices, num_qubits};

    qk_obs_add_term(obs, &term);
    qk_obs_add_term(obs, &term);

    double tol = 1e-6;
    QkObs *simplified = qk_obs_canonicalize(obs, tol);

    size_t num_terms = qk_obs_num_terms(obs);
    size_t num_terms_simplified = qk_obs_num_terms(simplified);

    qk_obs_free(obs);
    qk_obs_free(simplified);

    return (num_terms != 2 || num_terms_simplified != 1) ? EqualityError : Ok;
}

/**
 * Test getting the terms in an observable.
 */
int test_term(void) {
    uint32_t num_qubits = 100;
    QkObs *obs = qk_obs_identity(num_qubits);

    QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t qubits[3] = {0, 1, 2};
    QkComplex64 coeff = make_complex_double(1.0, 1.0);

    QkObsTerm term = {coeff, 3, bit_terms, qubits, num_qubits};
    int err = qk_obs_add_term(obs, &term);

    if (err != 0) {
        qk_obs_free(obs);
        return RuntimeError;
    }

    // some placeholders to store the results
    size_t nnis[2];
    QkBitTerm bits[3];
    uint32_t indices[3];

    size_t num_terms = qk_obs_num_terms(obs);
    for (size_t i = 0; i < num_terms; i++) {
        QkObsTerm view;
        qk_obs_term(obs, i, &view);
        size_t nni = view.len;
        nnis[i] = nni; // store to compare later

        for (uint32_t n = 0; n < nni; n++) {
            // this loop is only called once, so we can use ``n`` to index here
            bits[n] = view.bit_terms[n];
            indices[n] = view.indices[n];
        }
    }

    qk_obs_free(obs);

    int result = Ok;
    size_t expected_nnis[2] = {0, 3};
    QkBitTerm expected_bits[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t expected_indices[3] = {0, 1, 2};

    // check number of terms
    if (num_terms != 2) {
        result = EqualityError;
    }

    // check NNIs
    for (int i = 0; i < 2; i++) {
        if (nnis[i] != expected_nnis[i]) {
            result = EqualityError;
        }
    }

    // check bit terms and indices
    for (int n = 0; n < 3; n++) {
        if (indices[n] != expected_indices[n] || bits[n] != expected_bits[n]) {
            result = EqualityError;
        }
    }

    return result;
}

/**
 * Test copying and modifying a term.
 */
int test_copy_term(void) {
    // create an observable with the term X0 Y1 Z2
    uint32_t num_qubits = 100;
    QkObs *obs = qk_obs_zero(num_qubits);

    QkComplex64 coeff = make_complex_double(1.0, 0.0);
    QkBitTerm bits[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};

    QkObsTerm term = {coeff, 3, bits, indices, num_qubits};
    qk_obs_add_term(obs, &term);

    // now we add a modified copy of the first term -- we use use qk_obs_term(..., &borrowed) on
    // purpose here
    QkObsTerm borrowed;
    int error = qk_obs_term(obs, 0, &borrowed); // get view on 0th term
    if (error > 0) {
        qk_obs_free(obs);
        return RuntimeError;
    }

    // copy the term so we can safely modify it and add it onto the observable
    size_t len = borrowed.len;
    QkBitTerm *copied_bits = (QkBitTerm *)malloc(len * sizeof(QkBitTerm));
    uint32_t *copied_indices = (uint32_t *)malloc(len * sizeof(uint32_t));
    for (size_t i = 0; i < len; i++) {
        copied_bits[i] = borrowed.bit_terms[i];
        copied_indices[i] = borrowed.indices[i];
    }

    // modify the term and add it onto the observable
    QkComplex64 coeff2 = make_complex_double(0.0, 2.0);
    copied_indices[1] = 99;
    copied_bits[0] = QkBitTerm_Zero;
    QkObsTerm copied = {
        coeff2, borrowed.len, copied_bits, copied_indices, borrowed.num_qubits,
    };
    qk_obs_add_term(obs, &copied);

    free(copied_indices);
    free(copied_bits);

    // now we directly construct the expected observable
    QkBitTerm bits2[3] = {QkBitTerm_Zero, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t indices2[3] = {0, 99, 2};
    QkObsTerm term2 = {coeff2, 3, bits2, indices2, num_qubits};

    QkObs *expected = qk_obs_zero(num_qubits);
    qk_obs_add_term(expected, &term);
    qk_obs_add_term(expected, &term2);

    bool equal = qk_obs_equal(expected, obs);
    qk_obs_free(obs);
    qk_obs_free(expected);

    return equal ? Ok : EqualityError;
}

/**
 * Test getting the bit term labels.
 */
int test_bitterm_label(void) {
    char expected[9] = {'X', '+', '-', 'Y', 'l', 'r', 'Z', '0', '1'};
    QkBitTerm bits[9] = {QkBitTerm_X, QkBitTerm_Plus, QkBitTerm_Minus,
                         QkBitTerm_Y, QkBitTerm_Left, QkBitTerm_Right,
                         QkBitTerm_Z, QkBitTerm_Zero, QkBitTerm_One};

    for (int i = 0; i < 9; i++) {
        char label = qk_bitterm_label(bits[i]);
        if (label != expected[i]) {
            return EqualityError;
        }
    }

    return Ok;
}

/**
 * Test the coeffs access.
 */
int test_coeffs(void) {
    QkObs *obs = qk_obs_identity(2);
    QkComplex64 *coeffs = qk_obs_coeffs(obs);

    // read the first coefficient
    QkComplex64 first = coeffs[0];
    int result = Ok;
    if (creal(first) != 1.0 || cimag(first) != 0.0) {
        result = EqualityError;
    }

    // modify the coefficient by ref
    coeffs[0] = make_complex_double(0.0, 1.0);
    QkComplex64 later = qk_obs_coeffs(obs)[0];
    if (creal(later) != 0.0 || cimag(later) != 1.0) {
        result = EqualityError;
    }

    qk_obs_free(obs);
    return result;
}

/**
 * Test the bit term access.
 */
int test_bit_terms(void) {
    QkBitTerm bits[6] = {QkBitTerm_Left,  QkBitTerm_Right, QkBitTerm_Plus,
                         QkBitTerm_Minus, QkBitTerm_Zero,  QkBitTerm_One};
    uint32_t indices[6] = {9, 8, 7, 6, 5, 4};
    QkComplex64 coeff = make_complex_double(1.0, 0.0);
    QkObsTerm term = {coeff, 6, bits, indices, 10};

    QkObs *obs = qk_obs_zero(10);
    qk_obs_add_term(obs, &term);

    QkBitTerm *borrowed = qk_obs_bit_terms(obs);

    // test read access
    QkBitTerm element = borrowed[4];
    int result = Ok;
    if (element != QkBitTerm_Zero) {
        result = EqualityError;
    }

    // modify the element
    borrowed[4] = QkBitTerm_X;
    QkBitTerm later = qk_obs_bit_terms(obs)[4];
    if (later != QkBitTerm_X) {
        result = EqualityError;
    }

    qk_obs_free(obs);
    return result;
}

/**
 * Test the index access.
 */
int test_indices(void) {
    QkBitTerm bits[6] = {QkBitTerm_Left,  QkBitTerm_Right, QkBitTerm_Plus,
                         QkBitTerm_Minus, QkBitTerm_Zero,  QkBitTerm_One};
    uint32_t indices[6] = {9, 8, 7, 6, 5, 4};
    QkComplex64 coeff = make_complex_double(1.0, 0.0);
    QkObsTerm term = {coeff, 6, bits, indices, 10};

    QkObs *obs = qk_obs_zero(10);
    qk_obs_add_term(obs, &term);

    uint32_t *borrowed = qk_obs_indices(obs);

    // test read access
    uint32_t element = indices[2];
    int result = Ok;
    if (element != 7) {
        result = EqualityError;
    }

    // modify the element
    borrowed[0] = 0;
    uint32_t later = qk_obs_indices(obs)[0];
    if (later != 0) {
        result = EqualityError;
    }

    qk_obs_free(obs);
    return result;
}

/**
 * Test access to the term boundaries.
 */
int test_boundaries(void) {
    uint32_t num_qubits = 100;
    QkObs *obs = qk_obs_identity(num_qubits);

    QkComplex64 coeff = make_complex_double(1.0, 0.0);
    QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};
    QkObsTerm term = {coeff, 3, bit_terms, indices, num_qubits};
    qk_obs_add_term(obs, &term);

    size_t num_terms = qk_obs_num_terms(obs);
    size_t *boundaries = qk_obs_boundaries(obs);

    // the identity term has 0 bits, the XYZ has 3, therefore the terms are defined as
    // indices = [0, 1, 2]
    // bit_terms = [X, Y, Z]
    // boundaries = [0, 0, 3]
    size_t expected[] = {0, 0, 3};

    for (size_t i = 0; i < num_terms + 1; i++) {
        if (boundaries[i] != expected[i]) {
            return EqualityError;
        }
    }
    return Ok;
}

/**
 * Test direct setting.
 */
int test_direct_build(void) {
    // define the raw data for the 100-qubit observable |01><01|_{0, 1} - |+-><+-|_{98, 99}
    uint32_t num_qubits = 100;
    size_t num_terms = 2;
    size_t num_bits = 4;

    QkComplex64 coeffs[2] = {make_complex_double(1.0, 0.0), make_complex_double(-1.0, 0.0)};
    QkBitTerm bits[4] = {QkBitTerm_Zero, QkBitTerm_One, QkBitTerm_Plus, QkBitTerm_Minus};
    uint32_t indices[4] = {0, 1, 98, 99};
    size_t boundaries[3] = {0, 2, 4};

    // set the pointers to the new data
    QkObs *obs = qk_obs_new(num_qubits, num_terms, num_bits, coeffs, bits, indices, boundaries);

    // check the construction was successful
    if (!obs) {
        return NullptrError;
    }

    // check the data content
    int result = Ok;
    QkComplex64 *obs_coeffs = qk_obs_coeffs(obs);
    size_t *obs_boundaries = qk_obs_boundaries(obs);
    for (size_t i = 0; i < num_terms; i++) {
        if (creal(coeffs[i]) != creal(obs_coeffs[i]) || cimag(coeffs[i]) != cimag(obs_coeffs[i]) ||
            boundaries[i] != obs_boundaries[i]) {
            result = EqualityError;
        }
    }
    if (boundaries[num_terms] != obs_boundaries[num_terms])
        result = EqualityError;

    QkBitTerm *obs_bits = qk_obs_bit_terms(obs);
    uint32_t *obs_indices = qk_obs_indices(obs);
    for (size_t i = 0; i < num_bits; i++) {
        if (bits[i] != obs_bits[i] || indices[i] != obs_indices[i]) {
            result = EqualityError;
        }
    }

    qk_obs_free(obs);
    return result;
}

/**
 * Test direct setting fails.
 */
int test_direct_fail(void) {
    // define the faulty raw data
    uint32_t num_qubits = 100;
    size_t num_terms = 2;
    size_t num_bits = 4;

    QkComplex64 coeffs[2] = {make_complex_double(1.0, 0.0), make_complex_double(-1.0, 0.0)};
    QkBitTerm bits[4] = {QkBitTerm_Zero, QkBitTerm_One, QkBitTerm_Plus, QkBitTerm_Minus};
    uint32_t indices[4] = {0, 1, 99, 98}; // <-- needs to be ordered
    size_t boundaries[3] = {0, 2, 4};

    // set the pointers to the new data
    QkObs *obs = qk_obs_new(num_qubits, num_terms, num_bits, coeffs, bits, indices, boundaries);

    // check the construction failed
    if (!obs) {
        return Ok;
    }

    // if for some magical reason an observable was constructed, free it
    qk_obs_free(obs);
    return NullptrError;
}

/**
 * Test string generator for observable
 */
int test_obs_str(void) {
    QkObs *obs = qk_obs_identity(100);
    char *string = qk_obs_str(obs);
    char *expected = "SparseObservable { num_qubits: 100, coeffs: [Complex { re: 1.0, im: 0.0 }], "
                     "bit_terms: [], indices: [], boundaries: [0, 0] }";
    int result = strcmp(string, expected);
    qk_str_free(string);
    qk_obs_free(obs);

    return result;
}

/**
 * Test string generator for observable term
 */
int test_obsterm_str(void) {
    // Initialize observable and add a term
    uint32_t num_qubits = 100;
    QkObs *obs = qk_obs_identity(num_qubits);
    QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t qubits[3] = {0, 1, 2};
    QkComplex64 coeff = make_complex_double(1.0, 1.0);
    QkObsTerm term = {coeff, 3, bit_terms, qubits, num_qubits};
    int err = qk_obs_add_term(obs, &term);

    if (err != 0) {
        qk_obs_free(obs);
        return RuntimeError;
    }
    // Get string for term:
    QkObsTerm out_term;
    qk_obs_term(obs, 1, &out_term);
    char *string = qk_obsterm_str(&out_term);
    char *expected = "SparseTermView { num_qubits: 100, coeff: Complex { re: 1.0, im: 1.0 }, "
                     "bit_terms: [X, Y, Z], indices: [0, 1, 2] }";
    int result = strcmp(string, expected);
    qk_str_free(string);
    qk_obs_free(obs);

    return result;
}

int test_sparse_observable(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_zero);
    num_failed += RUN_TEST(test_identity);
    num_failed += RUN_TEST(test_add);
    num_failed += RUN_TEST(test_compose);
    num_failed += RUN_TEST(test_compose_map);
    num_failed += RUN_TEST(test_compose_scalar);
    num_failed += RUN_TEST(test_mult);
    num_failed += RUN_TEST(test_canonicalize);
    num_failed += RUN_TEST(test_copy);
    num_failed += RUN_TEST(test_num_terms);
    num_failed += RUN_TEST(test_num_qubits);
    num_failed += RUN_TEST(test_custom_build);
    num_failed += RUN_TEST(test_term);
    num_failed += RUN_TEST(test_copy_term);
    num_failed += RUN_TEST(test_bitterm_label);
    num_failed += RUN_TEST(test_coeffs);
    num_failed += RUN_TEST(test_bit_terms);
    num_failed += RUN_TEST(test_indices);
    num_failed += RUN_TEST(test_boundaries);
    num_failed += RUN_TEST(test_direct_build);
    num_failed += RUN_TEST(test_direct_fail);
    num_failed += RUN_TEST(test_obs_str);
    num_failed += RUN_TEST(test_obsterm_str);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
