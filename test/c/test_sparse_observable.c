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
#include "qiskit.h"
#include <complex.h>
#include <stdio.h>

/**
 * Test the zero constructor.
 */
int test_zero() {
    SparseObservable *obs = qk_obs_zero(100);
    uint64_t num_terms = qk_obs_num_terms(obs);
    uint32_t num_qubits = qk_obs_num_qubits(obs);
    qk_obs_free(obs);

    if (num_terms != 0 || num_qubits != 100) {
        return EqualityError;
    }
    return 0;
}

/**
 * Test the identity constructor.
 */
int test_identity() {
    SparseObservable *obs = qk_obs_identity(100);
    uint64_t num_terms = qk_obs_num_terms(obs);
    uint32_t num_qubits = qk_obs_num_qubits(obs);
    qk_obs_free(obs);

    if (num_terms != 1 || num_qubits != 100) {
        return EqualityError;
    }
    return 0;
}

/**
 * Test copying an observable.
 */
int test_copy() {
    SparseObservable *obs = qk_obs_identity(100);
    SparseObservable *copied = qk_obs_copy(obs);

    bool are_equal = qk_obs_equal(obs, copied);

    qk_obs_free(obs);
    qk_obs_free(copied);

    if (!are_equal) {
        return EqualityError;
    }

    return 0;
}

/**
 * Test adding two observables.
 */
int test_add() {
    SparseObservable *left = qk_obs_identity(100);
    SparseObservable *right = qk_obs_identity(100);
    SparseObservable *obs = qk_obs_add(left, right);

    uint64_t num_terms = qk_obs_num_terms(obs);

    qk_obs_free(left);
    qk_obs_free(right);
    qk_obs_free(obs);

    if (num_terms != 2) {
        return EqualityError;
    }

    return 0;
}

/**
 * Test multiplying two observables.
 */
int test_mult() {
    complex double coeffs[3] = {2, 2 * I, 2 + 2 * I};

    for (int i = 0; i < 3; i++) {
        SparseObservable *obs = qk_obs_identity(100);

        SparseObservable *result = qk_obs_multiply(obs, &coeffs[i]);

        // construct the expected observable: coeff * Id
        SparseObservable *expected = qk_obs_zero(100);
        BitTerm bit_terms[] = {};
        uint32_t indices[] = {};
        SparseTerm term = {&coeffs[i], 0, bit_terms, indices, 100};
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

    return 0;
}

/**
 * Test bringing an observable into canonical form.
 */
int test_canonicalize() {
    SparseObservable *left = qk_obs_identity(100);
    SparseObservable *right = qk_obs_identity(100);
    SparseObservable *obs = qk_obs_add(left, right);

    double tol = 1e-5;
    SparseObservable *simplified = qk_obs_canonicalize(obs, tol);

    // construct the expected observable: 2 * Id
    SparseObservable *expected = qk_obs_zero(100);
    BitTerm bit_terms[] = {};
    uint32_t indices[] = {};
    complex double coeff = 2.0;
    SparseTerm term = {&coeff, 0, bit_terms, indices, 100};
    qk_obs_add_term(expected, &term);

    bool is_equal = qk_obs_equal(expected, simplified);

    qk_obs_free(obs);
    qk_obs_free(right);
    qk_obs_free(left);
    qk_obs_free(simplified);
    qk_obs_free(expected);

    if (!is_equal) {
        return EqualityError;
    }

    return 0;
}

/**
 * Test getting the number of terms in an observable.
 */
int test_num_terms() {
    int result = Ok;
    uint64_t num_terms;

    SparseObservable *zero = qk_obs_zero(100);
    num_terms = qk_obs_num_terms(zero);
    if (num_terms != 0) {
        result = EqualityError;
    }
    qk_obs_free(zero);

    SparseObservable *iden = qk_obs_identity(100);
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
int test_num_qubits() {
    int result = Ok;
    uint32_t num_qubits;

    SparseObservable *obs = qk_obs_zero(1);
    num_qubits = qk_obs_num_qubits(obs);
    if (num_qubits != 1) {
        result = EqualityError;
    }
    qk_obs_free(obs);

    SparseObservable *obs100 = qk_obs_zero(100);
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
int test_custom_build() {
    u_int32_t num_qubits = 100;
    SparseObservable *obs = qk_obs_zero(num_qubits);

    complex double coeff = 1;
    BitTerm bit_terms[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};
    SparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};

    qk_obs_add_term(obs, &term);
    qk_obs_add_term(obs, &term);

    double tol = 1e-6;
    SparseObservable *simplified = qk_obs_canonicalize(obs, tol);

    uint64_t num_terms = qk_obs_num_terms(obs);
    uint64_t num_terms_simplified = qk_obs_num_terms(simplified);

    qk_obs_free(obs);
    qk_obs_free(simplified);

    if (num_terms != 2 || num_terms_simplified != 1) {
        return EqualityError;
    }

    return 0;
}

/**
 * Test getting the terms in an observable.
 */
int test_term() {
    uint32_t num_qubits = 100;
    SparseObservable *obs = qk_obs_identity(num_qubits);

    BitTerm bit_terms[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
    uint32_t qubits[3] = {0, 1, 2};
    complex double coeff = 1 + I;

    SparseTerm term = {&coeff, 3, bit_terms, qubits, num_qubits};
    int err = qk_obs_add_term(obs, &term);

    if (err != 0) {
        return RuntimeError;
    }

    // some placeholders to store the results
    int nnis[2] = {-1, -1};
    int bits[3] = {-1, -1, -1};
    int indices[3] = {-1, -1, -1};

    uint64_t num_terms = qk_obs_num_terms(obs);
    for (uint64_t i = 0; i < num_terms; i++) {
        SparseTerm view;
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

    int result = 0;
    int expected_nnis[2] = {0, 3};
    int expected_bits[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
    int expected_indices[3] = {0, 1, 2};

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
int test_copy_term() {
    // create an observable with the term X0 Y1 Z2
    u_int32_t num_qubits = 100;
    SparseObservable *obs = qk_obs_zero(num_qubits);

    complex double coeff = 1;
    BitTerm bits[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};

    SparseTerm term = {&coeff, 3, bits, indices, num_qubits};
    qk_obs_add_term(obs, &term);

    // now we add a modified copy of the first term -- we use use qk_obs_term(..., &borrowed) on
    // purpose here
    SparseTerm borrowed;
    int error = qk_obs_term(obs, 0, &borrowed); // get view on 0th term
    if (error > 0) {
        return RuntimeError;
    }

    // copy the term so we can safely modify it and add it onto the observable
    size_t len = borrowed.len;
    BitTerm *copied_bits = (BitTerm *)malloc(len * sizeof(BitTerm));
    uint32_t *copied_indices = (uint32_t *)malloc(len * sizeof(uint32_t));
    for (size_t i = 0; i < len; i++) {
        copied_bits[i] = borrowed.bit_terms[i];
        copied_indices[i] = borrowed.indices[i];
    }

    // modify the term and add it onto the observable
    complex double coeff2 = 2 * I;
    copied_indices[1] = 99;
    copied_bits[0] = BitTerm_Zero;
    SparseTerm copied = {
        &coeff2, borrowed.len, copied_bits, copied_indices, borrowed.num_qubits,
    };
    qk_obs_add_term(obs, &copied);

    free(copied_indices);
    free(copied_bits);

    // now we directly construct the expected observable
    BitTerm bits2[3] = {BitTerm_Zero, BitTerm_Y, BitTerm_Z};
    uint32_t indices2[3] = {0, 99, 2};
    SparseTerm term2 = {&coeff2, 3, bits2, indices2, num_qubits};

    SparseObservable *expected = qk_obs_zero(num_qubits);
    qk_obs_add_term(expected, &term);
    qk_obs_add_term(expected, &term2);

    bool equal = qk_obs_equal(expected, obs);
    qk_obs_free(obs);
    qk_obs_free(expected);

    if (!equal) {
        return EqualityError;
    }
    return 0;
}

/**
 * Test modifying an observable inplace, via the sparse term.
 */
int test_inplace_mut() {
    // create an observable with the term X0 Y1 Z2
    u_int32_t num_qubits = 100;
    SparseObservable *obs = qk_obs_zero(num_qubits);

    complex double coeff = 1;
    BitTerm bits[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};

    SparseTerm term = {&coeff, 3, bits, indices, num_qubits};
    qk_obs_add_term(obs, &term);

    // now we get the same term, referencing the observable data, and modify it
    SparseTerm borrowed;
    int error = qk_obs_term(obs, 0, &borrowed); // get view on 0th term
    if (error > 0) {
        return RuntimeError;
    }

    *borrowed.coeff = -5 * I;
    borrowed.bit_terms[2] = BitTerm_Zero;

    // compare to the expected observable
    SparseObservable *expected = qk_obs_zero(num_qubits);

    complex double new_coeff = -5 * I;
    BitTerm new_bits[3] = {BitTerm_X, BitTerm_Y, BitTerm_Zero};
    uint32_t new_indices[3] = {0, 1, 2};

    SparseTerm new_term = {&new_coeff, 3, new_bits, new_indices, num_qubits};
    qk_obs_add_term(expected, &new_term);

    bool equal = qk_obs_equal(expected, obs);
    qk_obs_free(expected);
    qk_obs_free(obs);

    if (!equal) {
        return EqualityError;
    }
    return 0;
}

/**
 * Test getting the bit term labels.
 */
int test_bitterm_label() {
    char expected[9] = {'X', '+', '-', 'Y', 'l', 'r', 'Z', '0', '1'};
    BitTerm bits[9] = {BitTerm_X,     BitTerm_Plus, BitTerm_Minus, BitTerm_Y,  BitTerm_Left,
                       BitTerm_Right, BitTerm_Z,    BitTerm_Zero,  BitTerm_One};

    for (int i = 0; i < 9; i++) {
        char label = qk_bitterm_label(&bits[i]);
        if (label != expected[i]) {
            return EqualityError;
        }
    }

    return 0;
}

/**
 * Test the coeffs access.
 */
int test_coeffs() {
    SparseObservable *obs = qk_obs_identity(2);
    complex double *coeffs = qk_obs_coeffs(obs);

    // read the first coefficient
    complex double first = coeffs[0];
    int result = 0;
    if (first != 1) {
        result = EqualityError;
    }

    // modify the coefficient by ref
    coeffs[0] = I;
    complex double later = qk_obs_coeffs(obs)[0];
    if (later != I) {
        result = EqualityError;
    }

    qk_obs_free(obs);
    return result;
}

/**
 * Test the bit term access.
 */
int test_bit_terms() {
    BitTerm bits[6] = {BitTerm_Left,  BitTerm_Right, BitTerm_Plus,
                       BitTerm_Minus, BitTerm_Zero,  BitTerm_One};
    uint32_t indices[6] = {9, 8, 7, 6, 5, 4};
    complex double coeff = 1;
    SparseTerm term = {&coeff, 6, bits, indices, 10};

    SparseObservable *obs = qk_obs_zero(10);
    qk_obs_add_term(obs, &term);

    BitTerm *borrowed = qk_obs_bit_terms(obs);

    // test read access
    BitTerm element = borrowed[4];
    int result = 0;
    if (element != BitTerm_Zero) {
        result = EqualityError;
    }

    // modify the element
    borrowed[4] = BitTerm_X;
    BitTerm later = qk_obs_bit_terms(obs)[4];
    if (later != BitTerm_X) {
        result = EqualityError;
    }

    qk_obs_free(obs);
    return result;
}

/**
 * Test the index access.
 */
int test_indices() {
    BitTerm bits[6] = {BitTerm_Left,  BitTerm_Right, BitTerm_Plus,
                       BitTerm_Minus, BitTerm_Zero,  BitTerm_One};
    uint32_t indices[6] = {9, 8, 7, 6, 5, 4};
    complex double coeff = 1;
    SparseTerm term = {&coeff, 6, bits, indices, 10};

    SparseObservable *obs = qk_obs_zero(10);
    qk_obs_add_term(obs, &term);

    uint32_t *borrowed = qk_obs_indices(obs);

    // test read access
    uint32_t element = indices[2];
    int result = 0;
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
int test_boundaries() {
    uint32_t num_qubits = 100;
    SparseObservable *obs = qk_obs_identity(num_qubits);

    complex double coeff = 1;
    BitTerm bit_terms[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};
    SparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
    qk_obs_add_term(obs, &term);

    uint64_t num_terms = qk_obs_num_terms(obs);
    size_t *boundaries = qk_obs_boundaries(obs);

    // the identity term has 0 bits, the XYZ has 3, therefore the terms are defined as
    // indices = [0, 1, 2]
    // bit_terms = [X, Y, Z]
    // boundaries = [0, 0, 3]
    size_t expected[] = {0, 0, 3};

    for (uint64_t i = 0; i < num_terms + 1; i++) {
        if (boundaries[i] != expected[i]) {
            return EqualityError;
        }
    }
    return EXIT_SUCCESS;
}

int test_sparse_observable() {
    int num_failed = 0;
    num_failed += RUN_TEST(test_zero);
    num_failed += RUN_TEST(test_identity);
    num_failed += RUN_TEST(test_add);
    num_failed += RUN_TEST(test_mult);
    num_failed += RUN_TEST(test_canonicalize);
    num_failed += RUN_TEST(test_copy);
    num_failed += RUN_TEST(test_num_terms);
    num_failed += RUN_TEST(test_num_qubits);
    num_failed += RUN_TEST(test_custom_build);
    num_failed += RUN_TEST(test_term);
    num_failed += RUN_TEST(test_copy_term);
    num_failed += RUN_TEST(test_inplace_mut);
    num_failed += RUN_TEST(test_bitterm_label);
    num_failed += RUN_TEST(test_coeffs);
    num_failed += RUN_TEST(test_bit_terms);
    num_failed += RUN_TEST(test_indices);
    num_failed += RUN_TEST(test_boundaries);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
