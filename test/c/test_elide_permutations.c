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

#include "common.h"
#include <complex.h>
#include <math.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/**
 * Test running the path with no elision.
 */
int test_elide_permutations_no_result(void) {
    const uint32_t num_qubits = 5;

    QkCircuit *qc = qk_circuit_new(num_qubits, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    int result = Ok;
    QkElidePermutationsResult *pass_result = qk_transpiler_pass_standalone_elide_permutations(qc);
    if (qk_elide_permutations_result_elided_gates(pass_result)) {
        printf("A gate was elided when one shouldn't have been");
        result = EqualityError;
        goto result_cleanup;
    }
result_cleanup:
    qk_elide_permutations_result_free(pass_result);
cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test running the path with no elision.
 */
int test_elide_permutations_swap_result(void) {
    const uint32_t num_qubits = 5;

    QkCircuit *qc = qk_circuit_new(5, 0);
    uint32_t swap_qargs[2] = {1, 3};
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
        if (i == 2) {
            qk_circuit_gate(qc, QkGate_Swap, swap_qargs, NULL);
        }
    }
    int result = Ok;
    QkElidePermutationsResult *pass_result = qk_transpiler_pass_standalone_elide_permutations(qc);
    if (!qk_elide_permutations_result_elided_gates(pass_result)) {
        printf("A gate wasn't elided when one should have been");
        result = EqualityError;
        goto result_cleanup;
    }
    QkCircuit *modified_circuit = qk_elide_permutations_result_circuit(pass_result);
    size_t *permutation = qk_elide_permutations_result_permutation(pass_result);
    size_t expected_permutation[5] = {0, 3, 2, 1, 4};
    for (int i = 0; i < 5; i++) {
        if (permutation[i] != expected_permutation[i]) {
            printf("Permutation doesn't match expected");
            result = EqualityError;
            goto result_cleanup;
        }
    }
    QkOpCounts op_counts = qk_circuit_count_ops(modified_circuit);
    if (op_counts.len != 1) {
        printf("More than 1 type of gates in circuit");
        result = EqualityError;
        goto result_cleanup;
    }
    for (int i = 0; i < op_counts.len; i++) {
        int swap_gate = strcmp(op_counts.data[i].name, "swap");
        if (swap_gate == 0) {
            printf("Swap gate in circuit which should have been elided");
            result = EqualityError;
            goto result_cleanup;
        }
    }

result_cleanup:
    qk_elide_permutations_result_free(pass_result);
cleanup:
    qk_circuit_free(qc);
    return result;
}

int test_elide_permutations(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_elide_permutations_no_result);
    num_failed += RUN_TEST(test_elide_permutations_swap_result);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
