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
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int test_split_2q_unitaries_no_unitaries(void) {
    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    QkSplit2qUnitariesResult *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (qk_split_2q_unitaries_result_permutation_len(split_result) != 0) {
        result = EqualityError;
        printf("Permutation returned for a circuit that shouldn't split");
        goto result_cleanup;
    }
    QkCircuit *res = qk_split_2q_unitaries_result_circuit(split_result);
    QkOpCounts counts = qk_circuit_count_ops(res);
    if (counts.len != 1) {
        printf("More than 1 type of gate in the circuit");
        result = EqualityError;
        goto result_cleanup;
    }
    for (int i = 0; i < counts.len; i++) {
        int gate = strcmp(counts.data[i].name, "cx");
        if (gate != 0) {
            printf("gates changed when there should be no circuit changes");
            result = EqualityError;
            goto result_cleanup;
        }
    }

result_cleanup:
    qk_split_2q_unitaries_result_free(split_result);
cleanup:
    qk_circuit_free(qc);
    return result;
}

int test_split_2q_unitaries_x_y_unitary(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    QkComplex64 c0 = {0., 0.};
    QkComplex64 neg_im = {0., -1.};
    QkComplex64 im = {0., 1.};
    QkComplex64 unitary[16] = {
        c0, c0, c0, neg_im, c0, c0, neg_im, c0, c0, im, c0, c0, im, c0, c0, c0,
    };
    uint32_t qargs[2] = {0, 1};
    qk_circuit_unitary(qc, unitary, qargs, 2, true);
    QkSplit2qUnitariesResult *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (qk_split_2q_unitaries_result_permutation_len(split_result) != 0) {
        result = EqualityError;
        printf("Permutation returned for a circuit that shouldn't split");
        goto result_cleanup;
    }
    QkCircuit *res = qk_split_2q_unitaries_result_circuit(split_result);
    QkOpCounts counts = qk_circuit_count_ops(res);
    if (counts.len != 1) {
        printf("More than 1 type of gate in the circuit");
        result = EqualityError;
        goto result_cleanup;
    }
    for (int i = 0; i < counts.len; i++) {
        int gate = strcmp(counts.data[i].name, "unitary");
        if (gate != 0) {
            printf("Gates outside expected set in output circuit");
            result = EqualityError;
            goto result_cleanup;
        }
        unsigned int count = counts.data[i].count;
        if (count != 2) {
            printf("Unexpected gate counts found");
            result = EqualityError;
            goto result_cleanup;
        }
    }
    QkCircuitInstruction inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(res); i++) {
        qk_circuit_get_instruction(res, i, &inst);
        if (inst.num_qubits != 1) {
            printf("Gate %d operates on more than 1 qubit: %zu", i, inst.num_qubits);
            result = EqualityError;
            goto result_cleanup;
        }
    }

result_cleanup:
    qk_split_2q_unitaries_result_free(split_result);
cleanup:
    qk_circuit_free(qc);
    return result;
}

int test_split_2q_unitaries_swap_x_y_unitary(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    QkComplex64 c0 = {0., 0.};
    QkComplex64 neg_im = {0., -1.};
    QkComplex64 im = {0., 1.};
    QkComplex64 unitary[16] = {
        c0, c0, c0, neg_im, c0, neg_im, c0, c0, c0, c0, im, c0, im, c0, c0, c0,
    };
    uint32_t qargs[2] = {0, 1};
    qk_circuit_unitary(qc, unitary, qargs, 2, true);
    QkSplit2qUnitariesResult *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (qk_split_2q_unitaries_result_permutation_len(split_result) != 2) {
        result = EqualityError;
        printf("Permutation returned for a circuit that shouldn't split");
        goto result_cleanup;
    }
    uint32_t *permutation = qk_split_2q_unitaries_result_permutation(split_result);
    uint32_t expected[2] = {1, 0};
    for (int i = 0; i < 2; i++) {
        if (permutation[i] != expected[i]) {
            printf("Permutation at position %d not as expected, found %zu expected %zu", i,
                   permutation[i], expected[i]);
            goto result_cleanup;
        }
    }

    QkCircuit *res = qk_split_2q_unitaries_result_circuit(split_result);
    QkOpCounts counts = qk_circuit_count_ops(res);
    if (counts.len != 1) {
        printf("More than 1 type of gate in the circuit");
        result = EqualityError;
        goto result_cleanup;
    }
    for (int i = 0; i < counts.len; i++) {
        int gate = strcmp(counts.data[i].name, "unitary");
        if (gate != 0) {
            printf("Gates outside expected set in output circuit");
            result = EqualityError;
            goto result_cleanup;
        }
        unsigned int count = counts.data[i].count;
        if (count != 2) {
            printf("Unexpected gate counts found");
            result = EqualityError;
            goto result_cleanup;
        }
    }
    QkCircuitInstruction inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(res); i++) {
        qk_circuit_get_instruction(res, i, &inst);
        if (inst.num_qubits != 1) {
            printf("Gate %d operates on more than 1 qubit: %zu", i, inst.num_qubits);
            result = EqualityError;
            goto result_cleanup;
        }
    }

result_cleanup:
    qk_split_2q_unitaries_result_free(split_result);
cleanup:
    qk_circuit_free(qc);
    return result;
}

int test_split_2q_unitaries(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_split_2q_unitaries_no_unitaries);
    num_failed += RUN_TEST(test_split_2q_unitaries_x_y_unitary);
    num_failed += RUN_TEST(test_split_2q_unitaries_swap_x_y_unitary);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
