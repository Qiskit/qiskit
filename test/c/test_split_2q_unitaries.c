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
    QkTranspileLayout *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (split_result != NULL) {
        result = EqualityError;
        printf("Permutation returned for a circuit that shouldn't split\n");
        qk_transpile_layout_free(split_result);
        goto cleanup;
    }
    QkOpCounts counts = qk_circuit_count_ops(qc);
    if (counts.len != 1) {
        printf("More than 1 type of gate in the circuit\n");
        result = EqualityError;
        goto ops_cleanup;
    }
    for (int i = 0; i < counts.len; i++) {
        int gate = strcmp(counts.data[i].name, "cx");
        if (gate != 0) {
            printf("gates changed when there should be no circuit changes\n");
            result = EqualityError;
            goto cleanup;
        }
    }
ops_cleanup:
    qk_opcounts_free(counts);
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
    QkTranspileLayout *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (split_result != NULL) {
        result = EqualityError;
        printf("Permutation returned for a circuit that shouldn't isn't a swap equivalent\n");
        qk_transpile_layout_free(split_result);
        goto cleanup;
    }
    QkOpCounts counts = qk_circuit_count_ops(qc);
    if (counts.len != 1) {
        printf("More than 1 type of gate in the circuit\n");
        result = EqualityError;
        goto ops_cleanup;
    }
    for (int i = 0; i < counts.len; i++) {
        int gate = strcmp(counts.data[i].name, "unitary");
        if (gate != 0) {
            printf("Gates outside expected set in output circuit\n");
            result = EqualityError;
            goto ops_cleanup;
        }
        unsigned int count = counts.data[i].count;
        if (count != 2) {
            printf("Unexpected gate counts found\n");
            result = EqualityError;
            goto ops_cleanup;
        }
    }
    QkCircuitInstruction inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(qc); i++) {
        qk_circuit_get_instruction(qc, i, &inst);
        if (inst.num_qubits != 1) {
            printf("Gate %zu operates on more than 1 qubit: %u\n", i, inst.num_qubits);
            result = EqualityError;
            goto ops_cleanup;
        }
        qk_circuit_instruction_clear(&inst);
    }

ops_cleanup:
    qk_opcounts_free(counts);
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
    QkTranspileLayout *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (split_result == NULL) {
        result = EqualityError;
        printf("Permutation returned for a circuit that shouldn't split\n");
        goto cleanup;
    }
    uint32_t permutation[2];
    qk_transpile_layout_output_permutation(split_result, permutation);
    uint32_t expected[2] = {1, 0};
    for (int i = 0; i < 2; i++) {
        if (permutation[i] != expected[i]) {
            printf("Permutation at position %d not as expected, found %u expected %u\n", i,
                   permutation[i], expected[i]);
            goto cleanup;
        }
    }

    QkOpCounts counts = qk_circuit_count_ops(qc);
    if (counts.len != 1) {
        printf("More than 1 type of gate in the circuit\n");
        result = EqualityError;
        goto ops_cleanup;
    }
    for (int i = 0; i < counts.len; i++) {
        int gate = strcmp(counts.data[i].name, "unitary");
        if (gate != 0) {
            printf("Gates outside expected set in output circuit\n");
            result = EqualityError;
            goto ops_cleanup;
        }
        unsigned int count = counts.data[i].count;
        if (count != 2) {
            printf("Unexpected gate counts found\n");
            result = EqualityError;
            goto ops_cleanup;
        }
    }
    QkCircuitInstruction inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(qc); i++) {
        qk_circuit_get_instruction(qc, i, &inst);
        if (inst.num_qubits != 1) {
            printf("Gate %zu operates on more than 1 qubit: %u\n", i, inst.num_qubits);
            result = EqualityError;
            goto ops_cleanup;
        }
        qk_circuit_instruction_clear(&inst);
    }

ops_cleanup:
    qk_opcounts_free(counts);
cleanup:
    qk_transpile_layout_free(split_result);
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
