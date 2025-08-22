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
 * Test running sabre layout that requires layout and routing
 */
int test_sabre_layout_applies_layout(void) {
    int result = Ok;

    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);

    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (num_qubits - i);
        double inst_duration = 0.020039;
        qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
    }

    qk_target_add_instruction(target, cx_entry);

    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {0, i + 1};
        qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
    }
    QkSabreLayoutOptions options = qk_sabre_layout_options_default();
    options.seed = 2025;
    QkTranspileLayout *layout_result =
        qk_transpiler_pass_standalone_sabre_layout(qc, target, &options);

    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    if (op_counts.len != 2) {
        printf("More than 2 types of gates in circuit, circuit's instructions are:\n");
        print_circuit(qc);
        result = EqualityError;
        goto layout_cleanup;
    }
    for (int i = 0; i < op_counts.len; i++) {
        int swap_gate = strcmp(op_counts.data[i].name, "swap");
        int cx_gate = strcmp(op_counts.data[i].name, "cx");
        if (cx_gate != 0 && swap_gate != 0) {
            printf("Gate type of %s found in the circuit which isn't expected\n");
            result = EqualityError;
            goto layout_cleanup;
        }
        if (swap_gate == 0 && op_counts.data[i].count != 2) {
            printf("Unexpected number of swaps %d found in the circuit.\n");
            result = EqualityError;
            goto layout_cleanup;
        }
    }
    uint32_t expected_initial_layout[5] = {1, 0, 2, 3, 4};
    uint32_t result_initial_layout[5];
    qk_transpile_layout_initial_layout(layout_result, false, result_initial_layout);
    for (uint32_t i = 0; i < 5; i++) {
        if (result_initial_layout[i] != expected_initial_layout[i]) {
            printf("Initial layout maps qubit %d to %d, expected %d instead\n", i,
                   result_initial_layout[i], expected_initial_layout[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }

    uint32_t expected_permutation[5] = {0, 3, 1, 2, 4};
    uint32_t result_permutation[5];
    qk_transpile_layout_output_permutation(layout_result, result_permutation);
    for (uint32_t i = 0; i < 5; i++) {
        if (result_permutation[i] != expected_permutation[i]) {
            printf("Output permutation maps qubit %d to %d, expected %d instead\n", i,
                   result_permutation[i], expected_permutation[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }
    QkCircuit *expected_circuit = qk_circuit_new(5, 0);
    uint32_t qargs[2] = {1, 0};
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    qargs[0] = 1;
    qargs[1] = 2;
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    qargs[0] = 2;
    qargs[1] = 1;
    qk_circuit_gate(expected_circuit, QkGate_Swap, qargs, NULL);
    qargs[0] = 2;
    qargs[1] = 3;
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    qargs[0] = 3;
    qargs[1] = 2;
    qk_circuit_gate(expected_circuit, QkGate_Swap, qargs, NULL);
    qargs[0] = 3;
    qargs[1] = 4;
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    bool compare_result = compare_circuits(qc, expected_circuit);
    if (!compare_result) {
        result = EqualityError;
    }
    qk_circuit_free(expected_circuit);

layout_cleanup:
    qk_opcounts_free(op_counts);
    qk_transpile_layout_free(layout_result);
circuit_cleanup:
    qk_circuit_free(qc);
cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test running sabre layout that performs no transformation.
 */
int test_sabre_layout_no_swap(void) {
    int result = Ok;

    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (num_qubits - i);
        double inst_duration = 0.020039;

        qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
    }

    qk_target_add_instruction(target, cx_entry);

    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    QkSabreLayoutOptions options = qk_sabre_layout_options_default();
    options.seed = 2025;
    QkTranspileLayout *layout_result =
        qk_transpiler_pass_standalone_sabre_layout(qc, target, &options);
    QkCircuit *expected_circuit = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
        }
    }
    bool circuit_eq = compare_circuits(qc, expected_circuit);
    if (!circuit_eq) {
        result = EqualityError;
        goto layout_cleanup;
    }
    uint32_t expected_initial_layout[5] = {0, 1, 2, 3, 4};
    uint32_t result_initial_layout[5];
    qk_transpile_layout_initial_layout(layout_result, false, result_initial_layout);
    for (uint32_t i = 0; i < 5; i++) {
        if (result_initial_layout[i] != expected_initial_layout[i]) {
            printf("Initial layout maps qubit %d to %d, expected %d instead\n", i,
                   result_initial_layout[i], expected_initial_layout[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }
    uint32_t expected_permutation[5] = {0, 1, 2, 3, 4};
    uint32_t result_permutation[5];
    qk_transpile_layout_output_permutation(layout_result, result_permutation);
    for (uint32_t i = 0; i < 5; i++) {
        if (result_permutation[i] != expected_permutation[i]) {
            printf("Output permutation maps qubit %d to %d, expected %d instead\n", i,
                   result_permutation[i], expected_permutation[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }

layout_cleanup:
    qk_circuit_free(expected_circuit);
    qk_transpile_layout_free(layout_result);
circuit_cleanup:
    qk_circuit_free(qc);
cleanup:
    qk_target_free(target);
    return result;
}

int test_sabre_layout(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_sabre_layout_no_swap);
    num_failed += RUN_TEST(test_sabre_layout_applies_layout);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
