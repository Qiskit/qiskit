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
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int test_basic_basis_translator(void) {
    // Create circuit
    int result = Ok;
    QkCircuit *circuit = qk_circuit_new(1, 0);
    qk_circuit_gate(circuit, QkGate_H, (uint32_t[1]){0}, NULL);

    // Create Target compatible with only U gates, with global props.
    QkTarget *target = qk_target_new(1);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));

    // Run pass
    qk_transpiler_pass_standalone_basis_translator(circuit, target, 0);

    QkOpCounts result_op_counts = qk_circuit_count_ops(circuit);

    if (result_op_counts.len != 1) {
        result = EqualityError;
        printf(
            "The number of gates resulting from the translation is incorrect. Expected 1, got %lu",
            result_op_counts.len);
        goto cleanup;
    }

    QkOpCount u_count = result_op_counts.data[0];
    if (u_count.count != 1 || strcmp(u_count.name, "u") != 0) {
        result = EqualityError;
        printf("The operation resulting from this translation was incorrect. Expected 'u' gate, "
               "got '%s'",
               u_count.name);
    }

cleanup:
    qk_opcounts_clear(&result_op_counts);
    qk_circuit_free(circuit);
    qk_target_free(target);
    return result;
}

int test_toffoli_basis_translator(void) {
    // Create circuit
    int result = Ok;
    QkCircuit *circuit = qk_circuit_new(3, 0);
    qk_circuit_gate(circuit, QkGate_CCX, (uint32_t[3]){0, 1, 2}, NULL);

    // Create Target compatible with only U gates, with global props.
    QkTarget *target = qk_target_new(3);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_H));
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_T));
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_Tdg));
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_CX));

    // Run pass
    qk_transpiler_pass_standalone_basis_translator(circuit, target, 0);

    QkOpCounts result_op_counts = qk_circuit_count_ops(circuit);

    if (result_op_counts.len != 4) {
        result = EqualityError;
        printf(
            "The number of gates resulting from the translation is incorrect. Expected 1, got %lu",
            result_op_counts.len);
        goto cleanup;
    }

    // Represent the Equivalence of CCX with op counts
    char *gates[4] = {"cx", "t", "tdg", "h"};
    size_t freqs[4] = {6, 4, 3, 2};
    for (int idx = 0; idx < 4; idx++) {
        QkOpCount gate_count = result_op_counts.data[idx];
        if (gate_count.count != freqs[idx] || strcmp(gate_count.name, gates[idx]) != 0) {
            result = EqualityError;
            printf(
                "The operation resulting from this translation was incorrect. Expected '%s' gate, "
                "got '%s'",
                gates[idx], gate_count.name);
        }
    }

cleanup:
    qk_target_free(target);
    qk_opcounts_clear(&result_op_counts);
    qk_circuit_free(circuit);
    return result;
}

int test_basis_translator(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_basic_basis_translator);
    num_failed += RUN_TEST(test_toffoli_basis_translator);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
