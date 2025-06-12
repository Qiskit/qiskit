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
 * Test adding an instruction to the Target.
 */
int test_vf2_layout_line(void) {
    const uint32_t num_qubits = 5;
    // Let's create a target with one qubit for now
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;

    // Add an X Gate.
    // This operation is global, no property map is provided
    QkExitCode result_x = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a global X gate.");
        result = EqualityError;
        goto cleanup;
    }

    // Re-add same gate, check if it fails
    QkExitCode result_x_readded = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x_readded != QkExitCode_TargetInstAlreadyExists) {
        printf("The addition of a repeated gate did not fail as expected.");
        result = EqualityError;
        goto cleanup;
    }

    // Number of qubits of the target should not change.
    uint32_t current_num_qubits = qk_target_num_qubits(target);

    size_t current_size = qk_target_num_instructions(target);
    if (current_size != 1) {
        printf("The size of this target is not correct: Expected 1, got %zu", current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (current_num_qubits - i);
        double inst_duration = 0.020039;

        QkExitCode result_cx_props =
            qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
        if (result_cx_props != QkExitCode_Success) {
            printf("Unexpected error occurred when adding property to a CX gate entry.");
            result = EqualityError;
            goto cleanup;
        }
    }

    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a CX gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    QkVF2LayoutResult *layout_result =
        qk_transpiler_pass_standalone_vf2_layout(qc, target, false, -1, NAN, -1);
    if (!qk_vf2_layout_has_match(layout_result)) {
        printf("No layout was found");
        result = EqualityError;
        goto layout_cleanup;
    }
    if (qk_vf2_layout_num_qubits(layout_result) != qk_circuit_num_qubits(qc)) {
        printf("Layout doesn't contain the same number of qubits as the circuit, %d != %d",
               qk_vf2_layout_num_qubits(layout_result), qk_circuit_num_qubits(qc));
        result = EqualityError;
        goto layout_cleanup;
    }
    // Higher indexed physical qubits have lower error rates and
    // higher indexed virtual qubits have more gates. VF2 should
    // chose a trivial layout since it's the lowest error rate
    // in mapping these lines together.
    uint32_t expected[5] = {0, 1, 2, 3, 4};
    for (uint32_t i = 0; i < qk_vf2_layout_num_qubits(layout_result); i++) {
        uint32_t phys = qk_vf2_layout_map_virtual_qubit(layout_result, i);
        if (phys != expected[i]) {
            printf("Unexpected layout result virtual qubit %d mapped to %d", phys, expected[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }

layout_cleanup:
    qk_vf2_layout_free(layout_result);
circuit_cleanup:
    qk_circuit_free(qc);
cleanup:
    qk_target_free(target);
    return result;
}

int test_vf2_layout(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_vf2_layout_line);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
