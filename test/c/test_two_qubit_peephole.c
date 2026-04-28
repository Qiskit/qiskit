// This code is part of Qiskit.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
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

static int build_unitary_target(QkTarget *target, uint32_t num_qubits) {
    // Create a target with cx connectivity in a line.
    QkExitCode result_x = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a global X gate.");
        return RuntimeError;
    }
    QkExitCode result_sx = qk_target_add_instruction(target, qk_target_entry_new(QkGate_SX));
    if (result_sx != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a global SX gate.");
        return RuntimeError;
    }

    QkExitCode result_rz = qk_target_add_instruction(target, qk_target_entry_new(QkGate_RZ));
    if (result_rz != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a global RZ gate.");
        return RuntimeError;
    }

    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (num_qubits - i);
        double inst_duration = 0.020039;

        QkExitCode result_cx_props =
            qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
        if (result_cx_props != QkExitCode_Success) {
            printf("Unexpected error occurred when adding property to a CX gate entry.");
            return RuntimeError;
        }
    }
    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a CX gate.");
        return RuntimeError;
    }
    return Ok;
}

/**
 * Test running two qubit peephole standalone
 */
static int test_peephole_standalone(void) {
    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    result = build_unitary_target(target, num_qubits);
    if (result != Ok) {
        goto cleanup;
    }
    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t forward[2] = {0, 1};
    uint32_t reverse[2] = {1, 0};
    for (int i = 0; i < 100; i++) {
        if (i % 2) {
            qk_circuit_gate(qc, QkGate_CX, forward, NULL);
        } else {
            qk_circuit_gate(qc, QkGate_CX, reverse, NULL);
        }
    }
    qk_transpiler_pass_standalone_two_qubit_peephole_optimization(qc, target, 1.0);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    if (num_instructions != 14) {
        printf("Circuit not simplified as expected");
        result = EqualityError;
    }
    qk_circuit_free(qc);

cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test running two qubit peephole
 */
static int test_peephole(void) {
    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    result = build_unitary_target(target, num_qubits);
    if (result != Ok) {
        goto cleanup;
    }
    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t forward[2] = {0, 1};
    uint32_t reverse[2] = {1, 0};
    for (int i = 0; i < 100; i++) {
        if (i % 2) {
            qk_circuit_gate(qc, QkGate_CX, forward, NULL);
        } else {
            qk_circuit_gate(qc, QkGate_CX, reverse, NULL);
        }
    }
    QkDag *dag = qk_circuit_to_dag(qc);
    qk_circuit_free(qc);
    qk_transpiler_pass_two_qubit_peephole_optimization(dag, target, 1.0);
    QkCircuit *out_circuit = qk_dag_to_circuit(dag);
    qk_dag_free(dag);
    size_t num_instructions = qk_circuit_num_instructions(out_circuit);
    if (num_instructions != 14) {
        printf("Circuit not simplified as expected");
        result = EqualityError;
    }
    qk_circuit_free(out_circuit);

cleanup:
    qk_target_free(target);
    return result;
}

int test_two_qubit_peephole(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_peephole_standalone);
    num_failed += RUN_TEST(test_peephole);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
