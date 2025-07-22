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

bool args_cmp(uint32_t *args1, uint32_t args1_len, uint32_t *args2, uint32_t args2_len);

/**
 * Test a small block of gates can be turned into a unitary on same wires
 */
int test_consolidate_small_block() {
    int result = Ok;

    // Build circuit
    QkCircuit *circuit = qk_circuit_new(2, 0);

    uint32_t p_qargs[1] = {0};
    uint32_t u_qargs[1] = {1};
    uint32_t cx_qargs[2] = {0, 1};

    double p_params[1] = {0.5};
    double u_params[3] = {1.5708, 0.2, 0.6};
    qk_circuit_gate(circuit, QkGate_Phase, p_qargs, p_params);
    qk_circuit_gate(circuit, QkGate_U, u_qargs, u_params);
    qk_circuit_gate(circuit, QkGate_CX, cx_qargs, NULL);

    // Run pass, without Target
    QkCircuit *circ_result =
        qk_transpiler_pass_standalone_consolidate_blocks(circuit, NULL, 1.0, true);

    QkOpCounts counts = qk_circuit_count_ops(circ_result);

    if (counts.len != 1) {
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(circuit);
    qk_circuit_free(circ_result);
    return result;
}

/**
 * Order of qubits and the corresponding unitary is correct
 */
int test_wire_order() {
    int result = Ok;

    // Build circuit
    QkCircuit *circuit = qk_circuit_new(2, 0);
    uint32_t cx_qargs[2] = {1, 0};

    qk_circuit_gate(circuit, QkGate_CX, cx_qargs, NULL);

    // Run pass, without Target
    QkCircuit *circ_result =
        qk_transpiler_pass_standalone_consolidate_blocks(circuit, NULL, 1.0, true);

    QkOpCounts counts = qk_circuit_count_ops(circ_result);

    if (counts.len != 1) {
        result = EqualityError;
        goto cleanup;
    }

    QkCircuitInstruction *instruction = malloc(sizeof(QkCircuitInstruction));
    qk_circuit_get_instruction(circ_result, 0, instruction);

    uint32_t new_cx_qargs[2] = {0, 1};
    if (!args_cmp(new_cx_qargs, 2, instruction->qubits, instruction->num_qubits)) {
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(circuit);
    qk_circuit_free(circ_result);
    qk_circuit_instruction_clear(instruction);
    return result;
}

/**
 * blocks of more than 2 qubits work.
 */
int test_3q_blocks() {
    int result = Ok;

    // Build circuit
    QkCircuit *circuit = qk_circuit_new(3, 0);

    uint32_t p_qargs[1] = {0};
    uint32_t u_qargs[1] = {1};
    uint32_t cx_qargs_0[2] = {2, 1};
    uint32_t cx_qargs_1[2] = {0, 1};

    double p_params[1] = {0.5};
    double u_params[3] = {1.5708, 0.2, 0.6};
    qk_circuit_gate(circuit, QkGate_Phase, p_qargs, p_params);
    qk_circuit_gate(circuit, QkGate_U, u_qargs, u_params);
    qk_circuit_gate(circuit, QkGate_CX, cx_qargs_0, NULL);
    qk_circuit_gate(circuit, QkGate_CX, cx_qargs_1, NULL);

    // Run pass, without Target
    QkCircuit *circ_result =
        qk_transpiler_pass_standalone_consolidate_blocks(circuit, NULL, 1.0, true);

    QkOpCounts counts = qk_circuit_count_ops(circ_result);
    if (counts.len != 1) {
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(circuit);
    qk_circuit_free(circ_result);
    return result;
}

int test_consolidate_blocks(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_consolidate_small_block);
    num_failed += RUN_TEST(test_wire_order);
    num_failed += RUN_TEST(test_3q_blocks);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}

bool args_cmp(uint32_t *args1, uint32_t args1_len, uint32_t *args2, uint32_t args2_len) {
    if (args1_len != args2_len) {
        return false;
    }
    for (size_t i = 0; i < args1_len; i++) {
        if (args1[i] != args2[i]) {
            return false;
        }
    }
    return true;
}
