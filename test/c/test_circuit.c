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

/**
 * Test the zero constructor.
 */
int test_empty() {
    QkCircuit *qc = qk_circuit_new(0, 0);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    qk_circuit_free(qc);

    return (num_qubits != 0 || num_clbits != 0 || num_instructions != 0) ? EqualityError : Ok;
}

int test_no_gate_1000_bits() {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    qk_circuit_free(qc);

    return (num_qubits != 1000 || num_clbits != 1000 || num_instructions != 0) ? EqualityError : Ok;
}

int test_get_gate_counts_bv_no_measure() {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i = 0;
    uint32_t qubits[1] = {999};
    qk_circuit_append_standard_gate(qc, QkStandardGate_XGate, qubits, params);
    for (i = 0; i < 1000; i++) {
        uint32_t qubits[1] = {i};
        qk_circuit_append_standard_gate(qc, QkStandardGate_HGate, qubits, params);
    }
    for (i = 0; i < 1000; i += 2) {
        uint32_t qubits[2] = {i, 999};
        qk_circuit_append_standard_gate(qc, QkStandardGate_CXGate, qubits, params);
    }
    for (i = 0; i < 999; i++) {
        uint32_t qubits[1] = {i};
        qk_circuit_append_standard_gate(qc, QkStandardGate_HGate, qubits, params);
    }
    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    int result = Ok;
    if (op_counts.len != 3) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[2].name, "x");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[2].count != 1) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[0].name, "h");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[0].count != 1999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[1].name, "cx");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[1].count != 500) {
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_free(op_counts);
    return result;
}

int test_get_gate_counts_bv_measures() {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i = 0;
    uint32_t qubits[1] = {999};
    qk_circuit_append_standard_gate(qc, QkStandardGate_XGate, qubits, params);
    for (i = 0; i < 1000; i++) {
        uint32_t qubits[1] = {i};
        qk_circuit_append_standard_gate(qc, QkStandardGate_HGate, qubits, params);
    }
    for (i = 0; i < 1000; i += 2) {
        uint32_t qubits[2] = {i, 999};
        qk_circuit_append_standard_gate(qc, QkStandardGate_CXGate, qubits, params);
    }
    for (i = 0; i < 999; i++) {
        uint32_t qubits[1] = {i};
        qk_circuit_append_standard_gate(qc, QkStandardGate_HGate, qubits, params);
    }
    for (i = 0; i < 999; i++) {
        qk_circuit_append_measure(qc, i, i);
    }
    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    int result = Ok;
    if (op_counts.len != 4) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[3].name, "x");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[3].count != 1) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[0].name, "h");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[0].count != 1999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[2].name, "cx");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[2].count != 500) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[1].name, "measure");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[1].count != 999) {
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_free(op_counts);
    return result;
}

int test_get_gate_counts_bv_barrier_and_measures() {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i = 0;
    uint32_t qubits[1] = {999};
    qk_circuit_append_standard_gate(qc, QkStandardGate_XGate, qubits, params);
    for (i = 0; i < 1000; i++) {
        uint32_t qubits[1] = {i};
        qk_circuit_append_standard_gate(qc, QkStandardGate_HGate, qubits, params);
    }
    uint32_t barrier_qubits[1000];
    for (i = 0; i < 1000; i++) {
        barrier_qubits[i] = i;
    }
    qk_circuit_append_barrier(qc, 1000, barrier_qubits);
    for (i = 0; i < 1000; i += 2) {
        uint32_t qubits[2] = {i, 999};
        qk_circuit_append_standard_gate(qc, QkStandardGate_CXGate, qubits, params);
    }
    qk_circuit_append_barrier(qc, 1000, barrier_qubits);
    for (i = 0; i < 999; i++) {
        uint32_t qubits[1] = {i};
        qk_circuit_append_standard_gate(qc, QkStandardGate_HGate, qubits, params);
    }
    for (i = 0; i < 999; i++) {
        qk_circuit_append_measure(qc, i, i);
    }
    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    int result = Ok;
    if (op_counts.len != 5) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[4].name, "x");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[4].count != 1) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[0].name, "h");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[0].count != 1999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[2].name, "cx");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[2].count != 500) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[1].name, "measure");
    if (result != 0) {
        return result;
    }
    if (op_counts.data[1].count != 999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[3].name, "barrier");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[3].count != 2) {
        return EqualityError;
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_free(op_counts);
    return result;
}

int test_get_gate_counts_bv_resets_barrier_and_measures() {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i = 0;
    uint32_t qubits[1] = {999};
    for (i = 0; i < 1000; i++) {
        uint32_t qubits[1] = {i};
        qk_circuit_append_reset(qc, i);
    }
    qk_circuit_append_standard_gate(qc, QkStandardGate_XGate, qubits, params);
    for (i = 0; i < 1000; i++) {
        uint32_t qubits[1] = {i};
        qk_circuit_append_standard_gate(qc, QkStandardGate_HGate, qubits, params);
    }
    uint32_t barrier_qubits[1000];
    for (i = 0; i < 1000; i++) {
        barrier_qubits[i] = i;
    }
    qk_circuit_append_barrier(qc, 1000, barrier_qubits);
    for (i = 0; i < 1000; i += 2) {
        uint32_t qubits[2] = {i, 999};
        qk_circuit_append_standard_gate(qc, QkStandardGate_CXGate, qubits, params);
    }
    qk_circuit_append_barrier(qc, 1000, barrier_qubits);
    for (i = 0; i < 999; i++) {
        uint32_t qubits[1] = {i};
        qk_circuit_append_standard_gate(qc, QkStandardGate_HGate, qubits, params);
    }
    for (i = 0; i < 999; i++) {
        qk_circuit_append_measure(qc, i, i);
    }
    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    int result = Ok;
    if (op_counts.len != 6) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[5].name, "x");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[5].count != 1) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[0].name, "h");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[0].count != 1999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[3].name, "cx");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[3].count != 500) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[2].name, "measure");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[2].count != 999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[4].name, "barrier");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[4].count != 2) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[1].name, "reset");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[1].count != 1000) {
        result = EqualityError;
        goto cleanup;
    }
    size_t num_instructions = qk_circuit_num_instructions(qc);
    if (num_instructions != 1000 + 2 + 999 + 500 + 1999 + 1) {
        result = EqualityError;
        goto cleanup;
    }
    for (size_t i = 0; i < num_instructions; i++) {

        QkCircuitInstruction inst = qk_circuit_get_instruction(qc, i);
        if (i < 1000) {
            result = strcmp(inst.name, "reset");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != i || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i == 1000) {
            result = strcmp(inst.name, "x");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != 999 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i < 2001) {
            result = strcmp(inst.name, "h");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != i - 1001 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i == 2001) {
            result = strcmp(inst.name, "barrier");
            if (result != 0) {
                goto loop_exit;
            }
            for (uint32_t j = 0; i < 1000; j++) {
                if (inst.qubits[i] != i) {
                    result = EqualityError;
                    goto loop_exit;
                }
            }
            if (inst.num_clbits > 0 || inst.num_params > 0 || inst.num_qubits != 1000) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i <= 2501) {
            result = strcmp(inst.name, "cx");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != (i - 2002) * 2) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.qubits[1] != 999 || inst.num_qubits != 2) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i == 2502) {
            result = strcmp(inst.name, "barrier");
            if (result != 0) {
                goto loop_exit;
            }
            for (uint32_t j = 0; i < 1000; j++) {
                if (inst.qubits[i] != i) {
                    result = EqualityError;
                    goto loop_exit;
                }
            }
            if (inst.num_clbits > 0 || inst.num_params > 0 || inst.num_qubits != 1000) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i <= 3501) {
            result = strcmp(inst.name, "h");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != i - 2503 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i <= 4500) {
            result = strcmp(inst.name, "measure");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != i - 3502 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.clbits[0] != i - 3502 || inst.num_clbits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        }
    loop_exit:
        qk_free_circuit_instruction(inst);
        if (result != 0) {
            break;
        }
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_free(op_counts);
    return result;
}

int test_circuit() {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty);
    num_failed += RUN_TEST(test_no_gate_1000_bits);
    num_failed += RUN_TEST(test_get_gate_counts_bv_no_measure);
    num_failed += RUN_TEST(test_get_gate_counts_bv_measures);
    num_failed += RUN_TEST(test_get_gate_counts_bv_barrier_and_measures);
    num_failed += RUN_TEST(test_get_gate_counts_bv_resets_barrier_and_measures);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
