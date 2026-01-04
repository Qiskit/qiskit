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
static int test_empty(void) {
    QkCircuit *qc = qk_circuit_new(0, 0);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);

    QkOpCounts counts = qk_circuit_count_ops(qc);
    size_t opcount = counts.len;

    qk_opcounts_clear(&counts);
    qk_circuit_free(qc);

    if (opcount != 0) {
        printf("The operation count %zu is not 0", opcount);
        return EqualityError;
    }
    if (num_qubits != 0) {
        printf("The number of qubits %d is not 0", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %d is not 0", num_clbits);
        return EqualityError;
    }
    if (num_instructions != 0) {
        printf("The number of instructions %zu is not 0", num_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_with_quantum_reg(void) {
    QkCircuit *qc = qk_circuit_new(0, 0);
    QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_little_register");
    qk_circuit_add_quantum_register(qc, qr);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    qk_circuit_free(qc);
    qk_quantum_register_free(qr);
    if (num_qubits != 1024) {
        printf("The number of qubits %d is not 1024", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %d is not 0", num_clbits);
        return EqualityError;
    }
    if (num_instructions != 0) {
        printf("The number of instructions %zu is not 0", num_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_copy(void) {
    QkCircuit *qc = qk_circuit_new(10, 10);
    QkCircuit *copy = qk_circuit_copy(qc);
    for (int i = 0; i < 10; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {
            i,
        };
        if (i % 2 == 0) {
            qk_circuit_gate(copy, QkGate_H, qubits, NULL);
        }
    }
    size_t num_instructions = qk_circuit_num_instructions(qc);
    size_t num_copy_instructions = qk_circuit_num_instructions(copy);
    qk_circuit_free(qc);
    qk_circuit_free(copy);
    if (num_instructions == num_copy_instructions) {
        printf("The number of instructions %zu is equal to the copied %zu", num_instructions,
               num_copy_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_with_classical_reg(void) {
    QkCircuit *qc = qk_circuit_new(0, 0);
    QkClassicalRegister *cr = qk_classical_register_new(2048, "my_less_little_register");
    qk_circuit_add_classical_register(qc, cr);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    qk_circuit_free(qc);
    qk_classical_register_free(cr);
    if (num_qubits != 0) {
        printf("The number of qubits %d is not 0", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 2048) {
        printf("The number of clbits %d is not 2048", num_clbits);
        return EqualityError;
    }
    if (num_instructions != 0) {
        printf("The number of instructions %zu is not 0", num_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_copy_with_instructions(void) {
    QkCircuit *qc = qk_circuit_new(10, 10);
    for (int i = 0; i < 10; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {
            i,
        };
        qk_circuit_gate(qc, QkGate_H, qubits, NULL);
    }
    QkCircuit *copy = qk_circuit_copy(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    size_t num_copy_instructions = qk_circuit_num_instructions(copy);
    if (num_instructions != num_copy_instructions) {
        printf("The number of instructions %zu does not equal the copied %zu", num_instructions,
               num_copy_instructions);
        return EqualityError;
    }

    for (int i = 0; i < 10; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {
            i,
        };
        qk_circuit_gate(qc, QkGate_Z, qubits, NULL);
    }
    for (int i = 0; i < 15; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {
            i,
        };
        qk_circuit_gate(copy, QkGate_X, qubits, NULL);
    }

    num_instructions = qk_circuit_num_instructions(qc);
    num_copy_instructions = qk_circuit_num_instructions(copy);
    qk_circuit_free(qc);
    qk_circuit_free(copy);
    if (num_instructions == num_copy_instructions) {
        printf("The number of instructions %zu is equal to the copied %zu", num_instructions,
               num_copy_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_copy_empty_like(void) {
    QkCircuit *qc = qk_circuit_new(10, 10);
    for (int i = 0; i < 10; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {i};
        qk_circuit_gate(qc, QkGate_H, qubits, NULL);
    }
    QkCircuit *copy = qk_circuit_copy_empty_like(qc, QkVarsMode_Alike, QkBlocksMode_Drop);

    size_t num_instructions = qk_circuit_num_instructions(qc);        // not 0
    size_t num_copy_instructions = qk_circuit_num_instructions(copy); // 0

    qk_circuit_free(qc);
    qk_circuit_free(copy);

    if (num_instructions == 0) {
        printf("Expected the original circuit to remain unchanged, but it is now empty\n");
        return EqualityError;
    }

    if (num_copy_instructions != 0) {
        printf("Expected no operations in the copied-empty-like circuit, but got %zu\n",
               num_copy_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_no_gate_1000_bits(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    qk_circuit_free(qc);

    if (num_qubits != 1000) {
        printf("The number of qubits %d is not 1000", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 1000) {
        printf("The number of clbits %d is not 1000", num_clbits);
        return EqualityError;
    }
    if (num_instructions != 0) {
        printf("The number of instructions %zu is not 0", num_instructions);
        return EqualityError;
    }

    return Ok;
}

static int test_gate_num_qubits(void) {
    for (uint8_t i = 0; i < 52; i++) {
        if (i == 0) {
            if (qk_gate_num_qubits(i) != 0) {

                return EqualityError;
            }
        } else if (i < 21) {
            if (qk_gate_num_qubits(i) != 1) {
                return EqualityError;
            }
        } else if (i <= 44) {
            if (qk_gate_num_qubits(i) != 2) {
                return EqualityError;
            }
        } else if (i <= 48) {
            if (qk_gate_num_qubits(i) != 3) {
                return EqualityError;
            }
        } else {
            if (qk_gate_num_qubits(i) != 4) {
                return EqualityError;
            }
        }
    }
    return Ok;
}

static bool value_in_array(uint8_t val, uint8_t *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (arr[i] == val)
            return true;
    }
    return false;
}

static int test_gate_num_params(void) {

    uint8_t zero_param_gates[29] = {1,  2,  3,  4,  5,  11, 12, 13, 14, 15, 16, 21, 22, 23, 24,
                                    25, 26, 27, 28, 33, 34, 35, 45, 46, 47, 48, 49, 50, 51};
    uint8_t one_param_gates[15] = {0, 6, 8, 9, 10, 18, 29, 30, 31, 32, 37, 39, 40, 41, 42};
    uint8_t two_param_gates[4] = {7, 19, 43, 44};

    for (uint8_t i = 0; i < 52; i++) {
        if (value_in_array(i, zero_param_gates, 29)) {
            if (qk_gate_num_params(i) != 0) {
                return EqualityError;
            }
        } else if (value_in_array(i, one_param_gates, 15)) {
            if (qk_gate_num_params(i) != 1) {
                return EqualityError;
            }
        } else if (value_in_array(i, two_param_gates, 4)) {
            if (qk_gate_num_params(i) != 2) {
                return EqualityError;
            }
        } else if (i == 36) {
            // CU takes 4 parameters theta, phi, lambda, gamma
            if (qk_gate_num_params(i) != 4) {
                return EqualityError;
            }
        } else {
            if (qk_gate_num_params(i) != 3) {
                return EqualityError;
            }
        }
    }
    return Ok;
}

/**
 * Test edge cases for getting the op counts.
 */
static int test_get_gate_counts(void) {
    QkCircuit *qc = qk_circuit_new(3, 3);

    // test empty circuit
    int result = Ok;
    QkOpCounts c1 = qk_circuit_count_ops(qc);
    if (c1.len != 0) {
        result = EqualityError;
        qk_opcounts_clear(&c1);
        goto circuit_cleanup;
    }
    qk_opcounts_clear(&c1);

    // add some instructions
    uint32_t qubits[2] = {1, 0};
    double params[1] = {0.12};
    qk_circuit_gate(qc, QkGate_CRX, qubits, params);
    QkOpCounts c2 = qk_circuit_count_ops(qc);

    if (c2.len != 1) {
        result = EqualityError;
        qk_opcounts_clear(&c1);
        goto circuit_cleanup;
    }
    qk_opcounts_clear(&c2);

    // check that after clearing, the object is still valid
    if (c2.len != 0 || c2.data != NULL) {
        result = EqualityError;
    }

circuit_cleanup:
    qk_circuit_free(qc);
    return result;
}

static int test_get_gate_counts_bv_no_measure(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i;
    uint32_t q1[1] = {999};
    uint32_t q2[2] = {0, 999};

    qk_circuit_gate(qc, QkGate_X, q1, params);
    for (i = 0; i < 1000; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (i = 0; i < 1000; i += 2) {
        q2[0] = i;
        qk_circuit_gate(qc, QkGate_CX, q2, params);
    }
    for (i = 0; i < 999; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
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
    qk_opcounts_clear(&op_counts);
    return result;
}

static int test_get_gate_counts_bv_measures(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i;
    uint32_t q1[1] = {999};
    uint32_t q2[2] = {0, 999};

    qk_circuit_gate(qc, QkGate_X, q1, params);
    for (i = 0; i < 1000; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (i = 0; i < 1000; i += 2) {
        q2[0] = i;
        qk_circuit_gate(qc, QkGate_CX, q2, params);
    }
    for (i = 0; i < 999; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (i = 0; i < 999; i++) {
        qk_circuit_measure(qc, i, i);
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
    qk_opcounts_clear(&op_counts);
    return result;
}

static int test_get_gate_counts_bv_barrier_and_measures(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i;
    uint32_t q1[1] = {999};
    uint32_t q2[2] = {0, 999};

    qk_circuit_gate(qc, QkGate_X, q1, params);
    for (i = 0; i < 1000; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    uint32_t barrier_qubits[1000];
    for (i = 0; i < 1000; i++) {
        barrier_qubits[i] = i;
    }
    qk_circuit_barrier(qc, barrier_qubits, 1000);
    for (i = 0; i < 1000; i += 2) {
        q2[0] = i;
        qk_circuit_gate(qc, QkGate_CX, q2, params);
    }
    qk_circuit_barrier(qc, barrier_qubits, 1000);
    for (i = 0; i < 999; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (i = 0; i < 999; i++) {
        qk_circuit_measure(qc, i, i);
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
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_clear(&op_counts);
    return result;
}

static int test_get_gate_counts_bv_resets_barrier_and_measures(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t q1[1] = {999};
    uint32_t q2[2] = {0, 999};

    for (uint32_t i = 0; i < 1000; i++) {
        qk_circuit_reset(qc, i);
    }
    qk_circuit_gate(qc, QkGate_X, q1, params);
    for (uint32_t i = 0; i < 1000; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    uint32_t barrier_qubits[1000];
    for (uint32_t i = 0; i < 1000; i++) {
        barrier_qubits[i] = i;
    }
    qk_circuit_barrier(qc, barrier_qubits, 1000);
    for (uint32_t i = 0; i < 1000; i += 2) {
        q2[0] = i;
        qk_circuit_gate(qc, QkGate_CX, q2, params);
    }
    qk_circuit_barrier(qc, barrier_qubits, 1000);
    for (uint32_t i = 0; i < 999; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (uint32_t i = 0; i < 999; i++) {
        qk_circuit_measure(qc, i, i);
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
    QkCircuitInstruction inst;
    for (size_t i = 0; i < num_instructions; i++) {

        qk_circuit_get_instruction(qc, i, &inst);
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
            for (uint32_t j = 0; j < 1000; j++) {
                if (inst.qubits[j] != j) {
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
            for (uint32_t j = 0; j < 1000; j++) {
                if (inst.qubits[j] != j) {
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
        qk_circuit_instruction_clear(&inst);
        if (result != 0) {
            break;
        }
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_clear(&op_counts);
    return result;
}

/**
 * Test appending a unitary gate.
 */
static int test_unitary_gate(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t qubits[2] = {0, 1};

    QkComplex64 c0 = {0.0, 0.0};
    QkComplex64 c1 = {1.0, 0.0};
    QkComplex64 matrix[16] = {c1, c0, c0, c0,  // this
                              c0, c1, c0, c0,  // is
                              c0, c0, c1, c0,  // for
                              c0, c0, c0, c1}; // formatting

    int ec = qk_circuit_unitary(qc, matrix, qubits, 2, false);
    if (ec != QkExitCode_Success) {
        qk_circuit_free(qc);
        return ec;
    }

    int result = Ok;

    size_t num_inst = qk_circuit_num_instructions(qc);
    if (num_inst != 1) {
        result = EqualityError;
        goto cleanup;
    }

    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    if (op_counts.len != 1 || strcmp(op_counts.data[0].name, "unitary") != 0 ||
        op_counts.data[0].count != 1) {
        result = EqualityError;
        qk_opcounts_clear(&op_counts);
        goto cleanup;
    }
    qk_opcounts_clear(&op_counts);

    QkCircuitInstruction inst;
    qk_circuit_get_instruction(qc, 0, &inst);
    if (strcmp(inst.name, "unitary") != 0 || inst.num_clbits != 0 || inst.num_params != 0 ||
        inst.num_qubits != 2) {
        result = EqualityError;
    }
    qk_circuit_instruction_clear(&inst);

cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test appending a unitary gate.
 */
static int test_unitary_gate_1q(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t qubits[1] = {0};

    QkComplex64 c0 = {0.0, 0.0};
    QkComplex64 c1 = {1.0, 0.0};
    QkComplex64 matrix[4] = {c1, c0,  // this
                             c0, c1}; // is

    int ec = qk_circuit_unitary(qc, matrix, qubits, 1, false);
    if (ec != QkExitCode_Success) {
        qk_circuit_free(qc);
        return ec;
    }

    int result = Ok;

    size_t num_inst = qk_circuit_num_instructions(qc);
    if (num_inst != 1) {
        result = EqualityError;
        goto cleanup;
    }

    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    if (op_counts.len != 1 || strcmp(op_counts.data[0].name, "unitary") != 0 ||
        op_counts.data[0].count != 1) {
        result = EqualityError;
        qk_opcounts_clear(&op_counts);
        goto cleanup;
    }
    qk_opcounts_clear(&op_counts);

    QkCircuitInstruction inst;
    qk_circuit_get_instruction(qc, 0, &inst);
    if (strcmp(inst.name, "unitary") != 0 || inst.num_clbits != 0 || inst.num_params != 0 ||
        inst.num_qubits != 1) {
        result = EqualityError;
    }
    qk_circuit_instruction_clear(&inst);

cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test appending a unitary gate.
 */
static int test_unitary_gate_3q(void) {
    QkCircuit *qc = qk_circuit_new(3, 0);
    uint32_t qubits[3] = {0, 1, 2};

    QkComplex64 c0 = {0.0, 0.0};
    QkComplex64 c1 = {1.0, 0.0};
    QkComplex64 matrix[64] = {c1, c0, c0, c0, c0, c0, c0, c0,  // this
                              c0, c1, c0, c0, c0, c0, c0, c0,  // is
                              c0, c0, c1, c0, c0, c0, c0, c0,  // for
                              c0, c0, c0, c1, c0, c0, c0, c0,  // formatting
                              c0, c0, c0, c0, c1, c0, c0, c0,  // this
                              c0, c0, c0, c0, c0, c1, c0, c0,  // to
                              c0, c0, c0, c0, c0, c0, c1, c0,  // look
                              c0, c0, c0, c0, c0, c0, c0, c1}; // like a matrix

    int ec = qk_circuit_unitary(qc, matrix, qubits, 3, false);
    if (ec != QkExitCode_Success) {
        qk_circuit_free(qc);
        return ec;
    }

    int result = Ok;

    size_t num_inst = qk_circuit_num_instructions(qc);
    if (num_inst != 1) {
        result = EqualityError;
        goto cleanup;
    }

    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    if (op_counts.len != 1 || strcmp(op_counts.data[0].name, "unitary") != 0 ||
        op_counts.data[0].count != 1) {
        result = EqualityError;
        qk_opcounts_clear(&op_counts);
        goto cleanup;
    }
    qk_opcounts_clear(&op_counts);
    QkCircuitInstruction inst;
    qk_circuit_get_instruction(qc, 0, &inst);
    if (strcmp(inst.name, "unitary") != 0 || inst.num_clbits != 0 || inst.num_params != 0 ||
        inst.num_qubits != 3) {
        result = EqualityError;
    }
    qk_circuit_instruction_clear(&inst);

cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test passing a non-unitary gate returns the correct exit code.
 */
static int test_not_unitary_gate(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t qubits[2] = {0, 1};

    QkComplex64 c0 = {0.0, 0.0};
    QkComplex64 c1 = {1.0, 0.0};
    QkComplex64 matrix[16] = {c1, c1, c0, c0,  // this
                              c1, c1, c0, c0,  // is
                              c0, c0, c1, c0,  // for
                              c0, c0, c0, c1}; // formatting

    int exit_code = qk_circuit_unitary(qc, matrix, qubits, 2, true);

    int result = Ok;
    if (exit_code != QkExitCode_ExpectedUnitary) {
        printf("Got exit code %i but expected %i", exit_code, QkExitCode_ExpectedUnitary);
        result = EqualityError;
        goto cleanup;
    }

    size_t num_inst = qk_circuit_num_instructions(qc);
    if (num_inst != 0) { // we expect no gate was added
        printf("Found gate when none should be added");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test adding delay instruction to a circuit.
 */
static int test_delay_instruction(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    int result = Ok;

    QkExitCode delay_s_code;

    delay_s_code = qk_circuit_delay(qc, 0, 0.001, QkDelayUnit_S);
    if (delay_s_code != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test circuit to dag conversion.
 */
static int test_circuit_to_dag(void) {
    QkCircuit *circuit = qk_circuit_new(2, 1);
    qk_circuit_gate(circuit, QkGate_H, (uint32_t[]){0}, NULL);
    qk_circuit_gate(circuit, QkGate_CX, (uint32_t[]){0, 1}, NULL);

    QkDag *dag = qk_circuit_to_dag(circuit);
    qk_circuit_free(circuit);

    int result = Ok;
    if (qk_dag_num_qubits(dag) != 2 || qk_dag_num_clbits(dag) != 1 ||
        qk_dag_num_op_nodes(dag) != 2) {
        printf("Circuit to DAG conversion encountered an issue\n");
        result = EqualityError;
    }

    qk_dag_free(dag);
    return result;
}

int test_circuit(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty);
    num_failed += RUN_TEST(test_circuit_with_quantum_reg);
    num_failed += RUN_TEST(test_circuit_with_classical_reg);
    num_failed += RUN_TEST(test_circuit_copy);
    num_failed += RUN_TEST(test_circuit_copy_with_instructions);
    num_failed += RUN_TEST(test_circuit_copy_empty_like);
    num_failed += RUN_TEST(test_no_gate_1000_bits);
    num_failed += RUN_TEST(test_get_gate_counts);
    num_failed += RUN_TEST(test_get_gate_counts_bv_no_measure);
    num_failed += RUN_TEST(test_get_gate_counts_bv_measures);
    num_failed += RUN_TEST(test_get_gate_counts_bv_barrier_and_measures);
    num_failed += RUN_TEST(test_get_gate_counts_bv_resets_barrier_and_measures);
    num_failed += RUN_TEST(test_gate_num_qubits);
    num_failed += RUN_TEST(test_gate_num_params);
    num_failed += RUN_TEST(test_delay_instruction);
    num_failed += RUN_TEST(test_unitary_gate);
    num_failed += RUN_TEST(test_not_unitary_gate);
    num_failed += RUN_TEST(test_unitary_gate_1q);
    num_failed += RUN_TEST(test_unitary_gate_3q);
    num_failed += RUN_TEST(test_circuit_to_dag);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
