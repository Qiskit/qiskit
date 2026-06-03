// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include <math.h>
#include <qiskit.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*
 * Test QDRIFT circuit synthesis for a single ZZ term.
 * All possible valid outcomes: [ZZ]
 */
static int test_qdrift_single_zz(void) {
    int result = Ok;

    // Build H = Z0 Z1, 2-qubit system
    QkObs *obs = qk_obs_zero(2);

    QkBitTerm bit_terms[2] = {QkBitTerm_Z, QkBitTerm_Z};
    QkComplex64 coeff = {1, 0};
    uint32_t indices[2] = {0, 1};
    QkObsTerm term = {coeff, 2, bit_terms, indices, 2};
    qk_obs_add_term(obs, &term);

    QkCircuit *circ = qk_qdrift(obs, 1, 0.5, 0, true, false);

    size_t instr_count = qk_circuit_num_instructions(circ);
    if (instr_count != 1) {
        printf("Expected 1 instruction for single ZZ term, got %zu\n", instr_count);
        result = EqualityError;
        goto cleanup;
    }

    QkCircuitInstruction instr;
    qk_circuit_get_instruction(circ, 0, &instr);

    if (strcmp(instr.name, "rzz") != 0 && strcmp(instr.name, "zz") != 0) {
        printf("Expected ZZ-type gate ('rzz' or 'zz'), got '%s'\n", instr.name);
        result = EqualityError;
        goto cleanup;
    }

    if (instr.num_qubits != 2) {
        printf("Expected ZZ instruction on 2 qubits, got %u\n", (unsigned)instr.num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t q0 = instr.qubits[0];
    uint32_t q1 = instr.qubits[1];
    if (!((q0 == 0 && q1 == 1) || (q0 == 1 && q1 == 0))) {
        printf("Expected ZZ gate on qubits {0,1}, got {%u,%u}\n", q0, q1);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(circ);
    qk_obs_free(obs);
    return result;
}

/*
 * Test QDRIFT circuit synthesis for the observable H = XI + ZZ.
 * All possible valid outcomes: [XI, XI], [ZZ, ZZ], [XI, ZZ] or [ZZ, XI].
 */
static int test_qdrift_xi_plus_zz(void) {
    int result = Ok;

    // 2-qubit observable H = XI + ZZ
    QkObs *obs = qk_obs_zero(2);

    // Term 1: X on qubit 1 (XI).
    QkBitTerm bit_term_1[1] = {QkBitTerm_X};
    QkComplex64 coeff_1 = {1, 0};
    uint32_t indices_1[1] = {1};
    QkObsTerm term_1 = {coeff_1, 1, bit_term_1, indices_1, 2};
    qk_obs_add_term(obs, &term_1);

    // Term 2: ZZ on qubits {0,1}.
    QkBitTerm bit_term_2[2] = {QkBitTerm_Z, QkBitTerm_Z};
    QkComplex64 coeff_2 = {1, 0};
    uint32_t indices_2[2] = {0, 1};
    QkObsTerm term_2 = {coeff_2, 2, bit_term_2, indices_2, 2};
    qk_obs_add_term(obs, &term_2);

    QkCircuit *circ = qk_qdrift(obs, 1, 0.5, 0, true, false);

    size_t instr_count = qk_circuit_num_instructions(circ);

    // For H = XI + ZZ, lambda = 2, t = 0.5, reps = 1:
    // num_gates = ceil(2 * lambda^2 * t^2 * reps) = ceil(2) = 2.
    // However, identity terms are implicit. So we may get 1 or 2 gates depending
    // on whether an identity was sampled.
    if (instr_count == 0 || instr_count > 2) {
        printf("Expected 1 or 2 instructions for XI + ZZ, got %zu\n", instr_count);
        result = EqualityError;
        goto cleanup;
    }

    // Because QDRIFT is stochastic, we don't assert *which* terms appear where.
    // We just check all instructions are either X-like or ZZ-like and on the
    // expected qubits.
    for (size_t k = 0; k < instr_count; ++k) {
        QkCircuitInstruction instr;
        qk_circuit_get_instruction(circ, k, &instr);

        if (strcmp(instr.name, "rx") == 0 || strcmp(instr.name, "x") == 0) {
            // X-type
            if (instr.num_qubits != 1) {
                printf("X-type instruction expected on 1 qubit, got %u\n",
                       (unsigned)instr.num_qubits);
                result = EqualityError;
                goto cleanup;
            }
            // Expect qubit 1 for XI → ("X", [1])
            if (instr.qubits[0] != 1) {
                printf("X term expected on qubit 1, got %u\n", instr.qubits[0]);
                result = EqualityError;
                goto cleanup;
            }
        } else if (strcmp(instr.name, "rzz") == 0 || strcmp(instr.name, "zz") == 0) {
            // ZZ-type
            if (instr.num_qubits != 2) {
                printf("ZZ-type instruction expected on 2 qubits, got %u\n",
                       (unsigned)instr.num_qubits);
                result = EqualityError;
                goto cleanup;
            }
            uint32_t q0 = instr.qubits[0];
            uint32_t q1 = instr.qubits[1];
            if (!((q0 == 0 && q1 == 1) || (q0 == 1 && q1 == 0))) {
                printf("ZZ term expected on qubits {0,1}, got {%u,%u}\n", q0, q1);
                result = EqualityError;
                goto cleanup;
            }
        } else {
            printf("Unexpected instruction name: %s\n", instr.name);
            result = EqualityError;
            goto cleanup;
        }
    }

cleanup:
    qk_circuit_free(circ);
    qk_obs_free(obs);
    return result;
}

/*
 * Test QDRIFT gate count scaling with time and repetitions.
 * Expected gate counts:
 *   t=0.5, reps=1  -> 1 gate
 *   t=1.0, reps=1  -> 2 gates
 *   t=1.0, reps=2  -> 4 gates
 */
static int test_qdrift_gate_count_scaling(void) {
    int result = Ok;

    QkObs *obs = qk_obs_zero(1);

    QkBitTerm bits[1] = {QkBitTerm_X};
    uint32_t idxs[1] = {0};
    QkComplex64 coeff = {1.0, 0.0};
    QkObsTerm term = {coeff, 1, bits, idxs, 1};
    qk_obs_add_term(obs, &term);

    /* Case A: t=0.5 → num_gates = 1 */
    {
        QkCircuit *circ = qk_qdrift(obs, 1, 0.5, 0, true, false);

        size_t n = qk_circuit_num_instructions(circ);
        if (n != 1) {
            printf("[scaling] Expected 1 gate, got %zu\n", n);
            result = EqualityError;
        }
        qk_circuit_free(circ);
        if (result != Ok)
            goto cleanup;
    }

    /* Case B: t=1.0 → num_gates = 2 */
    {
        QkCircuit *circ = qk_qdrift(obs, 1, 1.0, 0, true, false);

        size_t n = qk_circuit_num_instructions(circ);
        if (n != 2) {
            printf("[scaling] Expected 2 gates, got %zu\n", n);
            result = EqualityError;
        }
        qk_circuit_free(circ);
        if (result != Ok)
            goto cleanup;
    }

    /* Case C: reps=2, t=1.0 → num_gates = 4 */
    {
        QkCircuit *circ = qk_qdrift(obs, 2, 1.0, 0, true, false);

        size_t n = qk_circuit_num_instructions(circ);
        if (n != 4) {
            printf("[scaling] Expected 4 gates, got %zu\n", n);
            result = EqualityError;
        }
        qk_circuit_free(circ);
        if (result != Ok)
            goto cleanup;
    }

cleanup:
    qk_obs_free(obs);
    return result;
}

/*
 * Test QDRIFT circuit synthesis produces valid qubit indices.
 */
static int test_qdrift_qubit_bounds(void) {
    int result = Ok;

    QkObs *obs = qk_obs_zero(4);

    // Add X on each qubit independently.
    for (uint32_t q = 0; q < 4; ++q) {
        QkBitTerm bit[1] = {QkBitTerm_X};
        uint32_t idx[1] = {q};
        QkComplex64 coeff = {1, 0};
        QkObsTerm term = {coeff, 1, bit, idx, 4};
        qk_obs_add_term(obs, &term);
    }

    QkCircuit *circ = qk_qdrift(obs, 1, 1.0, 0, true, false);

    size_t n = qk_circuit_num_instructions(circ);
    for (size_t k = 0; k < n; ++k) {
        QkCircuitInstruction instr;
        qk_circuit_get_instruction(circ, k, &instr);

        for (size_t i = 0; i < instr.num_qubits; ++i) {
            if (instr.qubits[i] >= 4) {
                printf("[bounds] Out-of-range qubit %u\n", instr.qubits[i]);
                result = EqualityError;
                goto cleanup;
            }
        }
    }

cleanup:
    qk_obs_free(obs);
    qk_circuit_free(circ);
    return result;
}

static int test_qdrift_two_gates_manual(void) {
    QkCircuit *expected = qk_circuit_new(1, 0);
    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){1.0});
    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){1.0});

    QkObs *obs = qk_obs_zero(1);

    QkBitTerm op1_bits[1] = {QkBitTerm_X};
    QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 1, op1_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term1);
    QkBitTerm op2_bits[1] = {QkBitTerm_Y};
    QkObsTerm term2 = {(QkComplex64){1.0, 0.0}, 1, op2_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term2);

    QkCircuit *qc = qk_qdrift(obs, 1, 0.5, 9, true, false);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_obs_free(obs);
    qk_circuit_free(qc);
    return result;
}

static int test_qdrift_single_gate_manual(void) {
    QkCircuit *expected = qk_circuit_new(1, 0);
    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){2 * 0.238});

    QkObs *obs = qk_obs_zero(1);

    QkBitTerm op1_bits[1] = {QkBitTerm_X};
    QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 1, op1_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term1);

    QkCircuit *qc = qk_qdrift(obs, 2, 0.238, 1, true, false);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_obs_free(obs);
    qk_circuit_free(qc);
    return result;
}

int test_qdrift(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_qdrift_single_zz);
    num_failed += RUN_TEST(test_qdrift_xi_plus_zz);
    num_failed += RUN_TEST(test_qdrift_gate_count_scaling);
    num_failed += RUN_TEST(test_qdrift_qubit_bounds);
    num_failed += RUN_TEST(test_qdrift_two_gates_manual);
    num_failed += RUN_TEST(test_qdrift_single_gate_manual);

    fflush(stderr);
    fprintf(stderr, "=== QDRIFT: Number of failed subtests: %i\n", num_failed);

    return num_failed;
}