/**
* This code is part of Qiskit.
*
* (C) Copyright IBM 2025
*
* This code is licensed under the Apache License, Version 2.0. You may
* obtain a copy of this license in the LICENSE.txt file in the root directory
* of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
*
* Any modifications or derivative works of this code must retain this
* copyright notice, and modified files need to carry a notice indicating
* that they have been altered from the originals.
**/
#include "common.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <qiskit.h>

/*
* Test QDRIFT circuit synthesis for a single ZZ term.
* All possible valid outcomes: [ZZ]
*/
static int test_qdrift_single_zz(void) {
    int result = Ok;
    QkExitCode code;

    // Build H = Z0 Z1, 2-qubit system
    QkObs *obs = qk_obs_zero(2);

    QkBitTerm bit_terms[2] = {QkBitTerm_Z, QkBitTerm_Z};
    QkComplex64 coeff = {1, 0};
    uint32_t indices[2] = {0, 1};
    QkObsTerm term = {coeff, 2, bit_terms, indices, 2};
    code = qk_obs_add_term(obs, &term);

    QkCircuit *circ = NULL;
    code = qk_circuit_library_qdrift(obs, 1, 0.5, &circ);
    if (code != QkExitCode_Success || circ == NULL) {
        printf("qk_circuit_library_qdrift single ZZ failed with code %u\n", code);
        result = EqualityError;
        goto cleanup;
    }

    size_t instr_count = qk_circuit_num_instructions(circ);
    if (instr_count != 1) {
        printf("Expected 1 instruction for single ZZ term, got %lu\n",
               (unsigned long)instr_count);
        result = EqualityError;
        goto cleanup_circ;
    }

    QkCircuitInstruction instr;
    qk_circuit_get_instruction(circ, 0, &instr);

    if (strcmp(instr.name, "rzz") != 0 && strcmp(instr.name, "zz") != 0) {
        printf("Expected ZZ-type gate ('rzz' or 'zz'), got '%s'\n", instr.name);
        result = EqualityError;
        goto cleanup_circ;
    }

    if (instr.num_qubits != 2) {
        printf("Expected ZZ instruction on 2 qubits, got %u\n",
               (unsigned)instr.num_qubits);
        result = EqualityError;
        goto cleanup_circ;
    }

    uint32_t q0 = instr.qubits[0];
    uint32_t q1 = instr.qubits[1];
    if (!((q0 == 0 && q1 == 1) || (q0 == 1 && q1 == 0))) {
        printf("Expected ZZ gate on qubits {0,1}, got {%u,%u}\n", q0, q1);
        result = EqualityError;
        goto cleanup_circ;
    }

cleanup_circ:
    qk_circuit_free(circ);
cleanup:
    qk_obs_free(obs);
    return result;
}

/* 
* Test QDRIFT circuit synthesis for the observable H = XI + ZZ.
* All possible valid outcomes: [XI, XI], [ZZ, ZZ], [XI, ZZ] or [ZZ, XI]. 
*/
static int test_qdrift_xi_plus_zz(void) {
    int result = Ok;
    QkExitCode code;

    // 2-qubit observable H = XI + ZZ
    QkObs *obs = qk_obs_zero(2);

    // Term 1: X on qubit 1 (XI).
    QkBitTerm bit_term_1[1] = {QkBitTerm_X};
    QkComplex64 coeff_1 = {1, 0};
    uint32_t indices_1[1] = {1};
    QkObsTerm term_1 = {coeff_1, 1, bit_term_1, indices_1, 2};
    code = qk_obs_add_term(obs, &term_1);

    // Term 2: ZZ on qubits {0,1}.
    QkBitTerm bit_term_2[2] = {QkBitTerm_Z, QkBitTerm_Z};
    QkComplex64 coeff_2 = {1, 0};
    uint32_t indices_2[2] = {0, 1};
    QkObsTerm term_2 = {coeff_2, 2, bit_term_2, indices_2, 2};
    code = qk_obs_add_term(obs, &term_2);

    QkCircuit *circ = NULL;
    code = qk_circuit_library_qdrift(obs, 1, 0.5, &circ);
    if (code != QkExitCode_Success || circ == NULL) {
        printf("qk_circuit_library_qdrift XI+ZZ failed with code %u\n", code);
        result = EqualityError;
        goto cleanup;
    }

    size_t instr_count = qk_circuit_num_instructions(circ);

    // For H = XI + ZZ, lambda = 2, t = 0.5, reps = 1:
    // num_gates = ceil(2 * lambda^2 * t^2 * reps) = ceil(2) = 2.
    if (instr_count != 2) {
        printf("Expected 2 instructions for XI+ZZ, got %lu\n",
               (unsigned long)instr_count);
        result = EqualityError;
        goto cleanup_circ;
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
                goto cleanup_circ;
            }
            // Expect qubit 1 for XI → ("X", [1])
            if (instr.qubits[0] != 1) {
                printf("X term expected on qubit 1, got %u\n", instr.qubits[0]);
                result = EqualityError;
                goto cleanup_circ;
            }
        } else if (strcmp(instr.name, "rzz") == 0 || strcmp(instr.name, "zz") == 0) {
            // ZZ-type
            if (instr.num_qubits != 2) {
                printf("ZZ-type instruction expected on 2 qubits, got %u\n",
                       (unsigned)instr.num_qubits);
                result = EqualityError;
                goto cleanup_circ;
            }
            uint32_t q0 = instr.qubits[0];
            uint32_t q1 = instr.qubits[1];
            if (!((q0 == 0 && q1 == 1) || (q0 == 1 && q1 == 0))) {
                printf("ZZ term expected on qubits {0,1}, got {%u,%u}\n", q0, q1);
                result = EqualityError;
                goto cleanup_circ;
            }
        } else {
            printf("Unexpected instruction name: %s\n", instr.name);
            result = EqualityError;
            goto cleanup_circ;
        }
    }

cleanup_circ:
    qk_circuit_free(circ);
cleanup:
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
    QkExitCode code;

    QkObs *obs = qk_obs_zero(1);

    QkBitTerm bits[1] = {QkBitTerm_X};
    uint32_t idxs[1] = {0};
    QkComplex64 coeff = {1, 0};
    QkObsTerm term = {coeff, 1, bits, idxs, 1};
    qk_obs_add_term(obs, &term);

    /* Case A: t=0.5 → num_gates = 1 */
    {
        QkCircuit *circ = NULL;
        code = qk_circuit_library_qdrift(obs, 1, 0.5, &circ);
        if (code != QkExitCode_Success) return EqualityError;

        size_t n = qk_circuit_num_instructions(circ);
        if (n != 1) {
            printf("[scaling] Expected 1 gate, got %lu\n", (unsigned long)n);
            result = EqualityError;
        }
        qk_circuit_free(circ);
        if (result != Ok) goto cleanup;
    }

    /* Case B: t=1.0 → num_gates = 2 */
    {
        QkCircuit *circ = NULL;
        code = qk_circuit_library_qdrift(obs, 1, 1.0, &circ);
        if (code != QkExitCode_Success) return EqualityError;

        size_t n = qk_circuit_num_instructions(circ);
        if (n != 2) {
            printf("[scaling] Expected 2 gates, got %lu\n", (unsigned long)n);
            result = EqualityError;
        }
        qk_circuit_free(circ);
        if (result != Ok) goto cleanup;
    }

    /* Case C: reps=2, t=1.0 → num_gates = 4 */
    {
        QkCircuit *circ = NULL;
        code = qk_circuit_library_qdrift(obs, 2, 1.0, &circ);
        if (code != QkExitCode_Success) return EqualityError;

        size_t n = qk_circuit_num_instructions(circ);
        if (n != 4) {
            printf("[scaling] Expected 4 gates, got %lu\n", (unsigned long)n);
            result = EqualityError;
        }
        qk_circuit_free(circ);
        if (result != Ok) goto cleanup;
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
    QkExitCode code;

    QkObs *obs = qk_obs_zero(4);

    // Add X on each qubit independently.
    for (uint32_t q = 0; q < 4; ++q) {
        QkBitTerm bit[1] = {QkBitTerm_X};
        uint32_t idx[1] = {q};
        QkComplex64 coeff = {1, 0};
        QkObsTerm term = {coeff, 1, bit, idx, 1};
        code = qk_obs_add_term(obs, &term);
        if (code != QkExitCode_Success) {
            printf("[bounds] Failed add term at q=%u\n", q);
            result = EqualityError;
            goto cleanup_obs;
        }
    }

    QkCircuit *circ = NULL;
    code = qk_circuit_library_qdrift(obs, 1, 1.0, &circ);
    if (code != QkExitCode_Success || !circ) {
        printf("[bounds] QDRIFT failed\n");
        result = EqualityError;
        goto cleanup_obs;
    }

    size_t n = qk_circuit_num_instructions(circ);
    for (size_t k = 0; k < n; ++k) {
        QkCircuitInstruction instr;
        qk_circuit_get_instruction(circ, k, &instr);

        for (size_t i = 0; i < instr.num_qubits; ++i) {
            if (instr.qubits[i] >= 4) {
                printf("[bounds] Out-of-range qubit %u\n", instr.qubits[i]);
                result = EqualityError;
                goto cleanup_circ;
            }
        }
    }

cleanup_circ:
    qk_circuit_free(circ);
cleanup_obs:
    qk_obs_free(obs);
    return result;
}

/*
* Test QDRIFT rejects invalid repetition counts.
*/
static int test_qdrift_invalid_reps(void) {
    int result = Ok;
    QkExitCode code;

    QkObs *obs = qk_obs_zero(1);

    QkBitTerm bit_term[1] = {QkBitTerm_X};
    QkComplex64 coeff = {1, 0};
    uint32_t indices[1] = {0};
    QkObsTerm term = {coeff, 1, bit_term, indices, 1};

    code = qk_obs_add_term(obs, &term);

    QkCircuit *circ = NULL;
    code = qk_circuit_library_qdrift(obs, 0, 0.5, &circ);
    if (code == QkExitCode_Success) {
        printf("qk_circuit_library_qdrift unexpectedly succeeded with reps=0\n");
        result = EqualityError;
        // still free any circuit if allocated
        if (circ != NULL) {
            qk_circuit_free(circ);
        }
        goto cleanup;
    }

cleanup:
    qk_obs_free(obs);
    return result;
}

/*
* Test QDRIFT circuit synthesis for a non-Pauli observable, in this case the 
* projector onto the +1 eigenstate of X: |+><+| = (I + X)/2.
* and the projector onto the +1 eigenstate of Y: |R><R| = (I + Y)/2.
* All possible valid outcomes: [X, X], [X, Y], [Y, X], [Y, Y], [I, I], [I, X], [X, I], [I, Y], [Y, I]
* Paulis are chosen with weights I = 0.5, X = 0.25, Y = 0.25 since QDrift samples terms with probability 
* proportional to their coefficient magnitudes.
*/
static int test_qdrift_non_pauli_as_pauli(void) {
    int result = Ok;
    QkExitCode code;

    // 1-qubit observable: H = |+><+| = (I + X)/2
    QkObs *obs = qk_obs_zero(1);
    if (!obs) {
        printf("[plus] qk_obs_zero(1) failed\n");
        return EqualityError;
    }

    // Non-Pauli operator: Projector onto + eigenstate of X.
    // QkBitTerm_Plus is the projector.
    QkBitTerm bits[1] = { QkBitTerm_Plus };
    uint32_t idxs[1] = { 0 };
    QkComplex64 coeff = {1, 0};
    QkObsTerm term = {coeff, 1, bits, idxs, 1};

    QkBitTerm bits2[1] = { QkBitTerm_Right };
    uint32_t idxs2[1] = { 0 };
    QkObsTerm term2 = {coeff, 1, bits2, idxs2, 1};

    qk_obs_add_term(obs, &term);
    qk_obs_add_term(obs, &term2);

    // Run QDRIFT for time t=1.0
    QkCircuit *circ = NULL;
    code = qk_circuit_library_qdrift(obs, 1, 1.0, &circ);
    if (code != QkExitCode_Success || !circ) {
        printf("[plus] QDRIFT failed: %u\n", code);
        result = EqualityError;
        goto cleanup_obs;
    }

    size_t instr_count = qk_circuit_num_instructions(circ);
    printf("[plus] QDRIFT produced %lu instructions\n", (unsigned long)instr_count);

    for (size_t k = 0; k < instr_count; ++k) {
        QkCircuitInstruction instr;
        qk_circuit_get_instruction(circ, k, &instr);

        // Single-qubit only
        if (instr.num_qubits != 1) {
            printf("[plus] Instruction %lu has %u qubits (expected 1)\n",
                   (unsigned long)k, (unsigned)instr.num_qubits);
            result = EqualityError;
            goto cleanup_circ;
        }

        // Must be qubit 0
        if (instr.qubits[0] != 0) {
            printf("[plus] Instruction %lu acts on qubit %u (expected 0)\n",
                   (unsigned long)k, instr.qubits[0]);
            result = EqualityError;
            goto cleanup_circ;
        }

        // Only allowed gate: rx, x, ry, y
        if (strcmp(instr.name, "rx") != 0 && strcmp(instr.name, "x") != 0 &&
            strcmp(instr.name, "ry") != 0 && strcmp(instr.name, "y") != 0) {
            printf("[plus] Unexpected gate '%s' (expected 'rx' or 'x')\n",
                   instr.name);
            result = EqualityError;
            goto cleanup_circ;
        }
    }

cleanup_circ:
    qk_circuit_free(circ);
cleanup_obs:
    qk_obs_free(obs);
    return result;
}

static int test_qdrift_global_phase(void) {
    int result = Ok;
    QkExitCode code;

    // Build H = 1 * I on 1 qubit.
    QkObs *obs = qk_obs_zero(1);
    if (!obs) {
        printf("[global_phase] qk_obs_zero(1) failed\n");
        return EqualityError;
    }

    // Identity term: no bits, no indices.
    QkComplex64 coeff = {1, 0};
    // Dummy arrays of length 1, never used because len = 0, but non-null.
    QkBitTerm dummy_bits[1] = { QkBitTerm_X };   // value doesn’t matter
    uint32_t  dummy_idxs[1] = { 0 };             // value doesn’t matter

    QkObsTerm term_I = {
        coeff,
        1,            // num_qubits in the observable
        dummy_bits,   // non-null pointer
        dummy_idxs,   // non-null pointer
        0             // len = 0  → identity term
    };

    code = qk_obs_add_term(obs, &term_I);
    if (code != QkExitCode_Success) {
        printf("[global_phase] qk_obs_add_term(I) failed: %u\n", code);
        result = EqualityError;
        goto cleanup_obs;
    }

    // Evolve for time t = 1.0, reps = 1.
    double time = 1.0;
    QkCircuit *circ = NULL;
    code = qk_circuit_library_qdrift(obs, 1, time, &circ);
    if (code != QkExitCode_Success || !circ) {
        printf("[global_phase] qk_circuit_library_qdrift failed: %u\n", code);
        result = EqualityError;
        goto cleanup_obs;
    }

    // Expect no nontrivial instructions for H = I (pure global phase).
    size_t instr_count = qk_circuit_num_instructions(circ);
    if (instr_count != 0) {
        printf("[global_phase] Expected 0 instructions for identity Hamiltonian, got %lu\n",
               (unsigned long)instr_count);
        // Not necessarily a hard failure, but this likely indicates a bug.
        result = EqualityError;
        // continue anyway to inspect phase
    }

    // Global phase check 
    double phase = qk_circuit_get_global_phase(circ);

    // Mathematically, for H = I, global phase should be ~ -time.
    double expected = -time;

    if (fabs(phase - expected) > 1e-6) {
        printf("[global_phase] Expected global phase ~ %g, got %g\n",
               expected, phase);
        result = EqualityError;
    }

    qk_circuit_free(circ);
cleanup_obs:
    qk_obs_free(obs);
    return result;
}

int test_qdrift(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_qdrift_single_zz);
    num_failed += RUN_TEST(test_qdrift_xi_plus_zz);
    num_failed += RUN_TEST(test_qdrift_gate_count_scaling);
    num_failed += RUN_TEST(test_qdrift_qubit_bounds);
    num_failed += RUN_TEST(test_qdrift_invalid_reps);
    num_failed += RUN_TEST(test_qdrift_non_pauli_as_pauli);
    num_failed += RUN_TEST(test_qdrift_global_phase);

    fflush(stderr);
    fprintf(stderr, "=== QDRIFT: Number of failed subtests: %i\n", num_failed);

    return num_failed;
}