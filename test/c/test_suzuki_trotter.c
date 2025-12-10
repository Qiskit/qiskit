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
#include <math.h>

static int test_suzuki_trotter_4_order_no_reorder(void) {
    QkCircuit *expected = qk_circuit_new(1, 0);
    double time = 0.1;
    int reps = 1;
    double p_4 = 1 / (4 - pow(4, 1.0/3.0));
    
    for (size_t i = 0; i < 2; i++) {
        qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){p_4 * time});
        qk_circuit_gate(expected, QkGate_RY, (uint32_t[1]){0}, (double[1]){2 * p_4 * time});
        qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){p_4 * time});
    }

    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){(1 - 4 * p_4) * time});
    qk_circuit_gate(expected, QkGate_RY, (uint32_t[1]){0}, (double[1]){2 * (1 - 4 * p_4) * time});
    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){(1 - 4 * p_4) * time});

    for (size_t i = 0; i < 2; i++) {
        qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){p_4 * time});
        qk_circuit_gate(expected, QkGate_RY, (uint32_t[1]){0}, (double[1]){2 * p_4 * time});
        qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){p_4 * time});
    }

    QkObs *obs = qk_obs_zero(1);

    QkBitTerm op1_bits[1] = {QkBitTerm_X};
    QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 1, op1_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term1);

    QkBitTerm op2_bits[1] = {QkBitTerm_Y};
    QkObsTerm term2 = {(QkComplex64){1.0, 0.0}, 1, op2_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term2);

    QkCircuit *qc = qk_circuit_library_suzuki_trotter(obs, 4, reps, time, true, false);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_obs_free(obs);
    qk_circuit_free(qc);
    qk_circuit_free(expected);
    
    return result;
}

static int test_suzuki_trotter_2_order_reorder(void) {
    QkCircuit *expected = qk_circuit_new(4, 0);

    qk_circuit_gate(expected, QkGate_RZZ, (uint32_t[2]){0, 1}, (double[1]){0.1});
    qk_circuit_gate(expected, QkGate_RXX, (uint32_t[2]){2, 3}, (double[1]){0.1});
    qk_circuit_gate(expected, QkGate_RYY, (uint32_t[2]){1, 2}, (double[1]){0.2});
    qk_circuit_gate(expected, QkGate_RXX, (uint32_t[2]){2, 3}, (double[1]){0.1});
    qk_circuit_gate(expected, QkGate_RZZ, (uint32_t[2]){0, 1}, (double[1]){0.1});

    int num_qubits = 4;
    QkObs *obs = qk_obs_zero(num_qubits);

    QkBitTerm op1_bits[2] = {QkBitTerm_X, QkBitTerm_X};
    QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 2, op1_bits, (uint32_t[2]){2, 3}, num_qubits};
    qk_obs_add_term(obs, &term1);

    QkBitTerm op2_bits[2] = {QkBitTerm_Y, QkBitTerm_Y};
    QkObsTerm term2 = {(QkComplex64){1.0, 0.0}, 2, op2_bits, (uint32_t[2]){1, 2}, num_qubits};
    qk_obs_add_term(obs, &term2);

    QkBitTerm op3_bits[2] = {QkBitTerm_Z, QkBitTerm_Z};
    QkObsTerm term3 = {(QkComplex64){1.0, 0.0}, 2, op3_bits, (uint32_t[2]){0, 1}, num_qubits};
    qk_obs_add_term(obs, &term3);

    QkCircuit *qc = qk_circuit_library_suzuki_trotter(obs, 2, 1, 0.1, false, false);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_obs_free(obs);
    qk_circuit_free(qc);
    qk_circuit_free(expected);
    
    return result;
}

static int test_suzuki_trotter_with_barriers(void) {
    QkCircuit *expected = qk_circuit_new(1, 0);
    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){0.2});
    qk_circuit_barrier(expected, (uint32_t[1]){0}, 1);
    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){0.2});

    QkObs *obs = qk_obs_zero(1);

    QkBitTerm op1_bits[1] = {QkBitTerm_X};
    QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 1, op1_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term1);

    QkBitTerm op2_bits[1] = {QkBitTerm_X};
    QkObsTerm term2 = {(QkComplex64){1.0, 0.0}, 1, op2_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term2);

    QkCircuit *qc = qk_circuit_library_suzuki_trotter(obs, 1, 1, 0.1, true, true);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_obs_free(obs);
    qk_circuit_free(qc);
    qk_circuit_free(expected);
    
    return result;
}

int test_suzuki_trotter(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_suzuki_trotter_4_order_no_reorder);
    num_failed += RUN_TEST(test_suzuki_trotter_2_order_reorder);
    num_failed += RUN_TEST(test_suzuki_trotter_with_barriers);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (Suzuki-Trotter): %i\n", num_failed);

    return num_failed;
}