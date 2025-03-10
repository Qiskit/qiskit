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
    qk_circuit_free(qc);

    return (num_qubits != 0 || num_clbits != 0) ? EqualityError : Ok;
}

int test_no_gate_1000_bits() {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    qk_circuit_free(qc);

    return (num_qubits != 1000 || num_clbits != 1000) ? EqualityError : Ok;
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
    if (op_counts.len != 3) {
        return EqualityError;
    }
    int result = strcmp(op_counts.data[2].name, "x");
    if (result != 0) {
        return result;
    }
    if (op_counts.data[2].count != 1) {
        return EqualityError;
    }
    result = strcmp(op_counts.data[0].name, "h");
    if (result != 0) {
        return result;
    }
    if (op_counts.data[0].count != 1999) {
        return EqualityError;
    }
    result = strcmp(op_counts.data[1].name, "cx");
    if (result != 0) {
        return result;
    }
    if (op_counts.data[1].count != 500) {
        return EqualityError;
    }
    qk_circuit_free(qc);
    return 0;
}

int test_circuit() {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty);
    num_failed += RUN_TEST(test_no_gate_1000_bits);
    num_failed += RUN_TEST(test_get_gate_counts_bv_no_measure);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
