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

int test_remove_identity_equiv_removes_gates(void) {
    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;

    QkCircuit *qc = qk_circuit_new(1, 0);
    uint32_t qargs[1] = {
        0,
    };
    double params_zero[1] = {
        0.,
    };
    double params[1] = {
        1.23,
    };
    qk_circuit_gate(qc, QkGate_I, qargs, NULL);
    qk_circuit_gate(qc, QkGate_RZ, qargs, params_zero);
    qk_circuit_gate(qc, QkGate_RX, qargs, params);

    qk_transpiler_passes_remove_identity_equivalent_circuit(qc, target, 1.0);
    if (qk_circuit_num_instructions(qc) != 1) {
        result = EqualityError;
        printf("The gates weren't removed by this circuit");
    }

    qk_circuit_free(qc);
    qk_target_free(target);
    return result;
}

int test_remove_identity_equiv(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_remove_identity_equiv_removes_gates);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
