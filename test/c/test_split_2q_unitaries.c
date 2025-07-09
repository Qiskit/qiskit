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
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int test_split_2q_unitaries_no_unitaries(void) {
    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    QkSplit2qUnitariesResult *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (qk_split_2q_unitaries_result_permutation_len(split_result) != 0) {
        result = EqualityError;
        printf("Permutation returned for a circuit that shouldn't split");
        goto result_cleanup;
    }
result_cleanup:
    qk_split_2q_unitaries_result_free(split_result);
cleanup:
    qk_circuit_free(qc);
    return result;
}

int test_split_2q_unitaries(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_split_2q_unitaries_no_unitaries);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
