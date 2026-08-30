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
#include <qiskit.h>
#include <stdio.h>

static int test_round_trip(void) {
    const char *filename = "qiskit_c_api_test.qpy";
    QkCircuit *source = qk_circuit_new(2, 2);
    if (qk_circuit_gate(source, QkGate_H, (uint32_t[]){0}, NULL) != QkExitCode_Success ||
        qk_circuit_gate(source, QkGate_CX, (uint32_t[]){0, 1}, NULL) != QkExitCode_Success ||
        qk_circuit_measure(source, 0, 0) != QkExitCode_Success ||
        qk_circuit_measure(source, 1, 1) != QkExitCode_Success) {
        printf("Unexpected error encountered in QPY test_round_trip.");
        qk_circuit_free(source);
        return RuntimeError;
    }

    if (qk_qpy_dump_file(source, filename, 18) != QkExitCode_Success) {
        printf("Unexpected error encountered in QPY test_round_trip.");
        qk_circuit_free(source);
        return RuntimeError;
    }

    QkCircuit *loaded = NULL;
    QkExitCode load_result = qk_qpy_load_file(filename, &loaded);
    remove(filename);
    if (load_result != QkExitCode_Success || loaded == NULL) {
        printf("Unexpected error encountered in QPY test_round_trip.");
        qk_circuit_free(source);
        return RuntimeError;
    }

    int result = Ok;
    if (qk_circuit_num_qubits(loaded) != 2 || qk_circuit_num_clbits(loaded) != 2 ||
        qk_circuit_num_instructions(loaded) != 4) {
        result = EqualityError;
    }

cleanup:
    qk_circuit_free(loaded);
    qk_circuit_free(source);
    return result;
}

int test_qpy(void) { 
    int num_failed = 0;
    num_failed += RUN_TEST(test_round_trip);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
