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

/**
 * Test removing a Z gate before a measure.
 */
int test_remove_z_gate(void) {
    int result = Ok;
    QkCircuit *qc = qk_circuit_new(1, 1);
    qk_circuit_gate(qc, QkGate_Z, (int[]){0}, NULL);
    qk_circuit_measure(qc, 0, 0);

    if (2 != qk_circuit_num_instructions(qc)) {
        printf("Circuit build failure");
        result = RuntimeError;
        goto cleanup;
    }

    qk_transpiler_pass_standalone_remove_diagonal_gates_before_measure(qc);

    if (1 != qk_circuit_num_instructions(qc)) {
        printf("Circuit should only have a single instruction");
        result = EqualityError;
        goto cleanup;
    }

    QkCircuitInstruction inst;
    qk_circuit_get_instruction(qc, 0, &inst);

    if (0 != strcmp("measure", inst.name)) {
        printf("Circuit should contain a single 'measure' instruction. Instead, it has one: '%s'.",
               inst.name);
        result = EqualityError;
        goto cleanup_inst;
    }

cleanup_inst:
    qk_circuit_instruction_clear(&inst);
cleanup:
    qk_circuit_free(qc);
    return result;
}

int test_remove_diagonal_gates_before_measure(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_remove_z_gate);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
