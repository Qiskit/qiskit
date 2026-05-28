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
#include <string.h>

/**
 * Test setting the entanglement of the layers by entanglement strategy
 */
static int test_entanglement_by_strategy(void) {
    // int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;
    char *paramenter_prefix = "";

    QkGate rotation_blocks[1] = {QkGate_X};
    QkGate entanglement_blocks[1] = {QkGate_CCX};

    QkEntanglement *entanglement = qk_get_entanglement_with_strategy(
        5, 3, QkEntanglementStrategy_Linear, entanglement_blocks, 1);

    QkCircuit *qc = qk_n_local(rotation_blocks, 1, entanglement_blocks, 1, entanglement, 5, 3,
                               paramenter_prefix, false, false);

    

    qk_circuit_free(qc);

    return Ok;
}

int test_nlocal(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_entanglement_by_strategy);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (n_local): %i\n", num_failed);

    return num_failed;
}
