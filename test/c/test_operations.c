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
#include <math.h>
#include <qiskit.h>
#include <stdio.h>
#include <string.h>

const char *FOO_NAME = "foo";
const char *foo_name(const void *gate);
uint32_t foo_num_qubits(const void *gate);
uint32_t foo_num_clbits(const void *gate);
uint32_t foo_num_params(const void *gate);
bool foo_directive(const void *gate);
bool foo_is_unitary(const void *gate);
struct foo_gate {
    uint32_t num_qubits;
};

const char *foo_name(const void *gate) {
    struct foo_gate *_self = (struct foo_gate *)gate;
    (void)_self;
    return FOO_NAME;
}
uint32_t foo_num_qubits(const void *gate) {
    struct foo_gate *self = (struct foo_gate *)gate;
    return self->num_qubits;
}
uint32_t foo_num_clbits(const void *gate) {
    struct foo_gate *_self = (struct foo_gate *)gate;
    (void)_self;
    return 0;
}
uint32_t foo_num_params(const void *gate) {
    struct foo_gate *_self = (struct foo_gate *)gate;
    (void)_self;
    return 0;
}
bool foo_directive(const void *gate) {
    struct foo_gate *_self = (struct foo_gate *)gate;
    (void)_self;
    return false;
}
bool foo_is_unitary(const void *gate) {
    struct foo_gate *_self = (struct foo_gate *)gate;
    (void)_self;
    return true;
}

QkCustomOpVTableEntry entries[7] = {
    {.slot = 0, .func = foo_name},       {.slot = 1, .func = foo_num_qubits},
    {.slot = 2, .func = foo_num_clbits}, {.slot = 3, .func = foo_num_params},
    {.slot = 4, .func = foo_directive},  {.slot = 5, .func = foo_is_unitary},
    {.slot = -1, .func = NULL},
};

QkCustomOp wrap_foo(struct foo_gate *gate) {
    QkCustomOp op = {
        .orig = gate,
        .v_table = qk_custom_op_new_vtable(entries),
    };

    return op;
}

static int test_custom_operation_in_circuit(void) {
    int res = Ok;
    struct foo_gate gate = {.num_qubits = 3};
    QkCustomOp op = wrap_foo(&gate);

    QkCircuit *circuit = qk_circuit_new(3, 2);
    uint32_t qubits[3] = {0, 1, 2};

    qk_circuit_add_custom_operation(circuit, op, qubits, NULL, NULL);

    return res;
}

int test_operations(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_custom_operation_in_circuit);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    return num_failed;
}