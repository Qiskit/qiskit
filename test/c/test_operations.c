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
    uint32_t num_clbits;
    uint32_t num_params;
};

const char *foo_name(const void *gate) {
    struct foo_gate *_self = (struct foo_gate *)gate;
    // Void pointer.
    (void)_self;
    return FOO_NAME;
}
uint32_t foo_num_qubits(const void *gate) {
    struct foo_gate *self = (struct foo_gate *)gate;
    return self->num_qubits;
}
uint32_t foo_num_clbits(const void *gate) {
    struct foo_gate *self = (struct foo_gate *)gate;
    return self->num_clbits;
}
uint32_t foo_num_params(const void *gate) {
    struct foo_gate *self = (struct foo_gate *)gate;
    return self->num_params;
}
bool foo_directive(const void *gate) {
    struct foo_gate *_self = (struct foo_gate *)gate;
    // Void pointer.
    (void)_self;
    return false;
}
bool foo_is_unitary(const void *gate) {
    struct foo_gate *_self = (struct foo_gate *)gate;
    // Void pointer.
    (void)_self;
    return true;
}
bool foo_eq(const void *gate, const void *other) {
    struct foo_gate *_self = (struct foo_gate *)gate;
    struct foo_gate *_other = (struct foo_gate *)other;

    return (_self->num_qubits == _other->num_qubits && _self->num_clbits == _other->num_clbits &&
            _self->num_params == _other->num_params);
}

QkCustomOpVTableEntry entries[7] = {
    {.slot = 0, .func = foo_name},       {.slot = 1, .func = foo_num_qubits},
    {.slot = 2, .func = foo_num_clbits}, {.slot = 3, .func = foo_num_params},
    {.slot = 4, .func = foo_directive},  {.slot = 5, .func = foo_is_unitary},
    {.slot = -1, .func = NULL},
};

static QkCustomOpVtable *foo_vtable = NULL;

/// Test adding a custom operation in the cicuit;
static int test_custom_operation_in_circuit(void) {
    int res = Ok;

    struct foo_gate test_3q_op = {
        .num_qubits = 3,
        .num_clbits = 0,
        .num_params = 1,
    };
    struct foo_gate test_2q_op = {
        .num_qubits = 2,
        .num_clbits = 1,
        .num_params = 0,
    };

    // Initialize Vtable
    foo_vtable = qk_custom_op_new_vtable(entries);

    if (foo_vtable == NULL) {
        printf("Retrieved a Null pointer instead of a Vtable pointer.");
        res = NullptrError;
        goto exit;
    }

    QkCustomOp test_3q = {
        .orig = &test_3q_op,
        .v_table = foo_vtable,
    };
    QkCustomOp test_2q_1c = {
        .orig = &test_2q_op,
        .v_table = foo_vtable,
    };

    QkCircuit *circuit = qk_circuit_new(3, 2);
    uint32_t qubits[3] = {0, 1, 2};
    uint32_t qubits_2[2] = {1, 2};
    uint32_t clbits_2[1] = {1};
    QkParam *params[1] = {qk_param_from_double(3.14)};

    qk_circuit_add_custom_operation(circuit, test_3q, qubits, NULL, params);
    qk_circuit_add_custom_operation(circuit, test_2q_1c, qubits_2, clbits_2, NULL);

    // Retrieve operation from circuit
    QkCircuitInstruction inst;
    qk_circuit_get_instruction(circuit, 0, &inst);

    if (strcmp(inst.name, FOO_NAME)) {
        printf("Retrieved incorrect instruction name. Expected '%s', got '%s'.\n", FOO_NAME,
               inst.name);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_qubits != test_3q_op.num_qubits) {
        printf("Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n", inst.name,
               test_3q_op.num_qubits, inst.num_qubits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_clbits != test_3q_op.num_clbits) {
        printf("Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n", inst.name,
               test_3q_op.num_clbits, inst.num_clbits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_params != test_3q_op.num_params) {
        printf("Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n", inst.name,
               test_3q_op.num_params, inst.num_params);
        res = EqualityError;
        goto cleanup;
    }

    // Retrieve operation from circuit
    qk_circuit_get_instruction(circuit, 1, &inst);

    if (strcmp(inst.name, FOO_NAME)) {
        printf("Retrieved incorrect instruction name. Expected '%s', got '%s'.\n", FOO_NAME,
               inst.name);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_qubits != test_2q_op.num_qubits) {
        printf("Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n", inst.name,
               test_2q_op.num_qubits, inst.num_qubits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_clbits != test_2q_op.num_clbits) {
        printf("Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n", inst.name,
               test_2q_op.num_clbits, inst.num_clbits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_params != test_2q_op.num_params) {
        printf("Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n", inst.name,
               test_2q_op.num_params, inst.num_params);
        res = EqualityError;
        goto cleanup;
    }

    QkOperationKind kind = qk_circuit_instruction_kind(circuit, 0);

    if (kind != 8) {
        printf("Retrieved incorrect kind for '%s'. Expected %u, got %u.\n", inst.name, 8, kind);
        res = EqualityError;
        goto cleanup;
    }

    // Query based on the information retrieved
cleanup:
    qk_circuit_instruction_clear(&inst);
    qk_circuit_free(circuit);
exit:
    return res;
}

int test_operations(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_custom_operation_in_circuit);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    return num_failed;
}