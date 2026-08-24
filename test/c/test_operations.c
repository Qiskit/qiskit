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

QkCustomOpVTableEntry foo_entries[7] = {
    {.slot = 0, .func = foo_name},       {.slot = 1, .func = foo_num_qubits},
    {.slot = 2, .func = foo_num_clbits}, {.slot = 3, .func = foo_num_params},
    {.slot = 4, .func = foo_directive},  {.slot = 5, .func = foo_is_unitary},
    {.slot = -1, .func = NULL},
};

struct fee_gate {};

const char *fee_name(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return FOO_NAME;
}
uint32_t fee_num_qubits(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return 2;
}
uint32_t fee_num_clbits(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return 0;
}
uint32_t fee_num_params(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return 1;
}
bool fee_directive(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return false;
}
bool fee_is_unitary(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return true;
}

QkCircuit *fee_definition(const void *gate, const QkParam **params) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    QkCircuit *circuit = qk_circuit_new(2, 0);
    uint32_t hgate_args[1] = {0};
    uint32_t cxgate_args[2] = {0, 1};
    uint32_t rzgate_args[2] = {1};
    qk_circuit_gate(circuit, QkGate_H, hgate_args, NULL);
    qk_circuit_gate(circuit, QkGate_CX, cxgate_args, NULL);

    double params_fixed[1] = {qk_param_as_real(params[0])};
    qk_circuit_gate(circuit, QkGate_RZ, rzgate_args, params_fixed);

    return circuit;
}

QkCustomOpVTableEntry fee_entries[8] = {
    {.slot = 0, .func = fee_name},       {.slot = 1, .func = fee_num_qubits},
    {.slot = 2, .func = fee_num_clbits}, {.slot = 3, .func = fee_num_params},
    {.slot = 4, .func = fee_directive},  {.slot = 5, .func = fee_is_unitary},
    {.slot = 8, .func = fee_definition}, {.slot = -1, .func = NULL},
};

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
    const QkCustomOpVTable *foo_vtable = qk_custom_operation_vtable_new(entries);

    if (foo_vtable == NULL) {
        printf("Retrieved a Null pointer instead of a Vtable pointer.");
        res = NullptrError;
        goto exit;
    }

    QkCustomOperation *test_3q = qk_custom_operation_new(&test_3q_op, foo_vtable);
    QkCustomOperation *test_2q_1c = qk_custom_operation_new(&test_2q_op, foo_vtable);

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
    kind = qk_circuit_instruction_kind(circuit, 1);

    if (kind != 8) {
        printf("Retrieved incorrect kind for '%s'. Expected %u, got %u.\n", inst.name, 8, kind);
        res = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_circuit_instruction_clear(&inst);
    qk_circuit_free(circuit);
exit:
    return res;
}

/// Test adding a custom operation in the cicuit;
static int test_custom_operation_in_dag(void) {
    int res = Ok;

    struct foo_gate test_1q_op = {
        .num_qubits = 1,
        .num_clbits = 0,
        .num_params = 1,
    };
    struct foo_gate test_3q_op = {
        .num_qubits = 3,
        .num_clbits = 1,
        .num_params = 0,
    };

    // Initialize Vtable
    const QkCustomOpVTable *foo_vtable = qk_custom_operation_vtable_new(entries);

    if (foo_vtable == NULL) {
        printf("Retrieved a Null pointer instead of a Vtable pointer.");
        res = NullptrError;
        goto exit;
    }

    QkCustomOperation *test_1q = qk_custom_operation_new(&test_1q_op, foo_vtable);
    QkCustomOperation *test_3q_1c = qk_custom_operation_new(&test_3q_op, foo_vtable);

    QkDag *circuit = qk_dag_new();
    QkQuantumRegister *qreg = qk_quantum_register_new(3, "qreg0");
    QkClassicalRegister *creg = qk_classical_register_new(1, "creg0");
    qk_dag_add_quantum_register(circuit, qreg);
    qk_dag_add_classical_register(circuit, creg);

    uint32_t qubits_1[1] = {0};
    uint32_t qubits_3[3] = {0, 1, 2};
    uint32_t clbits_1[1] = {0};
    QkParam *params[1] = {qk_param_from_double(3.14)};

    uint32_t ind1;
    uint32_t ind2;
    if (qk_dag_apply_custom_operation(circuit, test_1q, qubits_1, NULL, params, &ind1, false) !=
        QkExitCode_Success) {
        printf("Unable to add operation 1q parametric custom operation to dag.");
        res = RuntimeError;
        goto cleanup;
    };
    if (qk_dag_apply_custom_operation(circuit, test_3q_1c, qubits_3, clbits_1, NULL, &ind2,
                                      false) != QkExitCode_Success) {
        printf("Unable to add operation 3q custom operation to dag.");
        res = RuntimeError;
        goto cleanup;
    };

    // Retrieve operation from circuit
    QkCircuitInstruction inst;
    qk_dag_get_instruction(circuit, ind1, &inst);

    if (strcmp(inst.name, FOO_NAME)) {
        printf("Retrieved incorrect instruction name. Expected '%s', got '%s'.\n", FOO_NAME,
               inst.name);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_qubits != test_1q_op.num_qubits) {
        printf("Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n", inst.name,
               test_1q_op.num_qubits, inst.num_qubits);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_clbits != test_1q_op.num_clbits) {
        printf("Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n", inst.name,
               test_1q_op.num_clbits, inst.num_clbits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_params != test_1q_op.num_params) {
        printf("Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n", inst.name,
               test_1q_op.num_params, inst.num_params);
        res = EqualityError;
        goto inst_cleanup;
    }

    QkOperationKind kind = qk_dag_op_node_kind(circuit, ind2);

    if (kind != 8) {
        printf("Retrieved incorrect kind for '%s'. Expected %u, got %u.\n", inst.name, 8, kind);
        res = EqualityError;
        goto cleanup;
    }

    // Retrieve operation from circuit
    qk_dag_get_instruction(circuit, ind2, &inst);

    if (strcmp(inst.name, FOO_NAME)) {
        printf("Retrieved incorrect instruction name. Expected '%s', got '%s'.\n", FOO_NAME,
               inst.name);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_qubits != test_3q_op.num_qubits) {
        printf("Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n", inst.name,
               test_3q_op.num_qubits, inst.num_qubits);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_clbits != test_3q_op.num_clbits) {
        printf("Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n", inst.name,
               test_3q_op.num_clbits, inst.num_clbits);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_params != test_3q_op.num_params) {
        printf("Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n", inst.name,
               test_3q_op.num_params, inst.num_params);
        res = EqualityError;
        goto inst_cleanup;
    }

    kind = qk_dag_op_node_kind(circuit, ind2);

    if (kind != 8) {
        printf("Retrieved incorrect kind for '%s'. Expected %u, got %u.\n", inst.name, 8, kind);
        res = EqualityError;
        goto cleanup;
    }
inst_cleanup:
    qk_circuit_instruction_clear(&inst);
cleanup:
    qk_quantum_register_free(qreg);
    qk_classical_register_free(creg);
    qk_dag_free(circuit);
exit:
    return res;
}

static int test_custom_operation_query(void) {
    int res = Ok;

    struct foo_gate test_3q_op = {
        .num_qubits = 3,
        .num_clbits = 0,
        .num_params = 0,
    };
    struct fee_gate test_2q_op;

    // Initialize Vtable
    const QkCustomOpVtable *foo_vtable = qk_custom_op_vtable_new(foo_entries);
    // Initialize Vtable
    const QkCustomOpVtable *fee_vtable = qk_custom_op_vtable_new(fee_entries);

    if (foo_vtable == NULL) {
        printf("Retrieved a Null pointer instead of a Vtable pointer for foo_gate.");
        res = NullptrError;
        goto exit;
    }
    if (fee_vtable == NULL) {
        printf("Retrieved a Null pointer instead of a Vtable pointer for fee_gate.");
        res = NullptrError;
        goto exit;
    }

    QkCustomOperation *test_3q = qk_custom_op_new(&test_3q_op, foo_vtable);
    QkCustomOperation *test_2q_1c = qk_custom_op_new(&test_2q_op, fee_vtable);

    QkCircuit *circuit = qk_circuit_new(3, 2);
    uint32_t qubits[3] = {0, 1, 2};
    uint32_t qubits_2[2] = {1, 2};
    uint32_t clbits_2[1] = {1};
    const QkParam *params[1] = {qk_param_from_double(3.14)};

    qk_circuit_add_custom_operation(circuit, test_3q, qubits, NULL, NULL);
    qk_circuit_add_custom_operation(circuit, test_2q_1c, qubits_2, clbits_2, NULL);

    void *gates[2] = {(void *)&test_3q_op, (void *)&test_2q_op};

    if (qk_circuit_instruction_kind(circuit, 0) != QkOperationKind_Unknown) {
        res = RuntimeError;
        goto exit;
    }
    const QkCustomOperation *op = qk_circuit_get_custom_operation(circuit, 0);
    void *gate = gates[0];

    const char *retrieved_name = qk_custom_operation_name(op);
    const char *orig_name = foo_name(gate);
    if (strcmp(retrieved_name, orig_name)) {
        printf("Retrieved incorrect instruction name. Expected '%s', got '%s'.\n", orig_name,
               retrieved_name);
        res = EqualityError;
        goto cleanup;
    }
    uint32_t retrieved_num_qubits = qk_custom_operation_num_qubits(op);
    uint32_t orig_num_qubits = foo_num_qubits(gate);
    if (retrieved_num_qubits != orig_num_qubits) {
        printf("Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n", retrieved_name,
               orig_num_qubits, retrieved_num_qubits);
        res = EqualityError;
        goto cleanup;
    }
    uint32_t retrieved_num_clbits = qk_custom_operation_num_clbits(op);
    uint32_t orig_num_clbits = foo_num_clbits(gate);
    if (retrieved_num_clbits != orig_num_clbits) {
        printf("Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n", retrieved_name,
               orig_num_clbits, retrieved_num_clbits);
        res = EqualityError;
        goto cleanup;
    }
    uint32_t retrieved_num_params = qk_custom_operation_num_params(op);
    uint32_t orig_num_params = foo_num_params(gate);
    if (retrieved_num_params != orig_num_params) {
        printf("Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n", retrieved_name,
               orig_num_params, retrieved_num_params);
        res = EqualityError;
        goto cleanup;
    }

    if (qk_custom_operation_is_unitary(op) != foo_is_unitary(gate)) {
        printf("Unexpected non-unitary instruction for '%s'.\n", retrieved_name);
        res = EqualityError;
        goto cleanup;
    }

    // No definition was made for this operation, therefore it should be NULL
    if (qk_custom_operation_definition(op, NULL) != NULL) {
        printf("Unexpected non-null definition for '%s'.\n", retrieved_name);
        res = EqualityError;
        goto cleanup;
    }

    if (qk_circuit_instruction_kind(circuit, 1) != QkOperationKind_Unknown) {
        res = RuntimeError;
        goto cleanup;
    }
    op = qk_circuit_get_custom_operation(circuit, 1);
    gate = &gates[1];

    const char *retrieved_name_1 = qk_custom_operation_name(op);
    const char *orig_name_1 = fee_name(gate);
    if (strcmp(retrieved_name_1, orig_name_1)) {
        printf("Retrieved incorrect instruction name. Expected '%s', got '%s'.\n", orig_name_1,
               retrieved_name_1);
        res = EqualityError;
        goto cleanup;
    }
    retrieved_num_qubits = qk_custom_operation_num_qubits(op);
    orig_num_qubits = fee_num_qubits(gate);
    if (retrieved_num_qubits != orig_num_qubits) {
        printf("Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n", retrieved_name_1,
               orig_num_qubits, retrieved_num_qubits);
        res = EqualityError;
        goto cleanup;
    }
    retrieved_num_clbits = qk_custom_operation_num_clbits(op);
    orig_num_clbits = fee_num_clbits(gate);
    if (retrieved_num_clbits != orig_num_clbits) {
        printf("Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n", retrieved_name_1,
               orig_num_clbits, retrieved_num_clbits);
        res = EqualityError;
        goto cleanup;
    }
    retrieved_num_params = qk_custom_operation_num_params(op);
    orig_num_params = fee_num_params(gate);
    if (retrieved_num_params != orig_num_params) {
        printf("Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n", retrieved_name_1,
               orig_num_params, retrieved_num_params);
        res = EqualityError;
        goto cleanup;
    }

    if (qk_custom_operation_is_unitary(op) != fee_is_unitary(gate)) {
        printf("Unexpected non-unitary instruction for '%s'.\n", retrieved_name_1);
        res = EqualityError;
        goto cleanup;
    }

    // No definition was made for this operation, therefore it should be NULL
    QkCircuit *retrieved_definition = qk_custom_operation_definition(op, (const QkParam **)params);
    QkCircuit *orig_definition = fee_definition(gate, (const QkParam **)params);
    char *retrieved_drawing = qk_circuit_draw(retrieved_definition, NULL);
    char *orig_drawing =
        qk_circuit_draw(qk_custom_operation_definition(op, (const QkParam **)params), NULL);
    if (strcmp(retrieved_drawing, orig_drawing) != 0) {
        printf("Definitions are not simlar for '%s'.\n", retrieved_name_1);
        printf("Expected.\n");
        print_circuit(retrieved_definition);
        printf("Got.\n");
        print_circuit(orig_definition);
        res = EqualityError;
        goto cleanup_definitions;
    }

cleanup_definitions:
    qk_circuit_free(retrieved_definition);
    qk_circuit_free(orig_definition);
cleanup:
    qk_circuit_free(circuit);
exit:
    return res;
}

int test_operations(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_custom_operation_in_circuit);
    num_failed += RUN_TEST(test_custom_operation_in_dag);
    num_failed += RUN_TEST(test_custom_operation_query);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    return num_failed;
}