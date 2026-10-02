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
#include <string.h>

static const char *FOO_NAME = "foo";

struct foo_gate {
    uint32_t num_qubits;
    uint32_t num_clbits;
    uint32_t num_params;
};

static const char *foo_name(const struct foo_gate *gate) {
    (void)gate; // Unused.
    return FOO_NAME;
}
static uint32_t foo_num_qubits(const struct foo_gate *gate) { return gate->num_qubits; }
static uint32_t foo_num_clbits(const struct foo_gate *gate) { return gate->num_clbits; }
static uint32_t foo_num_params(const void *gate) {
    struct foo_gate *self = (struct foo_gate *)gate;
    return self->num_params;
}
static bool foo_eq(const struct foo_gate *gate, const struct foo_gate *other) {
    return (gate->num_qubits == other->num_qubits && gate->num_clbits == other->num_clbits &&
            gate->num_params == other->num_params);
}
static struct foo_gate *foo_clone(const struct foo_gate *gate) {
    struct foo_gate *out = malloc(sizeof(*out));
    memcpy(out, gate, sizeof(*out));
    return out;
}

static QkVtableEntry foo_entries[] = {
    {.slot = QkCustomOpSlot_Name, .ptr = foo_name},
    {.slot = QkCustomOpSlot_Eq, .ptr = foo_eq},
    {.slot = QkCustomOpSlot_NumQubits, .ptr = foo_num_qubits},
    {.slot = QkCustomOpSlot_NumClbits, .ptr = foo_num_clbits},
    {.slot = QkCustomOpSlot_NumParams, .ptr = foo_num_params},
    {.slot = QkCustomOpSlot_Clone, .ptr = foo_clone},
    {.slot = QkCustomOpSlot_Delete, .ptr = free},
    {.slot = -1, .ptr = NULL},
};

typedef struct {
} Incomplete;

static QkVtableEntry incomplete_slots[] = {
    {.slot = QkCustomOpSlot_Name, .ptr = foo_name},
    {.slot = -1, .ptr = NULL},
};

/// Test adding a custom operation in the cicuit;
static int test_custom_operation_in_circuit(void) {
    int res = Ok;

    struct foo_gate test_3q_op = {
        .num_qubits = 3,
        .num_clbits = 0,
        .num_params = 2,
    };
    struct foo_gate test_2q_op = {
        .num_qubits = 2,
        .num_clbits = 1,
        .num_params = 0,
    };

    // Initialize Vtable
    const QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(foo_entries);

    if (foo_vtable == NULL) {
        printf("Retrieved a Null pointer instead of a Vtable pointer.");
        res = NullptrError;
        goto exit;
    }

    QkCustomOperation *test_3q = qk_custom_operation_new(foo_clone(&test_3q_op), foo_vtable);
    QkCustomOperation *test_2q_1c = qk_custom_operation_new(foo_clone(&test_2q_op), foo_vtable);

    QkCircuit *circuit = qk_circuit_new(3, 2);
    uint32_t qubits[3] = {0, 1, 2};
    uint32_t qubits_2[2] = {1, 2};
    uint32_t clbits_2[1] = {1};
    QkParam *params[2] = {qk_param_from_double(3.14), qk_param_from_double(1.57)};

    qk_circuit_custom_operation(circuit, test_3q, qubits, NULL, params);
    qk_circuit_custom_operation(circuit, test_2q_1c, qubits_2, clbits_2, NULL);

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
    qk_circuit_instruction_clear(&inst);
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
    qk_param_free(params[0]);
    qk_param_free(params[1]);
    qk_circuit_free(circuit);
    qk_custom_operation_vtable_free(foo_vtable);
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
    const QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(foo_entries);

    if (foo_vtable == NULL) {
        printf("Retrieved a Null pointer instead of a Vtable pointer.");
        res = NullptrError;
        goto exit;
    }

    QkCustomOperation *test_1q = qk_custom_operation_new(foo_clone(&test_1q_op), foo_vtable);
    QkCustomOperation *test_3q_1c = qk_custom_operation_new(foo_clone(&test_3q_op), foo_vtable);

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
    qk_circuit_instruction_clear(&inst);
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
    qk_param_free(params[0]);
    qk_dag_free(circuit);
    qk_custom_operation_vtable_free(foo_vtable);
exit:
    return res;
}

/**
 * Test passing an incomplete vtable returns NULL and exits gracefully.
 */
static int test_incomplete_vtable(void) {
    const QkCustomOpVtable *vtable = qk_custom_operation_vtable_new(incomplete_slots);
    int result = Ok;
    if (vtable != NULL) {
        // qk_custom_operation_vtable_free(vtable);
        result = EqualityError;
    }
    return result;
}

int test_operations(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_custom_operation_in_circuit);
    num_failed += RUN_TEST(test_custom_operation_in_dag);
    num_failed += RUN_TEST(test_incomplete_vtable);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    return num_failed;
}
