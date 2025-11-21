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

static inline bool instructions_equal(const QkCircuitInstruction *left,
                                      const QkCircuitInstruction *right) {
    if (strcmp(left->name, right->name) || left->num_qubits != right->num_qubits ||
        left->num_clbits != right->num_clbits || left->num_params != right->num_params) {
        return false;
    }
    for (uint32_t i = 0; i < left->num_qubits; i++) {
        if (left->qubits[i] != right->qubits[i]) {
            return false;
        }
    }
    for (uint32_t i = 0; i < left->num_clbits; i++) {
        if (left->clbits[i] != right->clbits[i]) {
            return false;
        }
    }
    for (uint32_t i = 0; i < left->num_params; i++) {
        if (left->params[i] != right->params[i]) {
            return false;
        }
    }
    return true;
}

static int test_empty(void) {
    QkDag *dag = qk_dag_new();
    uint32_t num_qubits = qk_dag_num_qubits(dag);
    uint32_t num_clbits = qk_dag_num_clbits(dag);
    qk_dag_free(dag);

    if (num_qubits != 0) {
        printf("The number of qubits %u is not 0\n", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %u is not 0\n", num_clbits);
        return EqualityError;
    }
    return Ok;
}

static int test_dag_with_quantum_reg(void) {
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_register");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t num_qubits = qk_dag_num_qubits(dag);
    uint32_t num_clbits = qk_dag_num_clbits(dag);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    if (num_qubits != 1024) {
        printf("The number of qubits %u is not 1024\n", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %u is not 0\n", num_clbits);
        return EqualityError;
    }
    return Ok;
}

static int test_dag_with_classical_reg(void) {
    QkDag *dag = qk_dag_new();
    QkClassicalRegister *cr = qk_classical_register_new(2048, "my_register");
    qk_dag_add_classical_register(dag, cr);
    uint32_t num_qubits = qk_dag_num_qubits(dag);
    uint32_t num_clbits = qk_dag_num_clbits(dag);
    qk_dag_free(dag);
    qk_classical_register_free(cr);
    if (num_qubits != 0) {
        printf("The number of qubits %u is not 0\n", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 2048) {
        printf("The number of clbits %u is not 2048\n", num_clbits);
        return EqualityError;
    }
    return Ok;
}

static int test_dag_apply_gate(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "my_register");
    qk_dag_add_quantum_register(dag, qr);

    size_t num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 0) {
        printf("The number of op nodes %zu is not 0\n", num_ops);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t h_bits[1] = {0};
    qk_dag_apply_gate(dag, QkGate_H, h_bits, NULL, false);

    num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 1) {
        printf("The number of op nodes %zu is not 1\n", num_ops);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t crx_bits[2] = {0, 1};
    double crx_params[1] = {0.12};
    uint32_t crx_node_idx = qk_dag_apply_gate(dag, QkGate_CRX, crx_bits, crx_params, true);

    num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 2) {
        printf("The number of op nodes %zu is not 2\n", num_ops);
        result = EqualityError;
        goto cleanup;
    }

    // Check the node's kind.
    if (qk_dag_op_node_kind(dag, crx_node_idx) != QkOperationKind_Gate) {
        printf("Expected gate kind\n");
        result = EqualityError;
        goto cleanup;
    }

    // Check the gate has the right number of params.
    uint32_t num_params = qk_dag_op_node_num_params(dag, crx_node_idx);
    if (num_params != 1) {
        printf("Expected num params 1 but got %u\n", num_params);
        result = EqualityError;
        goto cleanup;
    }

    // Make sure we can get the standard gate and params back from the node.
    double actual_crx_params[1];
    QkGate actual_gate = qk_dag_op_node_gate_op(dag, crx_node_idx, actual_crx_params);
    if (actual_gate != QkGate_CRX) {
        printf("Expected gate of type %u but got %u\n", QkGate_CRX, actual_gate);
        result = EqualityError;
        goto cleanup;
    }

    if (actual_crx_params[0] != crx_params[0]) {
        printf("Expected param %f but got %f\n", crx_params[0], actual_crx_params[0]);
        result = EqualityError;
    }
cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

static int test_op_node_bits_explicit(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "my_register");
    qk_dag_add_quantum_register(dag, qr);

    uint32_t h_bits[1] = {0};
    uint32_t h_node_idx = qk_dag_apply_gate(dag, QkGate_H, h_bits, NULL, false);

    uint32_t num_qubits = qk_dag_op_node_num_qubits(dag, h_node_idx);
    if (num_qubits != 1) {
        printf("The number of qubits %u is not 1\n", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    if (qk_dag_op_node_qubits(dag, h_node_idx)[0] != h_bits[0]) {
        printf("Expected a single qubit of value %u but got %u\n", h_bits[0],
               qk_dag_op_node_qubits(dag, h_node_idx)[0]);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t num_clbits = qk_dag_op_node_num_clbits(dag, h_node_idx);
    if (num_clbits != 0) {
        printf("The number of clbits %u is not 0\n", num_clbits);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t cx_bits[2] = {1, 0};
    uint32_t cx_node_idx = qk_dag_apply_gate(dag, QkGate_CX, cx_bits, NULL, true);

    num_qubits = qk_dag_op_node_num_qubits(dag, cx_node_idx);
    if (num_qubits != 2) {
        printf("The number of qubits %u is not 2\n", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    const uint32_t *actual_cx_bits = qk_dag_op_node_qubits(dag, cx_node_idx);
    for (uint32_t i = 0; i < num_qubits; i++) {
        if (actual_cx_bits[i] != cx_bits[i]) {
            printf("Expected a qubit of value %u in position %u but got %u\n", cx_bits[0], i,
                   qk_dag_op_node_qubits(dag, cx_node_idx)[0]);
            result = EqualityError;
            goto cleanup;
        }
    }

    num_clbits = qk_dag_op_node_num_clbits(dag, cx_node_idx);
    if (num_clbits != 0) {
        printf("The number of clbits %u is not 0\n", num_clbits);
        result = EqualityError;
    }
cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

static int test_dag_node_type(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1, "qr");
    QkClassicalRegister *cr = qk_classical_register_new(1, "cr");
    qk_dag_add_quantum_register(dag, qr);
    qk_dag_add_classical_register(dag, cr);

    // Get the wire node indices for the qubit.
    uint32_t qubit_in_idx = qk_dag_qubit_in_node(dag, 0);
    uint32_t qubit_out_idx = qk_dag_qubit_out_node(dag, 0);

    // Get the wire node indices for the clbit.
    uint32_t clbit_in_idx = qk_dag_clbit_in_node(dag, 0);
    uint32_t clbit_out_idx = qk_dag_clbit_out_node(dag, 0);

    // Add an operation node and save the node index.
    uint32_t h_bits[1] = {0};
    uint32_t h_node_idx = qk_dag_apply_gate(dag, QkGate_H, h_bits, NULL, false);

    QkDagNodeType node_type = qk_dag_node_type(dag, qubit_in_idx);
    if (node_type != QkDagNodeType_QubitIn) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_QubitIn, node_type);
        result = EqualityError;
        goto cleanup;
    }

    node_type = qk_dag_node_type(dag, qubit_out_idx);
    if (node_type != QkDagNodeType_QubitOut) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_QubitOut, node_type);
        result = EqualityError;
        goto cleanup;
    }

    node_type = qk_dag_node_type(dag, clbit_in_idx);
    if (node_type != QkDagNodeType_ClbitIn) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_ClbitIn, node_type);
        result = EqualityError;
        goto cleanup;
    }

    node_type = qk_dag_node_type(dag, clbit_out_idx);
    if (node_type != QkDagNodeType_ClbitOut) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_ClbitOut, node_type);
        result = EqualityError;
        goto cleanup;
    }

    node_type = qk_dag_node_type(dag, h_node_idx);
    if (node_type != QkDagNodeType_Operation) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_Operation, node_type);
        result = EqualityError;
    }

cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    qk_classical_register_free(cr);
    return result;
}

static int test_dag_endpoint_node_value(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    QkClassicalRegister *cr = qk_classical_register_new(2, "cr");
    qk_dag_add_quantum_register(dag, qr);
    qk_dag_add_classical_register(dag, cr);

    // For each bit (we have two qubits and two clbits!), check that the wire values
    // are correct.
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t qubit_in_idx = qk_dag_qubit_in_node(dag, i);
        uint32_t qubit_out_idx = qk_dag_qubit_out_node(dag, i);
        uint32_t clbit_in_idx = qk_dag_clbit_in_node(dag, i);
        uint32_t clbit_out_idx = qk_dag_clbit_out_node(dag, i);

        uint32_t actual = qk_dag_wire_node_value(dag, qubit_in_idx);
        if (actual != i) {
            printf("Expected wire endpoint qubit value to be %u but got %u\n", i, actual);
            result = EqualityError;
            goto cleanup;
        }
        actual = qk_dag_wire_node_value(dag, qubit_out_idx);
        if (actual != i) {
            printf("Expected wire endpoint qubit value to be %u but got %u\n", i, actual);
            result = EqualityError;
            goto cleanup;
        }
        actual = qk_dag_wire_node_value(dag, clbit_in_idx);
        if (actual != i) {
            printf("Expected wire endpoint clbit value to be %u but got %u\n", i, actual);
            result = EqualityError;
            goto cleanup;
        }
        actual = qk_dag_wire_node_value(dag, clbit_out_idx);
        if (actual != i) {
            printf("Expected wire endpoint clbit value to be %u but got %u\n", i, actual);
            result = EqualityError;
            goto cleanup;
        }
    }
cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    qk_classical_register_free(cr);
    return result;
}

static int test_dag_get_instruction(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    QkClassicalRegister *cr = qk_classical_register_new(2, "cr");
    qk_dag_add_quantum_register(dag, qr);
    qk_dag_add_classical_register(dag, cr);
    qk_quantum_register_free(qr);
    qk_classical_register_free(cr);

    uint32_t args[2] = {0, 1};
    uint32_t cx_node = qk_dag_apply_gate(dag, QkGate_CX, args, NULL, false);
    uint32_t h_node = qk_dag_apply_gate(dag, QkGate_H, args, NULL, true);
    // If the pointer is null, the number of qubits shouldn't be read.
    uint32_t final_barrier_node = qk_dag_apply_barrier(dag, NULL, 16000, false);
    uint32_t measure_0_node = qk_dag_apply_measure(dag, 0, 0, false);
    uint32_t measure_1_node = qk_dag_apply_measure(dag, 1, 1, false);
    uint32_t front_barrier_node = qk_dag_apply_barrier(dag, args, 2, true);
    uint32_t reset_0_node = qk_dag_apply_reset(dag, 0, true);

    uint32_t indices[] = {reset_0_node,       front_barrier_node, h_node,        cx_node,
                          final_barrier_node, measure_0_node,     measure_1_node};
    const char *desc[] = {"reset 0",       "front barrier", "h",        "cx",
                          "final barrier", "measure 0",     "measure 1"};
    QkCircuitInstruction expected[] = {
        {.name = "reset",
         .qubits = args,
         .num_qubits = 1,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "barrier",
         .qubits = args,
         .num_qubits = 2,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "h",
         .qubits = &args[0],
         .num_qubits = 1,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "cx",
         .qubits = args,
         .num_qubits = 2,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "barrier",
         .qubits = args,
         .num_qubits = 2,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "measure",
         .qubits = &args[0],
         .num_qubits = 1,
         .clbits = &args[0],
         .num_clbits = 1,
         .params = NULL,
         .num_params = 0},
        {.name = "measure",
         .qubits = &args[1],
         .num_qubits = 1,
         .clbits = &args[1],
         .num_clbits = 1,
         .params = NULL,
         .num_params = 0},
    };

    QkCircuitInstruction inst;
    for (size_t i = 0; i < sizeof(indices) / sizeof(*indices); i++) {
        qk_dag_get_instruction(dag, indices[i], &inst);
        if (!instructions_equal(&inst, &expected[i])) {
            printf("%s: mismatched instruction: %s\n", __func__, desc[i]);
            result = EqualityError;
            goto cleanup;
        }
        qk_circuit_instruction_clear(&inst);
    }
cleanup:
    qk_dag_free(dag);
    return result;
}

int test_dag(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty);
    num_failed += RUN_TEST(test_dag_with_quantum_reg);
    num_failed += RUN_TEST(test_dag_with_classical_reg);
    num_failed += RUN_TEST(test_dag_apply_gate);
    num_failed += RUN_TEST(test_dag_node_type);
    num_failed += RUN_TEST(test_dag_endpoint_node_value);
    num_failed += RUN_TEST(test_op_node_bits_explicit);
    num_failed += RUN_TEST(test_dag_get_instruction);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
