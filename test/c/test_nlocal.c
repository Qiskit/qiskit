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
#include <string.h>

static int test_nlocal_circuit_creation(void) {
    size_t num_qubits = 5;
    int reps = 2;

    QkCircuit *expected = qk_circuit_new(num_qubits, 0);

    // Add first rotation layer for X Gate
    for (size_t i = 0; i < num_qubits; i++) {
        qk_circuit_gate(expected, QkGate_X, (uint32_t[1]){i}, NULL);
    }

    qk_circuit_barrier(expected, (uint32_t[5]){0, 1, 2, 3, 4}, 5);

    // Add first entanglement layer for CCX Gate
    for (size_t i = 0; i < 3; i++) {
        qk_circuit_gate(expected, QkGate_CCX, (uint32_t[3]){i, i + 1, i + 2}, NULL);
    }

    qk_circuit_barrier(expected, (uint32_t[5]){0, 1, 2, 3, 4}, 5);

    // Add second rotation layer for X Gate
    for (size_t i = 0; i < num_qubits; i++) {
        qk_circuit_gate(expected, QkGate_X, (uint32_t[1]){i}, NULL);
    }

    qk_circuit_barrier(expected, (uint32_t[5]){0, 1, 2, 3, 4}, 5);

    // Add second entanglement layer for CCX Gate
    for (size_t i = 0; i < 3; i++) {
        qk_circuit_gate(expected, QkGate_CCX, (uint32_t[3]){i, i + 1, i + 2}, NULL);
    }
    qk_circuit_barrier(expected, (uint32_t[5]){0, 1, 2, 3, 4}, 5);

    // Add final rotation layer for X Gate
    for (size_t i = 0; i < num_qubits; i++) {
        qk_circuit_gate(expected, QkGate_X, (uint32_t[1]){i}, NULL);
    }

    QkGate entanglement_blocks[1] = {QkGate_CCX};
    QkGate rotation_blocks[1] = {QkGate_X};

    QkEntanglement *entanglement = qk_get_entanglement_with_strategy(
        num_qubits, reps, QkEntanglementStrategy_Linear, entanglement_blocks, 1);

    QkCircuit *qc = qk_n_local(rotation_blocks, 1, entanglement_blocks, 1, entanglement, num_qubits,
                               reps, NULL, true, false);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_circuit_free(qc);
    qk_circuit_free(expected);

    return result;
}

static int test_parameter_prefix(void) {
    int result = Ok;
    size_t num_qubits = 2;
    int reps = 1;

    QkGate rotation_blocks[1] = {QkGate_H};
    QkGate entanglement_blocks[1] = {QkGate_CRX};
    QkEntanglement *entanglement = qk_get_entanglement_with_strategy(
        num_qubits, reps, QkEntanglementStrategy_Linear, entanglement_blocks, 1);

    char *prefix = "myprefix";

    QkCircuit *qc = qk_n_local(rotation_blocks, 1, entanglement_blocks, 1, entanglement, num_qubits,
                               reps, prefix, false, true);

    QkCircuitInstruction inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(qc); i++) {
        qk_circuit_get_instruction(qc, i, &inst);
        if (strstr(inst.name, "crx") != NULL) {
            char *parameter_str = qk_param_str(inst.params[0]);
            if (strcmp(parameter_str, "myprefix[0]") != 0) {
                result = EqualityError;
                qk_str_free(parameter_str);
                break;
            }
            qk_str_free(parameter_str);
        }
    }

    qk_circuit_instruction_clear(&inst);
    qk_circuit_free(qc);

    return result;
}

/**
 * Test setting the entanglement of the layers by entanglement strategy
 */
static int test_entanglement_by_strategy(void) {
    int result = Ok;
    int reps = 3;

    QkGate entanglement_blocks[1] = {QkGate_CCX};

    uint32_t expected_connections_full[10][3] = {
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4},
        {0, 3, 4}, {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
    };
    uint32_t expected_connections_linear[3][3] = {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}};
    uint32_t expected_connections_reverse_linear[3][3] = {{2, 3, 4}, {1, 2, 3}, {0, 1, 2}};
    uint32_t expected_connections_circular[5][3] = {
        {3, 4, 0}, {4, 0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3, 4}};
    uint32_t expected_connections_sca[5][3] = {{0}};

    for (size_t strategy = 0; strategy < 5; strategy++) {

        QkEntanglement *entanglement =
            qk_get_entanglement_with_strategy(5, reps, strategy, entanglement_blocks, 1);

        if (qk_get_entanglement_layers_quantity(entanglement) != (size_t)reps) {
            result = EqualityError;
        }

        if (qk_get_entanglement_layer_blocks_quantity(entanglement, 0) != (size_t)1) {
            result = EqualityError;
        }

        bool break_loop = false;
        for (size_t i = 0; i < (size_t)reps; i++) {
            if (break_loop)
                break;

            uint32_t (*expected_connections)[3] = NULL;
            uint32_t expected_connections_length;
            switch (strategy) {
            case QkEntanglementStrategy_Full:
                expected_connections = expected_connections_full;
                expected_connections_length = 10;
                break;
            case QkEntanglementStrategy_Linear:
                expected_connections = expected_connections_linear;
                expected_connections_length = 3;
                break;
            case QkEntanglementStrategy_ReverseLinear:
                expected_connections = expected_connections_reverse_linear;
                expected_connections_length = 3;
                break;
            case QkEntanglementStrategy_Sca:
                expected_connections_length = 5;
                for (size_t k = 0; k < expected_connections_length; k++) {
                    int offset = ((int)k) - ((int)i);
                    int ind = offset >= 0 ? offset : expected_connections_length + offset;
                    for (size_t m = 0; m < 3; m++) {
                        expected_connections_sca[k][m] =
                            expected_connections_circular[ind][i % 2 == 1 ? 2 - m : m];
                    }
                }
                expected_connections = expected_connections_sca;
                break;
            case QkEntanglementStrategy_Circular:
                expected_connections = expected_connections_circular;
                expected_connections_length = 5;
                break;
            }

            if (qk_get_entanglement_qubit_connections_quantity(entanglement, i, 0) !=
                expected_connections_length) {
                result = EqualityError;
                break_loop = true;
                break;
            }

            for (size_t j = 0; j < expected_connections_length; j++) {
                uint32_t *connections = *(expected_connections + j);
                QkQubitConnection *expected_connection = qk_qubit_connection_new(3, connections);
                QkQubitConnection *connection =
                    qk_get_entanglement_qubit_connections(entanglement, i, 0, j);

                if (!qk_qubit_connection_equal(connection, expected_connection)) {
                    result = EqualityError;
                    break_loop = true;
                    break;
                }
            }
        }
    }
    return result;
}

static int test_pairwise_entanglement_strategy(void) {
    int result = Ok;
    int reps = 1;

    QkGate entanglement_blocks[1] = {QkGate_CX};
    QkEntanglement *entanglement = qk_get_entanglement_with_strategy(
        5, reps, QkEntanglementStrategy_Pairwise, entanglement_blocks, 1);

    if (qk_get_entanglement_layers_quantity(entanglement) != (size_t)reps) {
        result = EqualityError;
    }

    if (qk_get_entanglement_layer_blocks_quantity(entanglement, 0) != (size_t)1) {
        result = EqualityError;
    }

    if (qk_get_entanglement_qubit_connections_quantity(entanglement, 0, 0) != 4) {
        result = EqualityError;
    }

    uint32_t expected_connections_pairwise[4][2] = {{0, 1}, {2, 3}, {1, 2}, {3, 4}};

    for (size_t i = 0; i < 4; i++) {
        uint32_t *connections = expected_connections_pairwise[i];
        QkQubitConnection *expected_connection = qk_qubit_connection_new(2, connections);
        QkQubitConnection *connection =
            qk_get_entanglement_qubit_connections(entanglement, 0, 0, i);

        if (!qk_qubit_connection_equal(connection, expected_connection)) {
            result = EqualityError;
            break;
        }
    }

    return result;
}

int test_nlocal(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_nlocal_circuit_creation);
    num_failed += RUN_TEST(test_parameter_prefix);
    num_failed += RUN_TEST(test_entanglement_by_strategy);
    num_failed += RUN_TEST(test_pairwise_entanglement_strategy);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (n_local): %i\n", num_failed);

    return num_failed;
}
