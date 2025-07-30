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

void print_circuit(QkCircuit *qc) {
    size_t num_instructions = qk_circuit_num_instructions(qc);
    QkCircuitInstruction *inst = malloc(sizeof(QkCircuitInstruction));
    for (size_t i = 0; i < num_instructions; i++) {
        qk_circuit_get_instruction(qc, i, inst);
        printf("%s: qubits: (", inst->name);
        for (uint32_t j = 0; j < inst->num_qubits; j++) {
            printf("%d,", inst->qubits[j]);
        }
        printf(")");
        uint32_t num_clbits = inst->num_clbits;
        if (num_clbits > 0) {
            printf(", clbits: (");
            for (uint32_t j = 0; j < num_clbits; j++) {
                printf("%d,", inst->clbits[j]);
            }
            printf(")");
        }
        printf("\n");
        qk_circuit_instruction_clear(inst);
    }
    free(inst);
}

bool compare_circuits(QkCircuit *res, QkCircuit *expected) {
    if (qk_circuit_num_instructions(res) != qk_circuit_num_instructions(expected)) {
        printf("Number of instructions in circuit is mismatched");
        return false;
    }
    QkCircuitInstruction *res_inst = malloc(sizeof(QkCircuitInstruction));
    QkCircuitInstruction *expected_inst = malloc(sizeof(QkCircuitInstruction));
    for (size_t i = 0; i < qk_circuit_num_instructions(res); i++) {
        qk_circuit_get_instruction(res, i, res_inst);
        qk_circuit_get_instruction(expected, i, expected_inst);
        int result = strcmp(res_inst->name, expected_inst->name);
        if (result != 0) {
            printf("Gate %d have different gates %s was found and expected %s", i, res_inst->name,
                   expected_inst->name);
            qk_circuit_instruction_clear(res_inst);
            qk_circuit_instruction_clear(expected_inst);
            free(res_inst);
            free(expected_inst);
            return false;
        }
        if (res_inst->num_qubits != expected_inst->num_qubits) {
            printf("Gate %d have different number of qubits %d was found and expected %d", i,
                   res_inst->num_qubits, expected_inst->num_qubits);
            qk_circuit_instruction_clear(res_inst);
            qk_circuit_instruction_clear(expected_inst);
            free(res_inst);
            free(expected_inst);
            return false;
        }
        for (uint32_t j = 0; j < res_inst->num_qubits; j++) {
            if (res_inst->qubits[j] != expected_inst->qubits[j]) {
                printf("Qubit %d for gate %d are different %d was found and expected %d\n", j, i,
                       res_inst->qubits[j] != expected_inst->qubits[j]);
                printf("Expected circuit instructions:\n");
                print_circuit(expected);
                printf("Result circuit:\n");
                print_circuit(res);
                qk_circuit_instruction_clear(res_inst);
                qk_circuit_instruction_clear(expected_inst);
                free(res_inst);
                free(expected_inst);
                return false;
            }
        }
        if (res_inst->num_clbits != expected_inst->num_clbits) {
            printf("Gate %d have different number of clbits %d was found and expected %d", i,
                   res_inst->num_clbits, expected_inst->num_clbits);
            qk_circuit_instruction_clear(res_inst);
            qk_circuit_instruction_clear(expected_inst);
            free(res_inst);
            free(expected_inst);
            return false;
        }
        for (uint32_t j = 0; j < res_inst->num_clbits; j++) {
            if (res_inst->clbits[j] != expected_inst->clbits[j]) {
                printf("Clbit %d for gate %d are different %d was found and expected %d", j, i,
                       res_inst->clbits[j] != expected_inst->clbits[j]);
                qk_circuit_instruction_clear(res_inst);
                qk_circuit_instruction_clear(expected_inst);
                free(res_inst);
                free(expected_inst);
                return false;
            }
        }
        if (res_inst->num_params != expected_inst->num_params) {
            printf("Gate %d have different number of params %d was found and expected %d", i,
                   res_inst->num_params, expected_inst->num_params);
            qk_circuit_instruction_clear(res_inst);
            qk_circuit_instruction_clear(expected_inst);
            free(res_inst);
            free(expected_inst);
            return false;
        }
        for (uint32_t j = 0; j < res_inst->num_params; j++) {
            if (res_inst->params[j] != expected_inst->params[j]) {
                printf("Parameter %d for gate %d are different %d was found and expected %d", j, i,
                       res_inst->params[j] != expected_inst->params[j]);
                qk_circuit_instruction_clear(res_inst);
                qk_circuit_instruction_clear(expected_inst);
                free(res_inst);
                free(expected_inst);
                return false;
            }
        }
        qk_circuit_instruction_clear(res_inst);
        qk_circuit_instruction_clear(expected_inst);
    }
    free(res_inst);
    free(expected_inst);
    return true;
}

/**
 * Test running sabre layout that requires layout and routing
 */
int test_sabre_layout_applies_layout(void) {
    const uint32_t num_qubits = 5;
    // Let's create a target with one qubit for now
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;

    // Add an X Gate.
    // This operation is global, no property map is provided
    QkExitCode result_x = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a global X gate.");
        result = EqualityError;
        goto cleanup;
    }

    // Re-add same gate, check if it fails
    QkExitCode result_x_readded = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x_readded != QkExitCode_TargetInstAlreadyExists) {
        printf("The addition of a repeated gate did not fail as expected.");
        result = EqualityError;
        goto cleanup;
    }

    // Number of qubits of the target should not change.
    uint32_t current_num_qubits = qk_target_num_qubits(target);

    size_t current_size = qk_target_num_instructions(target);
    if (current_size != 1) {
        printf("The size of this target is not correct: Expected 1, got %zu", current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (current_num_qubits - i);
        double inst_duration = 0.020039;

        QkExitCode result_cx_props =
            qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
        if (result_cx_props != QkExitCode_Success) {
            printf("Unexpected error occurred when adding property to a CX gate entry.");
            result = EqualityError;
            goto cleanup;
        }
    }

    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a CX gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {0, i + 1};
        qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
    }
    QkSabreLayoutResult *layout_result =
        qk_transpiler_pass_standalone_sabre_layout(qc, target, 4, 20, 20, 2025);
    QkCircuit *result_circuit = qk_sabre_layout_result_circuit(layout_result);

    QkOpCounts op_counts = qk_circuit_count_ops(result_circuit);
    if (op_counts.len != 2) {
        printf("More than 2 types of gates in circuit, circuit's instructions are:\n");
        print_circuit(result_circuit);
        result = EqualityError;
        goto layout_cleanup;
    }
    for (int i = 0; i < op_counts.len; i++) {
        int swap_gate = strcmp(op_counts.data[i].name, "swap");
        int cx_gate = strcmp(op_counts.data[i].name, "cx");
        if (cx_gate != 0 && swap_gate != 0) {
            printf("Gate type of %s found in the circuit which isn't expected");
            result = EqualityError;
            goto layout_cleanup;
        }
        if (swap_gate == 0 && op_counts.data[i].count != 2) {
            printf("Unexpected number of swaps %d found in the circuit.");
            result = EqualityError;
            goto layout_cleanup;
        }
    }
    uint32_t expected_initial_layout[5] = {1, 0, 2, 3, 4};
    uint32_t init_layout_size = qk_sabre_layout_result_initial_layout_num_qubits(layout_result);
    if (init_layout_size != 5) {
        printf("Initial Layout is not the correct size, expected 5 qubit the layout contains %d",
               init_layout_size);
        result = EqualityError;
        goto layout_cleanup;
    }
    for (uint32_t i = 0; i < init_layout_size; i++) {
        uint32_t init_layout =
            qk_sabre_layout_result_map_virtual_qubit_initial_layout(layout_result, i);
        if (init_layout != expected_initial_layout[i]) {
            printf("Initial layout maps qubit %d to %d, expected %d instead", i, init_layout,
                   expected_initial_layout[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }

    uint32_t expected_final_layout[5] = {3, 0, 1, 2, 4};
    uint32_t final_layout_size = qk_sabre_layout_result_final_layout_num_qubits(layout_result);
    if (final_layout_size != 5) {
        printf("Final Layout is not the correct size, expected 5 qubit the layout contains %d",
               final_layout_size);
        result = EqualityError;
        goto layout_cleanup;
    }
    for (uint32_t i = 0; i < final_layout_size; i++) {
        uint32_t final_layout = qk_sabre_layout_result_map_qubit_final_layout(layout_result, i);
        if (final_layout != expected_final_layout[i]) {
            printf("Final layout maps qubit %d to %d, expected %d instead", i, final_layout,
                   expected_final_layout[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }
    QkCircuit *expected_circuit = qk_circuit_new(5, 0);
    uint32_t qargs[2] = {1, 0};
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    qargs[0] = 1;
    qargs[1] = 2;
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    qargs[0] = 2;
    qargs[1] = 1;
    qk_circuit_gate(expected_circuit, QkGate_Swap, qargs, NULL);
    qargs[0] = 2;
    qargs[1] = 3;
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    qargs[0] = 3;
    qargs[1] = 2;
    qk_circuit_gate(expected_circuit, QkGate_Swap, qargs, NULL);
    qargs[0] = 3;
    qargs[1] = 4;
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    bool compare_result = compare_circuits(result_circuit, expected_circuit);
    if (!compare_result) {
        result = EqualityError;
        goto layout_cleanup;
    }

layout_cleanup:
    qk_sabre_layout_result_free(layout_result);
circuit_cleanup:
    qk_circuit_free(qc);
cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test running sabre layout that performs no transformation.
 */
int test_sabre_layout_no_swap(void) {
    const uint32_t num_qubits = 5;
    // Let's create a target with one qubit for now
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;

    // Add an X Gate.
    // This operation is global, no property map is provided
    QkExitCode result_x = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a global X gate.");
        result = EqualityError;
        goto cleanup;
    }

    // Re-add same gate, check if it fails
    QkExitCode result_x_readded = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x_readded != QkExitCode_TargetInstAlreadyExists) {
        printf("The addition of a repeated gate did not fail as expected.");
        result = EqualityError;
        goto cleanup;
    }

    // Number of qubits of the target should not change.
    uint32_t current_num_qubits = qk_target_num_qubits(target);

    size_t current_size = qk_target_num_instructions(target);
    if (current_size != 1) {
        printf("The size of this target is not correct: Expected 1, got %zu", current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (current_num_qubits - i);
        double inst_duration = 0.020039;

        QkExitCode result_cx_props =
            qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
        if (result_cx_props != QkExitCode_Success) {
            printf("Unexpected error occurred when adding property to a CX gate entry.");
            result = EqualityError;
            goto cleanup;
        }
    }

    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a CX gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    QkSabreLayoutResult *layout_result =
        qk_transpiler_pass_standalone_sabre_layout(qc, target, 4, 20, 20, 42);
    QkCircuit *expected_circuit = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {4 - i, 4 - i - 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
        }
    }
    bool circuit_eq =
        compare_circuits(qk_sabre_layout_result_circuit(layout_result), expected_circuit);
    if (!circuit_eq) {
        result = EqualityError;
        goto layout_cleanup;
    }
    uint32_t init_layout_size = qk_sabre_layout_result_initial_layout_num_qubits(layout_result);
    if (init_layout_size != 5) {
        printf("Initial Layout is not the correct size, expected 5 qubit the layout contains %d",
               init_layout_size);
        result = EqualityError;
        goto layout_cleanup;
    }
    uint32_t expected_initial_layout[5] = {4, 3, 2, 1, 0};
    for (uint32_t i = 0; i < init_layout_size; i++) {
        uint32_t init_layout =
            qk_sabre_layout_result_map_virtual_qubit_initial_layout(layout_result, i);
        if (init_layout != expected_initial_layout[i]) {
            printf("Initial layout maps qubit %d to %d, expected %d instead", i, init_layout,
                   expected_initial_layout[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }
    uint32_t final_layout_size = qk_sabre_layout_result_final_layout_num_qubits(layout_result);
    if (final_layout_size != 5) {
        printf("Final Layout is not the correct size, expected 5 qubit the layout contains %d",
               final_layout_size);
        result = EqualityError;
        goto layout_cleanup;
    }
    uint32_t expected_final_layout[5] = {4, 3, 2, 1, 0};
    for (uint32_t i = 0; i < final_layout_size; i++) {
        uint32_t final_layout =
            qk_sabre_layout_result_map_virtual_qubit_initial_layout(layout_result, i);
        if (final_layout != expected_final_layout[i]) {
            printf("Final layout maps qubit %d to %d, expected %d instead", i, final_layout,
                   expected_final_layout[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }

layout_cleanup:
    qk_sabre_layout_result_free(layout_result);
circuit_cleanup:
    qk_circuit_free(qc);
cleanup:
    qk_target_free(target);
    return result;
}

int test_sabre_layout(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_sabre_layout_no_swap);
    num_failed += RUN_TEST(test_sabre_layout_applies_layout);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
