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

void print_circuit_list(QkCircuit *qc) {
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

bool compare_transpiled_circuits(QkCircuit *res, QkCircuit *expected) {
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
                print_circuit_list(expected);
                printf("Result circuit:\n");
                print_circuit_list(res);
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
int test_transpile_bv(void) {
    const uint32_t num_qubits = 100;
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;

    QkTargetEntry *x_entry = qk_target_entry_new(QkGate_X);
    for (int i = 0; i < num_qubits; i++) {
        uint32_t qargs[1] = {i,};
        double error = 0.8e-6 * (i + 1);
        double duration = 1.8e-9 * (i + 1);
        qk_target_entry_add_property(x_entry, qargs, 1, duration, error);

    }
    qk_target_add_instruction(target, x_entry);

    QkTargetEntry *sx_entry = qk_target_entry_new(QkGate_SX);
    for (int i = 0; i < num_qubits; i++) {
        uint32_t qargs[1] = {i,};
        double error = 0.8e-6 * (i + 1);
        double duration = 1.8e-9 * (i + 1);
        qk_target_entry_add_property(sx_entry, qargs, 1, duration, error);
    }
    qk_target_add_instruction(target, sx_entry);

    QkTargetEntry *rz_entry = qk_target_entry_new(QkGate_RZ);
    for (int i = 0; i < num_qubits; i++) {
        uint32_t qargs[1] = {i,};
        double error = 0.;
        double duration = 0.;
        qk_target_entry_add_property(rz_entry, qargs, 1, duration, error);
    }
    qk_target_add_instruction(target, rz_entry);

    QkTargetEntry *ecr_entry = qk_target_entry_new(QkGate_ECR);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (num_qubits - i);
        double inst_duration = 0.020039;

        qk_target_entry_add_property(ecr_entry, qargs, 2, inst_duration, inst_error);
    }
    qk_target_add_instruction(target, ecr_entry);

    QkCircuit *qc = qk_circuit_new(50, 0);
    uint32_t x_qargs[1] = {49,};
    qk_circuit_gate(qc, QkGate_X, x_qargs, NULL);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc); i++) {
        uint32_t qargs[1] = {i,};
        qk_circuit_gate(qc, QkGate_H, qargs, NULL);
    }
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i+=2) {
        uint32_t qargs[2] = {i, 49};
        qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
    }
    QkTranspileResult transpile_result = {NULL, NULL};
    char *error = NULL;
    QkTranspileOptions options = qk_transpiler_default_options();
    options.seed = 42;
    int result_code = qk_transpile(qc, target, &options, &transpile_result, &error);

    QkOpCounts op_counts = qk_circuit_count_ops(transpile_result.circuit);
    if (op_counts.len != 2) {
        printf("More than 2 types of gates in circuit, circuit's instructions are:\n");
        print_circuit_list(transpile_result.circuit);
        result = EqualityError;
        goto transpile_cleanup;
    }
    for (int i = 0; i < op_counts.len; i++) {
        int swap_gate = strcmp(op_counts.data[i].name, "swap");
        int cx_gate = strcmp(op_counts.data[i].name, "cx");
        if (cx_gate != 0 && swap_gate != 0) {
            printf("Gate type of %s found in the circuit which isn't expected");
            result = EqualityError;
            goto transpile_cleanup;
        }
        if (swap_gate == 0 && op_counts.data[i].count != 2) {
            printf("Unexpected number of swaps %d found in the circuit.");
            result = EqualityError;
            goto transpile_cleanup;
        }
    }

transpile_cleanup:
    qk_circuit_free(transpile_result.circuit);
    qk_transpile_layout_free(transpile_result.layout);

circuit_cleanup:
    qk_circuit_free(qc);
cleanup:
    qk_target_free(target);
    return result;
}

int test_transpiler(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_transpile_bv);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
