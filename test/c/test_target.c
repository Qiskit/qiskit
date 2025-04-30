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

/**
 * Test empty constructor for Target
 */
int test_empty_target(void) {
    QkTarget *target = qk_target_new(0);
    uint32_t num_qubits = qk_target_num_qubits(target);

    if (num_qubits != 0) {
        printf("The number of qubits %d is not 0.", num_qubits);
        return EqualityError;
    }
    return Ok;
}

/**
 * Test construction of an `InstructionProperties` instance
 */
int test_instruction_properties_construction(void) {
    const double duration = 0.090903;
    const double error = 0.0038484;

    const QkInstructionProps *inst_prop = qk_instruction_properties_new(duration, error);
    double inst_duration = qk_instruction_properties_get_duration(inst_prop);
    double inst_error = qk_instruction_properties_get_error(inst_prop);

    if (duration != inst_duration) {
        printf("The duration assigned to this instance: %f difference from the original: %f",
               duration, inst_duration);
    }
    if (error != inst_error) {
        printf("The duration assigned to this instance: %f difference from the original: %f", error,
               inst_error);
    }
    return Ok;
}

/**
 * Test construction of PropsMap
 */
int test_property_map_construction(void) {
    QkPropsMap *property_map = qk_propety_map_new();

    // Test length
    const size_t length = qk_property_map_length(property_map);
    if (length != 0) {
        printf("The initial length of the provided property map was not zero: %zu", length);
        return EqualityError;
    }

    // Add some qargs and properties
    uint32_t qargs[2] = {0, 1};
    const QkInstructionProps *inst_prop = qk_instruction_properties_new(0.00018, 0.00002);

    qk_property_map_add(property_map, qargs, 2, inst_prop);
    // Test length
    const size_t new_length = qk_property_map_length(property_map);
    if (new_length != 1) {
        printf("The initial length of the provided property map was not 1: %zu", length);
        return EqualityError;
    }
    if (!qk_property_map_contains_qargs(property_map, qargs, 2)) {
        printf("The qargs ");
        print_qargs(qargs, 2);
        printf(" not properly added to the property map.", qargs);
        return EqualityError;
    }
    return Ok;
}

/**
 * Test adding a global to the Target.
 */
int test_target_add_instruction(void) {
    // Let's create a target with one qubits for now
    QkTarget *target = qk_target_new(1);
    int result = Ok;
    // Add an X Gate.
    // Create prop_map for the instruction
    // This operation is global, so props_map stays empty
    QkPropsMap *property_map = qk_propety_map_new();
    // X Gate is not parametric.
    double *params = NULL;

    qk_target_add_instruction(target, QkGate_X, params, property_map);

    // Number of qubits of the target should not change.
    size_t current_size = qk_target_num_qubits(target);
    if (current_size != 1) {
        printf("The number of qubits this target is compatible with is not 1: %zu", current_size);
        return EqualityError;
    }
    // Check that the instruction exists in there
    if (!qk_target_contains_instr(target, "x")) {
        printf("This target did not correctly add the instruction 'x'");
        return RuntimeError;
    }

    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    property_map = qk_propety_map_new();
    uint32_t qargs[2] = {0, 1};
    double inst_error = 0.0090393;
    double inst_duration = 0.020039;
    QkInstructionProps *instruction_props =
        qk_instruction_properties_new(inst_error, inst_duration);
    qk_property_map_add(property_map, qargs, 2, instruction_props);
    // CX Gate is not paramtric. Re-use Null
    double *cx_params = NULL;

    qk_target_add_instruction(target, QkGate_CX, params, property_map);

    // Number of qubits of the target should change to 2.
    current_size = qk_target_num_qubits(target);
    if (current_size != 2) {
        printf("The number of qubits this target is compatible with is not 2: %zu", current_size);
        result = EqualityError;
        goto cleanup;
    }
    if (!qk_target_contains_instr(target, "cx")) {
        printf("This target did not correctly add the instruction 'cx'");
        result = RuntimeError;
        goto cleanup;
    }
    // Instruction should now show compatibility with (0,1)
    if (!qk_target_instruction_supported(target, "cx", qargs, 2)) {
        printf("This target did not correctly demonstrate compatibility with 'cx' and qargs [0,1]");
        result = RuntimeError;
        goto cleanup;
    }

    // Add a CRX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    const QkPropsMap *crx_property_map = qk_propety_map_new();
    uint32_t crx_qargs[2] = {1, 2};
    double crx_inst_error = 0.0129023;
    double crx_inst_duration = 0.92939;
    const QkInstructionProps *crx_instruction_props =
        qk_instruction_properties_new(crx_inst_error, crx_inst_duration);
    qk_property_map_add(property_map, crx_qargs, 2, crx_instruction_props);
    // CX Gate is not paramtric.
    double crx_params[1] = {3.14};

    qk_target_add_instruction(target, QkGate_CRX, crx_params, property_map);

    // Number of qubits of the target should change to 3.
    current_size = qk_target_num_qubits(target);
    if (current_size != 3) {
        printf("The number of qubits this target is compatible with is not 3: %ld", current_size);
        result = EqualityError;
        goto cleanup;
    }
    if (!qk_target_contains_instr(target, "crx")) {
        printf("This target did not correctly add the instruction 'cx'");
        result = RuntimeError;
        goto cleanup;
    }
    // Instruction should now show compatibility with (0,1)
    if (!qk_target_instruction_supported(target, "crx", crx_qargs, 2)) {
        printf(
            "This target did not correctly demonstrate compatibility with 'crx' and qargs ");
        print_qargs(crx_qargs, 2);
        printf(".");
        result = RuntimeError;
        goto cleanup;
    }

    const char *gate_names[3] = {"x", "cx", "crx"};
    char **comp_gate_names = qk_target_operation_names(target);

    for (int i = 0; i < 3; i++) {
        if (strcmp(gate_names[i], comp_gate_names[i]) != 0) {
            printf(
                "Gate comparison order is not correct in this target: At index %d, %s is not %s.",
                i, gate_names[i], comp_gate_names[i]);
            result = RuntimeError;
            goto cleanup;
        }
    }

cleanup:
    qk_target_free(target);
    qk_instruction_properties_free(instruction_props);
    qk_instruction_properties_free(crx_instruction_props);
    qk_propety_map_free(property_map);
    qk_propety_map_free(crx_property_map);
    return result;
}

int test_target_update_instruction(void) {
    // Let's create a target with one qubits for now
    QkTarget *target = qk_target_new(1);
    int result = Ok;
    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkPropsMap *property_map = qk_propety_map_new();
    uint32_t qargs[2] = {0, 1};
    double inst_error = 0.0090393;
    double inst_duration = 0.020039;
    QkInstructionProps *instruction_props =
        qk_instruction_properties_new(inst_duration, inst_error);
    qk_property_map_add(property_map, qargs, 2, instruction_props);
    // CX Gate is not paramtric. Re-use Null
    qk_target_add_instruction(target, QkGate_CX, NULL, property_map);

    // check current instruction property for cx at (0,1)
    const QkInstructionProps *retreived_inst = qk_target_get_inst_prop(target, "cx", qargs, 2);
    double retreived_duration = qk_instruction_properties_get_duration(retreived_inst);
    if (retreived_duration != inst_duration) {
        printf(
            "The incorrect duration was recorded for the instruction property, expected %f, got %f",
            retreived_duration, inst_duration);
        result = RuntimeError;
        goto cleanup;
    }
    double retreived_error = qk_instruction_properties_get_error(retreived_inst);
    if (retreived_error != inst_error) {
        printf(
            "The incorrect duration was recorded for the instruction property, expected %f, got %f",
            retreived_error, inst_error);
        result = RuntimeError;
        goto cleanup;
    }

    // change the intruction property of cx
    double cx_new_inst_error = 0.102902;
    double cx_new_inst_duration = 0.09457;
    QkInstructionProps *new_cx_instruction_props =
        qk_instruction_properties_new(cx_new_inst_duration, cx_new_inst_error);
    qk_target_update_instruction_prop(target, "cx", qargs, 2, new_cx_instruction_props);
    // check current instruction property for cx at (0,1)
    QkInstructionProps *new_cx_retreived_inst = qk_target_get_inst_prop(target, "cx", qargs, 2);
    double new_retreived_duration = qk_instruction_properties_get_duration(new_cx_retreived_inst);
    if (new_retreived_duration != cx_new_inst_duration) {
        printf(
            "The incorrect duration was recorded for the instruction property, expected %f, got %f",
            new_retreived_duration, cx_new_inst_duration);
        result = RuntimeError;
        goto cleanup;
    }
    double new_retreived_error = qk_instruction_properties_get_error(new_cx_retreived_inst);
    if (new_retreived_error != cx_new_inst_error) {
        printf(
            "The incorrect duration was recorded for the instruction property, expected %f, got %f",
            new_retreived_error, cx_new_inst_error);
        result = RuntimeError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    qk_instruction_properties_free(instruction_props);
    qk_instruction_properties_free(retreived_inst);
    qk_instruction_properties_free(new_cx_instruction_props);
    qk_instruction_properties_free(new_cx_retreived_inst);
    qk_propety_map_free(property_map);
    return result;
}

int test_target_non_global_op_names(void) {
    // Let's create a target with one qubits for now
    QkTarget *target = qk_target_new(1);
    int result = Ok;
    // Add an X Gate.
    // Create prop_map for the instruction
    // This operation is global, so props_map stays empty
    QkPropsMap *property_map = qk_propety_map_new();
    qk_target_add_instruction(target, QkGate_X, NULL, property_map);

    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    property_map = qk_propety_map_new();
    uint32_t qargs[2] = {0, 1};
    QkInstructionProps *instruction_props =
        qk_instruction_properties_new(0, 0);
    qk_property_map_add(property_map, qargs, 2, instruction_props);
    qk_target_add_instruction(target, QkGate_CX, NULL, property_map);

    // Add a CRX Gate.
    // Create prop_map for the instruction
    // Add property for (2, 1)
    const QkPropsMap *crx_property_map = qk_propety_map_new();
    uint32_t crx_qargs[2] = {2, 1};
    const QkInstructionProps *crx_instruction_props =
        qk_instruction_properties_new(0, 0);
    qk_property_map_add(crx_property_map, crx_qargs, 2, crx_instruction_props);
    // CX Gate is not paramtric.
    double crx_params[1] = {3.14};

    qk_target_add_instruction(target, QkGate_CRX, crx_params, crx_property_map);

    const char *non_local_gate_names[2] = {"cx", "crx"};
    char **non_local_comp_gate_names = qk_target_non_global_operation_names(target, false);
    for (int i = 0; i < 2; i++) {
        if (strcmp(non_local_gate_names[i], non_local_comp_gate_names[i]) != 0) {
            printf(
                "Gate comparison order is not correct in this target: At index %d, %s is not %s.",
                i, non_local_gate_names[i], non_local_comp_gate_names[i]);
            result = RuntimeError;
            goto cleanup;
        }
    }

    const char *non_local_gate_names_strict[2] = {"cx", "crx"};
    char **non_local_comp_gate_names_strict = qk_target_non_global_operation_names(target, true);
    for (int i = 0; i < 2; i++) {
        if (strcmp(non_local_gate_names_strict[i], non_local_comp_gate_names_strict[i]) != 0) {
            printf(
                "Gate comparison order is not correct in this target: At index %d, %s is not %s.",
                i, non_local_gate_names_strict[i], non_local_comp_gate_names_strict[i]);
            result = RuntimeError;
            goto cleanup;
        }
    }

cleanup:
    qk_target_free(target);
    qk_instruction_properties_free(instruction_props);
    qk_instruction_properties_free(crx_instruction_props);
    qk_propety_map_free(property_map);
    qk_propety_map_free(crx_property_map);
    return result;
}


// Helper function.
void print_qargs(uint32_t *qargs, uint32_t size) {
    printf("[");
    for (int i = 0; i < 2; i++) {
        printf("%d", qargs[i]);
        if (i < 1) {
            printf(", ");
        }
    }
    printf("]");
}

int test_target(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty_target);
    num_failed += RUN_TEST(test_instruction_properties_construction);
    num_failed += RUN_TEST(test_property_map_construction);
    num_failed += RUN_TEST(test_target_add_instruction);
    num_failed += RUN_TEST(test_target_update_instruction);
    num_failed += RUN_TEST(test_target_non_global_op_names);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
