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
        printf("The duration assigned to this instance: %f difference from the original: %f", duration, inst_duration);
    }
    if (error != inst_error) {
        printf("The duration assigned to this instance: %f difference from the original: %f", error, inst_error);
    }
    return Ok;
}

/**
 * Test construction of PropsMap
 */
int test_property_map_new(void) {
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
    return Ok;
}

/**
 * Test adding a global to the Target.
 */
int test_target_add_instruction() {
    // Let's create a target with zero qubits for now
    QkTarget *target = qk_target_new(1);

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
    QkInstructionProps *instruction_props = qk_instruction_properties_new(inst_error, inst_duration);
    qk_property_map_add(property_map, qargs, 2, instruction_props);
    // CX Gate is not paramtric. Re-use Null
    double *cx_params = NULL;

    qk_target_add_instruction(target, QkGate_CX, params, property_map);

    // Number of qubits of the target should change to 2.
    current_size = qk_target_num_qubits(target);
    if (current_size != 2) {
        printf("The number of qubits this target is compatible with is not 2: %zu", current_size);
        return EqualityError;
    }
    if (!qk_target_contains_instr(target, "cx")) {
        printf("This target did not correctly add the instruction 'cx'");
        return RuntimeError;
    }
    // Instruction should now show compatibility with (0,1)
    if (!qk_target_instruction_supported(target, "cx", qargs, 2)) {
        printf("This target did not correctly demonstrate compatibility with 'cx' and qargs [0,1]");
        return RuntimeError;
    }

    // Add a CRX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    const QkPropsMap *crx_property_map = qk_propety_map_new();
    uint32_t crx_qargs[2] = {1, 2};
    double crx_inst_error = 0.0129023;
    double crx_inst_duration = 0.92939;
    const QkInstructionProps *crx_instruction_props = qk_instruction_properties_new(crx_inst_error, crx_inst_duration);
    qk_property_map_add(property_map, crx_qargs, 2, crx_instruction_props);
    // CX Gate is not paramtric.
    double crx_params[1] = {3.14};

    qk_target_add_instruction(target, QkGate_CRX, crx_params, property_map);

    // Number of qubits of the target should change to 3.
    current_size = qk_target_num_qubits(target);
    if (current_size != 3) {
        printf("The number of qubits this target is compatible with is not 3: %d", current_size);
        return EqualityError;
    }
    if (!qk_target_contains_instr(target, "crx")) {
        printf("This target did not correctly add the instruction 'cx'");
        return RuntimeError;
    }
    // Instruction should now show compatibility with (0,1)
    if (!qk_target_instruction_supported(target, "crx", crx_qargs, 2)) {
        printf("This target did not correctly demonstrate compatibility with 'crx' and qargs [1,2]");
        return RuntimeError;
    }

    // TODO: Fix this
    const char *gate_names[3] = {"x", "cx", "crx"};
    const char **comp_gate_names = qk_target_operation_names(target);
    printf("%s", comp_gate_names[0]);
    for (int i = 0; i < 1; i++) {
        if (strcmp(gate_names[i], comp_gate_names[i]) != 0) {
            printf("Gate comparison order is not correct in this target: At index %d, %s is not %s.", i, gate_names[i], comp_gate_names[i]);
            return RuntimeError;
        }
    }
    return Ok;
}

int test_target(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty_target);
    num_failed += RUN_TEST(test_instruction_properties_construction);
    num_failed += RUN_TEST(test_property_map_new);
    num_failed += RUN_TEST(test_target_add_instruction);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}

