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

// Helper function to display qargs in case of failure.
void print_qargs(uint32_t *qargs, uint32_t size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%u", qargs[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]");
}

// Helper function to compare qargs
bool compare_qargs(uint32_t *lhs, uint32_t *rhs, size_t size) {
    for (int i = 0; i < size; i++) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

// Helper comp for qsort
int gate_comp_help(const void *lhs, const void *rhs) {
    const QkGate *lhs_as_pointer = lhs;
    const QkGate *rhs_as_pointer = rhs;
    return *lhs_as_pointer - *rhs_as_pointer;
}

void sort_gate_array(QkGate *gate_list, int length) {
    qsort(gate_list, length, sizeof(*gate_list), gate_comp_help);
}

/**
 * Test empty constructor for Target
 */
int test_empty_target(void) {
    QkTarget *target = qk_target_new(0);
    size_t num_qubits = qk_target_num_qubits(target);

    if (num_qubits != 0) {
        printf("The number of qubits %lu is not 0.", num_qubits);
        return EqualityError;
    }

    double retrieved_dt = qk_target_dt(target);
    if (retrieved_dt != 0) {
        printf("The dt value of this target %f is not %p.", retrieved_dt, NULL);
        return EqualityError;
    }

    uint32_t retrieved_granularity = qk_target_granularity(target);
    if (retrieved_granularity != 1) {
        printf("The granularity %u is not 1.", retrieved_granularity);
        return EqualityError;
    }

    size_t retrieved_min_length = qk_target_min_length(target);
    if (retrieved_min_length != 1) {
        printf("The min_length %lu is not 1.", retrieved_min_length);
        return EqualityError;
    }

    uint32_t pulse_alignment = qk_target_pulse_alignment(target);
    if (pulse_alignment != 1) {
        printf("The pulse_alignment values %u is not 1.", pulse_alignment);
        return EqualityError;
    }

    uint32_t acquire_alignment = qk_target_acquire_alignment(target);
    if (acquire_alignment != 0) {
        printf("The acquire_alignment values %u is not 0.", acquire_alignment);
        return EqualityError;
    }
    return Ok;
}

/**
 * Test constructor for Target
 */
int test_target_construct(void) {
    const size_t num_qubits = 2;
    const double dt = 10e-9;
    const uint32_t granularity = 2;
    const size_t min_length = 3;
    const uint32_t p_alignment = 4;
    const uint32_t a_alignment = 1;

    QkTarget *target = qk_target_new(num_qubits);

    qk_target_set_dt(target, dt);
    qk_target_set_granularity(target, granularity);
    qk_target_set_min_length(target, min_length);
    qk_target_set_pulse_alignment(target, p_alignment);
    qk_target_set_acquire_alignment(target, a_alignment);

    size_t retrieved_num_qubits = qk_target_num_qubits(target);
    if (retrieved_num_qubits != 2) {
        printf("The number of qubits %lu is not 0.", num_qubits);
        return EqualityError;
    }

    double retrieved_dt = qk_target_dt(target);
    if (retrieved_dt != dt) {
        printf("The dt value of this target %f is not %f.", retrieved_dt, dt);
        return EqualityError;
    }

    uint32_t retrieved_granularity = qk_target_granularity(target);
    if (retrieved_granularity != 2) {
        printf("The granularity %u is not 2.", retrieved_granularity);
        return EqualityError;
    }

    size_t retrieved_min_length = qk_target_min_length(target);
    if (retrieved_min_length != 3) {
        printf("The min_length %lu is not 3.", retrieved_min_length);
        return EqualityError;
    }

    uint32_t pulse_alignment = qk_target_pulse_alignment(target);
    if (pulse_alignment != 4) {
        printf("The pulse_alignment values %u is not 4.", pulse_alignment);
        return EqualityError;
    }

    uint32_t acquire_alignment = qk_target_acquire_alignment(target);
    if (acquire_alignment != 1) {
        printf("The acquire_alignment values %u is not 1.", acquire_alignment);
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
    QkPropsMap *property_map = qk_property_map_new();

    // Test length
    const size_t length = qk_property_map_length(property_map);
    if (length != 0) {
        printf("The initial length of the provided property map was not zero: %zu", length);
        return EqualityError;
    }

    // Add some qargs and properties
    uint32_t qargs[2] = {0, 1};
    QkInstructionProps *inst_prop = qk_instruction_properties_new(0.00018, 0.00002);

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
        printf(" not properly added to the property map.");
        return EqualityError;
    }

    qk_property_map_free(property_map);
    qk_instruction_properties_free(inst_prop);
    return Ok;
}

/**
 * Test adding an instruction to the Target.
 */
int test_target_add_instruction(void) {
    const size_t num_qubits = 1;
    // Let's create a target with one qubit for now
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    // Add an X Gate.
    // Create prop_map for the instruction
    // This operation is global, so props_map stays empty
    QkPropsMap *property_map = qk_property_map_new();
    // X Gate is not parametric.
    double *params = NULL;

    qk_target_add_instruction(target, QkGate_X, params, property_map);

    // Number of qubits of the target should not change.
    size_t current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 1) {
        printf("The number of qubits this target is compatible with is not 1: %zu",
               current_num_qubits);
        return EqualityError;
    }
    size_t current_size = qk_target_length(target);
    if (current_num_qubits != 1) {
        printf("The size of this target is not correct: Expected 1, got %zu", current_size);
        return EqualityError;
    }
    // Check that the instruction exists in there
    if (!qk_target_contains_instr(target, QkGate_X)) {
        printf("This target did not correctly add the instruction 'x'");
        return RuntimeError;
    }

    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    property_map = qk_property_map_new();
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
    current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 2) {
        printf("The number of qubits this target is compatible with is not 2: %zu",
               current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }
    current_size = qk_target_length(target);
    if (current_num_qubits != 2) {
        printf("The size of this target is not correct: Expected 2, got %zu", current_size);
        return EqualityError;
    }
    if (!qk_target_contains_instr(target, QkGate_CX)) {
        printf("This target did not correctly add the instruction 'cx'");
        result = RuntimeError;
        goto cleanup;
    }
    // Instruction should now show compatibility with (0,1)
    if (!qk_target_instruction_supported(target, QkGate_CX, qargs, 2)) {
        printf("This target did not correctly demonstrate compatibility with 'cx' and qargs [0,1]");
        result = RuntimeError;
        goto cleanup;
    }

    // Add a CRX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkPropsMap *crx_property_map = qk_property_map_new();
    uint32_t crx_qargs[2] = {1, 2};
    double crx_inst_error = 0.0129023;
    double crx_inst_duration = 0.92939;
    QkInstructionProps *crx_instruction_props =
        qk_instruction_properties_new(crx_inst_error, crx_inst_duration);
    qk_property_map_add(property_map, crx_qargs, 2, crx_instruction_props);
    // CX Gate is not paramtric.
    double crx_params[1] = {3.14};

    qk_target_add_instruction(target, QkGate_CRX, crx_params, property_map);

    // Number of qubits of the target should change to 3.
    current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 3) {
        printf("The number of qubits this target is compatible with is not 3: %ld",
               current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }
    current_size = qk_target_length(target);
    if (current_num_qubits != 3) {
        printf("The size of this target is not correct: Expected 3, got %zu", current_size);
        return EqualityError;
    }
    if (!qk_target_contains_instr(target, QkGate_CRX)) {
        printf("This target did not correctly add the instruction 'cx'");
        result = RuntimeError;
        goto cleanup;
    }
    // Instruction should now show compatibility with (0,1)
    if (!qk_target_instruction_supported(target, QkGate_CRX, crx_qargs, 2)) {
        printf("This target did not correctly demonstrate compatibility with 'crx' and qargs ");
        print_qargs(crx_qargs, 2);
        printf(".");
        result = RuntimeError;
        goto cleanup;
    }

    const QkGate gate_names[3] = {QkGate_X, QkGate_CX, QkGate_CRX};
    QkTargetOpsList comp_gate_names = qk_target_operations(target);

    for (int i = 0; i < comp_gate_names.length; i++) {
        if (gate_names[i] != comp_gate_names.list[i]) {
            printf("Gate comparison order is not correct in this target: At index %d, %hhu is not "
                   "%hhu.",
                   i, gate_names[i], comp_gate_names.list[i]);
            result = RuntimeError;
            goto cleanup;
        }
    }

cleanup:
    qk_target_free(target);
    qk_instruction_properties_free(instruction_props);
    qk_instruction_properties_free(crx_instruction_props);
    qk_property_map_free(property_map);
    qk_property_map_free(crx_property_map);
    return result;
}

/**
 * Test updating an instruction property in the Target using
 * `update_instruction_property`.
 */
int test_target_update_instruction(void) {
    const size_t num_qubits = 1;
    // Let's create a target with one qubit for now
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkPropsMap *property_map = qk_property_map_new();
    uint32_t qargs[2] = {0, 1};
    double inst_error = 0.0090393;
    double inst_duration = 0.020039;
    QkInstructionProps *instruction_props =
        qk_instruction_properties_new(inst_duration, inst_error);
    qk_property_map_add(property_map, qargs, 2, instruction_props);
    // CX Gate is not paramtric. Re-use Null
    qk_target_add_instruction(target, QkGate_CX, NULL, property_map);

    // check current instruction property for cx at (0,1)
    QkInstructionProps *retrieved_inst = qk_target_get_inst_prop(target, QkGate_CX, qargs, 2);
    double retrieved_duration = qk_instruction_properties_get_duration(retrieved_inst);
    if (retrieved_duration != inst_duration) {
        printf(
            "The incorrect duration was recorded for the instruction property, expected %f, got %f",
            retrieved_duration, inst_duration);
        result = RuntimeError;
        goto cleanup;
    }
    double retrieved_error = qk_instruction_properties_get_error(retrieved_inst);
    if (retrieved_error != inst_error) {
        printf(
            "The incorrect duration was recorded for the instruction property, expected %f, got %f",
            retrieved_error, inst_error);
        result = RuntimeError;
        goto cleanup;
    }

    // change the intruction property of cx
    double cx_new_inst_error = 0.102902;
    double cx_new_inst_duration = 0.09457;
    QkInstructionProps *new_cx_instruction_props =
        qk_instruction_properties_new(cx_new_inst_duration, cx_new_inst_error);
    qk_target_update_instruction_prop(target, QkGate_CX, qargs, 2, new_cx_instruction_props);
    // check current instruction property for cx at (0,1)
    QkInstructionProps *new_cx_retrieved_inst =
        qk_target_get_inst_prop(target, QkGate_CX, qargs, 2);
    double new_retrieved_duration = qk_instruction_properties_get_duration(new_cx_retrieved_inst);
    if (new_retrieved_duration != cx_new_inst_duration) {
        printf(
            "The incorrect duration was recorded for the instruction property, expected %f, got %f",
            new_retrieved_duration, cx_new_inst_duration);
        result = RuntimeError;
        goto cleanup;
    }
    double new_retrieved_error = qk_instruction_properties_get_error(new_cx_retrieved_inst);
    if (new_retrieved_error != cx_new_inst_error) {
        printf(
            "The incorrect duration was recorded for the instruction property, expected %f, got %f",
            new_retrieved_error, cx_new_inst_error);
        result = RuntimeError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    qk_instruction_properties_free(instruction_props);
    qk_instruction_properties_free(retrieved_inst);
    qk_instruction_properties_free(new_cx_instruction_props);
    qk_instruction_properties_free(new_cx_retrieved_inst);
    qk_property_map_free(property_map);
    return result;
}

/**
 * Test retrieving non global operation names from the Target.
 */
int test_target_non_global_op_names(void) {
    const size_t num_qubits = 1;
    // Let's create a target with one qubit for now
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    // Add an X Gate.
    // Create prop_map for the instruction
    // This operation is global, so props_map stays empty
    QkPropsMap *property_map = qk_property_map_new();
    qk_target_add_instruction(target, QkGate_X, NULL, property_map);

    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    property_map = qk_property_map_new();
    uint32_t qargs[2] = {0, 1};
    QkInstructionProps *instruction_props = qk_instruction_properties_new(0, 0);
    qk_property_map_add(property_map, qargs, 2, instruction_props);
    qk_target_add_instruction(target, QkGate_CX, NULL, property_map);

    // Add a CRX Gate.
    // Create prop_map for the instruction
    // Add property for (2, 1)
    QkPropsMap *crx_property_map = qk_property_map_new();
    uint32_t crx_qargs[2] = {2, 1};
    QkInstructionProps *crx_instruction_props = qk_instruction_properties_new(0, 0);
    qk_property_map_add(crx_property_map, crx_qargs, 2, crx_instruction_props);
    // CX Gate is not paramtric.
    double crx_params[1] = {3.14};

    qk_target_add_instruction(target, QkGate_CRX, crx_params, crx_property_map);

    QkGate non_local_gate_names[2] = {QkGate_CX, QkGate_CRX};
    QkTargetOpsList non_local_comp_gate_names = qk_target_non_global_operations(target, false);
    for (int i = 0; i < non_local_comp_gate_names.length; i++) {
        if (non_local_gate_names[i] != non_local_comp_gate_names.list[i]) {
            printf("Gate comparison order is not correct in this target: At index %u, %hhu is not "
                   "%hhu.",
                   i, non_local_gate_names[i], non_local_comp_gate_names.list[i]);
            result = RuntimeError;
            goto cleanup;
        }
    }

    QkGate non_local_gate_names_strict[2] = {QkGate_CX, QkGate_CRX};
    QkTargetOpsList non_local_comp_gate_names_strict =
        qk_target_non_global_operations(target, true);
    for (int i = 0; i < non_local_comp_gate_names_strict.length; i++) {
        if (non_local_gate_names_strict[i] != non_local_comp_gate_names_strict.list[i]) {
            printf("Gate comparison order is not correct in this target: At index %u, %hhu is not "
                   "%hhu.",
                   i, non_local_gate_names_strict[i], non_local_comp_gate_names_strict.list[i]);
            result = RuntimeError;
            goto cleanup;
        }
    }

cleanup:
    qk_target_free(target);
    qk_instruction_properties_free(instruction_props);
    qk_instruction_properties_free(crx_instruction_props);
    qk_property_map_free(property_map);
    qk_property_map_free(crx_property_map);
    return result;
}

/**
 * Test retrieving operation names based on their qargs.
 */
int test_target_operation_names_for_qargs(void) {
    const size_t num_qubits = 1;
    // Build sample target
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    QkPropsMap *i_property_map = qk_property_map_new();
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *i_props = qk_instruction_properties_new(35.5e-9, 0.);
        qk_property_map_add(i_property_map, qargs, 1, i_props);
    }
    qk_target_add_instruction(target, QkGate_I, NULL, i_property_map);

    QkPropsMap *rz_property_map = qk_property_map_new();
    double rz_params[1] = {3.14};
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *rz_props = qk_instruction_properties_new(0., 0.);
        qk_property_map_add(rz_property_map, qargs, 1, rz_props);
    }
    qk_target_add_instruction(target, QkGate_RZ, rz_params, i_property_map);

    QkPropsMap *sx_property_map = qk_property_map_new();
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *sx_props = qk_instruction_properties_new(35.5e-9, 0.);
        qk_property_map_add(sx_property_map, qargs, 1, sx_props);
    }
    qk_target_add_instruction(target, QkGate_SX, NULL, sx_property_map);

    QkPropsMap *x_property_map = qk_property_map_new();
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *x_props = qk_instruction_properties_new(35.5e-9, 0.0005);
        qk_property_map_add(x_property_map, qargs, 1, x_props);
    }
    qk_target_add_instruction(target, QkGate_X, NULL, x_property_map);

    QkPropsMap *cx_property_map = qk_property_map_new();
    uint32_t qarg_samples[8][2] = {
        {3, 4}, {4, 3}, {3, 1}, {1, 3}, {1, 2}, {2, 1}, {0, 1}, {1, 0},
    };
    QkInstructionProps *props[8] = {
        qk_instruction_properties_new(2.7022e-11, 0.00713),
        qk_instruction_properties_new(3.0577 - 11, 0.00713),
        qk_instruction_properties_new(4.6222 - 11, 0.00929),
        qk_instruction_properties_new(4.9777 - 11, 0.00929),
        qk_instruction_properties_new(2.2755 - 11, 0.00659),
        qk_instruction_properties_new(2.6311 - 11, 0.00659),
        qk_instruction_properties_new(5.1911 - 11, 0.01201),
        qk_instruction_properties_new(5.1911 - 11, 0.01201),
    };
    for (int i = 0; i < 8; i++) {
        qk_property_map_add(cx_property_map, qarg_samples[i], 2, props[i]);
    }
    qk_target_add_instruction(target, QkGate_CX, NULL, cx_property_map);

    // Add global Y Gate
    qk_target_add_instruction(target, QkGate_Y, NULL, qk_property_map_new());

    // Retrieve instruction names by qargs for {0,}
    uint32_t qargs_1[1] = {0};
    QkGate names[5] = {QkGate_I, QkGate_RZ, QkGate_SX, QkGate_X, QkGate_Y};
    QkTargetOpsList names_obtained = qk_target_operations_for_qargs(target, qargs_1, 1);
    // Sort arrays
    sort_gate_array(names, 5);
    sort_gate_array(names_obtained.list, names_obtained.length);
    for (int i = 0; i < names_obtained.length; i++) {
        if (names[i] != names_obtained.list[i]) {
            printf("Mismatch in operation names order: %hhu is not %hhu.", names[i],
                   names_obtained.list[i]);
            result = RuntimeError;
            goto cleanup;
        }
    }

    // Retrieve instruction names by qargs for {0,1}
    uint32_t qargs_2[2] = {0, 1};
    QkGate two_names[1] = {QkGate_CX};
    QkTargetOpsList two_names_obtained = qk_target_operations_for_qargs(target, qargs_2, 2);
    for (int i = 0; i < two_names_obtained.length; i++) {
        if (two_names[i] != two_names_obtained.list[i]) {
            printf("Mismatch in operation names order: %hhu is not %hhu.", two_names[i],
                   two_names_obtained.list[i]);
            result = RuntimeError;
            goto cleanup;
        }
    }

    QkGate global_names[1] = {QkGate_Y};
    QkTargetOpsList global_names_obtained = qk_target_operations_for_qargs(target, NULL, 1);
    for (int i = 0; i < global_names_obtained.length; i++) {
        if (global_names[i] != global_names_obtained.list[i]) {
            printf("Mismatch in operation names order: %hhu is not %hhu.", global_names[i],
                   global_names_obtained.list[i]);
            result = RuntimeError;
            goto cleanup;
        }
    }

cleanup:
    qk_target_free(target);
    qk_property_map_free(i_property_map);
    qk_property_map_free(sx_property_map);
    qk_property_map_free(x_property_map);
    qk_property_map_free(rz_property_map);
    qk_property_map_free(cx_property_map);

    for (int i = 0; i < 8; i++) {
        qk_instruction_properties_free(props[i]);
    }
    return result;
}

/**
 * Test retrieving qargs based on the name of an operation that has
 * defned properties for them.
 */
int test_target_qargs_for_operation_names(void) {
    // Build sample target
    QkTarget *target = qk_target_new(0);
    int result = Ok;
    QkPropsMap *i_property_map = qk_property_map_new();
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *i_props = qk_instruction_properties_new(35.5e-9, 0.);
        qk_property_map_add(i_property_map, qargs, 1, i_props);
    }
    qk_target_add_instruction(target, QkGate_I, NULL, i_property_map);

    QkPropsMap *rz_property_map = qk_property_map_new();
    double rz_params[1] = {3.14};
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *rz_props = qk_instruction_properties_new(0., 0.);
        qk_property_map_add(rz_property_map, qargs, 1, rz_props);
    }
    qk_target_add_instruction(target, QkGate_RZ, rz_params, i_property_map);

    QkPropsMap *sx_property_map = qk_property_map_new();
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *sx_props = qk_instruction_properties_new(35.5e-9, 0.);
        qk_property_map_add(sx_property_map, qargs, 1, sx_props);
    }
    qk_target_add_instruction(target, QkGate_SX, NULL, sx_property_map);

    QkPropsMap *x_property_map = qk_property_map_new();
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *x_props = qk_instruction_properties_new(35.5e-9, 0.0005);
        qk_property_map_add(x_property_map, qargs, 1, x_props);
    }
    qk_target_add_instruction(target, QkGate_X, NULL, x_property_map);

    QkPropsMap *cx_property_map = qk_property_map_new();
    uint32_t qarg_samples[8][2] = {
        {3, 4}, {4, 3}, {3, 1}, {1, 3}, {1, 2}, {2, 1}, {0, 1}, {1, 0},
    };
    QkInstructionProps *props[8] = {
        qk_instruction_properties_new(2.7022e-11, 0.00713),
        qk_instruction_properties_new(3.0577 - 11, 0.00713),
        qk_instruction_properties_new(4.6222 - 11, 0.00929),
        qk_instruction_properties_new(4.9777 - 11, 0.00929),
        qk_instruction_properties_new(2.2755 - 11, 0.00659),
        qk_instruction_properties_new(2.6311 - 11, 0.00659),
        qk_instruction_properties_new(5.1911 - 11, 0.01201),
        qk_instruction_properties_new(5.1911 - 11, 0.01201),
    };
    for (int i = 0; i < 8; i++) {
        qk_property_map_add(cx_property_map, qarg_samples[i], 2, props[i]);
    }
    qk_target_add_instruction(target, QkGate_CX, NULL, cx_property_map);

    // Add global Y Gate
    qk_target_add_instruction(target, QkGate_Y, NULL, qk_property_map_new());

    // Test all single qubit instructions
    QkGate names[4] = {QkGate_I, QkGate_X, QkGate_RZ, QkGate_SX};
    for (int i = 0; i < 4; i++) {
        QkTargetQargsList i_qargs = qk_target_qargs_for_operation(target, names[i]);
        for (int j = 0; j < 4; j++) {
            uint32_t qarg_sample[1] = {j};
            QkTargetQargs qargs_obtained = i_qargs.list[j];
            if (!compare_qargs(qarg_sample, qargs_obtained.args, qargs_obtained.length)) {
                printf("Mismatch of obtained qargs for instruction '%hhu': ", names[i]);
                print_qargs(qargs_obtained.args, qargs_obtained.length);
                printf(" is not ");
                print_qargs(qarg_sample, 1);
                printf(".");
                result = RuntimeError;
                goto cleanup;
            }
        }
    }

    // Test cx (only two qubit instruction).
    QkTargetQargsList cx_qargs = qk_target_qargs_for_operation(target, QkGate_CX);
    for (int j = 0; j < 8; j++) {
        QkTargetQargs qargs_obtained = cx_qargs.list[j];
        if (!compare_qargs(qargs_obtained.args, qarg_samples[j], 2)) {
            printf("Mismatch of obtained qargs for instruction 'cx': ");
            print_qargs(qargs_obtained.args, qargs_obtained.length);
            printf(" is not ");
            print_qargs(qarg_samples[j], 2);
            printf(".");
            result = RuntimeError;
            goto cleanup;
        }
    }

    // Test y (only global operation)
    QkTargetQargsList y_qargs = qk_target_qargs_for_operation(target, QkGate_Y);
    if (y_qargs.list != NULL) {
        printf("Mismatch of obtained qargs for instruction 'cx': Did not receive NULL but %p.",
               y_qargs.list);
        result = RuntimeError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    qk_property_map_free(i_property_map);
    qk_property_map_free(sx_property_map);
    qk_property_map_free(x_property_map);
    qk_property_map_free(rz_property_map);
    qk_property_map_free(cx_property_map);

    for (int i = 0; i < 8; i++) {
        qk_instruction_properties_free(props[i]);
    }
    return result;
}

/**
 * Test retrieving all of the qargs in the Target.
 */
int test_target_qargs(void) {
    // Build sample target
    QkTarget *target = qk_target_new(0);
    int result = Ok;
    QkPropsMap *i_property_map = qk_property_map_new();
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *i_props = qk_instruction_properties_new(35.5e-9, 0.);
        qk_property_map_add(i_property_map, qargs, 1, i_props);
    }
    qk_target_add_instruction(target, QkGate_I, NULL, i_property_map);

    QkPropsMap *rz_property_map = qk_property_map_new();
    double rz_params[1] = {3.14};
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *rz_props = qk_instruction_properties_new(0., 0.);
        qk_property_map_add(rz_property_map, qargs, 1, rz_props);
    }
    qk_target_add_instruction(target, QkGate_RZ, rz_params, i_property_map);

    QkPropsMap *sx_property_map = qk_property_map_new();
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *sx_props = qk_instruction_properties_new(35.5e-9, 0.);
        qk_property_map_add(sx_property_map, qargs, 1, sx_props);
    }
    qk_target_add_instruction(target, QkGate_SX, NULL, sx_property_map);

    QkPropsMap *x_property_map = qk_property_map_new();
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        QkInstructionProps *x_props = qk_instruction_properties_new(35.5e-9, 0.0005);
        qk_property_map_add(x_property_map, qargs, 1, x_props);
    }
    qk_target_add_instruction(target, QkGate_X, NULL, x_property_map);

    QkPropsMap *cx_property_map = qk_property_map_new();
    uint32_t qarg_samples[8][2] = {
        {3, 4}, {4, 3}, {3, 1}, {1, 3}, {1, 2}, {2, 1}, {0, 1}, {1, 0},
    };
    QkInstructionProps *props[8] = {
        qk_instruction_properties_new(2.7022e-11, 0.00713),
        qk_instruction_properties_new(3.0577 - 11, 0.00713),
        qk_instruction_properties_new(4.6222 - 11, 0.00929),
        qk_instruction_properties_new(4.9777 - 11, 0.00929),
        qk_instruction_properties_new(2.2755 - 11, 0.00659),
        qk_instruction_properties_new(2.6311 - 11, 0.00659),
        qk_instruction_properties_new(5.1911 - 11, 0.01201),
        qk_instruction_properties_new(5.1911 - 11, 0.01201),
    };
    for (int i = 0; i < 8; i++) {
        qk_property_map_add(cx_property_map, qarg_samples[i], 2, props[i]);
    }
    qk_target_add_instruction(target, QkGate_CX, NULL, cx_property_map);

    // Add global Y Gate
    qk_target_add_instruction(target, QkGate_Y, NULL, qk_property_map_new());

    // Check all qargs
    QkTargetQargsList all_qargs = qk_target_qargs(target);
    for (int i = 0; i < 13; i++) {
        QkTargetQargs qargs_obtained = all_qargs.list[i];
        if (i < 4) {
            // First qargs were single qubit operation and should preserve
            // their order.
            uint32_t qargs[1] = {
                i,
            };
            if (!compare_qargs(qargs_obtained.args, qargs, 1)) {
                printf("Mismatch of obtained qargs: ");
                print_qargs(qargs_obtained.args, 1);
                printf(" is not ");
                print_qargs(qargs, 1);
                printf(".");
                result = RuntimeError;
                goto cleanup;
            }
        } else if (i < 12) {
            // Next were from adding the cx gate, so all will be two
            // qubit operations.
            if (!compare_qargs(qargs_obtained.args, qarg_samples[i - 4], 2)) {
                printf("Mismatch of obtained qargs: ");
                print_qargs(qargs_obtained.args, 2);
                printf(" is not ");
                print_qargs(qarg_samples[i - 4], 2);
                printf(".");
                result = RuntimeError;
                goto cleanup;
            }
        } else {
            // Finally a NULL from the global operation Y.
            if (qargs_obtained.args != NULL) {
                printf("Mismatch of obtained qargs: %p is not NULL", qargs_obtained.args);
                result = RuntimeError;
                goto cleanup;
            }
        }
    }

cleanup:
    qk_target_free(target);
    qk_property_map_free(i_property_map);
    qk_property_map_free(sx_property_map);
    qk_property_map_free(x_property_map);
    qk_property_map_free(rz_property_map);
    qk_property_map_free(cx_property_map);

    for (int i = 0; i < 8; i++) {
        qk_instruction_properties_free(props[i]);
    }
    return result;
}

int test_target(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty_target);
    num_failed += RUN_TEST(test_target_construct);
    num_failed += RUN_TEST(test_instruction_properties_construction);
    num_failed += RUN_TEST(test_property_map_construction);
    num_failed += RUN_TEST(test_target_add_instruction);
    num_failed += RUN_TEST(test_target_update_instruction);
    num_failed += RUN_TEST(test_target_non_global_op_names);
    num_failed += RUN_TEST(test_target_operation_names_for_qargs);
    num_failed += RUN_TEST(test_target_qargs_for_operation_names);
    num_failed += RUN_TEST(test_target_qargs);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
