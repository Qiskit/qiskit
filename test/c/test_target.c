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

/**
 * Test empty constructor for Target
 */
int test_empty_target(void) {
    int result = Ok;
    QkTarget *target = qk_target_new(0);
    uint32_t num_qubits = qk_target_num_qubits(target);

    if (num_qubits != 0) {
        printf("The number of qubits %u is not 0.", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    double retrieved_dt = qk_target_dt(target);
    if (!isnan(retrieved_dt)) {
        printf("The dt value of this target %f is not %f.", retrieved_dt, NAN);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t retrieved_granularity = qk_target_granularity(target);
    if (retrieved_granularity != 1) {
        printf("The granularity %u is not 1.", retrieved_granularity);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t retrieved_min_length = qk_target_min_length(target);
    if (retrieved_min_length != 1) {
        printf("The min_length %u is not 1.", retrieved_min_length);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t pulse_alignment = qk_target_pulse_alignment(target);
    if (pulse_alignment != 1) {
        printf("The pulse_alignment values %u is not 1.", pulse_alignment);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t acquire_alignment = qk_target_acquire_alignment(target);
    if (acquire_alignment != 0) {
        printf("The acquire_alignment values %u is not 0.", acquire_alignment);
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test constructor for Target
 */
int test_target_construct(void) {
    int result = Ok;
    const uint32_t num_qubits = 2;
    const double dt = 10e-9;
    const uint32_t granularity = 2;
    const uint32_t min_length = 3;
    const uint32_t p_alignment = 4;
    const uint32_t a_alignment = 1;

    QkTarget *target = qk_target_new(num_qubits);

    qk_target_set_dt(target, dt);
    qk_target_set_granularity(target, granularity);
    qk_target_set_min_length(target, min_length);
    qk_target_set_pulse_alignment(target, p_alignment);
    qk_target_set_acquire_alignment(target, a_alignment);

    uint32_t retrieved_num_qubits = qk_target_num_qubits(target);
    if (retrieved_num_qubits != 2) {
        printf("The number of qubits %u is not 0.", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    double retrieved_dt = qk_target_dt(target);
    if (retrieved_dt != dt) {
        printf("The dt value of this target %f is not %f.", retrieved_dt, dt);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t retrieved_granularity = qk_target_granularity(target);
    if (retrieved_granularity != 2) {
        printf("The granularity %u is not 2.", retrieved_granularity);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t retrieved_min_length = qk_target_min_length(target);
    if (retrieved_min_length != 3) {
        printf("The min_length %u is not 3.", retrieved_min_length);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t pulse_alignment = qk_target_pulse_alignment(target);
    if (pulse_alignment != 4) {
        printf("The pulse_alignment values %u is not 4.", pulse_alignment);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t acquire_alignment = qk_target_acquire_alignment(target);
    if (acquire_alignment != 1) {
        printf("The acquire_alignment values %u is not 1.", acquire_alignment);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test construction of a QkTargetEntry
 */
int test_target_entry_construction(void) {
    int result = Ok;
    QkTargetEntry *property_map = qk_target_entry_new(QkGate_CX);

    // Test length
    const size_t length = qk_target_entry_num_properties(property_map);
    if (length != 0) {
        printf("The initial length of the provided property map was not zero: %zu", length);
        result = EqualityError;
        goto cleanup;
    }

    // Add some qargs and properties
    uint32_t qargs[2] = {0, 1};

    QkExitCode result_prop = qk_target_entry_add_property(property_map, qargs, 2, 0.00018, 0.00002);
    if (result_prop != QkExitCode_Success) {
        printf("Unexpected error occurred when adding entry.");
    }

    // Test length
    const size_t new_length = qk_target_entry_num_properties(property_map);
    if (new_length != 1) {
        printf("The initial length of the provided property map was not 1: %zu", length);
        result = EqualityError;
        goto cleanup;
    }

    // Add invalid qargs
    // Add some qargs and properties
    uint32_t invalid_qargs[3] = {0, 1, 2};

    QkExitCode error_result =
        qk_target_entry_add_property(property_map, invalid_qargs, 3, 0.00018, 0.00002);
    if (error_result != QkExitCode_TargetQargMismatch) {
        printf("The operation did not fail as expected for invalid qargs.");
    }

cleanup:
    qk_target_entry_free(property_map);
    return result;
}

/**
 * Test adding an instruction to the Target.
 */
int test_target_add_instruction(void) {
    const uint32_t num_qubits = 1;
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
    if (current_num_qubits != 1) {
        printf("The number of qubits this target is compatible with is not 1: %u",
               current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }
    size_t current_size = qk_target_num_instructions(target);
    if (current_num_qubits != 1) {
        printf("The size of this target is not correct: Expected 1, got %zu", current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    uint32_t qargs[2] = {0, 1};
    double inst_error = 0.0090393;
    double inst_duration = 0.020039;

    QkExitCode result_cx_props =
        qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
    if (result_cx_props != QkExitCode_Success) {
        printf("Unexpected error occurred when adding property to a CX gate entry.");
        result = EqualityError;
        goto cleanup;
    }

    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a CX gate.");
        result = EqualityError;
        goto cleanup;
    }

    // Number of qubits of the target should change to 2.
    current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 2) {
        printf("The number of qubits this target is compatible with is not 2: %u",
               current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }
    current_size = qk_target_num_instructions(target);
    if (current_size != 2) {
        printf("The size of this target is not correct: Expected 2, got %zu", current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a CRX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    double crx_params[1] = {3.14};
    QkTargetEntry *crx_entry = qk_target_entry_new_fixed(QkGate_CRX, crx_params);
    uint32_t crx_qargs[2] = {1, 2};
    double crx_inst_error = 0.0129023;
    double crx_inst_duration = 0.92939;
    QkExitCode result_crx_props =
        qk_target_entry_add_property(crx_entry, crx_qargs, 2, crx_inst_duration, crx_inst_error);
    if (result_crx_props != QkExitCode_Success) {
        printf("Unexpected error occurred when adding property to a CX gate entry.");
        result = EqualityError;
        goto cleanup;
    }

    QkExitCode result_crx = qk_target_add_instruction(target, crx_entry);
    if (result_crx != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a CX gate.");
        result = EqualityError;
        goto cleanup;
    }

    // Number of qubits of the target should change to 3.
    current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 3) {
        printf("The number of qubits this target is compatible with is not 3: %d",
               current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }
    current_size = qk_target_num_instructions(target);
    if (current_size != 3) {
        printf("The size of this target is not correct: Expected 3, got %zu", current_size);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test updating an instruction property in the Target using
 * `update_instruction_property`.
 */
int test_target_update_instruction(void) {
    const uint32_t num_qubits = 1;
    // Let's create a target with one qubit for now
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    uint32_t qargs[2] = {0, 1};
    double inst_error = 0.0090393;
    double inst_duration = 0.020039;
    qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
    // CX Gate is not paramtric. Re-use Null
    qk_target_add_instruction(target, cx_entry);

    // change the intruction property of cx
    double cx_new_inst_error = NAN;
    double cx_new_inst_duration = 0.09457;
    QkExitCode result_1 = qk_target_update_property(target, QkGate_CX, qargs, 2,
                                                    cx_new_inst_duration, cx_new_inst_error);
    if (result_1 != QkExitCode_Success) {
        printf("An unexpected error occured while modifying the property.");
        result = RuntimeError;
        goto cleanup;
    }

    // Try to modify wrong instruction
    QkExitCode result_2 = qk_target_update_property(target, QkGate_CH, qargs, 2,
                                                    cx_new_inst_duration, cx_new_inst_error);
    if (result_2 != QkExitCode_TargetInvalidInstKey) {
        printf("The function did not fail as expected when querying the wrong instruction.");
        result = RuntimeError;
        goto cleanup;
    }

    uint32_t new_qargs[2] = {1, 2};
    // Try to modify wrong qargs
    QkExitCode result_3 = qk_target_update_property(target, QkGate_CX, new_qargs, 2,
                                                    cx_new_inst_duration, cx_new_inst_error);
    if (result_3 != QkExitCode_TargetInvalidQargsKey) {
        printf("The function did not fail as expected when querying with wrong qargs.");
        result = RuntimeError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    return result;
}

int test_target(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty_target);
    num_failed += RUN_TEST(test_target_construct);
    num_failed += RUN_TEST(test_target_entry_construction);
    num_failed += RUN_TEST(test_target_add_instruction);
    num_failed += RUN_TEST(test_target_update_instruction);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
