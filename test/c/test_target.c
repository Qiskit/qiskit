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

QkTarget *create_sample_target(bool std_inst);
/**
 * Test empty constructor for Target
 */
static int test_empty_target(void) {
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
 * Test constructor for Target
 */
static int test_target_construct(void) {
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

/*
 * Test target construction with a parameterized gate
 */
static int test_target_construction_ibm_like_target(void) {
    int result = Ok;
    QkTarget *target = qk_target_new(5);
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    uint32_t cx_qargs[2] = {0, 1};
    QkExitCode result_prop = qk_target_entry_add_property(cx_entry, cx_qargs, 2, 2.2e-4, 6.2e-9);
    if (result_prop != QkExitCode_Success) {
        printf("Unexpected error occurred when adding entry.");
        qk_target_entry_free(cx_entry);
        result = EqualityError;
        goto cleanup;
    }
    cx_qargs[0] = 2;
    result_prop = qk_target_entry_add_property(cx_entry, cx_qargs, 2, 3.2e-4, 4.2e-9);
    if (result_prop != QkExitCode_Success) {
        printf("Unexpected error occurred when adding entry.");
        qk_target_entry_free(cx_entry);
        result = EqualityError;
        goto cleanup;
    }
    cx_qargs[1] = 3;
    result_prop = qk_target_entry_add_property(cx_entry, cx_qargs, 2, 1.2e-4, 4.2e-8);
    if (result_prop != QkExitCode_Success) {
        printf("Unexpected error occurred when adding entry.");
        qk_target_entry_free(cx_entry);
        result = EqualityError;
        goto cleanup;
    }
    cx_qargs[0] = 4;
    result_prop = qk_target_entry_add_property(cx_entry, cx_qargs, 2, 1.2e-3, 2.2e-8);
    if (result_prop != QkExitCode_Success) {
        printf("Unexpected error occurred when adding entry.");
        qk_target_entry_free(cx_entry);
        result = EqualityError;
        goto cleanup;
    }
    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a CX gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkTargetEntry *rz_entry = qk_target_entry_new(QkGate_RZ);
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t qargs[1] = {i};
        result_prop = qk_target_entry_add_property(rz_entry, qargs, 1, 0, 0);
        if (result_prop != QkExitCode_Success) {
            printf("Unexpected error occurred when adding entry.");
            result = EqualityError;
            qk_target_entry_free(rz_entry);
            goto cleanup;
        }
    }
    QkExitCode result_rz = qk_target_add_instruction(target, rz_entry);
    if (result_rz != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a parameterized RZ gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkTargetEntry *sx_entry = qk_target_entry_new(QkGate_SX);
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t qargs[1] = {i};
        result_prop = qk_target_entry_add_property(sx_entry, qargs, 1, 1.928e-10, 7.9829e-11);
        if (result_prop != QkExitCode_Success) {
            printf("Unexpected error occurred when adding entry.");
            result = EqualityError;
            qk_target_entry_free(sx_entry);
            goto cleanup;
        }
    }
    QkExitCode result_sx = qk_target_add_instruction(target, sx_entry);
    if (result_sx != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a parameterized RZ gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkTargetEntry *x_entry = qk_target_entry_new(QkGate_X);
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t qargs[1] = {i};
        result_prop = qk_target_entry_add_property(x_entry, qargs, 1, 1.928e-10, 7.9829e-11);
        if (result_prop != QkExitCode_Success) {
            printf("Unexpected error occurred when adding entry.");
            result = EqualityError;
            qk_target_entry_free(x_entry);
            goto cleanup;
        }
    }
    QkExitCode result_x = qk_target_add_instruction(target, x_entry);
    if (result_x != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a parameterized RZ gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkTargetEntry *measure_entry = qk_target_entry_new_measure();
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t qargs[1] = {i};
        result_prop = qk_target_entry_add_property(measure_entry, qargs, 1, 1.928e-10, 7.9829e-11);
        if (result_prop != QkExitCode_Success) {
            printf("Unexpected error occurred when adding entry.");
            result = EqualityError;
            qk_target_entry_free(measure_entry);
            goto cleanup;
        }
    }
    QkExitCode result_measure = qk_target_add_instruction(target, measure_entry);
    if (result_measure != QkExitCode_Success) {
        printf("Unexpected error occurred when adding a parameterized RZ gate.");
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
static int test_target_entry_construction(void) {
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

    // Set name for the entry
    QkExitCode set_name_error = qk_target_entry_set_name(property_map, "cx_gate");
    if (set_name_error != QkExitCode_Success) {
        printf("Unexpected error occurred when setting the name of the entry.\n");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_target_entry_free(property_map);
    return result;
}

/**
 * Test adding an instruction to the Target.
 */
static int test_target_add_instruction(void) {
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
    if (current_size != 1) {
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
    QkTargetEntry *crx_entry = qk_target_entry_new_fixed(QkGate_CRX, crx_params, "crx_fixed");
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

    // Add a measurement
    QkTargetEntry *meas = qk_target_entry_new_measure();
    for (uint32_t i = 0; i < 3; i++) {
        uint32_t q[1] = {i};
        qk_target_entry_add_property(meas, q, 1, 1e-6, 1e-4);
    }
    size_t num_meas = qk_target_entry_num_properties(meas);
    if (num_meas != 3) {
        printf("Expected 3 measurement entries but got: %zu", num_meas);
        result = EqualityError;
        qk_target_entry_free(meas);
        goto cleanup;
    }

    QkExitCode result_meas_props = qk_target_add_instruction(target, meas);
    if (result_meas_props != 0) {
        printf("Failed adding measurement instruction.");
        result = EqualityError;
        goto cleanup;
    }
    // Number of qubits of the target should remain 3.
    current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 3) {
        printf("The number of qubits this target is compatible with is not 3: %d",
               current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    current_size = qk_target_num_instructions(target);
    if (current_size != 4) {
        printf("The size of this target is not correct: Expected 4, got %zu", current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a reset
    QkTargetEntry *reset = qk_target_entry_new_reset();
    for (uint32_t i = 0; i < 3; i++) {
        uint32_t q[1] = {i};
        qk_target_entry_add_property(reset, q, 1, 2e-6, 2e-4);
    }
    size_t num_reset = qk_target_entry_num_properties(reset);
    if (num_reset != 3) {
        printf("Expected 3 reset entries but got: %zu", num_reset);
        result = EqualityError;
        qk_target_entry_free(reset);
        goto cleanup;
    }

    qk_target_add_instruction(target, reset);
    current_size = qk_target_num_instructions(target);
    if (current_size != 5) {
        printf("The size of this target is not correct: Expected 5, got %zu", current_size);
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
static int test_target_update_instruction(void) {
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
    // CX Gate is not parametric. Re-use Null
    qk_target_add_instruction(target, cx_entry);

    // Change the instruction property of cx
    double cx_new_inst_error = NAN;
    double cx_new_inst_duration = 0.09457;
    QkExitCode result_1 = qk_target_update_property(target, QkGate_CX, qargs, 2,
                                                    cx_new_inst_duration, cx_new_inst_error);
    if (result_1 != QkExitCode_Success) {
        printf("An unexpected error occurred while modifying the property.");
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

/**
 * Test if our target is compatible with certain instructions.
 */
int test_target_instruction_supported(void) {
    QkTarget *sample_target = create_sample_target(true);
    int result = Ok;

    char *gate_names[7] = {"x", "y", "id", "rz", "sx", "reset"};
    QkParam *rz_params[1] = {
        qk_param_from_double(3.14),
    };
    for (uint32_t qubit = 0; qubit < 5; qubit++) {
        uint32_t qargs[1] = {qubit};
        bool should_be_true = qubit < 4;

        for (int gate = 0; gate < 6; gate++) {
            // If i == 4 condition should be false unless we try with the y gate
            // since y is added as a global gate.
            if (qk_target_instruction_supported(sample_target, gate_names[gate], qargs,
                                                gate != 3 ? NULL : rz_params) !=
                (should_be_true || gate == 1)) {
                printf("This target did not correctly demonstrate compatibility with %s and qargs "
                       "[%d]",
                       gate_names[gate], qubit);
                result = EqualityError;
                goto cleanup;
            }
        }

        // Try checking with the wrong fixed parameter for RZ
        QkParam *param = qk_param_from_double(1.57);
        if (qk_target_instruction_supported(sample_target, "rz", qargs,
                                            (QkParam *[]){
                                                param,
                                            })) {
            printf("This target did not correctly demonstrate compatibility with 'rz' and qargs "
                   "[%d]",
                   qubit);
            result = EqualityError;
            qk_param_free(param);
            goto cleanup;
        }

        // Test standard instructions reset and measure
        if (!(qk_target_instruction_supported(sample_target, "measure", qargs, NULL) ==
              (qubit < 2))) {
            printf(
                "This target did not correctly demonstrate compatibility with 'measure' and qargs "
                "[%d]",
                qubit);
            result = EqualityError;
            goto cleanup;
        }
    }

    // Qarg samples for CX
    uint32_t qarg_samples[8][2] = {
        {3, 4}, {4, 3}, {3, 1}, {1, 3}, {1, 2}, {2, 1}, {0, 1}, {1, 0},
    };
    for (int i = 0; i < 8; i++) {
        if (!qk_target_instruction_supported(sample_target, "cx", qarg_samples[i], NULL)) {
            printf("This target did incorrectly demonstrate compatibility with 'cx' and qargs [%d, "
                   "%d]",
                   qarg_samples[i][0], qarg_samples[i][1]);
            result = EqualityError;
            goto cleanup;
        }
    }

    uint32_t cx_qargs[2] = {3, 2};
    // Instruction should not show compatibility with (3, 2)
    if (qk_target_instruction_supported(sample_target, "cx", cx_qargs, NULL)) {
        printf("This target did incorrectly demonstrate compatibility with 'cx' and qargs [3, 2]");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_param_free(rz_params[0]);
    qk_target_free(sample_target);
    return result;
}

int test_target(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty_target);
    num_failed += RUN_TEST(test_target_construct);
    num_failed += RUN_TEST(test_target_entry_construction);
    num_failed += RUN_TEST(test_target_add_instruction);
    num_failed += RUN_TEST(test_target_update_instruction);
    num_failed += RUN_TEST(test_target_construction_ibm_like_target);
    num_failed += RUN_TEST(test_target_instruction_supported);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}

QkTarget *create_sample_target(bool std_inst) {
    // Build sample target
    QkTarget *target = qk_target_new(0);
    QkTargetEntry *i_entry = qk_target_entry_new(QkGate_I);
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        qk_target_entry_add_property(i_entry, qargs, 1, 35.5e-9, 0.);
    }
    qk_target_add_instruction(target, i_entry);

    double rz_params[1] = {3.14};
    QkTargetEntry *rz_entry = qk_target_entry_new_fixed(QkGate_RZ, rz_params, NULL);
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        qk_target_entry_add_property(rz_entry, qargs, 1, 0., 0.);
    }
    qk_target_add_instruction(target, rz_entry);

    QkTargetEntry *sx_entry = qk_target_entry_new(QkGate_SX);
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        qk_target_entry_add_property(sx_entry, qargs, 1, 35.5e-9, 0.);
    }
    qk_target_add_instruction(target, sx_entry);

    QkTargetEntry *x_entry = qk_target_entry_new(QkGate_X);
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        qk_target_entry_add_property(x_entry, qargs, 1, 35.5e-9, 0.0005);
    }
    qk_target_add_instruction(target, x_entry);

    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    uint32_t qarg_samples[8][2] = {
        {3, 4}, {4, 3}, {3, 1}, {1, 3}, {1, 2}, {2, 1}, {0, 1}, {1, 0},
    };
    double props[8][2] = {
        {2.7022e-11, 0.00713}, {3.0577e-11, 0.00713}, {4.6222e-11, 0.00929}, {4.9777e-11, 0.00929},
        {2.2755e-11, 0.00659}, {2.6311e-11, 0.00659}, {5.1911e-11, 0.01201}, {5.1911e-11, 0.01201},
    };
    for (int i = 0; i < 8; i++) {
        qk_target_entry_add_property(cx_entry, qarg_samples[i], 2, props[i][0], props[i][1]);
    }
    qk_target_add_instruction(target, cx_entry);

    // Add global Y Gate
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_Y));

    if (std_inst) {
        QkTargetEntry *meas = qk_target_entry_new_measure();
        for (uint32_t i = 0; i < 2; i++) {
            uint32_t q[1] = {i};
            qk_target_entry_add_property(meas, q, 1, 1e-6, 1e-4);
        }
        qk_target_add_instruction(target, meas);

        QkTargetEntry *reset = qk_target_entry_new_reset();
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t q[1] = {i};
            qk_target_entry_add_property(reset, q, 1, 1e-6, 1e-4);
        }
        qk_target_add_instruction(target, reset);
    }

    return target;
}