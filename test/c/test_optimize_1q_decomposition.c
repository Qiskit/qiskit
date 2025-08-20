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

// Helper headers
QkTarget *get_u1_u2_u3_target(void);
QkTarget *get_rz_rx_target(void);
QkTarget *get_rz_sx_target(void);
QkTarget *get_rz_ry_u_target(void);
QkTarget *get_rz_ry_u_noerror_target(void);
bool compare_gate_counts(QkOpCounts counts, char **gates, uint32_t *freq, int num_gates);

/**
 * Test running pass on chains of h gates.
 *
 * Transpile: 0:--[H]-[H]-[H]--
 */
int inner_optimize_h_gates(QkTarget *target, char **gates, uint32_t *freq, int num_gates) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    for (int iter = 0; iter < 3; iter++) {
        qk_circuit_gate(circuit, QkGate_H, qubits, NULL);
    }

    // Run transpiler pass
    qk_transpiler_standalone_optimize_1q_gates_decomposition(circuit, target);
    QkOpCounts counts = qk_circuit_count_ops(circuit);
    if (!compare_gate_counts(counts, gates, freq, num_gates)) {
        result = EqualityError;
    }

    qk_opcounts_free(counts);
    qk_circuit_free(circuit);
    qk_target_free(target);
    return result;
}

int test_optimize_h_gates(void) {
    int num_failed = 0;
    QkTarget *targets[5] = {
        get_u1_u2_u3_target(), get_rz_rx_target(),           get_rz_sx_target(),
        get_rz_ry_u_target(),  get_rz_ry_u_noerror_target(),
    };
    char *gates[5][2] = {{
                             "u2",
                         },
                         {"rz", "rx"},
                         {"rz", "sx"},
                         {"u"},
                         {"u"}};

    uint32_t freq[5][2] = {
        {
            1,
        },
        {2, 1},
        {2, 1},
        {1},
        {1},
    };

    int num_gates[5] = {1, 2, 2, 1, 1};
    char *names[5] = {"u1_u2_u3", "rz_rx", "rz_sx", "rz_ry_u", "rz_ry_u_noerror"};
    printf("Optimize h gates tests.\n");
    for (int idx = 0; idx < 5; idx++) {
        int result = inner_optimize_h_gates(targets[idx], gates[idx], freq[idx], num_gates[idx]);
        printf("--- Run with %-21s: %s \n", names[idx], (bool)result ? "Fail" : "Ok");
        num_failed += result;
    }
    return num_failed;
}

int inner_optimize_identity_target(QkTarget *target) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    double params_pos[1] = {3.14 / 7.};
    double params_neg[1] = {-3.14 / 7.};
    qk_circuit_gate(circuit, QkGate_RY, qubits, params_pos);
    qk_circuit_gate(circuit, QkGate_RY, qubits, params_neg);

    QkOpCounts counts = qk_circuit_count_ops(circuit);
    // Run transpiler pass
    qk_transpiler_standalone_optimize_1q_sequences(circuit, target);
    if (counts.len != 0) {
        result = EqualityError;
    }

    qk_opcounts_free(counts);
    qk_circuit_free(circuit);
    qk_target_free(target);
    return result;
}

/**
 * Transpile: qr:--[RY(θ), RY(-θ)]-- to null.
 */
int test_optimize_identity_target(void) {
    int num_failed = 0;
    QkTarget *targets[4] = {
        get_u1_u2_u3_target(),
        get_rz_rx_target(),
        get_rz_sx_target(),
        get_rz_ry_u_target(),
    };
    char *names[4] = {
        "u1_u2_u3",
        "rz_rx",
        "rz_sx",
        "rz_ry_u",
    };
    printf("Optimize identities with target tests.\n");
    for (int idx = 0; idx < 4; idx++) {
        int result = inner_optimize_identity_target(targets[idx]);
        printf("--- Run with %-21s: %s \n", names[idx], (bool)result ? "Fail" : "Ok");
        num_failed += result;
    }
    return num_failed;
}

/**
 * Test identity run is removed for no target specified.
 */
int test_optimize_identity_no_target(void) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    for (int iter = 0; iter < 2; iter++) {
        qk_circuit_gate(circuit, QkGate_H, qubits, NULL);
    }

    QkOpCounts counts = qk_circuit_count_ops(circuit);
    // Run transpiler pass
    qk_transpiler_standalone_optimize_1q_sequences(circuit, NULL);
    if (counts.len != 0) {
        result = EqualityError;
    }

    qk_opcounts_free(counts);
    qk_circuit_free(circuit);
    return result;
}

/**
 * U is shorter than RZ-RY-RZ or RY-RZ-RY so use it when no error given.
 */
int test_optimize_error_over_target_3(void) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    double params[3] = {3.14 / 7., 3.14 / 4., 3.14 / 3.};
    qk_circuit_gate(circuit, QkGate_U, qubits, params);
    QkTarget *target = get_rz_ry_u_noerror_target();
    // Run transpiler pass
    qk_transpiler_standalone_optimize_1q_sequences(circuit, target);
    QkOpCounts counts = qk_circuit_count_ops(circuit);
    if (counts.len != 1) {
        result = EqualityError;
        goto cleanup;
    }
    if (strcmp(counts.data[0].name, "u") != 0 || counts.data[0].count != 1) {
        result = EqualityError;
    }

cleanup:
    qk_opcounts_free(counts);
    qk_target_free(target);
    qk_circuit_free(circuit);
    return result;
}

/// @brief Generates a typical target where u1 is cheaper than u2 is cheaper than u3.
/// @return The generated target instance.
QkTarget *get_u1_u2_u3_target(void) {
    QkTarget *target_u1_u2_u3 = qk_target_new(1);

    double u_errors[3] = {0., 1e-4, 1e-4};
    QkGate u_gates[3] = {QkGate_U1, QkGate_U2, QkGate_U3};
    for (int idx = 0; idx < 3; idx++) {
        QkTargetEntry *u_entry = qk_target_entry_new(u_gates[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(u_entry, qargs, 1, NAN, u_errors[idx]);
        qk_target_add_instruction(target_u1_u2_u3, u_entry);
    }
    return target_u1_u2_u3;
}

/// @brief Generates a typical target where continuous rz and rx are available; rz is cheaper.
/// @return The generated target instance.
QkTarget *get_rz_rx_target(void) {
    QkTarget *target_rz_rx = qk_target_new(1);

    double r_errors[2] = {0., 2.5e-4};
    double r_durations[2] = {0., 5e-9};
    QkGate r_gates[2] = {QkGate_RZ, QkGate_RX};

    for (int idx = 0; idx < 2; idx++) {
        QkTargetEntry *r_entry = qk_target_entry_new(r_gates[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(r_entry, qargs, 1, r_durations[idx], r_errors[idx]);
        qk_target_add_instruction(target_rz_rx, r_entry);
    }
    return target_rz_rx;
}

/// @brief Generates a typical target where continuous rz, and discrete sx are available; rz is
/// cheaper.
/// @return The generated target instance.
QkTarget *get_rz_sx_target(void) {
    QkTarget *target_rz_sx = qk_target_new(1);

    double inst_errors[2] = {0., 2.5e-4};
    double inst_durations[2] = {0., 5e-9};
    QkGate gates[2] = {QkGate_RZ, QkGate_SX};

    for (int idx = 0; idx < 2; idx++) {
        QkTargetEntry *entry;
        entry = qk_target_entry_new(gates[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(entry, qargs, 1, inst_durations[idx], inst_errors[idx]);
        qk_target_add_instruction(target_rz_sx, entry);
    }
    return target_rz_sx;
}

/// @brief Generates a target with overcomplete basis, rz is cheaper than ry is cheaper than u.
/// @return The generated target instance.
QkTarget *get_rz_ry_u_target(void) {
    QkTarget *target_rz_ry_u = qk_target_new(1);

    double gate_errors[3] = {1e-4, 2e-4, 5e-4};
    double gate_durations[3] = {1e-9, 5e-9, 9e-9};
    QkGate u_gates[3] = {QkGate_RZ, QkGate_RY, QkGate_U};

    for (int idx = 0; idx < 3; idx++) {
        QkTargetEntry *u_entry = qk_target_entry_new(u_gates[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(u_entry, qargs, 1, gate_durations[idx], gate_errors[idx]);
        qk_target_add_instruction(target_rz_ry_u, u_entry);
    }
    return target_rz_ry_u;
}

/// @brief Generates a target with rz, ry, and u. Error are not specified so we should prefer
/// shorter decompositions.
/// @return The generated target instance.
QkTarget *get_rz_ry_u_noerror_target(void) {
    QkTarget *target_rz_ry_u_noerror = qk_target_new(1);
    QkGate u_gates[3] = {QkGate_RZ, QkGate_RY, QkGate_U};

    for (int idx = 0; idx < 3; idx++) {
        QkTargetEntry *u_entry = qk_target_entry_new(u_gates[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_add_instruction(target_rz_ry_u_noerror, u_entry);
    }
    return target_rz_ry_u_noerror;
}

bool compare_gate_counts(QkOpCounts counts, char **gates, uint32_t *freq, int num_gates) {
    if (counts.len != num_gates) {
        return false;
    }
    for (int idx = 0; idx < counts.len; idx++) {
        QkOpCount current = counts.data[idx];
        if (strcmp(current.name, gates[idx]) != 0 || current.count != freq[idx]) {
            return false;
        }
    }
    return true;
}

int test_optimize_1q_decomposition(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_optimize_h_gates);
    num_failed += RUN_TEST(test_optimize_identity_target);
    num_failed += RUN_TEST(test_optimize_identity_no_target);
    num_failed += RUN_TEST(test_optimize_error_over_target_3);
    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}