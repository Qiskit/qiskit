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

QkTarget *get_u1_u2_u3_target(void);
QkTarget *get_rz_rx_target(void);
QkTarget *get_rz_sx_target(void);
QkTarget *get_rz_yx_u_target(void);
QkTarget *get_h_p_target(void);
QkTarget *get_rz_ry_u_noerror_target(void);
bool compare_gate_counts(QkOpCounts counts, char **gates, uint32_t *freq, int num_gates);
/**
 * Test running pass on chains of h gates.
 *
 * Transpile: 0:--[H]-[H]-[H]--
 */
int test_optimize_h_gates_inner(QkTarget *target, char **gates, uint32_t *freq, int num_gates) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    for (int iter = 0; iter < 3; iter++) {
        qk_circuit_gate(circuit, QkGate_H, qubits, NULL);
    }

    // Run transpiler pass
    QkCircuit *circuit_result =
        qk_transpiler_standalone_optimize_1q_gates_decomposition(circuit, target);
    if (!compare_gate_counts(qk_circuit_count_ops(circuit_result), gates, freq, num_gates)) {
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(circuit);
    qk_target_free(target);
    return result;
}

int test_optimize_h_gates(void) {
    int num_failed = 0;
    QkTarget *targets[6] = {
        get_u1_u2_u3_target(), get_rz_rx_target(), get_rz_sx_target(),
        get_rz_yx_u_target(),  get_h_p_target(),   get_rz_ry_u_noerror_target(),
    };
    char *gates[6][2] = {{
                             "u2",
                         },
                         {"rz", "rx"},
                         {"rz", "sx"},
                         {"u"},
                         {"h"},
                         {"u"}};

    uint32_t freq[6][2] = {
        {
            1,
        },
        {2, 1},
        {2, 1},
        {1},
        {3},
        {1},
    };

    int num_gates[6] = {1, 2, 2, 1, 1, 1};

    for (int idx = 0; idx < 6; idx++) {
        printf("Call #%u.\n", idx);
        num_failed +=
            test_optimize_h_gates_inner(targets[idx], gates[idx], freq[idx], num_gates[idx]);
    }
    return num_failed;
}

int test_optimize_identity_target_inner(QkTarget *target) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    double params_pos[1] = {3.14 / 7.};
    double params_neg[1] = {-3.14 / 7.};
    qk_circuit_gate(circuit, QkGate_RY, qubits, params_pos);
    qk_circuit_gate(circuit, QkGate_RY, qubits, params_neg);

    // Run transpiler pass
    QkCircuit *circuit_result =
        qk_transpiler_standalone_optimize_1q_gates_decomposition(circuit, target);
    if (qk_circuit_count_ops(circuit_result).len != 0) {
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(circuit);
    qk_target_free(target);
    return result;
}

int test_optimize_identity_target(void) {
    int num_failed = 0;
    QkTarget *targets[4] = {
        get_u1_u2_u3_target(),
        get_rz_rx_target(),
        get_rz_sx_target(),
        get_rz_yx_u_target(),
    };
    for (int idx = 0; idx < 4; idx++) {
        printf("Call #%u.\n", idx);
        num_failed += test_optimize_identity_target_inner(targets[idx]);
    }
    return num_failed;
}

int test_optimize_1q_decomposition(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_optimize_h_gates);
    num_failed += RUN_TEST(test_optimize_identity_target);
    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}

/// @brief Generates a typical target where u1 is cheaper than u2 is cheaper than u3.
/// @return The generated target instance.
QkTarget *get_u1_u2_u3_target(void) {
    QkTarget *target_u1_u2_u3 = qk_target_new(1);

    double u_errors[3] = {0., 1e-4, 1e-4};
    QkGate u_gates[3] = {QkGate_U1, QkGate_U2, QkGate_U3};
    // TODO: Update this part to use parameters once we support them.
    double u1_params[1] = {3.14};
    double u2_params[2] = {3.14, 3.14 / 2.};
    double u3_params[3] = {3.14, 3.14 / 2., 3.14 / 4.};

    double *u_params[3] = {u1_params, u2_params, u3_params};
    for (int idx = 0; idx < 3; idx++) {
        QkTargetEntry *u_entry = qk_target_entry_new_fixed(u_gates[idx], u_params[idx]);
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
    // TODO: Update this part to use parameters once we support them.
    double r_params[1] = {3.14};

    for (int idx = 0; idx < 2; idx++) {
        QkTargetEntry *r_entry = qk_target_entry_new_fixed(r_gates[idx], r_params);
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
    // TODO: Update this part to use parameters once we support them.
    double rz_params[1] = {3.14};

    for (int idx = 0; idx < 2; idx++) {
        QkTargetEntry *entry;
        if (gates[idx] == QkGate_RZ) {
            entry = qk_target_entry_new_fixed(gates[idx], rz_params);
        } else {
            entry = qk_target_entry_new(gates[idx]);
        }
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
QkTarget *get_rz_yx_u_target(void) {
    QkTarget *target_rz_ry_u = qk_target_new(1);

    double gate_errors[3] = {1e-4, 2e-4, 5e-4};
    double gate_durations[3] = {1e-9, 5e-9, 9e-9};
    QkGate u_gates[3] = {QkGate_RZ, QkGate_RY, QkGate_U};
    // TODO: Update this part to use parameters once we support them.
    double rz_params[1] = {3.14};
    double rx_params[1] = {3.14};
    double u_params[3] = {3.14, 3.14 / 2., 3.14 / 4.};

    double *gate_params[3] = {rz_params, rx_params, u_params};
    for (int idx = 0; idx < 3; idx++) {
        QkTargetEntry *u_entry = qk_target_entry_new_fixed(u_gates[idx], gate_params[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(u_entry, qargs, 1, gate_durations[idx], gate_errors[idx]);
        qk_target_add_instruction(target_rz_ry_u, u_entry);
    }
    return target_rz_ry_u;
}

/// @brief Generates a target with hadamard and phase, we don't yet have an explicit decomposer
/// but we can at least recognize circuits that are native for it
/// @return The generated target instance.
QkTarget *get_h_p_target(void) {
    QkTarget *target_h_p = qk_target_new(1);

    double inst_durations[2] = {3e-9, 0.};
    double inst_errors[2] = {3e-4, 0.};
    QkGate gates[2] = {QkGate_H, QkGate_Phase};
    // TODO: Update this part to use parameters once we support them.
    double phase_params[1] = {3.14};

    for (int idx = 0; idx < 2; idx++) {
        QkTargetEntry *entry;
        if (gates[idx] == QkGate_Phase) {
            entry = qk_target_entry_new_fixed(gates[idx], phase_params);
        } else {
            entry = qk_target_entry_new(gates[idx]);
        }
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(entry, qargs, 1, inst_durations[idx], inst_errors[idx]);
        qk_target_add_instruction(target_h_p, entry);
    }
    return target_h_p;
}

/// @brief Generates a target with rz, ry, and u. Error are not specified so we should prefer
/// shorter decompositions.
/// @return The generated target instance.
QkTarget *get_rz_ry_u_noerror_target(void) {
    QkTarget *target_rz_ry_u_noerror = qk_target_new(1);
    QkGate u_gates[3] = {QkGate_RZ, QkGate_RY, QkGate_U};
    // TODO: Update this part to use parameters once we support them.
    double rz_params[1] = {3.14};
    double rx_params[1] = {3.14};
    double u_params[3] = {3.14, 3.14 / 2., 3.14 / 4.};

    double *gate_params[3] = {rz_params, rx_params, u_params};
    for (int idx = 0; idx < 3; idx++) {
        QkTargetEntry *u_entry = qk_target_entry_new_fixed(u_gates[idx], gate_params[idx]);
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
