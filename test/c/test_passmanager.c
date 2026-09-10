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
#include <qiskit.h>
#include <stdio.h>
#include <string.h>

#define UNUSED_VARIABLE(x) (void)(x)

/// A struct to keep the configuration of the RemoveIdentity pass.
typedef struct {
    QkTarget *target;
} RemoveIdentity;

/// The execution function for RemoveIdentity.
void *run_remove_identity(void *self, void *ir, QkPassContext *context) {
    UNUSED_VARIABLE(context);

    RemoveIdentity *self_ = (RemoveIdentity *)self;
    qk_transpiler_pass_standalone_remove_identity_equivalent((QkCircuit *)ir, self_->target, 1.0);
    return (void *)ir;
}

/// The execution function for a circuit-to-dag pass.
void *run_circuit_to_dag(void *self, void *ir, QkPassContext *context) {
    UNUSED_VARIABLE(self);
    UNUSED_VARIABLE(context);

    QkCircuit *circuit = (QkCircuit *)ir;
    QkDag *dag = qk_circuit_to_dag(circuit);
    qk_circuit_free(circuit);
    return (void *)dag;
}

/// A callback struct keeping track of the op count.
typedef struct {
    QkOpCounts *counts;
    size_t num_counts;
    size_t current_index;
} CounterCallback;

bool trigger(void *self, uint8_t hookpoint) {
    UNUSED_VARIABLE(self);
    return (hookpoint == 0); // post pass
}

void ir_and_context(void *self, void *ir, QkPassManager *context) {
    UNUSED_VARIABLE(context);

    // cast self and the IR to the expected types
    CounterCallback *slf = (CounterCallback *)self;
    QkCircuit *circuit = (QkCircuit *)ir;

    // store the opcount in the current index and increase by one for the next call
    slf->counts[slf->current_index] = qk_circuit_count_ops(circuit);
    slf->current_index++;
}

/**
 * Test running a single RemoveIdentity pass on a circuit.
 */
static int test_circuit(void) {
    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};
    QkVTableSlot slots[1] = {{.index = 0, .ptr = (void *)(&run_remove_identity)}};
    QkPass *remove_identity = qk_pass_new((void *)(&remove_identity_config), slots, 1);

    QkPassManager *pm = qk_passmanager_new();

    int result = Ok;
    if (qk_passmanager_push_pass(pm, remove_identity) != QkExitCode_Success) {
        printf("Failed pushing pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    QkCircuit *circuit = qk_circuit_new(10, 0);
    uint32_t q0[1] = {0};
    double almost_zero[1] = {1e-20};
    double nonzero[1] = {1.23};
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit, QkGate_RX, q0, almost_zero);
    qk_circuit_gate(circuit, QkGate_RZ, q0, nonzero);
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);

    QkPassManagerResult pm_result = {NULL, NULL};
    QkExitCode exit = qk_passmanager_run(pm, (void *)circuit, NULL, &pm_result);
    if (exit != QkExitCode_Success) {
        printf("Exited with %i\n", exit);
        result = RuntimeError;
        goto cleanup_circuit;
    }

    if (pm_result.ir == NULL) {
        printf("Compilation returned NULL.\n");
        result = RuntimeError;
        goto cleanup_circuit;
    }

    QkCircuit *out = (QkCircuit *)pm_result.ir;
    QkOpCounts counts = qk_circuit_count_ops(out);

    for (size_t i = 0; i < counts.len; i++) {
        QkOpCount count = counts.data[i];
        if (strcmp(count.name, "h") == 0) {
            if (count.count != 2) {
                printf("Expected 2 H gates, found %zu\n", count.count);
                result = EqualityError;
                goto cleanup_circuit;
            }
        } else if (strcmp(count.name, "rz") == 0) {
            if (count.count != 1) {
                printf("Expected 1 RZ gate, found %zu\n", count.count);
                result = EqualityError;
                goto cleanup_circuit;
            }
        } else {
            printf("Unexpected gate.\n");
            result = EqualityError;
            goto cleanup_circuit;
        }
    }

cleanup_circuit:
    qk_circuit_free(circuit);
    qk_passmanager_context_free(pm_result.context);
cleanup:
    qk_passmanager_free(pm);

    return result;
}

/**
 * Test a pass manager lowering from circuit to dag.
 */
static int test_lowering(void) {
    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};

    QkVTableSlot remove_identity_slots[1] = {{.index = 0, .ptr = (void *)(&run_remove_identity)}};
    QkPass *remove_identity =
        qk_pass_new((void *)(&remove_identity_config), remove_identity_slots, 1);

    QkVTableSlot circuit_to_dag_slots[1] = {{.index = 0, .ptr = (void *)(&run_circuit_to_dag)}};
    QkPass *circuit_to_dag = qk_pass_new(NULL, circuit_to_dag_slots, 1);

    QkCircuit *circuit = qk_circuit_new(10, 0);
    uint32_t q0[1] = {0};
    double almost_zero[1] = {1e-20};
    double nonzero[1] = {1.23};
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit, QkGate_RX, q0, almost_zero);
    qk_circuit_gate(circuit, QkGate_RZ, q0, nonzero);
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);

    QkPassManager *pm = qk_passmanager_new();
    int result = Ok;
    if (qk_passmanager_push_pass(pm, remove_identity) != QkExitCode_Success) {
        printf("Failed pushing pass.\n");
        result = RuntimeError;
        qk_circuit_free(circuit);
        goto cleanup;
    }
    if (qk_passmanager_push_pass(pm, circuit_to_dag) != QkExitCode_Success) {
        printf("Failed pushing pass.\n");
        result = RuntimeError;
        qk_circuit_free(circuit);
        goto cleanup;
    }

    QkPassManagerResult pm_result = {NULL, NULL};
    // note: as the passmanager is set up, it takes ownership of the input IR, which no longer
    // needs to be freed -- only the output IR must be freed
    if (qk_passmanager_run(pm, (void *)circuit, NULL, &pm_result) != QkExitCode_Success) {
        printf("Failed running pass.\n");
        result = RuntimeError;
        qk_circuit_free(circuit);
        goto cleanup;
    }

    QkDag *out = (QkDag *)pm_result.ir;

    // iterate over the DAG and ensure it matches the expected ops
    size_t num_ops = qk_dag_num_op_nodes(out);
    uint32_t *op_indices = malloc(num_ops * sizeof(*op_indices));
    qk_dag_topological_op_nodes(out, op_indices);

    for (size_t i = 0; i < num_ops; i++) {
        QkCircuitInstruction inst;
        qk_dag_get_instruction(out, op_indices[i], &inst);

        if (i == 0 || i == 2) {
            if (strcmp(inst.name, "h") != 0) {
                printf("Expected h at %zu, but got %s\n", i, inst.name);
                result = EqualityError;
                goto dag_cleanup;
            }
        } else if (i == 1) {
            if (strcmp(inst.name, "rz") != 0) {
                printf("Expected rz at %zu, but got %s\n", i, inst.name);
                result = EqualityError;
                goto dag_cleanup;
            }
        } else {
            printf("Unexpected number of operations.\n");
            result = EqualityError;
            goto dag_cleanup;
        }
    }

dag_cleanup:
    free(op_indices);
    qk_dag_free(out);
    qk_passmanager_context_free(pm_result.context);
cleanup:
    qk_passmanager_free(pm);

    return result;
}

static int test_callback(void) {
    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};

    QkVTableSlot pass_slots[1] = {{.index = 0, .ptr = (void *)(&run_remove_identity)}};
    QkPass *remove_identity = qk_pass_new((void *)(&remove_identity_config), pass_slots, 1);

    QkVTableSlot cb_slots[2] = {{.index = 0, .ptr = (void *)(&trigger)},
                                {.index = 1, .ptr = (void *)(&ir_and_context)}};

    CounterCallback cb_config = {malloc(sizeof(QkOpCounts) * 1),
                                 1, // 1 pass run
                                 0};
    QkCCallback *callback = qk_callback_new((void *)(&cb_config), cb_slots, 2);

    QkPassManager *pm = qk_passmanager_new();

    int result = Ok;
    if (qk_passmanager_push_pass(pm, remove_identity) != QkExitCode_Success) {
        printf("Failed pushing pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    QkCircuit *circuit = qk_circuit_new(10, 0);
    uint32_t q0[1] = {0};
    double almost_zero[1] = {1e-20};
    double nonzero[1] = {1.23};
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit, QkGate_RX, q0, almost_zero);
    qk_circuit_gate(circuit, QkGate_RZ, q0, nonzero);
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);

    QkPassManagerResult pm_result = {NULL, NULL};
    QkExitCode exit = qk_passmanager_run(pm, (void *)circuit, callback, &pm_result);
    if (exit != QkExitCode_Success) {
        printf("Exited with %i\n", exit);
        result = RuntimeError;
        goto cleanup_circuit;
    }

    if (pm_result.ir == NULL) {
        printf("Compilation returned NULL.\n");
        result = RuntimeError;
        goto cleanup_circuit;
    }

    QkOpCounts counts = cb_config.counts[0];
    for (size_t i = 0; i < counts.len; i++) {
        QkOpCount count = counts.data[i];
        if (strcmp(count.name, "h") == 0) {
            if (count.count != 2) {
                printf("Expected 2 H gates, found %zu\n", count.count);
                result = EqualityError;
                goto cleanup_circuit;
            }
        } else if (strcmp(count.name, "rz") == 0) {
            if (count.count != 1) {
                printf("Expected 1 RZ gate, found %zu\n", count.count);
                result = EqualityError;
                goto cleanup_circuit;
            }
        } else {
            printf("Unexpected gate.\n");
            result = EqualityError;
            goto cleanup_circuit;
        }
    }

cleanup_circuit:
    qk_circuit_free(circuit);
    qk_passmanager_context_free(pm_result.context);
cleanup:
    qk_passmanager_free(pm);

    return result;
}

int test_passmanager(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_circuit);
    num_failed += RUN_TEST(test_lowering);
    num_failed += RUN_TEST(test_callback);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (passmanager): %i\n", num_failed);

    return num_failed;
}
