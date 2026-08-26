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

/**
 * Test running a single RemoveIdentity pass on a circuit.
 */
static int test_circuit(void) {
    QkPass *remove_identity = qk_pass_new();
    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};

    qk_pass_set_self(remove_identity, (void *)(&remove_identity_config));
    qk_pass_set_run(remove_identity, run_remove_identity);

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

    QkPassManagerResult pm_result = {NULL};
    QkExitCode exit = qk_passmanager_run(pm, (void *)circuit, &pm_result);
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
cleanup:
    qk_passmanager_free(pm);
    qk_pass_free(remove_identity);

    return result;
}

/**
 * Test a pass manager lowering from circuit to dag.
 */
static int test_lowering(void) {
    QkPass *remove_identity = qk_pass_new();
    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};
    qk_pass_set_self(remove_identity, (void *)(&remove_identity_config));
    qk_pass_set_run(remove_identity, run_remove_identity);

    QkPass *circuit_to_dag = qk_pass_new();
    qk_pass_set_self(circuit_to_dag, NULL); // no self struct
    qk_pass_set_run(circuit_to_dag, run_circuit_to_dag);

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

    QkPassManagerResult pm_result = {NULL};
    // note: as the passmanager is set up, it takes ownership of the input IR, which no longer
    // needs to be freed -- only the output IR must be freed
    if (qk_passmanager_run(pm, (void *)circuit, &pm_result) != QkExitCode_Success) {
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
cleanup:
    qk_pass_free(remove_identity);
    qk_pass_free(circuit_to_dag);
    qk_passmanager_free(pm);

    return result;
}

int test_passmanager(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_circuit);
    num_failed += RUN_TEST(test_lowering);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (passmanager): %i\n", num_failed);

    return num_failed;
}
