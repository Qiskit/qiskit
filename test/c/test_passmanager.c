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
void *run_remove_identity(void *self, void *ir) {
    RemoveIdentity *self_ = (RemoveIdentity *)self;
    qk_transpiler_pass_standalone_remove_identity_equivalent((QkCircuit *)ir, self_->target, 1.0);
    return (void *)ir;
}

/// The execution function for a circuit-to-dag pass.
void *run_circuit_to_dag(void *self, void *ir) {
    UNUSED_VARIABLE(self);

    QkCircuit *circuit = (QkCircuit *)ir;
    QkDag *dag = qk_circuit_to_dag(circuit);
    qk_circuit_free(circuit);
    return (void *)dag;
}

/**
 * Test running a single RemoveIdentity pass on a circuit.
 */
static int test_circuit(void) {
    QkIrHandle *ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkVtableEntry table[2] = {
        // Slot 0 for run -- slot 1 for delete (which we don't have here)
        {.slot = 0, .flags = 0, .ptr = run_remove_identity},
        {.slot = -1, .flags = 0, .ptr = NULL},
    };
    QkPassVtable *vtable = qk_pass_vtable_new("remove_identity", ir, ir, table);
    QkTarget *target = qk_target_new(10);
    RemoveIdentity this = {target};
    QkPass *pass = qk_pass_new((void *)(&this), vtable);
    qk_pass_vtable_free(vtable);

    QkPassManager *pm = qk_passmanager_new();
    int result = Ok;

    if (qk_passmanager_push_pass(pm, pass) != QkExitCode_Success) {
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

    QkCircuit *out = (QkCircuit *)qk_passmanager_run_simple(pm, (void *)circuit, ir, ir, NULL);

    QkOpCounts counts = qk_circuit_count_ops(out);
    qk_circuit_free(out);

    for (size_t i = 0; i < counts.len; i++) {
        QkOpCount count = counts.data[i];
        if (strcmp(count.name, "h") == 0) {
            if (count.count != 2) {
                printf("Expected 2 H gates, found %zu\n", count.count);
                result = EqualityError;
                goto cleanup;
            }
        } else if (strcmp(count.name, "rz") == 0) {
            if (count.count != 1) {
                printf("Expected 1 RZ gate, found %zu\n", count.count);
                result = EqualityError;
                goto cleanup;
            }
        } else {
            printf("Unexpected gate.\n");
            result = EqualityError;
            goto cleanup;
        }
    }

cleanup:
    qk_target_free(target);
    qk_passmanager_free(pm);
    qk_ir_handle_free(ir);
    return result;
}

/**
 * Test a pass manager lowering from circuit to dag.
 */
static int test_lowering(void) {
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkIrHandle *dag_ir = qk_ir_handle_builtin(QkIrBuiltin_Dag);

    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};
    const QkVtableEntry remove_identity_slots[2] = {
        {.slot = 0, .flags = 0, .ptr = (void *)(&run_remove_identity)},
        {.slot = -1, .flags = 0, .ptr = NULL}};
    const QkPassVtable *remove_identity_vtable =
        qk_pass_vtable_new("remove_identity", circuit_ir, circuit_ir, remove_identity_slots);
    QkPass *remove_identity =
        qk_pass_new((void *)(&remove_identity_config), remove_identity_vtable);

    const QkVtableEntry circuit_to_dag_slots[2] = {
        {.slot = 0, .flags = 0, .ptr = (void *)(&run_circuit_to_dag)},
        {.slot = -1, .flags = 0, .ptr = NULL}};
    QkPassVtable *circuit_to_dag_vtable =
        qk_pass_vtable_new("circuit_to_dag", circuit_ir, dag_ir, circuit_to_dag_slots);
    QkPass *circuit_to_dag = qk_pass_new(NULL, circuit_to_dag_vtable);
    qk_pass_vtable_free(circuit_to_dag_vtable);

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

    // note: as the passmanager is set up, it takes ownership of the input IR, which no longer
    // needs to be freed -- only the output IR must be freed
    const QkCompilationError *error = NULL;
    void *out_ir = qk_passmanager_run_simple(pm, (void *)circuit, circuit_ir, dag_ir, &error);
    if (error != NULL) {
        printf("Failed running pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    QkDag *out = (QkDag *)out_ir;

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
    qk_target_free(target);
    qk_passmanager_free(pm);
    qk_ir_handle_free(circuit_ir);
    qk_ir_handle_free(dag_ir);

    return result;
}

/// A pass that appends one H gate to qubit 0.
void *run_add_h(void *self, void *ir) {
    UNUSED_VARIABLE(self);

    QkCircuit *circuit = (QkCircuit *)ir;
    uint32_t q0[1] = {0};
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    return (void *)circuit;
}

/// The configuration of a C-defined predicate that stops once the circuit is long enough.
typedef struct {
    size_t target;
} AtLeastGates;

/// A predicate function: stop once the circuit has at least `target` instructions.
bool evaluate_at_least_gates(void *self, const void *ir, const void *context,
                             QkCompilationError **error) {
    UNUSED_VARIABLE(context);
    UNUSED_VARIABLE(error);

    AtLeastGates *self_ = (AtLeastGates *)self;
    const QkCircuit *circuit = (const QkCircuit *)ir;
    return qk_circuit_num_instructions(circuit) >= self_->target;
}

/// Run a pass manager holding a single `while` task over an `add_h` body, and return the number of
/// instructions the loop left behind.  Returns `-1` if the run failed.
static int64_t run_while_loop(QkIrHandle *ir, QkPredicate *predicate, size_t max_iterations) {
    const QkVtableEntry add_h_slots[2] = {{.slot = 0, .flags = 0, .ptr = (void *)(&run_add_h)},
                                          {.slot = -1, .flags = 0, .ptr = NULL}};
    QkPassVtable *add_h_vtable = qk_pass_vtable_new("add_h", ir, ir, add_h_slots);
    QkPass *add_h = qk_pass_new(NULL, add_h_vtable);
    qk_pass_vtable_free(add_h_vtable);

    QkPassManager *pm = qk_passmanager_new();
    if (qk_passmanager_push_while(pm, add_h, predicate, max_iterations) != QkExitCode_Success) {
        printf("Failed pushing the while task.\n");
        qk_passmanager_free(pm);
        return -1;
    }

    QkCircuit *circuit = qk_circuit_new(1, 0);
    const QkCompilationError *error = NULL;
    void *out_ir = qk_passmanager_run_simple(pm, (void *)circuit, ir, ir, &error);
    qk_passmanager_free(pm);
    if (error != NULL) {
        qk_compilation_error_free((QkCompilationError *)error);
        return -1;
    }

    QkCircuit *out = (QkCircuit *)out_ir;
    int64_t num_instructions = (int64_t)qk_circuit_num_instructions(out);
    qk_circuit_free(out);
    return num_instructions;
}

/**
 * Test a loop governed by a predicate whose behavior is defined in C.
 */
static int test_while_c_predicate(void) {
    QkIrHandle *ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);

    AtLeastGates predicate_config = {4};
    const QkVtableEntry predicate_slots[2] = {
        // Slot 0 for evaluate -- slot 1 for delete (which we don't have here)
        {.slot = 0, .flags = 0, .ptr = (void *)(&evaluate_at_least_gates)},
        {.slot = -1, .flags = 0, .ptr = NULL}};
    QkPredicateVtable *predicate_vtable =
        qk_predicate_vtable_new("at_least_gates", ir, predicate_slots);
    QkPredicate *predicate = qk_predicate_new((void *)(&predicate_config), predicate_vtable);
    qk_predicate_vtable_free(predicate_vtable);

    int result = Ok;
    int64_t num_instructions = run_while_loop(ir, predicate, 100);
    if (num_instructions < 0) {
        printf("Failed running the pass manager.\n");
        result = RuntimeError;
        goto cleanup;
    }
    // The predicate is checked before each run, so the loop stops at exactly the target.
    if (num_instructions != 4) {
        printf("Expected 4 instructions, found %lld\n", (long long)num_instructions);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_ir_handle_free(ir);
    return result;
}

/**
 * Test a loop governed by one of Qiskit's own predicates, retrieved from C.
 */
static int test_while_builtin_predicate(void) {
    QkIrHandle *ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkPredicate *predicate = qk_predicate_builtin(QkPredicateBuiltin_UntilStable, ir);

    int result = Ok;
    if (predicate == NULL) {
        printf("Failed retrieving the built-in predicate.\n");
        result = RuntimeError;
        goto cleanup;
    }

    // `UntilStable` never returns true because we always report the IR changes from C
    int64_t num_instructions = run_while_loop(ir, predicate, 3);
    if (num_instructions >= 0) {
        printf("Expected the iteration limit to fail the run, got %lld instructions\n",
               (long long)num_instructions);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_ir_handle_free(ir);
    return result;
}

int test_passmanager(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_circuit);
    num_failed += RUN_TEST(test_lowering);
    num_failed += RUN_TEST(test_while_c_predicate);
    num_failed += RUN_TEST(test_while_builtin_predicate);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (passmanager): %i\n", num_failed);

    return num_failed;
}
