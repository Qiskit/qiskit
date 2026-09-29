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

/// A custom integer IR.
///
/// This holds an array of uint32_t and is assumed to flip the bits at where there is 
/// a 1-bit.
typedef struct {
    size_t capacity;
    size_t len;
    uint32_t *integers;
} Flips;

Flips *new_flips(size_t capacity) {
    Flips *flips = malloc(sizeof(Flips));
    flips->capacity = capacity;
    flips->len = 0;
    flips->integers = malloc(capacity * sizeof(uint32_t));
    return flips;
}

void free_flips(Flips *flips) {
    free(flips->integers);
    free(flips);
}

/// Methods for the IR.
int apply_flip(Flips *flips, uint32_t flip) {
    if (flips->capacity <= flips->len) 
        return 1; // cannot append

    // i++ increases _after_ reading the value, hence this is correct
    flips->integers[flips->len++] = flip;
    return 0;
}

size_t num_flips(Flips *flips) {
    size_t n = 0;
    for (size_t i = 0; i < flips->len; i++) {
        uint32_t integer = flips->integers[i];
        for (uint32_t b = 0; b < 32; b++) {
            n += (integer >> b) & 1u;
        }
    }
    return n;
}

/// Passes for the integers IR.
void *inverse_cancellation(void *self, void *ir) {
    UNUSED_VARIABLE(self);
    Flips *flips = (Flips *)ir;

    if (flips->len < 2) 
        return ir;

    size_t write_index = 0;
    size_t read_index = 0;
    while (read_index < flips->len - 1) {
        // If the next one is not the same, write the current one.
        if (flips->integers[read_index] != flips->integers[read_index + 1]) {
            flips->integers[write_index++] = flips->integers[read_index];
            read_index++;
        } else {
            // .. if they were the same, increase the read index by 2
            read_index += 2;
        }
    }
    // Handle the last element: if it was skipped, then read_index equals flips->len,
    // if not we are one element short.
    if (read_index == flips->len - 1) {
        flips->integers[write_index++] = flips->integers[read_index];
    }
    flips->len = write_index;

    return (void *)flips;
}


/**
 * Test running a single RemoveIdentity pass on a circuit.
 */
int test_circuit_ir(void) {
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
int test_lowering(void) {
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
    QkCompilationError *error = NULL;
    void *out_ir = qk_passmanager_run_simple(pm, (void *)circuit, circuit_ir, dag_ir, &error);
    if (error != NULL) {
        printf("Failed running pass.\n");
        result = RuntimeError;
        qk_compilation_error_free(error);
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

int test_custom_ir(void) {
    Flips *program = new_flips(5);
    apply_flip(program, 1);
    apply_flip(program, 1);
    apply_flip(program, 2);
    apply_flip(program, 3);

    size_t flipped = num_flips(program);

    int result = Ok;
    if (flipped != 5) {
        printf("Wrong number of initial bitflips, expected 5 got %zu\n", flipped);
        result = EqualityError;
    }

    QkVtableEntry flip_methods[3] = {
        { .slot = 0, .flags = 0, .ptr = (void*)(&apply_flip) },
        { .slot = 1, .flags = 0, .ptr = (void*)(&num_flips) },
        { .slot = -1, .flags = 0, .ptr = NULL },
    };
    QkIrHandle *ir = qk_ir_handle_new("flips", flip_methods);
    QkPassManager *pm = qk_passmanager_new();

    QkVtableEntry cancellation_slots[2] = {
        { .slot = 0, .flags = 0, .ptr = (void*)(&inverse_cancellation)},
        { .slot = -1, .flags = 0, .ptr = NULL },
    };
    const QkPassVtable *cancellation_vtable = qk_pass_vtable_new("cancellation", ir, ir, cancellation_slots);
    QkPass *cancellation = qk_pass_new(NULL, cancellation_vtable);

    if (qk_passmanager_push_pass(pm, cancellation) != QkExitCode_Success) {
        printf("Failed pushing pass.\n");
        result = RuntimeError;
        free_flips(program);
        goto cleanup;
    }

    Flips *out = (Flips *)qk_passmanager_run_simple(pm, program, ir, ir, NULL);

    flipped = num_flips(out);
    free_flips(out);
    if (flipped != 3) {
        printf("Wrong number of bitflips, expected 3 got %zu\n", flipped);
        result = EqualityError;
    }

cleanup:
    qk_passmanager_free(pm);
    qk_ir_handle_free(ir);

    return result;
}

/**
 * Test the pipeline being incoherent.
 */
static int test_invalid_pipeline(void) {
    // A pass on Flips->Flips IR.
    QkVtableEntry flip_methods[3] = {
        { .slot = 0, .flags = 0, .ptr = (void*)(&apply_flip) },
        { .slot = 1, .flags = 0, .ptr = (void*)(&num_flips) },
        { .slot = -1, .flags = 0, .ptr = NULL },
    };
    QkIrHandle *flip_ir = qk_ir_handle_new("flips", flip_methods);

    QkVtableEntry cancellation_slots[2] = {
        { .slot = 0, .flags = 0, .ptr = (void*)(&inverse_cancellation)},
        { .slot = -1, .flags = 0, .ptr = NULL },
    };
    const QkPassVtable *cancellation_vtable = qk_pass_vtable_new("cancellation", flip_ir, flip_ir, cancellation_slots);
    QkPass *flip_pass = qk_pass_new(NULL, cancellation_vtable);

    // A pass on Circuit->Circuit IR.
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};
    const QkVtableEntry remove_identity_slots[2] = {
        {.slot = 0, .flags = 0, .ptr = (void *)(&run_remove_identity)},
        {.slot = -1, .flags = 0, .ptr = NULL}};
    const QkPassVtable *remove_identity_vtable =
        qk_pass_vtable_new("remove_identity", circuit_ir, circuit_ir, remove_identity_slots);
    QkPass *circuit_pass =
        qk_pass_new((void *)(&remove_identity_config), remove_identity_vtable);

    int result = Ok;
    QkPassManager *pm = qk_passmanager_new();
    if (qk_passmanager_push_pass(pm, flip_pass) != QkExitCode_Success) {
        printf("Failed pushing pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    // Now try pushing the circuit pass onto the flip pass. This should error.
    if (qk_passmanager_push_pass(pm, circuit_pass) != QkExitCode_IncompatibleTypes) {
        printf("Expected QkExitCode_IncompatibleTypes.\n");
        result = EqualityError;
    }

cleanup:
    qk_passmanager_free(pm);
    qk_ir_handle_free(flip_ir);
    qk_ir_handle_free(circuit_ir);

    return result;
}

/**
 * Test the input/output IR types being correct, but not matching the pipeline type.
 */
static int test_mismatching_input(void) {
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkIrHandle *dag_ir = qk_ir_handle_builtin(QkIrBuiltin_Dag);

    QkVtableEntry table[2] = {
        // Slot 0 for run -- slot 1 for delete (which we don't have here)
        {.slot = 0, .flags = 0, .ptr = run_remove_identity},
        {.slot = -1, .flags = 0, .ptr = NULL},
    };
    QkPassVtable *vtable = qk_pass_vtable_new("remove_identity", circuit_ir, circuit_ir, table);
    QkTarget *target = qk_target_new(10);
    RemoveIdentity this = {target};
    QkPass *pass = qk_pass_new((void *)(&this), vtable);
    qk_pass_vtable_free(vtable);

    // This PM now has a pipeline built on QkCircuit
    QkPassManager *pm = qk_passmanager_new();
    int result = Ok;

    if (qk_passmanager_push_pass(pm, pass) != QkExitCode_Success) {
        printf("Failed pushing pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    // .. and now we call it on QkDag
    QkDag *dag = qk_dag_new();
    QkCompilationError *error = NULL;
    QkDag *out = (QkDag *)qk_passmanager_run_simple(pm, (void *)dag, dag_ir, dag_ir, &error);

    if (out != NULL) {
        printf("Expected out pointer to be NULL, but it is not.\n");
        result = EqualityError;
        goto cleanup;
    }

    if (error == NULL) {
        printf("Expected error to be written, but the pointer is NULL.\n");
        result = EqualityError;
        goto cleanup;
    } else {
        const char *error_msg = qk_compilation_error_str(error);
        // The C11 standard does not provide regex match functionality, so we match
        // on 3 words we expect to be in this message. We can't give the full message since the 
        // type description is not stable and e.g. the path inclusion might change.
        if (strcmp(error_msg, "failed to cast to expected input type") != 0) {
            printf("Wrong error message: %s\n", error_msg);
            result = EqualityError;
            goto cleanup;
        }
        qk_compilation_error_free(error);
    }

cleanup:
    qk_passmanager_free(pm);
    qk_ir_handle_free(dag_ir);
    qk_ir_handle_free(circuit_ir);

    return result;
}

int test_passmanager(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_circuit_ir);
    num_failed += RUN_TEST(test_lowering);
    num_failed += RUN_TEST(test_custom_ir);
    num_failed += RUN_TEST(test_invalid_pipeline);
    num_failed += RUN_TEST(test_mismatching_input);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (passmanager): %i\n", num_failed);

    return num_failed;
}
