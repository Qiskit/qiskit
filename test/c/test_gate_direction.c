#include "common.h"
#include <qiskit.h>
#include <stdio.h>

static QkTarget* build_target() {
    QkTarget *target = qk_target_new(3);
    uint32_t qargs[3] = {0,1,2};
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);

    // TODO: handle return values and free
    qk_target_entry_add_property(cx_entry, qargs, 2, 0.0, 0.0);
    qk_target_entry_add_property(cx_entry, &qargs[1], 2, 0.0, 0.0);
    qk_target_add_instruction(target, cx_entry);

    double rzx_params[1] = {1.5};
    QkTargetEntry *rzx_entry = qk_target_entry_new_fixed(QkGate_RZX, rzx_params);
    qk_target_entry_add_property(rzx_entry, qargs, 2, 0.0, 0.0);
    qk_target_add_instruction(target, rzx_entry);

    return target;
}

int test_check_gate_direction(void) {
    enum TestResult result = Ok;

    QkCircuit *circuit = qk_circuit_new(3, 0);
    uint32_t qargs[4] = {0, 1, 2, 1};
    qk_circuit_gate(circuit, QkGate_CX, qargs, NULL);
    qk_circuit_gate(circuit, QkGate_CX, &qargs[1], NULL);

    QkTarget *target = build_target();

    bool check_pass = qk_transpiler_pass_standalone_check_gate_direction(circuit, target);
    if ( !check_pass )
        result = EqualityError;
    else {
        qk_circuit_gate(circuit, QkGate_CX, &qargs[2], NULL);
        check_pass = qk_transpiler_pass_standalone_check_gate_direction(circuit, target);
        if ( check_pass )
            result = EqualityError;
    }

    qk_target_free(target);
    qk_circuit_free(circuit);
    return result;
}

int test_gate_direction(void) {
    enum TestResult result = Ok;

    QkCircuit *circuit = qk_circuit_new(3, 0);
    uint32_t qargs[5] = {0, 1, 2, 1, 0};
    qk_circuit_gate(circuit, QkGate_CX, qargs, NULL);
    qk_circuit_gate(circuit, QkGate_CX, &qargs[1], NULL);
    qk_circuit_gate(circuit, QkGate_CX, &qargs[2], NULL);
    double params[1] = {1.5};
    qk_circuit_gate(circuit, QkGate_RZX, &qargs[3], params);

    QkTarget *target = build_target();

    QkCircuit *fixed_circuit = qk_transpiler_pass_standalone_gate_direction(circuit, target);

    qk_target_free(target);
    qk_circuit_free(circuit);
    return result;
}

int test_gate_direction_passes(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_check_gate_direction);
    num_failed += RUN_TEST(test_gate_direction);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
