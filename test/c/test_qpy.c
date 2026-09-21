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

static int test_round_trip(void) {
    const char *filename = "qiskit_c_api_test.qpy";
    QkCircuit *source = qk_circuit_new(2, 2);
    if (qk_circuit_gate(source, QkGate_H, (uint32_t[]){0}, NULL) != QkExitCode_Success ||
        qk_circuit_gate(source, QkGate_CX, (uint32_t[]){0, 1}, NULL) != QkExitCode_Success ||
        qk_circuit_measure(source, 0, 0) != QkExitCode_Success ||
        qk_circuit_measure(source, 1, 1) != QkExitCode_Success) {
        printf("Unexpected error encountered in QPY test_round_trip.");
        qk_circuit_free(source);
        return RuntimeError;
    }

    const QkCircuit *sources[] = {source, source};
    if (qk_qpy_dump_file(sources, 2, filename, NULL) != QkExitCode_Success) {
        printf("Unexpected error encountered in QPY test_round_trip.");
        qk_circuit_free(source);
        return RuntimeError;
    }

    QkCircuit **loaded = NULL;
    size_t num_loaded = 0;
    QkExitCode load_result = qk_qpy_load_file(&loaded, &num_loaded, filename, NULL);
    remove(filename);
    if (load_result != QkExitCode_Success || loaded == NULL || num_loaded != 2) {
        printf("Unexpected error encountered in QPY test_round_trip.");
        if (loaded != NULL) {
            qk_qpy_free_circuits(loaded, num_loaded);
        }
        qk_circuit_free(source);
        return RuntimeError;
    }

    if (qk_qpy_dump_file_with_version(sources, 2, filename, 18, NULL) != QkExitCode_Success) {
        qk_qpy_free_circuits(loaded, num_loaded);
        qk_circuit_free(source);
        return RuntimeError;
    }
    remove(filename);

    int result = Ok;
    if (qk_circuit_num_qubits(loaded[0]) != 2 || qk_circuit_num_clbits(loaded[0]) != 2 ||
        qk_circuit_num_instructions(loaded[0]) != 4 ||
        qk_circuit_num_instructions(loaded[1]) != 4) {
        result = EqualityError;
    }

    qk_qpy_free_circuits(loaded, num_loaded);
    qk_circuit_free(source);
    return result;
}

static int test_buffer_round_trip(void) {
    QkCircuit *source = qk_circuit_new(2, 0);
    if (qk_circuit_gate(source, QkGate_H, (uint32_t[]){0}, NULL) != QkExitCode_Success) {
        qk_circuit_free(source);
        return RuntimeError;
    }

    uint8_t *buffer = NULL;
    size_t size = 0;
    const QkCircuit *sources[] = {source, source};
    if (qk_qpy_dump_buffer(sources, 2, &buffer, &size, NULL) != QkExitCode_Success || buffer == NULL ||
        size == 0) {
        qk_circuit_free(source);
        return RuntimeError;
    }
    qk_qpy_free_buffer(buffer, size);

    buffer = NULL;
    size = 0;
    if (qk_qpy_dump_buffer_with_version(sources, 2, &buffer, &size, 18, NULL) != QkExitCode_Success ||
        buffer == NULL || size == 0) {
        qk_circuit_free(source);
        return RuntimeError;
    }

    QkCircuit **loaded = NULL;
    size_t num_loaded = 0;
    QkExitCode result = qk_qpy_load_buffer(&loaded, &num_loaded, buffer, size, NULL);
    qk_qpy_free_buffer(buffer, size);
    if (result != QkExitCode_Success || loaded == NULL || num_loaded != 2) {
        if (loaded != NULL) {
            qk_qpy_free_circuits(loaded, num_loaded);
        }
        qk_circuit_free(source);
        return RuntimeError;
    }

    int test_result = qk_circuit_num_qubits(loaded[0]) == 2 &&
                              qk_circuit_num_instructions(loaded[0]) == 1 &&
                              qk_circuit_num_instructions(loaded[1]) == 1
                          ? Ok
                          : EqualityError;
    qk_qpy_free_circuits(loaded, num_loaded);
    qk_circuit_free(source);
    return test_result;
}

static int test_error_message(void) {
    uint8_t invalid_payload[] = {0};
    QkCircuit **loaded = NULL;
    size_t num_loaded = 0;
    char *error = NULL;
    QkExitCode result =
        qk_qpy_load_buffer(&loaded, &num_loaded, invalid_payload, sizeof(invalid_payload), &error);
    if (result != QkExitCode_QpyError || error == NULL) {
        if (error != NULL) {
            qk_str_free(error);
        }
        return RuntimeError;
    }
    qk_str_free(error);
    return Ok;
}

int test_qpy(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_round_trip);
    num_failed += RUN_TEST(test_buffer_round_trip);
    num_failed += RUN_TEST(test_error_message);

    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    fflush(stderr);

    return num_failed;
}
