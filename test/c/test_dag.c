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
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int test_empty(void) {
    QkDag *dag = qk_dag_new();
    uint32_t num_qubits = qk_dag_num_qubits(dag);
    uint32_t num_clbits = qk_dag_num_clbits(dag);
    qk_dag_free(dag);

    if (num_qubits != 0) {
        printf("The number of qubits %ul is not 0\n", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %ul is not 0\n", num_clbits);
        return EqualityError;
    }
    return Ok;
}

static int test_dag_with_quantum_reg(void) {
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_register");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t num_qubits = qk_dag_num_qubits(dag);
    uint32_t num_clbits = qk_dag_num_clbits(dag);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    if (num_qubits != 1024) {
        printf("The number of qubits %ul is not 1024\n", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %ul is not 0\n", num_clbits);
        return EqualityError;
    }
    return Ok;
}

static int test_dag_with_classical_reg(void) {
    QkDag *dag = qk_dag_new();
    QkClassicalRegister *cr = qk_classical_register_new(2048, "my_register");
    qk_dag_add_classical_register(dag, cr);
    uint32_t num_qubits = qk_dag_num_qubits(dag);
    uint32_t num_clbits = qk_dag_num_clbits(dag);
    qk_dag_free(dag);
    qk_classical_register_free(cr);
    if (num_qubits != 0) {
        printf("The number of qubits %ul is not 0\n", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 2048) {
        printf("The number of clbits %ul is not 2048\n", num_clbits);
        return EqualityError;
    }
    return Ok;
}

int test_dag(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty);
    num_failed += RUN_TEST(test_dag_with_quantum_reg);
    num_failed += RUN_TEST(test_dag_with_classical_reg);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
