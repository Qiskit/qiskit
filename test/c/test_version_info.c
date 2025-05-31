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

// Test that QISKIT_VERSION_MAJOR, _MINOR, _PATCH are defined and print them
int test_version_macros() {
    if (QISKIT_VERSION_MAJOR < 0 || QISKIT_VERSION_MINOR < 0 || QISKIT_VERSION_PATCH < 0) {
        return EqualityError;
    }
    if (QISKIT_VERSION_NUMERIC(QISKIT_VERSION_MAJOR, QISKIT_VERSION_MINOR, QISKIT_VERSION_PATCH) !=
        QISKIT_VERSION) {
        fprintf(stderr, "QISKIT_VERSION_NUMERIC does not match QISKIT_VERSION\n");
        return EqualityError;
    }
    return Ok;
}

// Main test function
int test_version_info() {
    int num_failed = 0;
    num_failed += RUN_TEST(test_version_macros);

    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    fflush(stderr);
    return num_failed;
}