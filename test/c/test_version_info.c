#include "common.h"

// Test that QISKIT_VERSION_MAJOR, _MINOR, _PATCH are defined and print them
int test_version_macros() {
    printf("Qiskit version: %d.%d.%d\n", QISKIT_VERSION_MAJOR, QISKIT_VERSION_MINOR,
           QISKIT_VERSION_PATCH);
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