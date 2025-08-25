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
#include <qiskit.h>
#include <string.h>

/**
 * Build the version a string, based on the version numbers.
 */
char *build_version_string(void) {
    char suffix[4]; // 3 chars + end
    switch (QISKIT_RELEASE_LEVEL) {
    case QISKIT_RELEASE_LEVEL_DEV:
        sprintf(suffix, "dev");
        break;
    case QISKIT_RELEASE_LEVEL_RC:
        sprintf(suffix, "rc%u", QISKIT_RELEASE_SERIAL);
        break;
    default:
        // no suffix
        break;
    }

    char *version = malloc(9 * sizeof(char));
    sprintf(version, "%u.%u.%u%s", QISKIT_VERSION_MAJOR, QISKIT_VERSION_MINOR, QISKIT_VERSION_PATCH,
            suffix);
    return version;
}

/**
 * Test the string version.
 */
int test_version(void) {
    char *ref = build_version_string();
    int result;
    if (strcmp(ref, QISKIT_VERSION) == 0)
        result = Ok;
    else {
        printf("Qiskit version (%s) didn't match the expectation (%s)", QISKIT_VERSION, ref);
        result = EqualityError;
    }

    free(ref);
    return result;
}

/**
 * Test the version macro and HEX version.
 */
int test_version_macros(void) {
    if (QISKIT_VERSION_MAJOR < 0 || QISKIT_VERSION_MINOR < 0 || QISKIT_VERSION_PATCH < 0) {
        return EqualityError;
    }
    if (QISKIT_GET_VERSION_HEX(QISKIT_VERSION_MAJOR, QISKIT_VERSION_MINOR, QISKIT_VERSION_PATCH,
                               QISKIT_RELEASE_LEVEL, QISKIT_RELEASE_SERIAL) != QISKIT_VERSION_HEX) {
        fprintf(stderr, "QISKIT_VERSION_NUMERIC does not match QISKIT_VERSION\n");
        return EqualityError;
    }
    return Ok;
}

int test_version_info(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_version);
    num_failed += RUN_TEST(test_version_macros);

    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    fflush(stderr);
    return num_failed;
}
