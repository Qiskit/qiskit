// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include <stdio.h>

#ifdef _MSC_VER
QkComplex64 make_complex_double(double real, double imag) { return (QkComplex64){real, imag}; }
#else
QkComplex64 make_complex_double(double real, double imag) { return real + I * imag; }
#endif

// A function to run a test function of a given name. This function will also
// post-process the returned `TestResult` to product a minimal info message for
// the developer running the test suite.
int run(const char *name, int (*test_function)(void)) {
    // TODO: we could consider to change the return value of our test functions
    // to be a struct containing the integer return value and a custom error
    // message which could then be included below.
    int result = test_function();
    int did_fail = 1;
    char *msg;
    if (result == Ok) {
        did_fail = 0;
        msg = "Ok";
    } else if (result == EqualityError) {
        msg = "FAILED with an EqualityError";
    } else {
        msg = "FAILED with unknown error";
    }
    fprintf(stderr, "--- %-30s: %s\n", name, msg);
    fflush(stderr);

    return did_fail;
}
