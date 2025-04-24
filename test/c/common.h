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

#include <complex.h>
#include <qiskit.h>
#include <stdio.h>

QkComplex64 make_complex_double(double real, double imag);

// An enumeration of test results. These should be returned by test functions to
// indicate what kind of error occurred. This will be used to produce more
// helpful messages for the developer running the test suite.
enum TestResult {
    Ok,
    EqualityError,
    RuntimeError,
    NullptrError, // an unexpected null pointer
};

// A macro for running a test function. This calls the run function below with
// the provided function and its name.
#define RUN_TEST(f) run(#f, f)

// A function to run a test function of a given name. This function will also
// post-process the returned `TestResult` to product a minimal info message for
// the developer running the test suite.
int run(const char *name, int (*test_function)(void));
