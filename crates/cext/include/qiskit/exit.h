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

#ifndef QISKIT__EXIT_H
#define QISKIT__EXIT_H

#include <stdlib.h>

/// Integer exit codes returned to C.
enum QkExitCode
#ifdef __cplusplus
    : uint32_t
#endif // __cplusplus
{
    /// Success.
    QkExitCode_Success = 0,
    /// Error related to data input.
    QkExitCode_CInputError = 100,
    /// Unexpected null pointer.
    QkExitCode_NullPointerError = 101,
    /// Pointer is not aligned to expected data.
    QkExitCode_AlignmentError = 102,
    /// Index out of bounds.
    QkExitCode_IndexError = 103,
    /// Error related to arithmetic operations or similar.
    QkExitCode_ArithmeticError = 200,
    /// Mismatching number of qubits.
    QkExitCode_MismatchedQubits = 201,
};
#ifndef __cplusplus
typedef uint32_t QkExitCode;
#endif // __cplusplus

#endif // QISKIT__EXIT_H
