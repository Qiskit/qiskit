// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#if !defined(QISKIT_FUNCS_PY_H)
#define QISKIT_FUNCS_PY_H

// We rely on `qiskit.h` to `#include <Python.h>` on our behalf, since it needs to be included
// before all other includes. (Though really, a user should also have included it themselves,
// surely, if they're delcaring a Python extension module.)

static void **_Qk_API_Circuit;
static void **_Qk_API_Transpile;
static void **_Qk_API_QI;

/**
 * Import the Qiskit C API.
 *
 * @return 0 on success, or non-zero on failure.  If a failure occurs, the Python exception state
 *     will be set.
 *
 * This function must be called before any attempt to use the Qiskit C API within this translation
 * unit.
 */
static int qk_import(void) {
    PyObject *accelerate = PyImport_ImportModule("qiskit._accelerate");
    if (!accelerate)
        return -1;
    // We don't actually need a handle to `accelerate` ourselves, we just need to have ensured it's
    // already been imported.
    Py_DECREF(accelerate);

    _Qk_API_Circuit = (void **)PyCapsule_Import("qiskit._accelerate.capi.QK_FFI_CIRCUIT", 0);
    if (!_Qk_API_Circuit)
        return -1;
    _Qk_API_Transpile = (void **)PyCapsule_Import("qiskit._accelerate.capi.QK_FFI_TRANSPILE", 0);
    if (!_Qk_API_Transpile)
        return -1;
    _Qk_API_QI = (void **)PyCapsule_Import("qiskit._accelerate.capi.QK_FFI_QI", 0);
    if (!_Qk_API_QI)
        return -1;

    // TODO: any validity checks on the version of the Qiskit API?
    return 0;
}

#include "qiskit/funcs_py_generated.h"

#endif // QISKIT_FUNCS_PY_H
