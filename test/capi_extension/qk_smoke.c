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

// This file implements a minimal Python extension module that uses the Qiskit
// C API in the same way a third-party package would: through `qk_import()`
// and the function tables exported by `qiskit._accelerate.capi`. It is
// used in a smoke test to help avoid breaking downstream extensions.

#define PY_SSIZE_T_CLEAN
#define QISKIT_PYTHON_EXTENSION
#include <Python.h>
#include <qiskit.h>

static PyObject *api_version(PyObject *self, PyObject *args) {
    return PyLong_FromUnsignedLong(qk_api_version());
}

// Build a Bell circuit in C and pass it to Python as a full `QuantumCircuit`.
static PyObject *bell(PyObject *self, PyObject *args) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t q0[1] = {0};
    uint32_t q01[2] = {0, 1};
    if (qk_circuit_gate(qc, QkGate_H, q0, NULL) != QkExitCode_Success ||
        qk_circuit_gate(qc, QkGate_CX, q01, NULL) != QkExitCode_Success) {
        qk_circuit_free(qc);
        PyErr_SetString(PyExc_RuntimeError, "qk_circuit_gate failed");
        return NULL;
    }
    return qk_circuit_to_python_full(qc);
}

// Borrow the `CircuitData` of a Python circuit and read it from C.
static PyObject *num_instructions(PyObject *self, PyObject *circuit_data) {
    QkCircuit *qc = qk_circuit_borrow_from_python(circuit_data);
    if (!qc)
        return NULL;
    return PyLong_FromSize_t(qk_circuit_num_instructions(qc));
}

static PyMethodDef methods[] = {
    {"api_version", api_version, METH_NOARGS, NULL},
    {"bell", bell, METH_NOARGS, NULL},
    {"num_instructions", num_instructions, METH_O, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "qk_smoke",
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit_qk_smoke(void) {
    if (qk_import() < 0)
        return NULL;
    return PyModule_Create(&module);
}
