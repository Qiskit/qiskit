# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,broad-except,redefined-builtin

"""Test Qiskit Result class."""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import compile, get_backend
from .common import QiskitTestCase


class TestResult(QiskitTestCase):
    """Test Qiskit Result class."""

    def test_qubitpol(self):
        """Test the results of the qubitpol function in Results.

        Do two 2Q circuits: on 1st do nothing, and on 2nd do X on the first qubit.
        """
        backend = get_backend('local_qasm_simulator')
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc1 = QuantumCircuit(q, c)
        qc2 = QuantumCircuit(q, c)
        qc2.x(q[0])
        qc1.measure(q, c)
        qc2.measure(q, c)
        circuits = [qc1, qc2]
        xvals_dict = {circuits[0].name: 0, circuits[1].name: 1}
        qobj = compile(circuits, backend)
        job = backend.run(qobj)
        result = job.result()
        yvals, xvals = result.get_qubitpol_vs_xval(2, xvals_dict=xvals_dict)
        self.assertTrue(np.array_equal(yvals, [[-1, -1], [1, -1]]))
        self.assertTrue(np.array_equal(xvals, [0, 1]))

    def test_average_data(self):
        """Test average_data."""
        backend = get_backend('local_qasm_simulator')
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c, name="qc")
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        shots = 10000
        qobj = compile(qc, backend, shots=shots)
        job = backend.run(qobj)
        result = job.result()
        observable = {"00": 1, "11": 1, "01": -1, "10": -1}
        mean_zz = result.average_data("qc", observable)
        observable = {"00": 1, "11": -1, "01": 1, "10": -1}
        mean_zi = result.average_data("qc", observable)
        observable = {"00": 1, "11": -1, "01": -1, "10": 1}
        mean_iz = result.average_data("qc", observable)
        self.assertAlmostEqual(mean_zz, 1, places=1)
        self.assertAlmostEqual(mean_zi, 0, places=1)
        self.assertAlmostEqual(mean_iz, 0, places=1)

    def test_combine_results(self):
        """Test combining two results."""
        backend = get_backend('local_qasm_simulator')
        q = QuantumRegister(1)
        c = ClassicalRegister(1)
        qc1 = QuantumCircuit(q, c)
        qc2 = QuantumCircuit(q, c)
        qc1.measure(q[0], c[0])
        qc2.x(q[0])
        qc2.measure(q[0], c[0])
        qobj1 = compile(qc1, backend)
        qobj2 = compile(qc2, backend)
        job1 = backend.run(qobj1)
        job2 = backend.run(qobj2)
        result1 = job1.result()
        result2 = job2.result()
        counts1 = result1.get_counts(qc1)
        counts2 = result2.get_counts(qc2)
        result1 += result2
        counts12 = [result1.get_counts(qc1), result1.get_counts(qc2)]
        self.assertEqual(counts12, [counts1, counts2])
