# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,broad-except

"""Test IBMQ online qasm simulator.
TODO: Must expand tests. Re-evaluate after Aer."""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import transpiler
from qiskit.backends.ibmq import IBMQProvider
from .common import requires_qe_access, QiskitTestCase


class TestIbmqQasmSimulator(QiskitTestCase):
    """Test IBM Q Qasm Simulator."""

    @requires_qe_access
    def test_execute_one_circuit_simulator_online(self, qe_token, qe_url):
        """Test execute_one_circuit_simulator_online.

        If all correct should return correct counts.
        """
        provider = IBMQProvider(qe_token, qe_url)
        backend = provider.get_backend('ibmq_qasm_simulator')
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qobj = transpiler.compile(qc, backend, seed=73846087)
        shots = qobj.config.shots
        job = backend.run(qobj)
        result = job.result()
        counts = result.get_counts(qc)
        target = {'0': shots / 2, '1': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    @requires_qe_access
    def test_execute_several_circuits_simulator_online(self, qe_token, qe_url):
        """Test execute_several_circuits_simulator_online.

        If all correct should return correct counts.
        """
        provider = IBMQProvider(qe_token, qe_url)
        backend = provider.get_backend('ibmq_qasm_simulator')
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr)
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc1.measure(qr[0], cr[0])
        qc1.measure(qr[1], cr[1])
        qc2.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        shots = 1024
        qobj = transpiler.compile([qc1, qc2], backend, seed=73846087, shots=shots)
        job = backend.run(qobj)
        result = job.result()
        counts1 = result.get_counts(qc1)
        counts2 = result.get_counts(qc2)
        target1 = {'00': shots / 4, '01': shots / 4,
                   '10': shots / 4, '11': shots / 4}
        target2 = {'00': shots / 2, '11': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts1, target1, threshold)
        self.assertDictAlmostEqual(counts2, target2, threshold)

    @requires_qe_access
    def test_online_qasm_simulator_two_registers(self, qe_token, qe_url):
        """Test online_qasm_simulator_two_registers.

        If all correct should return correct counts.
        """
        provider = IBMQProvider(qe_token, qe_url)
        backend = provider.get_backend('ibmq_qasm_simulator')
        q1 = QuantumRegister(2)
        c1 = ClassicalRegister(2)
        q2 = QuantumRegister(2)
        c2 = ClassicalRegister(2)
        qc1 = QuantumCircuit(q1, q2, c1, c2)
        qc2 = QuantumCircuit(q1, q2, c1, c2)
        qc1.x(q1[0])
        qc2.x(q2[1])
        qc1.measure(q1[0], c1[0])
        qc1.measure(q1[1], c1[1])
        qc1.measure(q2[0], c2[0])
        qc1.measure(q2[1], c2[1])
        qc2.measure(q1[0], c1[0])
        qc2.measure(q1[1], c1[1])
        qc2.measure(q2[0], c2[0])
        qc2.measure(q2[1], c2[1])
        shots = 1024
        qobj = transpiler.compile([qc1, qc2], backend, seed=8458, shots=shots)
        job = backend.run(qobj)
        result = job.result()
        result1 = result.get_counts(qc1)
        result2 = result.get_counts(qc2)
        self.assertEqual(result1, {'00 01': 1024})
        self.assertEqual(result2, {'10 00': 1024})
