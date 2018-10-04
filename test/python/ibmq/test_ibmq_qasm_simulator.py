# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,broad-except

"""Test IBMQ online qasm simulator.
TODO: Must expand tests. Re-evaluate after Aer."""

from unittest import skip
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import transpiler
from qiskit import IBMQ
from ..common import requires_qe_access, QiskitTestCase


class TestIbmqQasmSimulator(QiskitTestCase):
    """Test IBM Q Qasm Simulator."""

    @requires_qe_access
    def test_execute_one_circuit_simulator_online(self, qe_token, qe_url):
        """Test execute_one_circuit_simulator_online.

        If all correct should return correct counts.
        """
        IBMQ.enable_account(qe_token, qe_url)
        backend = IBMQ.get_backend('ibmq_qasm_simulator')

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
        IBMQ.enable_account(qe_token, qe_url)
        backend = IBMQ.get_backend('ibmq_qasm_simulator')

        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qcr1 = QuantumCircuit(qr, cr)
        qcr2 = QuantumCircuit(qr, cr)
        qcr1.h(qr)
        qcr2.h(qr[0])
        qcr2.cx(qr[0], qr[1])
        qcr1.measure(qr[0], cr[0])
        qcr1.measure(qr[1], cr[1])
        qcr2.measure(qr[0], cr[0])
        qcr2.measure(qr[1], cr[1])
        shots = 1024
        qobj = transpiler.compile([qcr1, qcr2], backend, seed=73846087, shots=shots)
        job = backend.run(qobj)
        result = job.result()
        counts1 = result.get_counts(qcr1)
        counts2 = result.get_counts(qcr2)
        target1 = {'00': shots / 4, '01': shots / 4,
                   '10': shots / 4, '11': shots / 4}
        target2 = {'00': shots / 2, '11': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts1, target1, threshold)
        self.assertDictAlmostEqual(counts2, target2, threshold)

    # TODO: Investigate why this test is failing in master:
    # https://github.com/Qiskit/qiskit-terra/issues/1016
    @skip('Intermitent failure, see: https://github.com/Qiskit/qiskit-terra/issues/1016 ')
    @requires_qe_access
    def test_online_qasm_simulator_two_registers(self, qe_token, qe_url):
        """Test online_qasm_simulator_two_registers.

        If all correct should return correct counts.
        """
        IBMQ.enable_account(qe_token, qe_url)
        backend = IBMQ.get_backend('ibmq_qasm_simulator')

        qr1 = QuantumRegister(2)
        cr1 = ClassicalRegister(2)
        qr2 = QuantumRegister(2)
        cr2 = ClassicalRegister(2)
        qcr1 = QuantumCircuit(qr1, qr2, cr1, cr2)
        qcr2 = QuantumCircuit(qr1, qr2, cr1, cr2)
        qcr1.x(qr1[0])
        qcr2.x(qr2[1])
        qcr1.measure(qr1[0], cr1[0])
        qcr1.measure(qr1[1], cr1[1])
        qcr1.measure(qr2[0], cr2[0])
        qcr1.measure(qr2[1], cr2[1])
        qcr2.measure(qr1[0], cr1[0])
        qcr2.measure(qr1[1], cr1[1])
        qcr2.measure(qr2[0], cr2[0])
        qcr2.measure(qr2[1], cr2[1])
        shots = 1024
        qobj = transpiler.compile([qcr1, qcr2], backend, seed=8458, shots=shots)
        job = backend.run(qobj)
        result = job.result()
        result1 = result.get_counts(qcr1)
        result2 = result.get_counts(qcr2)
        self.assertEqual(result1, {'00 01': 1024})
        self.assertEqual(result2, {'10 00': 1024})
