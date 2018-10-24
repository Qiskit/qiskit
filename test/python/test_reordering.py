# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=no-value-for-parameter,broad-except

"""Tests for bit reordering fix."""

import unittest
import qiskit

from qiskit import transpiler
from qiskit.backends.ibmq import least_busy
from .common import requires_qe_access, QiskitTestCase, slow_test


class TestBitReordering(QiskitTestCase):
    """Test QISKit's fix for the ibmq hardware reordering bug.

    The bug will be fixed with the introduction of qobj,
    in which case these tests can be used to verify correctness.
    """
    @slow_test
    @requires_qe_access
    def test_basic_reordering(self, qe_token, qe_url):
        """a simple reordering within a 2-qubit register"""
        sim, real = self._get_backends(qe_token, qe_url)
        if not sim or not real:
            raise unittest.SkipTest('no remote device available')

        qr = qiskit.QuantumRegister(2)
        cr = qiskit.ClassicalRegister(2)
        circuit = qiskit.QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[1])
        circuit.measure(qr[1], cr[0])

        shots = 2000
        qobj_real = transpiler.compile(circuit, real, shots=shots)
        qobj_sim = transpiler.compile(circuit, sim, shots=shots)
        result_real = real.run(qobj_real).result(timeout=600)
        result_sim = sim.run(qobj_sim).result(timeout=600)
        counts_real = result_real.get_counts()
        counts_sim = result_sim.get_counts()
        self.log.info(counts_real)
        self.log.info(counts_sim)
        threshold = 0.1 * shots
        self.assertDictAlmostEqual(counts_real, counts_sim, threshold)

    @slow_test
    @requires_qe_access
    def test_multi_register_reordering(self, qe_token, qe_url):
        """a more complicated reordering across 3 registers of different sizes"""
        sim, real = self._get_backends(qe_token, qe_url)
        if not sim or real:
            raise unittest.SkipTest('no remote device available')

        qr0 = qiskit.QuantumRegister(2)
        qr1 = qiskit.QuantumRegister(2)
        qr2 = qiskit.QuantumRegister(1)
        cr0 = qiskit.ClassicalRegister(2)
        cr1 = qiskit.ClassicalRegister(2)
        cr2 = qiskit.ClassicalRegister(1)
        circuit = qiskit.QuantumCircuit(qr0, qr1, qr2, cr0, cr1, cr2)
        circuit.h(qr0[0])
        circuit.cx(qr0[0], qr2[0])
        circuit.x(qr1[1])
        circuit.h(qr2[0])
        circuit.cx(qr2[0], qr1[0])
        circuit.barrier()
        circuit.measure(qr0[0], cr2[0])
        circuit.measure(qr0[1], cr0[1])
        circuit.measure(qr1[0], cr0[0])
        circuit.measure(qr1[1], cr1[0])
        circuit.measure(qr2[0], cr1[1])

        shots = 4000
        qobj_real = transpiler.compile(circuit, real, shots=shots)
        qobj_sim = transpiler.compile(circuit, sim, shots=shots)
        result_real = real.run(qobj_real).result(timeout=600)
        result_sim = sim.run(qobj_sim).result(timeout=600)
        counts_real = result_real.get_counts()
        counts_sim = result_sim.get_counts()
        threshold = 0.2 * shots
        self.assertDictAlmostEqual(counts_real, counts_sim, threshold)

    def _get_backends(self, qe_token, qe_url):
        sim_backend = qiskit.Aer.get_backend('qasm_simulator')
        try:
            qiskit.IBMQ.enable_account(qe_token, qe_url)
            real_backends = qiskit.IBMQ.backends(simulator=False)
            real_backend = least_busy(real_backends)
        except Exception:
            real_backend = None

        return sim_backend, real_backend


if __name__ == '__main__':
    unittest.main(verbosity=2)
