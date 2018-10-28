# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Tests for verifying the correctness of simulator extension instructions."""

import unittest
import qiskit
import qiskit.extensions.simulator
from qiskit import Aer
from qiskit.tools.qi.qi import state_fidelity
from qiskit import execute
from ..common import QiskitTestCase, requires_cpp_simulator


@requires_cpp_simulator
class TestExtensionsSimulator(QiskitTestCase):
    """Test instruction extensions for aer simulators:
    save, load, noise, snapshot, wait
    """
    _desired_fidelity = 0.99

    def test_save_load(self):
        """save |+>|0>, do some stuff, then load"""
        qr = qiskit.QuantumRegister(2)
        cr = qiskit.ClassicalRegister(2)
        circuit = qiskit.QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.save(1)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.h(qr[1])
        circuit.load(1)

        sim = Aer.get_backend('statevector_simulator')
        result = execute(circuit, sim).result()
        statevector = result.get_statevector()
        target = [0.70710678 + 0.j, 0.70710678 + 0.j, 0. + 0.j, 0. + 0.j]
        fidelity = state_fidelity(statevector, target)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "save-load statevector has low fidelity{0:.2g}.".format(fidelity))

    def test_snapshot(self):
        """snapshot a bell state in the middle of circuit"""
        qr = qiskit.QuantumRegister(2)
        cr = qiskit.ClassicalRegister(2)
        circuit = qiskit.QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.snapshot(3)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[1])

        sim = Aer.get_backend('statevector_simulator')
        result = execute(circuit, sim).result()
        snapshot = result.get_snapshot(slot='3')
        target = [0.70710678 + 0.j, 0. + 0.j, 0. + 0.j, 0.70710678 + 0.j]
        fidelity = state_fidelity(snapshot, target)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "snapshot has low fidelity{0:.2g}.".format(fidelity))

    def test_noise(self):
        """turn on a pauli x noise for qubits 0 and 2"""
        qr = qiskit.QuantumRegister(3)
        cr = qiskit.ClassicalRegister(3)
        circuit = qiskit.QuantumCircuit(qr, cr)
        circuit.iden(qr[0])
        circuit.noise(0)
        circuit.iden(qr[1])
        circuit.noise(1)
        circuit.iden(qr[2])
        circuit.measure(qr, cr)

        config = {
            'noise_params': {
                'id': {'p_pauli': [1.0, 0.0, 0.0]}
            }
        }
        sim = Aer.get_backend('qasm_simulator')
        shots = 1000
        result = execute(circuit, sim, config=config, shots=shots).result()
        counts = result.get_counts()
        target = {'101': shots}
        self.assertEqual(counts, target)


if __name__ == '__main__':
    unittest.main(verbosity=2)
