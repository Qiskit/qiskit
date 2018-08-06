# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,unused-import

"""Tests for verifying the correctness of simulator extension instructions."""

import unittest
import qiskit
import qiskit.extensions.simulator
from qiskit.tools.qi.qi import state_fidelity
from qiskit.wrapper import execute
from .common import QiskitTestCase, requires_cpp_simulator


@requires_cpp_simulator
class TestExtensionsSimulator(QiskitTestCase):
    """Test instruction extensions for simulators:
    save, load, noise, snapshot, wait
    """
    _desired_fidelity = 0.99

    def test_save_load(self):
        """save |+>|0>, do some stuff, then load"""
        q = qiskit.QuantumRegister(2)
        c = qiskit.ClassicalRegister(2)
        circ = qiskit.QuantumCircuit(q, c)
        circ.h(q[0])
        circ.save(1)
        circ.cx(q[0], q[1])
        circ.cx(q[1], q[0])
        circ.h(q[1])
        circ.load(1)

        sim = 'local_statevector_simulator_cpp'
        result = execute(circ, sim).result()
        statevector = result.get_statevector()
        target = [0.70710678 + 0.j, 0.70710678 + 0.j, 0. + 0.j, 0. + 0.j]
        fidelity = state_fidelity(statevector, target)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "save-load statevector has low fidelity{0:.2g}.".format(fidelity))

    def test_snapshot(self):
        """snapshot a bell state in the middle of circuit"""
        q = qiskit.QuantumRegister(2)
        c = qiskit.ClassicalRegister(2)
        circ = qiskit.QuantumCircuit(q, c)
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.snapshot(3)
        circ.cx(q[0], q[1])
        circ.h(q[1])

        sim = 'local_statevector_simulator_cpp'
        result = execute(circ, sim).result()
        snapshot = result.get_snapshot(slot='3')
        target = [0.70710678 + 0.j, 0. + 0.j, 0. + 0.j, 0.70710678 + 0.j]
        fidelity = state_fidelity(snapshot, target)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "snapshot has low fidelity{0:.2g}.".format(fidelity))

    def test_noise(self):
        """turn on a pauli x noise for qubits 0 and 2"""
        q = qiskit.QuantumRegister(3)
        c = qiskit.ClassicalRegister(3)
        circ = qiskit.QuantumCircuit(q, c)
        circ.iden(q[0])
        circ.noise(0)
        circ.iden(q[1])
        circ.noise(1)
        circ.iden(q[2])
        circ.measure(q, c)

        config = {
            'noise_params': {
                'id': {'p_pauli': [1.0, 0.0, 0.0]}
            }
        }
        sim = 'local_qasm_simulator_cpp'
        shots = 1000
        result = execute(circ, sim, config=config, shots=shots).result()
        counts = result.get_counts()
        target = {'101': shots}
        self.assertEqual(counts, target)


if __name__ == '__main__':
    unittest.main(verbosity=2)
