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
from qiskit.quantum_info import state_fidelity
from qiskit.result.postprocess import format_statevector
from qiskit import execute
from qiskit.test import QiskitTestCase, requires_aer_provider


@requires_aer_provider
class TestExtensionsSimulator(QiskitTestCase):
    """Test instruction extensions for builtinsimulators simulators:
    save, load, noise, snapshot, wait
    """
    _desired_fidelity = 0.99

    def test_snapshot(self):
        """snapshot a bell state in the middle of circuit"""
        basis_gates = ['cx', 'u1', 'u2', 'u3', 'snapshot']
        qr = qiskit.QuantumRegister(2)
        cr = qiskit.ClassicalRegister(2)
        circuit = qiskit.QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.snapshot('3')
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[1])

        sim = qiskit.providers.aer.StatevectorSimulator()
        result = execute(circuit, sim, basis_gates=basis_gates).result()
        # TODO: rely on Result.get_statevector() postprocessing rather than manual
        snapshots = result.data(0)['snapshots']['statevector']['3']
        snapshot = format_statevector(snapshots[0])
        target = [0.70710678 + 0.j, 0. + 0.j, 0. + 0.j, 0.70710678 + 0.j]
        fidelity = state_fidelity(snapshot, target)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "snapshot has low fidelity{0:.2g}.".format(fidelity))


if __name__ == '__main__':
    unittest.main(verbosity=2)
