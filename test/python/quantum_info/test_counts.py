# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,missing-docstring

"""Quick program to test the quantum information states modules."""

import unittest

from qiskit import execute, QuantumCircuit, BasicAer
from qiskit.quantum_info.states import state_to_counts
from qiskit.test import QiskitTestCase


class TestStates(QiskitTestCase):
    """Tests for qiskit.quantum_info.states.counts"""

    def test_state_to_counts(self):
        """Test statevector to counts"""
        qc = QuantumCircuit(5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(3, 4)
        sim = BasicAer.get_backend('statevector_simulator')
        res = execute(qc, sim).result()
        vec = res.get_statevector()
        counts = state_to_counts(vec)
        self.assertAlmostEqual(counts['00000'], 0.5)
        self.assertAlmostEqual(counts['11111'], 0.5)

    def test_counts_from_result(self):
        """Get counts from statevector result"""
        qc = QuantumCircuit(5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(3, 4)
        sim = BasicAer.get_backend('statevector_simulator')
        res = execute(qc, sim).result()
        counts = res.get_counts()
        self.assertAlmostEqual(counts['00000'], 0.5)
        self.assertAlmostEqual(counts['11111'], 0.5)


if __name__ == '__main__':
    unittest.main()
