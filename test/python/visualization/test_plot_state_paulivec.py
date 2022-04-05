# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for plot_state_paulivec."""

import unittest
import matplotlib as mpl

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.quantum_info import Statevector
from qiskit.tools.visualization import plot_state_paulivec


class TestPlotStatePaulivec(QiskitTestCase):
    """Qiskit plot_state_paulivec tests."""

    def test_bell_state(self):
        """Test plotting Pauli vector of the bell state"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        state = Statevector.from_instruction(qc)
        fig = plot_state_paulivec(state, title="New PauliVec plot")
        self.assertIsInstance(fig, mpl.figure.Figure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
