# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test MCX synthesis."""

import unittest

from qiskit.quantum_info import Operator
from qiskit.circuit.library import C3XGate
from qiskit.synthesis.multi_controlled import synth_c3x

from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestMCXSynth(QiskitTestCase):
    """Test MCX synthesis methods."""

    def test_c3x(self):
        """Test the default synthesis method for C3XGate."""
        # ToDo: it might be nicer to compare with the actual matrix
        self.assertEqual(Operator(C3XGate().definition), Operator(synth_c3x()))


if __name__ == "__main__":
    unittest.main()
