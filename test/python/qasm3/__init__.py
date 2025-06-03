# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""QASM3 tests."""
import unittest
from qiskit import qasm3
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestQASM3load(QiskitTestCase):
    """Test qasm3 load and loads."""

    def test_num_qubits(self):
        """Test num_qubits equal the loaded circuit number of qubits"""
        program = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nh $0;\ncx $2, $1;\n'
        out = qasm3.loads(program, num_qubits=5)
        self.assertEqual(out.num_qubits, 5)


if __name__ == "__main__":
    unittest.main()
