# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test RemoveDiagonalGatesBeforeMeasure pass"""

import unittest

from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveSmallRotations
from qiskit.transpiler import PassManager
from qiskit.test import QiskitTestCase


class TesROptimizeSmallRotations(QiskitTestCase):
    """Test optimize_small_rotations"""

    def test_remove_small_rotation_gates(self):
        """Remove a 1-qubit small rotations"""

        c = QuantumCircuit(2)
        c.rz(1e-16, 0)
        c.crz(0, 0, 1)
        c.rx(0, 1)
        c.rx(3.141592, 1)

        pm = PassManager(RemoveSmallRotations())
        c = pm.run(c)
        self.assertEqual(len(c), 2)

        pm = PassManager(RemoveSmallRotations(epsilon=2e-16))
        c = pm.run(c)
        self.assertEqual(len(c), 1)


if __name__ == "__main__":
    unittest.main()
