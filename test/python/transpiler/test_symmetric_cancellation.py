# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Testing SymmetricCancellation
"""

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes import SymmetricCancellation


class TestSymmetricCancellation(QiskitTestCase):
    """Test the InverseCancellation transpiler pass."""

    def test_swap_gate_cancellation(self):
        qc = QuantumCircuit(3)
        # same qargs
        qc.swap(0, 1)
        qc.swap(0, 1)
        qc.x(2)
        qc.h([1, 2])
        # different qargs
        qc.swap(1, 2)
        qc.swap(2, 1)
        qc.y(2)
        qc.x(0)

        pass_ = SymmetricCancellation()
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("swap", gates_after)
