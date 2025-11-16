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

"""Test Discretize Rotations optimization pass"""

from ddt import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import DiscretizeRotations
from qiskit.quantum_info import Operator, get_clifford_gate_names
from qiskit.circuit.library import RXGate, RYGate, RZGate
from test import combine, QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestDiscretizeRotations(QiskitTestCase):
    """Test the Discretize Rotations optimization pass."""

    @combine(multiple=[*range(0, 32)], gate=[RXGate, RYGate, RZGate])
    def test_rotation_gates(self, multiple, gate):
        """Test circuit with rotations gates are translated into Clifford+T+Tdg correctly."""
        qc = QuantumCircuit(1)
        angle = np.pi / 4 * multiple
        qc.append(gate(angle), [0])
        qct = DiscretizeRotations()(qc)
        clifford_t_names = get_clifford_gate_names() + ["t"] + ["tdg"]
        self.assertEqual(Operator(qct), Operator(qc))
        self.assertLessEqual(len(qct.count_ops()), 4)
        self.assertTrue(set(qct.count_ops().keys()).issubset(set(clifford_t_names)))
        self.assertLessEqual(
            len(set(qct.count_ops().keys()).intersection({"t", "tdg"})), 1
        )  # at most one t/tdg gate
        if multiple % 2 == 0:  # only clifford gates
            self.assertLessEqual(len(qct.count_ops()), 3)
            self.assertTrue(set(qct.count_ops().keys()).issubset(get_clifford_gate_names()))
