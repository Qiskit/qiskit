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

"""Test Litinski transformation pass"""

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import QFTGate
from qiskit.compiler import transpile
from qiskit.transpiler.passes import LitinskiTransformation
from qiskit.quantum_info import Operator
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestLitinskiTransformation(QiskitTestCase):
    """Test the Litinski Transformation pass."""

    def test_t_tdg_gates(self):
        """Test circuit with T/Tdg gates."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.t(0)
        qc.cx(0, 2)
        qc.t(1)
        qc.tdg(0)
        qc.s(2)
        qc.t(2)

        qct = LitinskiTransformation()(qc)

        self.assertEqual(qct.count_ops(), {"PauliEvolution": 4, "cx": 2, "h": 1, "s": 1})
        self.assertEqual(Operator(qct), Operator(qc))

    def test_rz_gates(self):
        """Test circuit with RZ-rotation gates."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.1, 0)
        qc.cx(0, 2)
        qc.rz(-0.4, 1)
        qc.s(2)
        qc.rz(0.1, 1)

        qct = LitinskiTransformation()(qc)

        self.assertEqual(qct.count_ops(), {"PauliEvolution": 3, "cx": 2, "h": 1, "s": 1})
        self.assertEqual(Operator(qct), Operator(qc))

    def test_parametric_rz_gates(self):
        """Test circuit with parameterized RZ-rotation gates."""
        alpha = Parameter("alpha")
        beta = Parameter("beta")

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(alpha, 0)
        qc.cx(0, 2)
        qc.rz(beta, 1)
        qc.s(2)
        qc.rz(0.1, 1)

        qct = LitinskiTransformation()(qc)
        self.assertEqual(qct.count_ops(), {"PauliEvolution": 3, "cx": 2, "h": 1, "s": 1})

        qc_bound = qc.assign_parameters([0.123, -1.234])
        qct_bound = qct.assign_parameters([0.123, -1.234])
        self.assertEqual(Operator(qct_bound), Operator(qc_bound))

    @data(2, 3, 4, 5, 6, 7, 8)
    def test_qft_circuits(self, num_qubits):
        """Test more complex circuits produced by transpiling QFT gates into [cx, sx, rz] basis."""
        qc = QuantumCircuit(num_qubits)
        qc.append(QFTGate(num_qubits), range(num_qubits))

        # transpile the circuit into ["cx", "rz", "sx"] so that Litinski's transform can be applied
        qc = transpile(qc, basis_gates=["cx", "rz", "sx"])

        # apply Litinski's transform
        qc_litinski = LitinskiTransformation()(qc)

        # make sure the transform was applied
        self.assertNotIn("rz", qc_litinski.count_ops())
        # make sure the result is correct
        self.assertEqual(Operator(qc_litinski), Operator(qc))
