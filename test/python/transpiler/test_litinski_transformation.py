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

import numpy as np

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import QFTGate, PauliEvolutionGate, PauliProductMeasurement
from qiskit.circuit.random import random_clifford_circuit
from qiskit.compiler import transpile
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import LitinskiTransformation
from qiskit.quantum_info import Operator, Pauli, SparseObservable
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

    def test_omit_clifford_gates(self):
        """Test fix_clifford."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.1, 0)
        qc.cx(0, 2)
        qc.rz(-0.4, 1)
        qc.s(2)
        qc.rz(0.1, 1)

        qct = LitinskiTransformation(fix_clifford=False)(qc)

        self.assertEqual(qct.count_ops(), {"PauliEvolution": 3})

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

    def test_all_supported_clifford_gates(self):
        """Test circuit with all of the supported clifford gates."""

        qc = QuantumCircuit(4)

        # Put all possible Clifford gates at the front of the circuit,
        # so that the algorithm will need to combine these into a Clifford,
        # and commute through the rotation gates.
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.s(0)
        qc.sdg(1)
        qc.cz(0, 2)
        qc.cz(1, 3)
        qc.z(0)
        qc.h(1)
        qc.x(2)
        qc.y(3)
        qc.swap(0, 1)
        qc.sx(0)
        qc.sxdg(1)
        qc.cy(1, 2)
        qc.id(3)
        qc.ecr(0, 3)
        qc.iswap(1, 2)
        qc.dcx(1, 3)

        # Rotations
        qc.t(0)
        qc.rz(0.1, 1)
        qc.tdg(2)
        qc.rz(-0.2, 3)

        qc_litinski = LitinskiTransformation()(qc)
        ops_litinski = qc_litinski.count_ops()

        # make sure the transform was applied
        self.assertNotIn("t", ops_litinski)
        self.assertNotIn("tdg", ops_litinski)
        self.assertNotIn("rz", ops_litinski)

        # make sure the result is correct
        self.assertEqual(Operator(qc_litinski), Operator(qc))

    def test_random_circuits(self):
        """Test on random Clifford+T circuits."""

        for trial in range(10):
            start_seed = 1234 + 10 * trial

            # create a circuit with multiple layers of Clifford and T/Tdg gates
            qc = QuantumCircuit(5)
            for layer in range(5):
                clifford_circuit = random_clifford_circuit(
                    num_qubits=5, num_gates=20, gates="all", seed=start_seed + layer
                )
                qc.compose(clifford_circuit, inplace=True)
                qc.t(0)
                qc.tdg(1)

            # apply the transform
            qc_litinski = LitinskiTransformation()(qc)
            ops_litinski = qc_litinski.count_ops()

            # make sure the transform was applied
            self.assertNotIn("t", ops_litinski)
            self.assertNotIn("tdg", ops_litinski)

            # make sure the result is correct
            self.assertEqual(Operator(qc_litinski), Operator(qc))

    def test_raises_on_unsupported_gates(self):
        """Test that the pass returns an error when it runs on unsupported gates."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.1, 0)
        qc.cp(0.1, 0, 1)  # unsupported
        qc.cx(0, 2)

        with self.assertRaises(TranspilerError):
            _ = LitinskiTransformation()(qc)

    def test_t(self):
        """Test the transform on a circuit with a T-gate."""
        qc = QuantumCircuit(2)
        qc.t(0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 8)
        expected.append(PauliEvolutionGate(SparseObservable("Z"), np.pi / 8), [0])

        self.assertEqual(qct, expected)

    def test_tdg(self):
        """Test the transform on a circuit with a Tdg-gate."""
        qc = QuantumCircuit(2)
        qc.tdg(0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, global_phase=-np.pi / 8)
        expected.append(PauliEvolutionGate(SparseObservable("Z"), -np.pi / 8), [0])

        self.assertEqual(qct, expected)

    def test_h_t(self):
        """Test the transform on a circuit with an H-gate and a T-gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.t(0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 8)
        expected.append(PauliEvolutionGate(SparseObservable("X"), np.pi / 8), [0])
        expected.h(0)

        self.assertEqual(qct, expected)

    def test_h_tdg(self):
        """Test the transform on a circuit with an H-gate and a Tdg-gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.tdg(0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, global_phase=-np.pi / 8)
        expected.append(PauliEvolutionGate(SparseObservable("X"), -np.pi / 8), [0])
        expected.h(0)

        self.assertEqual(qct, expected)

    def test_sx_t(self):
        """Test the transform on a circuit with an SX-gate and a T-gate."""
        qc = QuantumCircuit(2)
        qc.sx(0)
        qc.t(0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 8)
        expected.append(PauliEvolutionGate(SparseObservable("Y"), np.pi / 8), [0])
        expected.sx(0)

        self.assertEqual(qct, expected)

    def test_sx_tdg(self):
        """Test the transform on a circuit with an SX-gate and a Tdg-gate."""
        qc = QuantumCircuit(2)
        qc.sx(0)
        qc.tdg(0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, global_phase=-np.pi / 8)
        expected.append(PauliEvolutionGate(SparseObservable("Y"), -np.pi / 8), [0])
        expected.sx(0)

        self.assertEqual(qct, expected)

    def test_cx_t(self):
        """Test the transform on a circuit with a CX-gate and a T-gate."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.t(1)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 8)
        expected.append(PauliEvolutionGate(SparseObservable("ZZ"), np.pi / 8), [0, 1])
        expected.cx(0, 1)

        self.assertEqual(qct, expected)

    def test_cx_tdg(self):
        """Test the transform on a circuit with a CX-gate and a Tdg-gate."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.tdg(1)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, global_phase=-np.pi / 8)
        expected.append(PauliEvolutionGate(SparseObservable("ZZ"), -np.pi / 8), [0, 1])
        expected.cx(0, 1)

        self.assertEqual(qct, expected)

    def test_measure(self):
        """Test the transform on a circuit with a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("Z")), [0], [0])

        self.assertEqual(qct, expected)

    def test_h_measure(self):
        """Test the transform on a circuit with an H-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("X")), [0], [0])
        expected.h(0)

        self.assertEqual(qct, expected)

    def test_sx_measure(self):
        """Test the transform on a circuit with an SX-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.sx(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("Y")), [0], [0])
        expected.sx(0)

        self.assertEqual(qct, expected)

    def test_cx_measure(self):
        """Test the transform on a circuit with an CX-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.measure(1, 1)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("ZZ")), [0, 1], [1])
        expected.cx(0, 1)

        self.assertEqual(qct, expected)

    def test_x_measure(self):
        """Test the transform on a circuit with a X-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("-Z")), [0], [0])
        expected.x(0)

        self.assertEqual(qct, expected)

    def test_h_x_measure(self):
        """Test the transform on a circuit with an H-gate, an X-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.x(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("-X")), [0], [0])
        expected.h(0)
        expected.x(0)

        self.assertEqual(qct, expected)

    def test_sx_x_measure(self):
        """Test the transform on a circuit with an H-gate, an X-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.sx(0)
        qc.x(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation()(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("-Y")), [0], [0])
        expected.sx(0)
        expected.x(0)

        self.assertEqual(qct, expected)

    def test_on_circuits_with_measures(self):
        """Test the Litinski transformation pass on a more complex with Clifford gates,
        T gates and Z-measures.
        """
        # This is the example from Figure 4 in the paper "A Game of Surface Codes" by Litinski.

        # The original circuit (as shown at the top-left of the figure).
        qc = QuantumCircuit(4, 4)
        qc.t(0)
        qc.cx(2, 1)
        qc.sxdg(3)
        qc.cx(1, 0)
        qc.sx(2)
        qc.t(3)
        qc.cx(3, 0)
        qc.t(0)
        qc.s(1)
        qc.t(2)
        qc.s(3)
        qc.sxdg(0)
        qc.sx(1)
        qc.sx(2)
        qc.sx(3)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        qc.measure(3, 3)

        # Apply the Litinski transform with fix_cliffords=False (ignoring the Clifford gates
        # at the end of the transformed circuit, and clearing the global phase).
        qct = LitinskiTransformation(fix_clifford=False)(qc)
        qct.global_phase = 0

        # The transformed circuit (as shown at the bottom-right of the figure).
        expected = QuantumCircuit(4, 4)
        expected.append(PauliEvolutionGate(SparseObservable.from_list([("Z", 1)]), np.pi / 8), [0])
        expected.append(
            PauliEvolutionGate(SparseObservable.from_list([("YX", 1)]), np.pi / 8), [1, 2]
        )
        expected.append(PauliEvolutionGate(SparseObservable.from_list([("Y", 1)]), -np.pi / 8), [3])
        expected.append(
            PauliEvolutionGate(SparseObservable.from_list([("YZZZ", 1)]), -np.pi / 8), [0, 1, 2, 3]
        )
        expected.append(PauliProductMeasurement(Pauli("YZZY")), [0, 1, 2, 3], [0])
        expected.append(PauliProductMeasurement(Pauli("XX")), [0, 1], [1])
        expected.append(PauliProductMeasurement(Pauli("-Z")), [2], [2])
        expected.append(PauliProductMeasurement(Pauli("XX")), [0, 3], [3])

        self.assertEqual(qct, expected)
