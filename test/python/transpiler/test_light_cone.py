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

"""Test the LightCone pass"""

import unittest
from test import QiskitTestCase  # pylint: disable=wrong-import-order

import ddt

from qiskit.circuit import (
    ClassicalRegister,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library import real_amplitudes
from qiskit.circuit.library.n_local.efficient_su2 import efficient_su2
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler.passes.optimization.light_cone import LightCone
from qiskit.transpiler.passmanager import PassManager


@ddt.ddt
class TestLightConePass(QiskitTestCase):
    """Test the LightCone pass."""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    @ddt.data("Y", "Z")
    def test_nonparameterized_noncommuting(self, pauli_label):
        """Test for a non-commuting, asymmetric, weight-one Pauli."""
        bit_terms = pauli_label
        light_cone = LightCone(bit_terms=bit_terms, indices=[1])
        pm = PassManager([light_cone])

        q0 = QuantumRegister(2, "q0")
        qc = QuantumCircuit(q0)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)

        new_circuit = pm.run(qc)

        expected = qc

        self.assertEqual(expected, new_circuit)

    def test_nonparameterized_commuting(self):
        """Test for a commuting, asymmetric, weight-one Pauli."""
        light_cone = LightCone(bit_terms="X", indices=[1])
        pm = PassManager([light_cone])

        q0 = QuantumRegister(2, "q0")
        qc = QuantumCircuit(q0)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(q0)
        expected.h(1)

        self.assertEqual(expected, new_circuit)

    @ddt.data("Y", "Z")
    def test_parameterized_noncommuting(self, pauli_label):
        """Test for a non-commuting, asymmetric, weight-one Pauli on a parameterized circuit."""
        bit_terms = pauli_label
        light_cone = LightCone(bit_terms=bit_terms, indices=[1])
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        q0 = QuantumRegister(2, "q0")
        qc = QuantumCircuit(q0)
        qc.rx(theta, 0)
        qc.ry(theta, 1)
        qc.cx(0, 1)

        new_circuit = pm.run(qc)

        expected = qc

        self.assertEqual(expected, new_circuit)

    def test_parameterized_commuting(self):
        """Test for a commuting, asymmetric, weight-one Pauli on a parameterized circuit."""
        light_cone = LightCone(bit_terms="X", indices=[1])
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        q0 = QuantumRegister(2, "q0")
        qc = QuantumCircuit(q0)
        qc.rx(theta, 0)
        qc.ry(theta, 1)
        qc.cx(0, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(q0)
        expected.ry(theta, 1)

        self.assertEqual(expected, new_circuit)

    def test_parameterized_symmetric(self):
        """Test for a double symmetric `Z` observable on a parameterized circuit."""
        light_cone = LightCone(bit_terms="ZZ", indices=[0, 3])
        pm = PassManager([light_cone])

        qc = real_amplitudes(4, entanglement="pairwise", reps=1)

        new_circuit = pm.run(qc)

        q0 = QuantumRegister(4, "q")
        expected = QuantumCircuit(q0)
        theta = qc.parameters
        expected.ry(theta[0], 0)
        expected.ry(theta[1], 1)
        expected.ry(theta[2], 2)
        expected.ry(theta[3], 3)
        expected.cx(0, 1)
        expected.cx(2, 3)
        expected.ry(theta[4], 0)
        expected.ry(theta[7], 3)

        self.assertEqual(expected, new_circuit)

    def test_parameterized_asymmetric(self):
        """Test for a double asymmetric observable on a parameterized circuit."""
        light_cone = LightCone(bit_terms="XZ", indices=[0, 2])
        pm = PassManager([light_cone])

        qc = efficient_su2(4, entanglement="circular", reps=1)

        new_circuit = pm.run(qc)

        q0 = QuantumRegister(4, "q")
        expected = QuantumCircuit(q0)
        theta = qc.parameters
        expected.ry(theta[0], 0)
        expected.ry(theta[1], 1)
        expected.ry(theta[2], 2)
        expected.ry(theta[3], 3)
        expected.rz(theta[4], 0)
        expected.rz(theta[5], 1)
        expected.rz(theta[6], 2)
        expected.rz(theta[7], 3)
        expected.cx(3, 0)
        expected.cx(0, 1)
        expected.cx(1, 2)
        expected.cx(2, 3)
        expected.ry(theta[8], 0)
        expected.ry(theta[10], 2)
        expected.rz(theta[14], 2)

        self.assertEqual(expected, new_circuit)

    def test_all_commuting(self):
        """Test for a circuit that fully commutes with an observable."""
        light_cone = LightCone(bit_terms="Z", indices=[0])
        pm = PassManager([light_cone])

        q0 = QuantumRegister(4, "q0")
        qc = QuantumCircuit(q0)
        qc.s(0)
        qc.z(0)
        qc.h(2)
        qc.cx(0, 1)
        qc.cx(2, 3)

        new_circuit = pm.run(qc)

        self.assertEqual(sum(new_circuit.count_ops().values()), 0)

    def test_commuting_block(self):
        """Test for a commuting block. Currently, gates are checked
        one by one and commuting blocks are thus ignored."""
        light_cone = LightCone(bit_terms="X", indices=[2])
        pm = PassManager([light_cone])

        q0 = QuantumRegister(5, "q0")
        qc = QuantumCircuit(q0)
        qc.cx(2, 1)
        qc.cx(3, 4)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(2, 1)
        qc.x(1)
        qc.cx(2, 1)

        new_circuit = pm.run(qc)

        q0 = QuantumRegister(5, "q0")
        expected = QuantumCircuit(q0)
        expected.cx(2, 1)
        expected.cx(3, 4)
        expected.cx(1, 0)
        expected.cx(2, 3)
        expected.cx(2, 1)
        expected.cx(2, 1)

        self.assertEqual(expected, new_circuit)

    def test_measurement_barriers(self):
        """Test for measurement and barriers."""
        light_cone = LightCone()
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        q0 = QuantumRegister(2, "q0")
        c0 = ClassicalRegister(1, "c0")
        qc = QuantumCircuit(q0, c0)
        qc.rz(theta, 1)
        qc.ry(theta, 0)
        qc.barrier()
        qc.cx(0, 1)
        qc.barrier()
        qc.rz(theta, 1)
        qc.measure(0, 0)

        new_circuit = pm.run(qc)

        q0 = QuantumRegister(2, "q0")
        c0 = ClassicalRegister(1, "c0")
        expected = QuantumCircuit(q0, c0)
        expected.rz(theta, 1)
        expected.ry(theta, 0)
        expected.barrier()
        expected.cx(0, 1)
        expected.barrier()
        expected.measure(0, 0)

        self.assertEqual(expected, new_circuit)

    def test_parameter_expression(self):
        """Test for Parameter expressions."""
        light_cone = LightCone(bit_terms="X", indices=[0])
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        q0 = QuantumRegister(2, "q0")
        qc = QuantumCircuit(q0)
        qc.rz(theta + 2, 1)
        qc.ry(theta - 2, 0)
        qc.cx(0, 1)
        qc.rz(theta * 2, 1)
        qc.rz(theta / 2, 1)

        new_circuit = pm.run(qc)

        q0 = QuantumRegister(2, "q0")
        expected = QuantumCircuit(q0)
        expected.rz(theta + 2, 1)
        expected.ry(theta - 2, 0)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    def test_big_circuit(self):
        """Test for large circuit and observable."""
        num_qubits = 120
        observable = Pauli("X" + "I" * (num_qubits - 4) + "YII")
        bit_terms, indices, _ = SparsePauliOp(observable).to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        q0 = QuantumRegister(num_qubits, "q0")
        qc = QuantumCircuit(q0)
        for i in range(num_qubits):
            qc.h(i)
        for i in range(0, num_qubits, 2):
            qc.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(num_qubits):
            qc.rz(theta, i)

        new_circuit = pm.run(qc)

        q0 = QuantumRegister(num_qubits, "q0")
        expected = QuantumCircuit(q0)
        expected.h(0)
        expected.h(1)
        expected.h(2)
        expected.h(3)
        expected.h(num_qubits - 2)
        expected.h(num_qubits - 1)
        expected.cx(0, 1)
        expected.cx(2, 3)
        expected.cx(1, 2)
        expected.rz(theta, 2)
        expected.cx(num_qubits - 2, num_qubits - 1)
        expected.rz(theta, num_qubits - 1)

        self.assertEqual(expected, new_circuit)


if __name__ == "__main__":
    unittest.main()
