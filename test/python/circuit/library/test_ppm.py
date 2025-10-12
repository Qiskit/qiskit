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

"""A Test for Pauli Product Measurement instruction."""

import io

from ddt import ddt, data
from qiskit.circuit import QuantumCircuit, CircuitError
from qiskit.circuit.library import PauliProductMeasurement
from qiskit.quantum_info import Pauli, Clifford
from qiskit.qpy import dump, load
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestPauliProductMeasurement(QiskitTestCase):
    """Test the PauliProductMeasurement Instruction."""

    @data("-XIYZ", "ZIYXY", "-Y", "-ZXII")
    def test_pauli_evolution(self, p):
        """Asserts that the pauli evolution is correct and that
        the circuit around the measure reduces to identity."""
        pauli = Pauli(p)
        num_qubits = pauli.num_qubits
        ppm = PauliProductMeasurement(pauli)
        qc_before_meas = QuantumCircuit(num_qubits)
        qc_no_meas = QuantumCircuit(num_qubits)
        for inst in ppm.definition.data:
            if inst.operation.name != "measure":
                qc_no_meas.append(inst.operation, inst.qubits)
        for inst in ppm.definition.data:
            if inst.operation.name == "measure":
                break
            qc_before_meas.append(inst.operation, inst.qubits)

        ind_z = 0
        for q in pauli:
            if Pauli(q) != Pauli("I"):
                break
            ind_z += 1
        cliff = Clifford(qc_before_meas)
        pauli_z = Pauli((num_qubits - 1 - ind_z) * "I" + "Z" + ind_z * "I")
        self.assertEqual(pauli_z.evolve(cliff), pauli)
        self.assertEqual(Clifford(qc_no_meas), Clifford(QuantumCircuit(num_qubits)))

    @data("-iX", "iZY")
    def test_wrong_phase_raises(self, p):
        """Test that a Pauli with phase i or -i raises an error."""
        with self.assertRaises(CircuitError):
            _ = PauliProductMeasurement(Pauli(p))

    def test_inverse_raises(self):
        """Test that the inverse method raises an error."""
        with self.assertRaises(CircuitError):
            _ = PauliProductMeasurement(Pauli("XYZ")).inverse()

    def test_qpy(self):
        """Test qpy for circuits with PauliProductMeasurement objects."""
        qc = QuantumCircuit(6, 2)
        qc.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [1])
        qc.append(PauliProductMeasurement(Pauli("Z")), [2], [0])
        qc.append(PauliProductMeasurement(Pauli("ZZ")), [3, 2], [0])
        qc.h(0)

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_gate_equality(self):
        """Test checking equality of PauliProductMeasurement objects."""
        self.assertEqual(PauliProductMeasurement(Pauli("XZ")), PauliProductMeasurement(Pauli("XZ")))
        self.assertNotEqual(
            PauliProductMeasurement(Pauli("XZ")), PauliProductMeasurement(Pauli("XX"))
        )

    def test_circuit_with_gate_equality(self):
        """Test checking equality of circuits with PauliProductMeasurement objects."""
        qc1 = QuantumCircuit(5, 2)
        qc1.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [1])

        qc2 = QuantumCircuit(5, 2)
        qc2.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [1])

        qc3 = QuantumCircuit(5, 2)
        qc3.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [0])

        qc4 = QuantumCircuit(5, 2)
        qc4.append(PauliProductMeasurement(Pauli("ZX")), [4, 1], [1])

        self.assertEqual(qc1, qc2)
        self.assertNotEqual(qc1, qc3)
        self.assertNotEqual(qc1, qc4)
