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

from ddt import ddt, data
from qiskit.circuit import QuantumCircuit, CircuitError
from qiskit.circuit.library import PauliProductMeasurement
from qiskit.quantum_info import Pauli, Clifford
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
