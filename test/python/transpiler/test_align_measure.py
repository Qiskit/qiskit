# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Transpiler testing"""

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.passes import AlignMeasures


class TestAlignMeasures(QiskitTestCase):

    def setUp(self):
        super().setUp()
        instruction_durations = InstructionDurations()
        instruction_durations.update(
            [
                ("rz", (0,), 0),
                ("rz", (1,), 0),
                ("x", (0,), 160),
                ("x", (1,), 160),
                ("sx", (0,), 160),
                ("sx", (1,), 160),
                ("cx", (0, 1), 800),
                ("cx", (1, 0), 800),
                ("measure", (0,), 1600),
                ("measure", (1,), 1600),
            ]
        )
        self.align_measure_pass = AlignMeasures(alignment=16, durations=instruction_durations)

    def test_t1_experiment_type(self):
        """Test T1 experiment type circuit."""
        circuit = QuantumCircuit(1, 1)
        circuit.x(0)
        circuit.delay(100, 0, unit="dt")
        circuit.measure(0, 0)

        transpiled = self.align_measure_pass(circuit, property_set={"time_unit": "dt"})

        ref_circuit = QuantumCircuit(1, 1)
        ref_circuit.x(0)
        ref_circuit.delay(112, 0, unit="dt")
        ref_circuit.measure(0, 0)

        self.assertEqual(transpiled, ref_circuit)

    def test_hanh_echo_experiment_type(self):
        """Test Hahn echo experiment type circuit."""
        circuit = QuantumCircuit(1, 1)
        circuit.sx(0)
        circuit.delay(100, 0, unit="dt")
        circuit.x(0)
        circuit.delay(100, 0, unit="dt")
        circuit.sx(0)
        circuit.measure(0, 0)

        transpiled = self.align_measure_pass(circuit, property_set={"time_unit": "dt"})

        ref_circuit = QuantumCircuit(1, 1)
        ref_circuit.sx(0)
        ref_circuit.delay(100, 0, unit="dt")
        ref_circuit.x(0)
        ref_circuit.delay(100, 0, unit="dt")
        ref_circuit.sx(0)
        ref_circuit.delay(8, 0, unit="dt")
        ref_circuit.measure(0, 0)

        self.assertEqual(transpiled, ref_circuit)

    def test_mid_circuit_measure(self):
        """Test circuit with mid circuit measurement."""
        circuit = QuantumCircuit(1, 2)
        circuit.x(0)
        circuit.delay(100, 0, unit="dt")
        circuit.measure(0, 0)
        circuit.delay(10, 0, unit="dt")
        circuit.x(0)
        circuit.delay(120, 0, unit="dt")
        circuit.measure(0, 1)

        transpiled = self.align_measure_pass(circuit, property_set={"time_unit": "dt"})

        ref_circuit = QuantumCircuit(1, 2)
        ref_circuit.x(0)
        ref_circuit.delay(112, 0, unit="dt")
        ref_circuit.measure(0, 0)
        ref_circuit.delay(10, 0, unit="dt")
        ref_circuit.x(0)
        ref_circuit.delay(134, 0, unit="dt")
        ref_circuit.measure(0, 1)

        self.assertEqual(transpiled, ref_circuit)

    def test_mid_circuit_multiq_gates(self):
        """Test circuit with mid circuit measurement and multi qubit gates."""
        circuit = QuantumCircuit(2, 2)
        circuit.x(0)
        circuit.delay(100, 0, unit="dt")
        circuit.measure(0, 0)
        circuit.cx(0, 1)
        circuit.measure(1, 1)
        circuit.cx(0, 1)
        circuit.measure(0, 0)

        transpiled = self.align_measure_pass(circuit, property_set={"time_unit": "dt"})

        ref_circuit = QuantumCircuit(2, 2)
        ref_circuit.x(0)
        ref_circuit.delay(112, 0, unit="dt")
        ref_circuit.measure(0, 0)
        ref_circuit.delay(160 + 112 + 1600, 1, unit="dt")
        ref_circuit.cx(0, 1)
        ref_circuit.delay(1600, 0, unit="dt")
        ref_circuit.measure(1, 1)
        ref_circuit.cx(0, 1)
        ref_circuit.delay(1600, 1, unit="dt")
        ref_circuit.measure(0, 0)

        self.assertEqual(transpiled, ref_circuit)
