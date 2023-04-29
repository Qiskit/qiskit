# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring


"""
Tests for singleton gate behavior
"""

import copy

from qiskit.circuit.library import HGate
from qiskit.circuit import Clbit, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit.test.base import QiskitTestCase


class TestSingletonGate(QiskitTestCase):
    """Qiskit SingletonGate tests."""

    def test_default_singleton(self):
        gate = HGate()
        new_gate = HGate()
        self.assertIs(gate, new_gate)

    def test_label_not_singleton(self):
        gate = HGate()
        label_gate = HGate(label="special")
        self.assertIsNot(gate, label_gate)

    def test_condition_not_singleton(self):
        gate = HGate()
        condition_gate = HGate().c_if(Clbit(), 0)
        self.assertIsNot(gate, condition_gate)

    def test_raise_on_state_mutation(self):
        gate = HGate()
        with self.assertRaises(NotImplementedError):
            gate.label = "foo"
        with self.assertRaises(NotImplementedError):
            gate.condition = (Clbit(), 0)

    def test_labeled_condition(self):
        singleton_gate = HGate()
        clbit = Clbit()
        gate = HGate(label="conditionally special").c_if(clbit, 0)
        self.assertIsNot(singleton_gate, gate)
        self.assertEqual(gate.label, "conditionally special")
        self.assertEqual(gate.condition, (clbit, 0))

    def test_default_singleton_copy(self):
        gate = HGate()
        copied = gate.copy()
        self.assertIs(gate, copied)

    def test_label_copy(self):
        gate = HGate(label="special")
        copied = gate.copy()
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)

    def test_label_copy_new(self):
        gate = HGate()
        label_gate = HGate(label="special")
        self.assertIsNot(gate, label_gate)
        self.assertNotEqual(gate.label, label_gate.label)
        copied = gate.copy()
        copied_label = label_gate.copy()
        self.assertIs(gate, copied)
        self.assertIsNot(copied, label_gate)
        self.assertIsNot(copied_label, gate)
        self.assertIsNot(copied_label, label_gate)
        self.assertNotEqual(copied.label, label_gate.label)
        self.assertEqual(copied_label, label_gate)
        self.assertNotEqual(copied.label, "special")
        self.assertEqual(copied_label.label, "special")

    def test_condition_copy(self):
        gate = HGate().c_if(Clbit(), 0)
        copied = gate.copy()
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)

    def test_condition_label_copy(self):
        clbit = Clbit()
        gate = HGate(label="conditionally special").c_if(clbit, 0)
        copied = gate.copy()
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)
        self.assertEqual(copied.label, "conditionally special")
        self.assertEqual(copied.condition, (clbit, 0))

    def test_deepcopy(self):
        gate = HGate()
        copied = copy.deepcopy(gate)
        self.assertIs(gate, copied)

    def test_deepcopy_with_label(self):
        gate = HGate(label="special")
        copied = copy.deepcopy(gate)
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)
        self.assertEqual(copied.label, "special")

    def test_deepcopy_with_condition(self):
        gate = HGate().c_if(Clbit(), 0)
        copied = copy.deepcopy(gate)
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)

    def test_condition_label_deepcopy(self):
        clbit = Clbit()
        gate = HGate(label="conditionally special").c_if(clbit, 0)
        copied = copy.deepcopy(gate)
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)
        self.assertEqual(copied.label, "conditionally special")
        self.assertEqual(copied.condition, (clbit, 0))

    def test_label_deepcopy_new(self):
        gate = HGate()
        label_gate = HGate(label="special")
        self.assertIsNot(gate, label_gate)
        self.assertNotEqual(gate.label, label_gate.label)
        copied = copy.deepcopy(gate)
        copied_label = copy.deepcopy(label_gate)
        self.assertIs(gate, copied)
        self.assertIsNot(copied, label_gate)
        self.assertIsNot(copied_label, gate)
        self.assertIsNot(copied_label, label_gate)
        self.assertNotEqual(copied.label, label_gate.label)
        self.assertEqual(copied_label, label_gate)
        self.assertNotEqual(copied.label, "special")
        self.assertEqual(copied_label.label, "special")

    def test_control_a_singleton(self):
        singleton_gate = HGate()
        gate = HGate(label="special")
        ch = gate.control(label="my_ch")
        self.assertEqual(ch.base_gate.label, "special")
        self.assertIsNot(ch.base_gate, singleton_gate)

    def test_round_trip_dag_conversion(self):
        qc = QuantumCircuit(1)
        gate = HGate()
        qc.append(gate, [0])
        dag = circuit_to_dag(qc)
        out = dag_to_circuit(dag)
        self.assertIs(qc.data[0].operation, out.data[0].operation)

    def test_round_trip_dag_conversion_with_label(self):
        gate = HGate(label="special")
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        dag = circuit_to_dag(qc)
        out = dag_to_circuit(dag)
        self.assertIsNot(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(out.data[0].operation.label, "special")

    def test_round_trip_dag_conversion_with_condition(self):
        qc = QuantumCircuit(1, 1)
        gate = HGate().c_if(qc.cregs[0], 0)
        qc.append(gate, [0])
        dag = circuit_to_dag(qc)
        out = dag_to_circuit(dag)
        self.assertIsNot(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(out.data[0].operation.condition, (qc.cregs[0], 0))

    def test_round_trip_dag_conversion_condition_label(self):
        qc = QuantumCircuit(1, 1)
        gate = HGate(label="conditionally special").c_if(qc.cregs[0], 0)
        qc.append(gate, [0])
        dag = circuit_to_dag(qc)
        out = dag_to_circuit(dag)
        self.assertIsNot(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(out.data[0].operation.condition, (qc.cregs[0], 0))
        self.assertEqual(out.data[0].operation.label, "conditionally special")

    def test_condition_via_instructionset(self):
        gate = HGate()
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0]).c_if(cr, 1)
        self.assertIsNot(gate, circuit.data[0].operation)
        self.assertEqual(circuit.data[0].operation.condition, (cr, 1))
