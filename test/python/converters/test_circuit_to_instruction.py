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

"""Tests for the converters."""

import unittest

from qiskit.converters import circuit_to_instruction
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit, Instruction
from qiskit.circuit import Parameter
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError


class TestCircuitToInstruction(QiskitTestCase):
    """Test Circuit to Instruction."""

    def test_flatten_circuit_registers(self):
        """Check correct flattening"""
        qr1 = QuantumRegister(4, "qr1")
        qr2 = QuantumRegister(3, "qr2")
        qr3 = QuantumRegister(3, "qr3")
        cr1 = ClassicalRegister(4, "cr1")
        cr2 = ClassicalRegister(1, "cr2")
        circ = QuantumCircuit(qr1, qr2, qr3, cr1, cr2)
        circ.cx(qr1[1], qr2[2])
        circ.measure(qr3[0], cr2[0])

        inst = circuit_to_instruction(circ)
        q = QuantumRegister(10, "q")
        c = ClassicalRegister(5, "c")

        self.assertEqual(inst.definition[0][1], [q[1], q[6]])
        self.assertEqual(inst.definition[1][1], [q[7]])
        self.assertEqual(inst.definition[1][2], [c[4]])

    def test_flatten_registers_of_circuit_single_bit_cond(self):
        """Check correct mapping of registers gates conditioned on single classical bits."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(3, "qr2")
        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        circ = QuantumCircuit(qr1, qr2, cr1, cr2)
        circ.h(qr1[0]).c_if(cr1[1], True)
        circ.h(qr2[1]).c_if(cr2[0], False)
        circ.cx(qr1[1], qr2[2]).c_if(cr2[2], True)
        circ.measure(qr2[2], cr2[0])

        inst = circuit_to_instruction(circ)
        q = QuantumRegister(5, "q")
        c = ClassicalRegister(6, "c")

        self.assertEqual(inst.definition[0][1], [q[0]])
        self.assertEqual(inst.definition[1][1], [q[3]])
        self.assertEqual(inst.definition[2][1], [q[1], q[4]])

        self.assertEqual(inst.definition[0][0].condition, (c[1], True))
        self.assertEqual(inst.definition[1][0].condition, (c[3], False))
        self.assertEqual(inst.definition[2][0].condition, (c[5], True))

    def test_flatten_circuit_registerless(self):
        """Test that the conversion works when the given circuit has bits that are not contained in
        any register."""
        qr1 = QuantumRegister(2)
        qubits = [Qubit(), Qubit(), Qubit()]
        qr2 = QuantumRegister(3)
        cr1 = ClassicalRegister(2)
        clbits = [Clbit(), Clbit(), Clbit()]
        cr2 = ClassicalRegister(3)
        circ = QuantumCircuit(qr1, qubits, qr2, cr1, clbits, cr2)
        circ.cx(3, 5)
        circ.measure(4, 4)

        inst = circuit_to_instruction(circ)
        self.assertEqual(inst.num_qubits, len(qr1) + len(qubits) + len(qr2))
        self.assertEqual(inst.num_clbits, len(cr1) + len(clbits) + len(cr2))
        inst_definition = inst.definition
        _, cx_qargs, cx_cargs = inst_definition.data[0]
        _, measure_qargs, measure_cargs = inst_definition.data[1]
        self.assertEqual(cx_qargs, [inst_definition.qubits[3], inst_definition.qubits[5]])
        self.assertEqual(cx_cargs, [])
        self.assertEqual(measure_qargs, [inst_definition.qubits[4]])
        self.assertEqual(measure_cargs, [inst_definition.clbits[4]])

    def test_flatten_circuit_overlapping_registers(self):
        """Test that the conversion works when the given circuit has bits that are contained in more
        than one register."""
        qubits = [Qubit() for _ in [None] * 10]
        qr1 = QuantumRegister(bits=qubits[:6])
        qr2 = QuantumRegister(bits=qubits[4:])
        clbits = [Clbit() for _ in [None] * 10]
        cr1 = ClassicalRegister(bits=clbits[:6])
        cr2 = ClassicalRegister(bits=clbits[4:])
        circ = QuantumCircuit(qubits, clbits, qr1, qr2, cr1, cr2)
        circ.cx(3, 5)
        circ.measure(4, 4)

        inst = circuit_to_instruction(circ)
        self.assertEqual(inst.num_qubits, len(qubits))
        self.assertEqual(inst.num_clbits, len(clbits))
        inst_definition = inst.definition
        _, cx_qargs, cx_cargs = inst_definition.data[0]
        _, measure_qargs, measure_cargs = inst_definition.data[1]
        self.assertEqual(cx_qargs, [inst_definition.qubits[3], inst_definition.qubits[5]])
        self.assertEqual(cx_cargs, [])
        self.assertEqual(measure_qargs, [inst_definition.qubits[4]])
        self.assertEqual(measure_cargs, [inst_definition.clbits[4]])

    def test_flatten_parameters(self):
        """Verify parameters from circuit are moved to instruction.params"""
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_ = theta + phi

        qc.rz(theta, qr[0])
        qc.rz(phi, qr[1])
        qc.u(theta, phi, 0, qr[2])
        qc.rz(sum_, qr[0])

        inst = circuit_to_instruction(qc)

        self.assertEqual(inst.params, [phi, theta])
        self.assertEqual(inst.definition[0][0].params, [theta])
        self.assertEqual(inst.definition[1][0].params, [phi])
        self.assertEqual(inst.definition[2][0].params, [theta, phi, 0])
        self.assertEqual(str(inst.definition[3][0].params[0]), "phi + theta")

    def test_underspecified_parameter_map_raises(self):
        """Verify we raise if not all circuit parameters are present in parameter_map."""
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_ = theta + phi

        gamma = Parameter("gamma")

        qc.rz(theta, qr[0])
        qc.rz(phi, qr[1])
        qc.u(theta, phi, 0, qr[2])
        qc.rz(sum_, qr[0])

        self.assertRaises(QiskitError, circuit_to_instruction, qc, {theta: gamma})

        # Raise if provided more parameters than present in the circuit
        delta = Parameter("delta")
        self.assertRaises(
            QiskitError, circuit_to_instruction, qc, {theta: gamma, phi: phi, delta: delta}
        )

    def test_parameter_map(self):
        """Verify alternate parameter specification"""
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_ = theta + phi

        gamma = Parameter("gamma")

        qc.rz(theta, qr[0])
        qc.rz(phi, qr[1])
        qc.u(theta, phi, 0, qr[2])
        qc.rz(sum_, qr[0])

        inst = circuit_to_instruction(qc, {theta: gamma, phi: phi})

        self.assertEqual(inst.params, [gamma, phi])
        self.assertEqual(inst.definition[0][0].params, [gamma])
        self.assertEqual(inst.definition[1][0].params, [phi])
        self.assertEqual(inst.definition[2][0].params, [gamma, phi, 0])
        self.assertEqual(str(inst.definition[3][0].params[0]), "gamma + phi")

    def test_registerless_classical_bits(self):
        """Test that conditions on registerless classical bits can be handled during the conversion.

        Regression test of gh-7394."""
        expected = QuantumCircuit([Qubit(), Clbit()])
        expected.h(0).c_if(expected.clbits[0], 0)
        test = circuit_to_instruction(expected)

        self.assertIsInstance(test, Instruction)
        self.assertIsInstance(test.definition, QuantumCircuit)

        self.assertEqual(len(test.definition.data), 1)
        test_instruction, _, _ = test.definition.data[0]
        expected_instruction, _, _ = expected.data[0]
        self.assertIs(type(test_instruction), type(expected_instruction))
        self.assertEqual(test_instruction.condition, (test.definition.clbits[0], 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
