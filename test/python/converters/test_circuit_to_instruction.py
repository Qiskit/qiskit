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

import math
import unittest

import numpy as np

from qiskit.converters import circuit_to_instruction
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit, Instruction
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
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

        self.assertEqual(inst.definition[0].qubits, (q[1], q[6]))
        self.assertEqual(inst.definition[1].qubits, (q[7],))
        self.assertEqual(inst.definition[1].clbits, (c[4],))

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

        self.assertEqual(inst.definition[0].qubits, (q[0],))
        self.assertEqual(inst.definition[1].qubits, (q[3],))
        self.assertEqual(inst.definition[2].qubits, (q[1], q[4]))

        self.assertEqual(inst.definition[0].operation.condition, (c[1], True))
        self.assertEqual(inst.definition[1].operation.condition, (c[3], False))
        self.assertEqual(inst.definition[2].operation.condition, (c[5], True))

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
        cx = inst_definition.data[0]
        measure = inst_definition.data[1]
        self.assertEqual(cx.qubits, (inst_definition.qubits[3], inst_definition.qubits[5]))
        self.assertEqual(cx.clbits, ())
        self.assertEqual(measure.qubits, (inst_definition.qubits[4],))
        self.assertEqual(measure.clbits, (inst_definition.clbits[4],))

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
        cx = inst_definition.data[0]
        measure = inst_definition.data[1]
        self.assertEqual(cx.qubits, (inst_definition.qubits[3], inst_definition.qubits[5]))
        self.assertEqual(cx.clbits, ())
        self.assertEqual(measure.qubits, (inst_definition.qubits[4],))
        self.assertEqual(measure.clbits, (inst_definition.clbits[4],))

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
        self.assertEqual(inst.definition[0].operation.params, [theta])
        self.assertEqual(inst.definition[1].operation.params, [phi])
        self.assertEqual(inst.definition[2].operation.params, [theta, phi, 0])
        self.assertEqual(str(inst.definition[3].operation.params[0]), "phi + theta")

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
        self.assertEqual(inst.definition[0].operation.params, [gamma])
        self.assertEqual(inst.definition[1].operation.params, [phi])
        self.assertEqual(inst.definition[2].operation.params, [gamma, phi, 0])
        self.assertEqual(str(inst.definition[3].operation.params[0]), "gamma + phi")

    def test_registerless_classical_bits(self):
        """Test that conditions on registerless classical bits can be handled during the conversion.

        Regression test of gh-7394."""
        expected = QuantumCircuit([Qubit(), Clbit()])
        expected.h(0).c_if(expected.clbits[0], 0)
        test = circuit_to_instruction(expected)

        self.assertIsInstance(test, Instruction)
        self.assertIsInstance(test.definition, QuantumCircuit)

        self.assertEqual(len(test.definition.data), 1)
        test_instruction = test.definition.data[0]
        expected_instruction = expected.data[0]
        self.assertIs(type(test_instruction.operation), type(expected_instruction.operation))
        self.assertEqual(test_instruction.operation.condition, (test.definition.clbits[0], 0))

    def test_zero_operands(self):
        """Test that an instruction can be created, even if it has zero operands."""
        base = QuantumCircuit(global_phase=math.pi)
        instruction = base.to_instruction()
        self.assertEqual(instruction.num_qubits, 0)
        self.assertEqual(instruction.num_clbits, 0)
        self.assertEqual(instruction.definition, base)
        compound = QuantumCircuit(1)
        compound.append(instruction, [], [])
        np.testing.assert_allclose(-np.eye(2), Operator(compound), atol=1e-16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
