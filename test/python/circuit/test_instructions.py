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

"""Test Qiskit's Instruction class."""

import unittest.mock

import numpy as np

from qiskit.circuit import Gate
from qiskit.circuit import Parameter
from qiskit.circuit import Instruction, InstructionSet
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.library.standard_gates.s import SGate
from qiskit.circuit.library.standard_gates.t import TGate
from qiskit.test import QiskitTestCase
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.random import random_circuit


class TestInstructions(QiskitTestCase):
    """Instructions tests."""

    def test_instructions_equal(self):
        """Test equality of two instructions."""
        hop1 = Instruction("h", 1, 0, [])
        hop2 = Instruction("s", 1, 0, [])
        hop3 = Instruction("h", 1, 0, [])

        self.assertFalse(hop1 == hop2)
        self.assertTrue(hop1 == hop3)

        uop1 = Instruction("u", 1, 0, [0.4, 0.5, 0.5])
        uop2 = Instruction("u", 1, 0, [0.4, 0.6, 0.5])
        uop3 = Instruction("v", 1, 0, [0.4, 0.5, 0.5])
        uop4 = Instruction("u", 1, 0, [0.4, 0.5, 0.5])

        self.assertFalse(uop1 == uop2)
        self.assertTrue(uop1 == uop4)
        self.assertFalse(uop1 == uop3)

        self.assertTrue(HGate() == HGate())
        self.assertFalse(HGate() == CXGate())
        self.assertFalse(hop1 == HGate())

        eop1 = Instruction("kraus", 1, 0, [np.array([[1, 0], [0, 1]])])
        eop2 = Instruction("kraus", 1, 0, [np.array([[0, 1], [1, 0]])])
        eop3 = Instruction("kraus", 1, 0, [np.array([[1, 0], [0, 1]])])
        eop4 = Instruction("kraus", 1, 0, [np.eye(4)])

        self.assertTrue(eop1 == eop3)
        self.assertFalse(eop1 == eop2)
        self.assertFalse(eop1 == eop4)

    def test_instructions_equal_with_parameters(self):
        """Test equality of instructions for cases with Parameters."""
        theta = Parameter("theta")
        phi = Parameter("phi")

        # Verify we can check params including parameters
        self.assertEqual(
            Instruction("u", 1, 0, [theta, phi, 0.4]), Instruction("u", 1, 0, [theta, phi, 0.4])
        )

        # Verify we can test for correct parameter order
        self.assertNotEqual(
            Instruction("u", 1, 0, [theta, phi, 0]), Instruction("u", 1, 0, [phi, theta, 0])
        )

        # Verify we can still find a wrong fixed param if we use parameters
        self.assertNotEqual(
            Instruction("u", 1, 0, [theta, phi, 0.4]), Instruction("u", 1, 0, [theta, phi, 0.5])
        )

        # Verify we can find cases when param != float
        self.assertNotEqual(
            Instruction("u", 1, 0, [0.3, phi, 0.4]), Instruction("u", 1, 0, [theta, phi, 0.5])
        )

    def test_instructions_soft_compare(self):
        """Test soft comparison between instructions."""
        theta = Parameter("theta")
        phi = Parameter("phi")

        # Verify that we are insensitive when there are parameters.
        self.assertTrue(
            Instruction("u", 1, 0, [0.3, phi, 0.4]).soft_compare(
                Instruction("u", 1, 0, [theta, phi, 0.4])
            )
        )

        # Verify that normal equality still holds.
        self.assertTrue(
            Instruction("u", 1, 0, [0.4, 0.5]).soft_compare(Instruction("u", 1, 0, [0.4, 0.5]))
        )

        # Test that when names differ we get False.
        self.assertFalse(
            Instruction("u", 1, 0, [0.4, phi]).soft_compare(Instruction("v", 1, 0, [theta, phi]))
        )

        # Test cutoff precision.
        self.assertFalse(
            Instruction("v", 1, 0, [0.401, phi]).soft_compare(Instruction("v", 1, 0, [0.4, phi]))
        )

        # Test cutoff precision.
        self.assertTrue(
            Instruction("v", 1, 0, [0.4 + 1.0e-20, phi]).soft_compare(
                Instruction("v", 1, 0, [0.4, phi])
            )
        )

    def test_instructions_equal_with_parameter_expressions(self):
        """Test equality of instructions for cases with ParameterExpressions."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_ = theta + phi
        product_ = theta * phi

        # Verify we can check params including parameters
        self.assertEqual(
            Instruction("u", 1, 0, [sum_, product_, 0.4]),
            Instruction("u", 1, 0, [sum_, product_, 0.4]),
        )

        # Verify we can test for correct parameter order
        self.assertNotEqual(
            Instruction("u", 1, 0, [product_, sum_, 0]), Instruction("u", 1, 0, [sum_, product_, 0])
        )

        # Verify we can still find a wrong fixed param if we use parameters
        self.assertNotEqual(
            Instruction("u", 1, 0, [sum_, phi, 0.4]), Instruction("u", 1, 0, [sum_, phi, 0.5])
        )

        # Verify we can find cases when param != float
        self.assertNotEqual(
            Instruction("u", 1, 0, [0.3, sum_, 0.4]), Instruction("u", 1, 0, [product_, sum_, 0.5])
        )

    def circuit_instruction_circuit_roundtrip(self):
        """test converting between circuit and instruction and back
        preserves the circuit"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ1 = QuantumCircuit(q, c, name="circuit1")
        circ1.h(q[0])
        circ1.crz(0.1, q[0], q[1])
        circ1.i(q[1])
        circ1.u(0.1, 0.2, -0.2, q[0])
        circ1.barrier()
        circ1.measure(q, c)
        circ1.rz(0.8, q[0]).c_if(c, 6)
        inst = circ1.to_instruction()

        circ2 = QuantumCircuit(q, c, name="circ2")
        circ2.append(inst, q[:])

        self.assertEqual(circ1, circ2)

    def test_append_opaque_wrong_dimension(self):
        """test appending opaque gate to wrong dimension wires."""
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        opaque_gate = Gate(name="crz_2", num_qubits=2, params=[0.5])
        self.assertRaises(CircuitError, circ.append, opaque_gate, [qr[0]])

    def test_opaque_gate(self):
        """test opaque gate functionality"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name="circ")
        opaque_gate = Gate(name="crz_2", num_qubits=2, params=[0.5])
        circ.append(opaque_gate, [q[2], q[0]])
        self.assertEqual(circ.data[0][0].name, "crz_2")
        self.assertEqual(circ.decompose(), circ)

    def test_opaque_instruction(self):
        """test opaque instruction does not decompose"""
        q = QuantumRegister(4)
        c = ClassicalRegister(2)
        circ = QuantumCircuit(q, c)
        opaque_inst = Instruction(name="my_inst", num_qubits=3, num_clbits=1, params=[0.5])
        circ.append(opaque_inst, [q[3], q[1], q[0]], [c[1]])
        self.assertEqual(circ.data[0][0].name, "my_inst")
        self.assertEqual(circ.decompose(), circ)

    def test_reverse_gate(self):
        """test reversing a composite gate"""
        q = QuantumRegister(4)
        circ = QuantumCircuit(q, name="circ")
        circ.h(q[0])
        circ.crz(0.1, q[0], q[1])
        circ.i(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        gate = circ.to_gate()

        circ = QuantumCircuit(q, name="circ")
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.i(q[1])
        circ.crz(0.1, q[0], q[1])
        circ.h(q[0])
        gate_reverse = circ.to_gate()
        self.assertEqual(gate.reverse_ops().definition, gate_reverse.definition)

    def test_reverse_instruction(self):
        """test reverseing an instruction with conditionals"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name="circ")
        circ.t(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.barrier()
        circ.measure(q[0], c[0])
        circ.rz(0.8, q[0]).c_if(c, 6)
        inst = circ.to_instruction()

        circ = QuantumCircuit(q, c, name="circ")
        circ.rz(0.8, q[0]).c_if(c, 6)
        circ.measure(q[0], c[0])
        circ.barrier()
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.t(q[1])
        inst_reverse = circ.to_instruction()

        self.assertEqual(inst.reverse_ops().definition, inst_reverse.definition)

    def test_reverse_opaque(self):
        """test opaque gates reverse to themselves"""
        opaque_gate = Gate(name="crz_2", num_qubits=2, params=[0.5])
        self.assertEqual(opaque_gate.reverse_ops(), opaque_gate)
        hgate = HGate()
        self.assertEqual(hgate.reverse_ops(), hgate)

    def test_inverse_and_append(self):
        """test appending inverted gates to circuits"""
        q = QuantumRegister(1)
        circ = QuantumCircuit(q, name="circ")
        circ.s(q)
        circ.append(SGate().inverse(), q[:])
        circ.append(TGate().inverse(), q[:])
        circ.t(q)
        gate = circ.to_instruction()
        circ = QuantumCircuit(q, name="circ")
        circ.inverse()
        circ.tdg(q)
        circ.t(q)
        circ.s(q)
        circ.sdg(q)
        gate_inverse = circ.to_instruction()
        self.assertEqual(gate.inverse().definition, gate_inverse.definition)

    def test_inverse_composite_gate(self):
        """test inverse of composite gate"""
        q = QuantumRegister(4)
        circ = QuantumCircuit(q, name="circ")
        circ.h(q[0])
        circ.crz(0.1, q[0], q[1])
        circ.i(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        gate = circ.to_instruction()
        circ = QuantumCircuit(q, name="circ")
        circ.u(-0.1, 0.2, -0.2, q[0])
        circ.i(q[1])
        circ.crz(-0.1, q[0], q[1])
        circ.h(q[0])
        gate_inverse = circ.to_instruction()
        self.assertEqual(gate.inverse().definition, gate_inverse.definition)

    def test_inverse_recursive(self):
        """test that a hierarchical gate recursively inverts"""
        qr0 = QuantumRegister(2)
        circ0 = QuantumCircuit(qr0, name="circ0")
        circ0.t(qr0[0])
        circ0.rx(0.4, qr0[1])
        circ0.cx(qr0[1], qr0[0])
        little_gate = circ0.to_instruction()

        qr1 = QuantumRegister(4)
        circ1 = QuantumCircuit(qr1, name="circuit1")
        circ1.cp(-0.1, qr1[0], qr1[2])
        circ1.i(qr1[1])
        circ1.append(little_gate, [qr1[2], qr1[3]])

        circ_inv = QuantumCircuit(qr1, name="circ1_dg")
        circ_inv.append(little_gate.inverse(), [qr1[2], qr1[3]])
        circ_inv.i(qr1[1])
        circ_inv.cp(0.1, qr1[0], qr1[2])

        self.assertEqual(circ1.inverse(), circ_inv)

    def test_inverse_instruction_with_measure(self):
        """test inverting instruction with measure fails"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name="circ")
        circ.t(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.barrier()
        circ.measure(q[0], c[0])
        inst = circ.to_instruction()
        self.assertRaises(CircuitError, inst.inverse)

    def test_inverse_instruction_with_conditional(self):
        """test inverting instruction with conditionals fails"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name="circ")
        circ.t(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.barrier()
        circ.measure(q[0], c[0])
        circ.rz(0.8, q[0]).c_if(c, 6)
        inst = circ.to_instruction()
        self.assertRaises(CircuitError, inst.inverse)

    def test_inverse_opaque(self):
        """test inverting opaque gate fails"""
        opaque_gate = Gate(name="crz_2", num_qubits=2, params=[0.5])
        self.assertRaises(CircuitError, opaque_gate.inverse)

    def test_inverse_empty(self):
        """test inverting empty gate works"""
        q = QuantumRegister(3)
        c = ClassicalRegister(3)
        empty_circ = QuantumCircuit(q, c, name="empty_circ")
        empty_gate = empty_circ.to_instruction()
        self.assertEqual(empty_gate.inverse().definition, empty_gate.definition)

    def test_inverse_with_global_phase(self):
        """test inverting instruction with global phase in definition."""
        q = QuantumRegister(1)
        circ = QuantumCircuit(q, name="circ", global_phase=np.pi / 3)
        circ.x(q)
        gate = circ.to_instruction()
        circ = QuantumCircuit(q, name="circ", global_phase=-np.pi / 3)
        circ.x(q)
        gate_inverse = circ.to_instruction()
        self.assertEqual(gate.inverse().definition, gate_inverse.definition)

    def test_inverse_with_label(self):
        """test inverting gate initialized with label attribute."""
        q = QuantumRegister(2)
        qc = QuantumCircuit(q, name="circ")
        qc.cx(0, 1)
        qc_gate = qc.to_gate()
        qc_gate_inverse = qc_gate.inverse()
        self.assertEqual(qc_gate.name + "_dg", qc_gate_inverse.name)
        qc_gate_inverse_inverse = qc_gate_inverse.inverse()
        self.assertEqual(qc_gate_inverse_inverse.name, qc_gate.name)

    def test_no_broadcast(self):
        """See https://github.com/Qiskit/qiskit-terra/issues/2777
        When creating custom instructions, do not broadcast parameters"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        subcircuit = QuantumCircuit(qr, cr, name="subcircuit")

        subcircuit.x(qr[0])
        subcircuit.h(qr[1])
        subcircuit.measure(qr[0], cr[0])
        subcircuit.measure(qr[1], cr[1])

        inst = subcircuit.to_instruction()
        circuit = QuantumCircuit(qr, cr, name="circuit")
        circuit.append(inst, qr[:], cr[:])
        self.assertEqual(circuit.qregs, [qr])
        self.assertEqual(circuit.cregs, [cr])
        self.assertEqual(circuit.qubits, [qr[0], qr[1]])
        self.assertEqual(circuit.clbits, [cr[0], cr[1]])

    def test_modifying_copied_params_leaves_orig(self):
        """Verify modifying the parameters of a copied instruction does not
        affect the original."""

        inst = Instruction("test", 2, 1, [0, 1, 2])

        cpy = inst.copy()

        cpy.params[1] = 7

        self.assertEqual(inst.params, [0, 1, 2])

    def test_instance_of_instruction(self):
        """Test correct error message is raised when invalid instruction
        is passed to append"""

        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        with self.assertRaisesRegex(CircuitError, r"Object is a subclass of Instruction"):
            qc.append(HGate, qr[:], [])

    def test_repr_of_instructions(self):
        """Test the __repr__ method of the Instruction
        class"""

        ins1 = Instruction("test_instruction", 3, 5, [0, 1, 2, 3])
        self.assertEqual(
            repr(ins1),
            "Instruction(name='{}', num_qubits={}, num_clbits={}, params={})".format(
                ins1.name, ins1.num_qubits, ins1.num_clbits, ins1.params
            ),
        )

        ins2 = random_circuit(num_qubits=4, depth=4, measure=True).to_instruction()
        self.assertEqual(
            repr(ins2),
            "Instruction(name='{}', num_qubits={}, num_clbits={}, params={})".format(
                ins2.name, ins2.num_qubits, ins2.num_clbits, ins2.params
            ),
        )

    def test_instructionset_c_if_direct_resource(self):
        """Test that using :meth:`.InstructionSet.c_if` with an exact classical resource always
        works, and produces the expected condition."""
        cr1 = ClassicalRegister(3)
        qubits = [Qubit()]
        loose_clbits = [Clbit(), Clbit(), Clbit()]
        # These bits are going into registers which overlap.
        register_clbits = [Clbit(), Clbit(), Clbit()]
        cr2 = ClassicalRegister(bits=register_clbits[:2])
        cr3 = ClassicalRegister(bits=register_clbits[1:])

        def case(resource):
            qc = QuantumCircuit(cr1, qubits, loose_clbits, cr2, cr3)
            qc.x(0).c_if(resource, 0)
            c_if_resource = qc.data[0][0].condition[0]
            self.assertIs(c_if_resource, resource)

        with self.subTest("classical register"):
            case(cr1)
        with self.subTest("bit from classical register"):
            case(cr1[0])
        with self.subTest("loose bit"):
            case(loose_clbits[0])
        with self.subTest("overlapping register left"):
            case(cr2)
        with self.subTest("overlapping register right"):
            case(cr3)
        with self.subTest("bit in two different registers"):
            case(register_clbits[1])

    def test_instructionset_c_if_indexing(self):
        """Test that using :meth:`.InstructionSet.c_if` with an index for the classical resource
        resolves to the same value that :obj:`.QuantumCircuit` would resolve it to.

        Regression test for gh-7246."""
        cr1 = ClassicalRegister(3)
        qubits = [Qubit()]
        loose_clbits = [Clbit(), Clbit(), Clbit()]
        # These bits are going into registers which overlap.
        register_clbits = [Clbit(), Clbit(), Clbit()]
        cr2 = ClassicalRegister(bits=register_clbits[:2])
        cr3 = ClassicalRegister(bits=register_clbits[1:])

        qc = QuantumCircuit(cr1, qubits, loose_clbits, cr2, cr3)
        for index, clbit in enumerate(qc.clbits):
            with self.subTest(index=index):
                qc.x(0).c_if(index, 0)
                qc.measure(0, index)
                from_c_if = qc.data[-2][0].condition[0]
                from_measure = qc.data[-1][2][0]
                self.assertIs(from_c_if, from_measure)
                # Sanity check that the bit is also the one we expected.
                self.assertIs(from_c_if, clbit)

    def test_instructionset_c_if_size_1_classical_register(self):
        """Test that there is a distinction between conditioning on a single bit and a classical
        register, even if the register in question has a size of one."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        with self.subTest("classical register"):
            qc.x(0).c_if(cr, 0)
            self.assertIs(qc.data[-1][0].condition[0], cr)
        with self.subTest("classical bit by value"):
            qc.x(0).c_if(cr[0], 0)
            self.assertIs(qc.data[-1][0].condition[0], cr[0])
        with self.subTest("classical bit by index"):
            qc.x(0).c_if(0, 0)
            self.assertIs(qc.data[-1][0].condition[0], cr[0])

    def test_instructionset_c_if_no_classical_registers(self):
        """Test that using :meth:`.InstructionSet.c_if` works if there are no classical registers
        defined on the circuit.

        Regression test for gh-7250."""
        bits = [Qubit(), Clbit()]
        qc = QuantumCircuit(bits)
        with self.subTest("by value"):
            qc.x(0).c_if(bits[1], 0)
            self.assertIs(qc.data[-1][0].condition[0], bits[1])
        with self.subTest("by index"):
            qc.x(0).c_if(0, 0)
            self.assertIs(qc.data[-1][0].condition[0], bits[1])

    def test_instructionset_c_if_rejects_invalid_specifiers(self):
        """Test that calling the :meth:`.InstructionSet.c_if` method on instructions added to a
        circuit raises a suitable exception if an invalid specifier is passed to it."""

        qreg = QuantumRegister(1)
        creg = ClassicalRegister(2)

        def case(specifier, message):
            qc = QuantumCircuit(qreg, creg)
            instruction = qc.x(0)
            with self.assertRaisesRegex(CircuitError, message):
                instruction.c_if(specifier, 0)

        with self.subTest("absent bit"):
            case(Clbit(), r"Clbit .* is not present in this circuit\.")
        with self.subTest("absent register"):
            case(ClassicalRegister(2), r"Register .* is not present in this circuit\.")
        with self.subTest("index out of range"):
            case(2, r"Classical bit index .* is out-of-range\.")
        with self.subTest("list of bits"):
            case(list(creg), r"Unknown classical resource specifier: .*")
        with self.subTest("tuple of bits"):
            case(tuple(creg), r"Unknown classical resource specifier: .*")
        with self.subTest("float"):
            case(1.0, r"Unknown classical resource specifier: .*")

    def test_instructionset_c_if_with_no_requester(self):
        """Test that using a raw :obj:`.InstructionSet` with no classical-resource resoluer accepts
        arbitrary :obj:`.Clbit` and `:obj:`.ClassicalRegister` instances, but rejects integers."""

        with self.subTest("accepts arbitrary register"):
            instruction = HGate()
            instructions = InstructionSet()
            instructions.add(instruction, [Qubit()], [])
            register = ClassicalRegister(2)
            instructions.c_if(register, 0)
            self.assertIs(instruction.condition[0], register)
        with self.subTest("accepts arbitrary bit"):
            instruction = HGate()
            instructions = InstructionSet()
            instructions.add(instruction, [Qubit()], [])
            bit = Clbit()
            instructions.c_if(bit, 0)
            self.assertIs(instruction.condition[0], bit)
        with self.subTest("rejects index"):
            instruction = HGate()
            instructions = InstructionSet()
            instructions.add(instruction, [Qubit()], [])
            with self.assertRaisesRegex(CircuitError, r"Cannot pass an index as a condition .*"):
                instructions.c_if(0, 0)

    def test_instructionset_c_if_deprecated_resolution(self):
        r"""Test that the deprecated path of passing an iterable of :obj:`.ClassicalRegister`\ s to
        :obj:`.InstructionSet` works, issues a deprecation warning, and resolves indices in the
        simple cases it was meant to handle."""
        # The deprecated path can only cope with non-overlapping classical registers, with no loose
        # clbits in the mix.
        registers = [ClassicalRegister(2), ClassicalRegister(3), ClassicalRegister(1)]
        bits = [bit for register in registers for bit in register]

        deprecated_regex = r"The 'circuit_cregs' argument to 'InstructionSet' is deprecated .*"

        def dummy_requester(specifier):
            """A dummy requester that technically fulfills the spec."""
            raise CircuitError

        with self.subTest("cannot pass both registers and requester"):
            with self.assertRaisesRegex(
                CircuitError, r"Cannot pass both 'circuit_cregs' and 'resource_requester'\."
            ):
                InstructionSet(registers, resource_requester=dummy_requester)

        with self.subTest("classical register"):
            instruction = HGate()
            with self.assertWarnsRegex(DeprecationWarning, deprecated_regex):
                instructions = InstructionSet(registers)
            instructions.add(instruction, [Qubit()], [])
            instructions.c_if(registers[0], 0)
            self.assertIs(instruction.condition[0], registers[0])
        with self.subTest("classical bit"):
            instruction = HGate()
            with self.assertWarnsRegex(DeprecationWarning, deprecated_regex):
                instructions = InstructionSet(registers)
            instructions.add(instruction, [Qubit()], [])
            instructions.c_if(registers[0][1], 0)
            self.assertIs(instruction.condition[0], registers[0][1])
        for i, bit in enumerate(bits):
            with self.subTest("bit index", index=i):
                instruction = HGate()
                with self.assertWarnsRegex(DeprecationWarning, deprecated_regex):
                    instructions = InstructionSet(registers)
                instructions.add(instruction, [Qubit()], [])
                instructions.c_if(i, 0)
                self.assertIs(instruction.condition[0], bit)

        with self.subTest("raises on bad register"):
            instruction = HGate()
            with self.assertWarnsRegex(DeprecationWarning, deprecated_regex):
                instructions = InstructionSet(registers)
            instructions.add(instruction, [Qubit()], [])
            with self.assertRaisesRegex(
                CircuitError, r"Condition register .* is not one of the registers known here: .*"
            ):
                instructions.c_if(ClassicalRegister(2), 0)
        with self.subTest("raises on bad bit"):
            instruction = HGate()
            with self.assertWarnsRegex(DeprecationWarning, deprecated_regex):
                instructions = InstructionSet(registers)
            instructions.add(instruction, [Qubit()], [])
            with self.assertRaisesRegex(
                CircuitError, "Condition bit .* is not in the registers known here: .*"
            ):
                instructions.c_if(Clbit(), 0)
        with self.subTest("raises on bad index"):
            instruction = HGate()
            with self.assertWarnsRegex(DeprecationWarning, deprecated_regex):
                instructions = InstructionSet(registers)
            instructions.add(instruction, [Qubit()], [])
            with self.assertRaisesRegex(CircuitError, r"Bit index .* is out-of-range\."):
                instructions.c_if(len(bits), 0)
        with self.subTest("raises on bad type"):
            instruction = HGate()
            with self.assertWarnsRegex(DeprecationWarning, deprecated_regex):
                instructions = InstructionSet(registers)
            instructions.add(instruction, [Qubit()], [])
            with self.assertRaisesRegex(CircuitError, r"Invalid classical condition\. .*"):
                instructions.c_if([0], 0)

    def test_instructionset_c_if_calls_custom_requester(self):
        """Test that :meth:`.InstructionSet.c_if` calls a custom requester, and uses its output."""
        # This isn't expected to be useful to end users, it's more about the principle that you can
        # control the resolution paths, so future blocking constructs can forbid the method from
        # accessing certain resources.

        sentinel_bit = Clbit()
        sentinel_register = ClassicalRegister(2)

        def dummy_requester(specifier):
            """A dummy requester that returns sentinel values."""
            if not isinstance(specifier, (int, Clbit, ClassicalRegister)):
                raise CircuitError
            return sentinel_bit if isinstance(specifier, (int, Clbit)) else sentinel_register

        dummy_requester = unittest.mock.MagicMock(wraps=dummy_requester)

        with self.subTest("calls requester with bit"):
            dummy_requester.reset_mock()
            instruction = HGate()
            instructions = InstructionSet(resource_requester=dummy_requester)
            instructions.add(instruction, [Qubit()], [])
            bit = Clbit()
            instructions.c_if(bit, 0)
            dummy_requester.assert_called_once_with(bit)
            self.assertIs(instruction.condition[0], sentinel_bit)
        with self.subTest("calls requester with index"):
            dummy_requester.reset_mock()
            instruction = HGate()
            instructions = InstructionSet(resource_requester=dummy_requester)
            instructions.add(instruction, [Qubit()], [])
            index = 0
            instructions.c_if(index, 0)
            dummy_requester.assert_called_once_with(index)
            self.assertIs(instruction.condition[0], sentinel_bit)
        with self.subTest("calls requester with register"):
            dummy_requester.reset_mock()
            instruction = HGate()
            instructions = InstructionSet(resource_requester=dummy_requester)
            instructions.add(instruction, [Qubit()], [])
            register = ClassicalRegister(2)
            instructions.c_if(register, 0)
            dummy_requester.assert_called_once_with(register)
            self.assertIs(instruction.condition[0], sentinel_register)
        with self.subTest("calls requester only once when broadcast"):
            dummy_requester.reset_mock()
            instruction_list = [HGate(), HGate(), HGate()]
            instructions = InstructionSet(resource_requester=dummy_requester)
            for instruction in instruction_list:
                instructions.add(instruction, [Qubit()], [])
            register = ClassicalRegister(2)
            instructions.c_if(register, 0)
            dummy_requester.assert_called_once_with(register)
            for instruction in instruction_list:
                self.assertIs(instruction.condition[0], sentinel_register)


if __name__ == "__main__":
    unittest.main()
