# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test the BasisTranslator pass"""


from numpy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.circuit import Gate, Parameter, EquivalenceLibrary
from qiskit.converters import circuit_to_dag, dag_to_circuit, circuit_to_instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.basis import BasisTranslator, UnrollCustomDefinitions


from qiskit.circuit.library.standard_gates.equivalence_library \
    import StandardEquivalenceLibrary as std_eqlib


class OneQubitZeroParamGate(Gate):
    """Mock one qubit zero param gate."""
    def __init__(self):
        super().__init__('1q0p', 1, [])


class OneQubitOneParamGate(Gate):
    """Mock one qubit one param gate."""
    def __init__(self, theta):
        super().__init__('1q1p', 1, [theta])


class OneQubitOneParamPrimeGate(Gate):
    """Mock one qubit one param gate."""
    def __init__(self, alpha):
        super().__init__('1q1p_prime', 1, [alpha])


class OneQubitTwoParamGate(Gate):
    """Mock one qubit two param gate."""
    def __init__(self, phi, lam):
        super().__init__('1q2p', 1, [phi, lam])


class TwoQubitZeroParamGate(Gate):
    """Mock one qubit zero param gate."""
    def __init__(self):
        super().__init__('2q0p', 2, [])


class VariadicZeroParamGate(Gate):
    """Mock variadic zero param gate."""
    def __init__(self, num_qubits):
        super().__init__('vq0p', num_qubits, [])


class TestBasisTranslator(QiskitTestCase):
    """Test the BasisTranslator pass."""

    def test_circ_in_basis_no_op(self):
        """Verify we don't change a circuit already in the target basis."""
        eq_lib = EquivalenceLibrary()
        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = circuit_to_dag(qc)

        pass_ = BasisTranslator(eq_lib, ['1q0p'])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected)

    def test_raise_if_target_basis_unreachable(self):
        """Verify we raise if the circuit cannot be transformed to the target."""
        eq_lib = EquivalenceLibrary()

        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        pass_ = BasisTranslator(eq_lib, ['1q1p'])

        with self.assertRaises(TranspilerError):
            pass_.run(dag)

    def test_single_substitution(self):
        """Verify we correctly unroll gates through a single equivalence."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1)
        expected.append(OneQubitOneParamGate(pi), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ['1q1p'])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_double_substitution(self):
        """Verify we correctly unroll gates through multiple equivalences."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        theta = Parameter('theta')
        gate = OneQubitOneParamGate(theta)
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitTwoParamGate(theta, pi/2), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1)
        expected.append(OneQubitTwoParamGate(pi, pi/2), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ['1q2p'])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_multiple_variadic(self):
        """Verify circuit with multiple instances of variadic gate."""
        eq_lib = EquivalenceLibrary()

        # e.g. MSGate
        oneq_gate = VariadicZeroParamGate(1)
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitZeroParamGate(), [0])
        eq_lib.add_equivalence(oneq_gate, equiv)

        twoq_gate = VariadicZeroParamGate(2)
        equiv = QuantumCircuit(2)
        equiv.append(TwoQubitZeroParamGate(), [0, 1])
        eq_lib.add_equivalence(twoq_gate, equiv)

        qc = QuantumCircuit(2)
        qc.append(VariadicZeroParamGate(1), [0])
        qc.append(VariadicZeroParamGate(2), [0, 1])

        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(2)
        expected.append(OneQubitZeroParamGate(), [0])
        expected.append(TwoQubitZeroParamGate(), [0, 1])

        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ['1q0p', '2q0p'])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_diamond_path(self):
        """Verify we find a path when there are multiple paths to the target basis."""
        eq_lib = EquivalenceLibrary()

        # Path 1: 1q0p -> 1q1p(pi) -> 1q2p(pi, pi/2)

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        theta = Parameter('theta')
        gate = OneQubitOneParamGate(theta)
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitTwoParamGate(theta, pi/2), [0])

        eq_lib.add_equivalence(gate, equiv)

        # Path 2: 1q0p -> 1q1p_prime(pi/2) -> 1q2p(2 * pi/2, pi/2)

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitOneParamPrimeGate(pi/2), [0])

        eq_lib.add_equivalence(gate, equiv)

        alpha = Parameter('alpha')
        gate = OneQubitOneParamPrimeGate(alpha)
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitTwoParamGate(2 * alpha, pi/2), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1)
        expected.append(OneQubitTwoParamGate(pi, pi/2), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ['1q2p'])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)


class TestUnrollerCompatability(QiskitTestCase):
    """Tests backward compatability with the Unroller pass.

    Duplicate of TestUnroller from test.python.transpiler.test_unroller with
    Unroller replaced by UnrollCustomDefinitions -> BasisTranslator.
    """

    def test_basic_unroll(self):
        """Test decompose a single H into u2.
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, ['u2'])
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ['u2'])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, 'u2')

    def test_unroll_toffoli(self):
        """Test unroll toffoli on multi regs to h, t, tdg, cx.
        """
        qr1 = QuantumRegister(2, 'qr1')
        qr2 = QuantumRegister(1, 'qr2')
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, ['h', 't', 'tdg', 'cx'])
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ['h', 't', 'tdg', 'cx'])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            self.assertIn(node.name, ['h', 't', 'tdg', 'cx'])

    def test_unroll_1q_chain_conditional(self):
        """Test unroll chain of 1-qubit gates interrupted by conditional.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.tdg(qr)
        circuit.z(qr)
        circuit.t(qr)
        circuit.ry(0.5, qr)
        circuit.rz(0.3, qr)
        circuit.rx(0.1, qr)
        circuit.measure(qr, cr)
        circuit.x(qr).c_if(cr, 1)
        circuit.y(qr).c_if(cr, 1)
        circuit.z(qr).c_if(cr, 1)
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, ['u1', 'u2', 'u3'])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, ['u1', 'u2', 'u3'])
        unrolled_dag = pass_.run(dag)

        ref_circuit = QuantumCircuit(qr, cr)
        ref_circuit.u2(0, pi, qr[0])
        ref_circuit.u1(-pi/4, qr[0])
        ref_circuit.u1(pi, qr[0])
        ref_circuit.u1(pi/4, qr[0])
        ref_circuit.u3(0.5, 0, 0, qr[0])
        ref_circuit.u1(0.3, qr[0])
        ref_circuit.u3(0.1, -pi/2, pi/2, qr[0])
        ref_circuit.measure(qr[0], cr[0])
        ref_circuit.u3(pi, 0, pi, qr[0]).c_if(cr, 1)
        ref_circuit.u3(pi, pi/2, pi/2, qr[0]).c_if(cr, 1)
        ref_circuit.u1(pi, qr[0]).c_if(cr, 1)
        ref_dag = circuit_to_dag(ref_circuit)
        self.assertEqual(unrolled_dag, ref_dag)

    def test_unroll_no_basis(self):
        """Test when a given gate has no decompositions.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, [])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, [])

        with self.assertRaises(QiskitError):
            pass_.run(dag)

    def test_unroll_all_instructions(self):
        """Test unrolling a circuit containing all standard instructions.
        """

        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(3, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.crx(0.5, qr[1], qr[2])
        circuit.cry(0.5, qr[1], qr[2])
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.ch(qr[0], qr[2])
        circuit.crz(0.5, qr[1], qr[2])
        circuit.cswap(qr[1], qr[0], qr[2])
        circuit.cu1(0.1, qr[0], qr[2])
        circuit.cu3(0.2, 0.1, 0.0, qr[1], qr[2])
        circuit.cx(qr[1], qr[0])
        circuit.cy(qr[1], qr[2])
        circuit.cz(qr[2], qr[0])
        circuit.h(qr[1])
        circuit.i(qr[0])
        circuit.rx(0.1, qr[0])
        circuit.ry(0.2, qr[1])
        circuit.rz(0.3, qr[2])
        circuit.rzz(0.6, qr[1], qr[0])
        circuit.s(qr[0])
        circuit.sdg(qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.t(qr[2])
        circuit.tdg(qr[0])
        circuit.u1(0.1, qr[1])
        circuit.u2(0.2, -0.1, qr[0])
        circuit.u3(0.3, 0.0, -0.1, qr[2])
        circuit.x(qr[2])
        circuit.y(qr[1])
        circuit.z(qr[0])
        # circuit.snapshot('0')
        # circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, ['u3', 'cx', 'id'])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, ['u3', 'cx', 'id'])
        unrolled_dag = pass_.run(dag)

        ref_circuit = QuantumCircuit(qr, cr)
        ref_circuit.u3(0, 0, pi/2, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(-0.25, 0, 0, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0.25, -pi/2, 0, qr[2])
        ref_circuit.u3(0.25, 0, 0, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(-0.25, 0, 0, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, -pi/4, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[1])
        ref_circuit.u3(0, 0, -pi/4, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.u3(0, 0, pi/4, qr[0])
        ref_circuit.u3(0, 0, -pi/4, qr[1])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.u3(0, 0, pi/2, qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, -pi/4, qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.u3(0, 0, -pi/2, qr[2])
        ref_circuit.u3(0, 0, 0.25, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, -0.25, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, -pi/4, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[0])
        ref_circuit.u3(0, 0, -pi/4, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(0, 0, -pi/4, qr[0])
        ref_circuit.u3(0, 0, pi/4, qr[1])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(0, 0, 0.05, qr[1])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.u3(0, 0, 0.05, qr[0])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, -0.05, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, 0.05, qr[2])
        ref_circuit.u3(0, 0, -0.05, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(-0.1, 0, -0.05, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(pi/2, 0, pi, qr[0])
        ref_circuit.u3(0.1, 0.1, 0, qr[2])
        ref_circuit.u3(0, 0, -pi/2, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[1])
        ref_circuit.u3(0.2, 0, 0, qr[1])
        ref_circuit.u3(0, 0, pi/2, qr[2])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.u3(pi/2, 0, pi, qr[0])
        ref_circuit.i(qr[0])
        ref_circuit.u3(0.1, -pi/2, pi/2, qr[0])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(0, 0, 0.6, qr[0])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(0, 0, pi/2, qr[0])
        ref_circuit.u3(0, 0, -pi/4, qr[0])
        ref_circuit.u3(pi/2, 0.2, -0.1, qr[0])
        ref_circuit.u3(0, 0, pi, qr[0])
        ref_circuit.u3(0, 0, -pi/2, qr[1])
        ref_circuit.u3(0, 0, 0.3, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[1])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, 0.1, qr[1])
        ref_circuit.u3(pi, pi/2, pi/2, qr[1])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.u3(0.3, 0.0, -0.1, qr[2])
        ref_circuit.u3(pi, 0, pi, qr[2])
        # ref_circuit.snapshot('0')
        # ref_circuit.measure(qr, cr)
        # ref_dag = circuit_to_dag(ref_circuit)

        self.assertTrue(
            Operator(dag_to_circuit(unrolled_dag)).equiv(ref_circuit))

    def test_simple_unroll_parameterized_without_expressions(self):
        """Verify unrolling parameterized gates without expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')

        qc.rz(theta, qr[0])
        dag = circuit_to_dag(qc)

        pass_ = UnrollCustomDefinitions(std_eqlib, ['u1', 'cx'])
        dag = pass_.run(dag)

        unrolled_dag = BasisTranslator(std_eqlib, ['u1', 'cx']).run(dag)

        expected = QuantumCircuit(qr)
        expected.u1(theta, qr[0])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_simple_unroll_parameterized_with_expressions(self):
        """Verify unrolling parameterized gates with expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')
        phi = Parameter('phi')
        sum_ = theta + phi

        qc.rz(sum_, qr[0])
        dag = circuit_to_dag(qc)
        pass_ = UnrollCustomDefinitions(std_eqlib, ['u1', 'cx'])
        dag = pass_.run(dag)

        unrolled_dag = BasisTranslator(std_eqlib, ['u1', 'cx']).run(dag)

        expected = QuantumCircuit(qr)
        expected.u1(sum_, qr[0])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_definition_unroll_parameterized(self):
        """Verify that unrolling complex gates with parameters does not raise."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')

        qc.cu1(theta, qr[1], qr[0])
        qc.cu1(theta * theta, qr[0], qr[1])
        dag = circuit_to_dag(qc)
        pass_ = UnrollCustomDefinitions(std_eqlib, ['u1', 'cx'])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(std_eqlib, ['u1', 'cx']).run(dag)

        self.assertEqual(out_dag.count_ops(), {'u1': 6, 'cx': 4})

    def test_unrolling_parameterized_composite_gates(self):
        """Verify unrolling circuits with parameterized composite gates."""
        mock_sel = EquivalenceLibrary(base=std_eqlib)

        qr1 = QuantumRegister(2)
        subqc = QuantumCircuit(qr1)

        theta = Parameter('theta')

        subqc.rz(theta, qr1[0])
        subqc.cx(qr1[0], qr1[1])
        subqc.rz(theta, qr1[1])

        # Expanding across register with shared parameter
        qr2 = QuantumRegister(4)
        qc = QuantumCircuit(qr2)

        sub_instr = circuit_to_instruction(subqc, equivalence_library=mock_sel)
        qc.append(sub_instr, [qr2[0], qr2[1]])
        qc.append(sub_instr, [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        pass_ = UnrollCustomDefinitions(mock_sel, ['u1', 'cx'])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(mock_sel, ['u1', 'cx']).run(dag)

        expected = QuantumCircuit(qr2)
        expected.u1(theta, qr2[0])
        expected.u1(theta, qr2[2])
        expected.cx(qr2[0], qr2[1])
        expected.cx(qr2[2], qr2[3])
        expected.u1(theta, qr2[1])
        expected.u1(theta, qr2[3])

        self.assertEqual(circuit_to_dag(expected), out_dag)

        # Expanding across register with shared parameter
        qc = QuantumCircuit(qr2)

        phi = Parameter('phi')
        gamma = Parameter('gamma')

        sub_instr = circuit_to_instruction(subqc, {theta: phi}, mock_sel)
        qc.append(sub_instr, [qr2[0], qr2[1]])
        sub_instr = circuit_to_instruction(subqc, {theta: gamma}, mock_sel)
        qc.append(sub_instr, [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        pass_ = UnrollCustomDefinitions(mock_sel, ['u1', 'cx'])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(mock_sel, ['u1', 'cx']).run(dag)

        expected = QuantumCircuit(qr2)
        expected.u1(phi, qr2[0])
        expected.u1(gamma, qr2[2])
        expected.cx(qr2[0], qr2[1])
        expected.cx(qr2[2], qr2[3])
        expected.u1(phi, qr2[1])
        expected.u1(gamma, qr2[3])

        self.assertEqual(circuit_to_dag(expected), out_dag)


class TestBasisExamples(QiskitTestCase):
    """Test example circuits targeting example bases over the StandardEquivalenceLibrary."""

    def test_cx_bell_to_cz(self):
        """Verify we can translate a CX bell circuit to CZ,RX,RZ."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        in_dag = circuit_to_dag(bell)
        out_dag = BasisTranslator(std_eqlib, ['cz', 'rx', 'rz']).run(in_dag)

        qr = QuantumRegister(2, 'q')
        expected = QuantumCircuit(qr)
        expected.rz(pi, qr)
        expected.rx(pi / 2, qr)
        expected.rz(3 * pi / 2, qr)
        expected.rx(pi / 2, qr)
        expected.rz(3 * pi, qr)
        expected.cz(qr[0], qr[1])
        expected.rz(pi, qr[1])
        expected.rx(pi / 2, qr[1])
        expected.rz(3 * pi / 2, qr[1])
        expected.rx(pi / 2, qr[1])
        expected.rz(3 * pi, qr[1])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out_dag, expected_dag)

    def test_cx_bell_to_iswap(self):
        """Verify we can translate a CX bell to iSwap,U3."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        in_dag = circuit_to_dag(bell)
        out_dag = BasisTranslator(std_eqlib, ['iswap', 'u3']).run(in_dag)

        qr = QuantumRegister(2, 'q')
        expected = QuantumCircuit(2)
        expected.u3(pi / 2, 0, pi, qr[0])
        expected.u3(pi, 0, pi, qr[1])
        expected.u3(pi / 2, 0, pi, qr)
        expected.iswap(qr[0], qr[1])
        expected.u3(pi, 0, pi, qr)
        expected.u3(pi / 2, 0, pi, qr[1])
        expected.iswap(qr[0], qr[1])
        expected.u3(pi / 2, 0, pi, qr[0])
        expected.u3(0, 0, pi / 2, qr)
        expected.u3(pi, 0, pi, qr[1])
        expected.u3(pi / 2, 0, pi, qr[1])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out_dag, expected_dag)
