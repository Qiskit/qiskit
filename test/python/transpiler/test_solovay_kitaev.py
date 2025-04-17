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

"""Test the Solovay Kitaev transpilation pass."""

import os
import unittest
import math
import tempfile
import numpy as np
import scipy

from ddt import ddt, data

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library import (
    TGate,
    TdgGate,
    HGate,
    SGate,
    SdgGate,
    IGate,
    QFT,
    RXGate,
    RYGate,
    RZGate,
)
from qiskit.circuit import QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.synthesis.discrete_basis.generate_basis_approximations import (
    generate_basic_approximations,
)
from qiskit.synthesis.discrete_basis.commutator_decompose import commutator_decompose
from qiskit.synthesis.discrete_basis.gate_sequence import GateSequence
from qiskit.synthesis.discrete_basis.solovay_kitaev import SolovayKitaevCompiler
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnitarySynthesis, Collect1qRuns, ConsolidateBlocks
from qiskit.transpiler.passes.synthesis import SolovayKitaev, SolovayKitaevSynthesis
from test import QiskitTestCase  # pylint: disable=wrong-import-order


def _trace_distance(circuit1, circuit2):
    """Return the trace distance of the two input circuits."""
    op1, op2 = Operator(circuit1), Operator(circuit2)
    return 0.5 * np.trace(scipy.linalg.sqrtm(np.conj(op1 - op2).T.dot(op1 - op2))).real


def _generate_x_rotation(angle: float) -> np.ndarray:
    return np.array(
        [[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]
    )


def _generate_y_rotation(angle: float) -> np.ndarray:
    return np.array(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]]
    )


def _generate_z_rotation(angle: float) -> np.ndarray:
    return np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )


def is_so3_matrix(array: np.ndarray) -> bool:
    """Check if the input array is a SO(3) matrix."""
    if array.shape != (3, 3):
        return False

    if abs(np.linalg.det(array) - 1.0) > 1e-10:
        return False

    if False in np.isreal(array):
        return False

    return True


@ddt
class TestSolovayKitaev(QiskitTestCase):
    """Test the Solovay Kitaev algorithm and transformation pass."""

    def setUp(self):
        super().setUp()

        self.basic_approx = generate_basic_approximations([HGate(), TGate(), TdgGate()], 3)

    def test_unitary_synthesis(self):
        """Test the unitary synthesis transpiler pass with Solovay-Kitaev."""
        circuit = QuantumCircuit(2)
        circuit.rx(0.8, 0)
        circuit.cx(0, 1)
        circuit.x(1)

        _1q = Collect1qRuns()
        _cons = ConsolidateBlocks()
        _synth = UnitarySynthesis(["h", "t", "tdg"], method="sk")
        passes = PassManager([_1q, _cons, _synth])
        compiled = passes.run(circuit)

        diff = _trace_distance(circuit, compiled)
        self.assertLess(diff, 1e-5)
        self.assertEqual(set(compiled.count_ops().keys()), {"h", "t", "tdg", "cx"})

    def test_plugin(self):
        """Test calling the plugin directly."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        unitary = Operator(circuit).data

        plugin = SolovayKitaevSynthesis()
        out = plugin.run(unitary, basis_gates=["h", "s"], depth=10, recursion_degree=3)

        reference = QuantumCircuit(1, global_phase=7 * np.pi / 4)
        reference.h(0)
        reference.s(0)
        reference.h(0)

        self.assertEqual(dag_to_circuit(out), reference)

    def test_i_returns_empty_circuit(self):
        """Test that ``SolovayKitaev`` returns an empty circuit when
        it approximates the I-gate."""
        circuit = QuantumCircuit(1)
        circuit.id(0)

        skd = SolovayKitaev(3, self.basic_approx)

        decomposed_circuit = skd(circuit)
        self.assertEqual(QuantumCircuit(1), decomposed_circuit)

    def test_exact_decomposition_acts_trivially(self):
        """Test that the a circuit that can be represented exactly is represented exactly."""
        circuit = QuantumCircuit(1)
        circuit.t(0)
        circuit.h(0)
        circuit.tdg(0)

        synth = SolovayKitaev(3, self.basic_approx)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)
        self.assertEqual(circuit, decomposed_circuit)

    def test_str_basis_gates(self):
        """Test specifying the basis gates by string works."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        basic_approx = generate_basic_approximations(["h", "t", "s"], 3)
        synth = SolovayKitaev(2, basic_approx)

        dag = circuit_to_dag(circuit)
        discretized = dag_to_circuit(synth.run(dag))

        reference = QuantumCircuit(1, global_phase=7 * np.pi / 8)
        reference.h(0)
        reference.t(0)
        reference.h(0)

        self.assertEqual(discretized, reference)

    def test_approximation_on_qft(self):
        """Test the Solovay-Kitaev decomposition on the QFT circuit."""
        qft = QFT(3)
        transpiled = transpile(qft, basis_gates=["u", "cx"], optimization_level=1)

        skd = SolovayKitaev(5)

        with self.subTest("1 recursion"):
            discretized = skd(transpiled)
            self.assertLess(_trace_distance(transpiled, discretized), 15)

        skd.recursion_degree = 5
        with self.subTest("2 recursions"):
            discretized = skd(transpiled)
            self.assertLess(_trace_distance(transpiled, discretized), 7)

    def test_u_gates_work(self):
        """Test SK works on Qiskit's UGate.

        Regression test of Qiskit/qiskit-terra#9437.
        """
        circuit = QuantumCircuit(1)
        circuit.u(np.pi / 2, -np.pi, -np.pi, 0)
        circuit.u(np.pi / 2, np.pi / 2, -np.pi, 0)
        circuit.u(-np.pi / 4, 0, -np.pi / 2, 0)
        circuit.u(np.pi / 4, -np.pi / 16, 0, 0)
        circuit.u(0, 0, np.pi / 16, 0)
        circuit.u(0, np.pi / 4, np.pi / 4, 0)
        circuit.u(np.pi / 2, 0, -15 * np.pi / 16, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.u(np.pi / 2, 0, -3 * np.pi / 4, 0)
        circuit.u(0, 0, -np.pi / 16, 0)
        circuit.u(np.pi / 2, 0, 15 * np.pi / 16, 0)

        depth = 4
        basis_gates = ["h", "t", "tdg", "s", "z"]
        gate_approx_library = generate_basic_approximations(basis_gates=basis_gates, depth=depth)

        skd = SolovayKitaev(recursion_degree=2, basic_approximations=gate_approx_library)
        discretized = skd(circuit)

        included_gates = set(discretized.count_ops().keys())
        self.assertEqual(set(basis_gates), included_gates)

    def test_load_from_file(self):
        """Test loading basic approximations from a file works.

        Regression test of Qiskit/qiskit#12576.
        """
        filename = "approximations.npy"

        with tempfile.TemporaryDirectory() as tmp_dir:
            fullpath = os.path.join(tmp_dir, filename)

            # dump approximations to file
            generate_basic_approximations(basis_gates=["h", "s", "sdg"], depth=3, filename=fullpath)

            # circuit to decompose and reference decomp
            circuit = QuantumCircuit(1)
            circuit.rx(0.8, 0)

            reference = QuantumCircuit(1, global_phase=3 * np.pi / 4)
            reference.h(0)
            reference.s(0)
            reference.h(0)

            # load the decomp and compare to reference
            skd = SolovayKitaev(basic_approximations=fullpath)
            # skd = SolovayKitaev(basic_approximations=filename)
            discretized = skd(circuit)

        self.assertEqual(discretized, reference)

    def test_measure(self):
        """Test the Solovay-Kitaev transpiler pass on circuits with measure operators."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        transpiled = SolovayKitaev()(qc)
        self.assertEqual(set(transpiled.count_ops()), {"h", "t", "measure"})

    def test_barrier(self):
        """Test the Solovay-Kitaev transpiler pass on circuits with barriers."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.barrier(0)
        transpiled = SolovayKitaev()(qc)
        self.assertEqual(set(transpiled.count_ops()), {"h", "t", "barrier"})

    def test_parameterized_gates(self):
        """Test the Solovay-Kitaev transpiler pass on circuits with parameterized gates."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.rz(Parameter("t"), 0)
        transpiled = SolovayKitaev()(qc)
        self.assertEqual(set(transpiled.count_ops()), {"h", "t", "rz"})

    def test_control_flow_if(self):
        """Test the Solovay-Kitaev transpiler pass on circuits with control flow ops"""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        with qc.if_test((cr[0], 0)) as else_:
            qc.y(0)
        with else_:
            qc.z(0)
        transpiled = SolovayKitaev()(qc)

        # check that we still have an if-else block and all the operations within
        # have been recursively synthesized
        self.assertEqual(transpiled[0].name, "if_else")
        for block in transpiled[0].operation.blocks:
            self.assertLessEqual(set(block.count_ops()), {"h", "t", "tdg"})

    def test_no_to_matrix(self):
        """Test the Solovay-Kitaev transpiler pass ignores gates without to_matrix."""
        qc = QuantumCircuit(1)
        qc.initialize("0")

        transpiled = SolovayKitaev()(qc)
        self.assertEqual(set(transpiled.count_ops()), {"initialize"})


@ddt
class TestGateSequence(QiskitTestCase):
    """Test the ``GateSequence`` class."""

    def test_append(self):
        """Test append."""
        seq = GateSequence([IGate()])
        seq.append(HGate())

        ref = GateSequence([IGate(), HGate()])
        self.assertEqual(seq, ref)

    def test_eq(self):
        """Test equality."""
        base = GateSequence([HGate(), HGate()])
        seq1 = GateSequence([HGate(), HGate()])
        seq2 = GateSequence([IGate()])
        seq3 = GateSequence([HGate(), HGate()])
        seq3.global_phase = 0.12
        seq4 = GateSequence([IGate(), HGate()])

        with self.subTest("equal"):
            self.assertEqual(base, seq1)

        with self.subTest("same product, but different repr (-> false)"):
            self.assertNotEqual(base, seq2)

        with self.subTest("differing global phase (-> false)"):
            self.assertNotEqual(base, seq3)

        with self.subTest("same num gates, but different gates (-> false)"):
            self.assertNotEqual(base, seq4)

    def test_to_circuit(self):
        """Test converting a gate sequence to a circuit."""
        seq = GateSequence([HGate(), HGate(), TGate(), SGate(), SdgGate()])
        ref = QuantumCircuit(1)
        ref.h(0)
        ref.h(0)
        ref.t(0)
        ref.s(0)
        ref.sdg(0)
        # a GateSequence is SU(2), so add the right phase
        z = 1 / np.sqrt(np.linalg.det(Operator(ref)))
        ref.global_phase = np.arctan2(np.imag(z), np.real(z))

        self.assertEqual(seq.to_circuit(), ref)

    def test_adjoint(self):
        """Test adjoint."""
        seq = GateSequence([TGate(), SGate(), HGate(), IGate()])
        inv = GateSequence([IGate(), HGate(), SdgGate(), TdgGate()])

        self.assertEqual(seq.adjoint(), inv)

    def test_copy(self):
        """Test copy."""
        seq = GateSequence([IGate()])
        copied = seq.copy()
        seq.gates.append(HGate())

        self.assertEqual(len(seq.gates), 2)
        self.assertEqual(len(copied.gates), 1)

    @data(0, 1, 10)
    def test_len(self, n):
        """Test __len__."""
        seq = GateSequence([IGate()] * n)
        self.assertEqual(len(seq), n)

    def test_getitem(self):
        """Test __getitem__."""
        seq = GateSequence([IGate(), HGate(), IGate()])
        self.assertEqual(seq[0], IGate())
        self.assertEqual(seq[1], HGate())
        self.assertEqual(seq[2], IGate())

        self.assertEqual(seq[-2], HGate())

    def test_from_su2_matrix(self):
        """Test from_matrix with an SU2 matrix."""
        matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        matrix /= np.sqrt(np.linalg.det(matrix))
        seq = GateSequence.from_matrix(matrix)

        ref = GateSequence([HGate()])

        self.assertEqual(seq.gates, [])
        self.assertTrue(np.allclose(seq.product, ref.product))
        self.assertEqual(seq.global_phase, 0)

    def test_from_so3_matrix(self):
        """Test from_matrix with an SO3 matrix."""
        matrix = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        seq = GateSequence.from_matrix(matrix)

        ref = GateSequence([HGate()])

        self.assertEqual(seq.gates, [])
        self.assertTrue(np.allclose(seq.product, ref.product))
        self.assertEqual(seq.global_phase, 0)

    def test_from_invalid_matrix(self):
        """Test from_matrix with invalid matrices."""
        with self.subTest("2x2 but not SU2"):
            matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            with self.assertRaises(ValueError):
                _ = GateSequence.from_matrix(matrix)

        with self.subTest("not 2x2 or 3x3"):
            with self.assertRaises(ValueError):
                _ = GateSequence.from_matrix(np.array([[1]]))

    def test_dot(self):
        """Test dot."""
        seq1 = GateSequence([HGate()])
        seq2 = GateSequence([TGate(), SGate()])
        composed = seq1.dot(seq2)

        ref = GateSequence([TGate(), SGate(), HGate()])

        # check the product matches
        self.assertTrue(np.allclose(ref.product, composed.product))

        # check the circuit & phases are equivalent
        self.assertTrue(Operator(ref.to_circuit()).equiv(composed.to_circuit()))


@ddt
class TestSolovayKitaevUtils(QiskitTestCase):
    """Test the public functions in the Solovay Kitaev utils."""

    @data(
        _generate_x_rotation(0.1),
        _generate_y_rotation(0.2),
        _generate_z_rotation(0.3),
        np.dot(_generate_z_rotation(0.5), _generate_y_rotation(0.4)),
        np.dot(_generate_y_rotation(0.5), _generate_x_rotation(0.4)),
    )
    def test_commutator_decompose_return_type(self, u_so3: np.ndarray):
        """Test that ``commutator_decompose`` returns two SO(3) gate sequences."""
        v, w = commutator_decompose(u_so3)
        self.assertTrue(is_so3_matrix(v.product))
        self.assertTrue(is_so3_matrix(w.product))
        self.assertIsInstance(v, GateSequence)
        self.assertIsInstance(w, GateSequence)

    @data(
        _generate_x_rotation(0.1),
        _generate_y_rotation(0.2),
        _generate_z_rotation(0.3),
        np.dot(_generate_z_rotation(0.5), _generate_y_rotation(0.4)),
        np.dot(_generate_y_rotation(0.5), _generate_x_rotation(0.4)),
    )
    def test_commutator_decompose_decomposes_correctly(self, u_so3):
        """Test that ``commutator_decompose`` exactly decomposes the input."""
        v, w = commutator_decompose(u_so3)
        v_so3 = v.product
        w_so3 = w.product
        actual_commutator = np.dot(v_so3, np.dot(w_so3, np.dot(np.conj(v_so3).T, np.conj(w_so3).T)))
        self.assertTrue(np.allclose(actual_commutator, u_so3))

    def test_generate_basis_approximation_gates(self):
        """Test the basis approximation generation works for all supported gates.

        Regression test of Qiskit/qiskit-terra#9585.
        """
        basis = ["i", "x", "y", "z", "h", "t", "tdg", "s", "sdg", "sx", "sxdg"]
        approx = generate_basic_approximations(basis, depth=2)

        # This mainly checks that there are no errors in the generation (like
        # in computing the inverse as described in #9585), so a simple check is enough.
        self.assertGreater(len(approx), len(basis))


@ddt
class TestSolovayKitaevCompiler(QiskitTestCase):
    """Test the Solovay Kitaev Compiler."""

    def setUp(self):
        super().setUp()
        self.default_sk = SolovayKitaevCompiler(depth=12)

    def test_i_returns_empty_circuit(self):
        """Test an empty circuit is returned for the identity."""
        synth = self.default_sk.synthesize(IGate(), 1)

        self.assertEqual(QuantumCircuit(1), synth)
        self.assertAlmostEqual(_trace_distance(IGate(), synth), 0)

    def test_exact_decomposition_acts_trivially(self):
        """Test that the a circuit that can be represented exactly is represented exactly."""
        for gate in [HGate(), TGate(), TdgGate()]:
            synth = self.default_sk.synthesize(gate, 1)

            expected = QuantumCircuit(1)
            expected.append(gate, [0])

            self.assertEqual(expected, synth)
            self.assertAlmostEqual(_trace_distance(expected, synth), 0)

    @data(True, False)
    def test_approximation(self, use_matrix):
        """Test the approximation works."""
        recursion = 4
        for angle in np.pi / np.arange(1, 10, 2):
            with self.subTest(angle=angle):
                gate = RZGate(angle)
                if use_matrix:
                    synth = self.default_sk.synthesize_matrix(gate.to_matrix(), recursion)
                else:
                    synth = self.default_sk.synthesize(gate, recursion)

                error = np.linalg.norm(Operator(gate).data - Operator(synth).data)
                self.assertLess(error, 2e-3)

    def test_find_basic_approximation(self):
        """Test finding the basic approximations for some gates."""
        # exactly representable for depth of 12
        for gate_cls in [RXGate, RYGate, RZGate]:
            for angle in [np.pi, np.pi / 2, np.pi / 4]:
                with self.subTest(gate_cls=gate_cls, angle=angle):
                    gate = gate_cls(angle)
                    synth = self.default_sk.find_basic_approximation(gate)
                    error = np.linalg.norm(Operator(gate).data - Operator(synth).data)
                    self.assertLess(error, 1e-10)

        # approximately representable
        for gate_cls in [RXGate, RYGate, RZGate]:
            for angle in [np.pi / 8, np.pi / 10, np.pi / 20]:
                with self.subTest(gate_cls=gate_cls, angle=angle):
                    gate = gate_cls(angle)
                    synth = self.default_sk.find_basic_approximation(gate)
                    error = np.linalg.norm(Operator(gate).data - Operator(synth).data)
                    self.assertLess(error, 0.5)

    @data(True, False)
    def test_basis_gates(self, use_str):
        """Test specifying the basis gates by string or gate works."""
        if use_str:
            basis_gates = ["h", "s", "sdg"]
        else:
            basis_gates = [HGate(), SGate(), SdgGate()]

        sk = SolovayKitaevCompiler(basis_gates, 3)

        synth = sk.synthesize(RXGate(np.pi / 2), 3)

        expected = QuantumCircuit(1, global_phase=7 * np.pi / 4)
        expected.h(0)
        expected.s(0)
        expected.h(0)

        self.assertEqual(synth, expected)

    # def test_approximation_on_qft(self):
    #     """Test the Solovay-Kitaev decomposition on the QFT circuit."""
    #     qft = QFT(3)
    #     transpiled = transpile(
    #         qft, basis_gates=["u1", "u2", "u3", "u", "cx", "id"], optimization_level=1
    #     )

    #     skd = SolovayKitaev(1)

    #     with self.subTest("1 recursion"):
    #         discretized = skd(transpiled)
    #         self.assertLess(_trace_distance(transpiled, discretized), 15)

    #     skd.recursion_degree = 2
    #     with self.subTest("2 recursions"):
    #         discretized = skd(transpiled)
    #         self.assertLess(_trace_distance(transpiled, discretized), 7)

    # def test_u_gates_work(self):
    #     """Test SK works on Qiskit's UGate.

    #     Regression test of Qiskit/qiskit-terra#9437.
    #     """
    #     circuit = QuantumCircuit(1)
    #     circuit.u(np.pi / 2, -np.pi, -np.pi, 0)
    #     circuit.u(np.pi / 2, np.pi / 2, -np.pi, 0)
    #     circuit.u(-np.pi / 4, 0, -np.pi / 2, 0)
    #     circuit.u(np.pi / 4, -np.pi / 16, 0, 0)
    #     circuit.u(0, 0, np.pi / 16, 0)
    #     circuit.u(0, np.pi / 4, np.pi / 4, 0)
    #     circuit.u(np.pi / 2, 0, -15 * np.pi / 16, 0)
    #     circuit.p(-np.pi / 4, 0)
    #     circuit.p(np.pi / 4, 0)
    #     circuit.u(np.pi / 2, 0, -3 * np.pi / 4, 0)
    #     circuit.u(0, 0, -np.pi / 16, 0)
    #     circuit.u(np.pi / 2, 0, 15 * np.pi / 16, 0)

    #     depth = 4
    #     basis_gates = ["h", "t", "tdg", "s", "z"]
    #     gate_approx_library = generate_basic_approximations(basis_gates=basis_gates, depth=depth)

    #     skd = SolovayKitaev(recursion_degree=2, basic_approximations=gate_approx_library)
    #     discretized = skd(circuit)

    #     included_gates = set(discretized.count_ops().keys())
    #     self.assertEqual(set(basis_gates), included_gates)

    # def test_load_from_file(self):
    #     """Test loading basic approximations from a file works.

    #     Regression test of Qiskit/qiskit#12576.
    #     """
    #     filename = "approximations.npy"

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         fullpath = os.path.join(tmp_dir, filename)

    #         # dump approximations to file
    #         generate_basic_approximations(basis_gates=["h", "s", "sdg"], depth=3, filename=fullpath)

    #         # circuit to decompose and reference decomp
    #         circuit = QuantumCircuit(1)
    #         circuit.rx(0.8, 0)

    #         reference = QuantumCircuit(1, global_phase=3 * np.pi / 4)
    #         reference.h(0)
    #         reference.s(0)
    #         reference.h(0)

    #         # load the decomp and compare to reference
    #         skd = SolovayKitaev(basic_approximations=fullpath)
    #         # skd = SolovayKitaev(basic_approximations=filename)
    #         discretized = skd(circuit)

    #     self.assertEqual(discretized, reference)

    # def test_measure(self):
    #     """Test the Solovay-Kitaev transpiler pass on circuits with measure operators."""
    #     qc = QuantumCircuit(1, 1)
    #     qc.x(0)
    #     qc.measure(0, 0)
    #     transpiled = SolovayKitaev()(qc)
    #     self.assertEqual(set(transpiled.count_ops()), {"h", "t", "measure"})

    # def test_barrier(self):
    #     """Test the Solovay-Kitaev transpiler pass on circuits with barriers."""
    #     qc = QuantumCircuit(1)
    #     qc.x(0)
    #     qc.barrier(0)
    #     transpiled = SolovayKitaev()(qc)
    #     self.assertEqual(set(transpiled.count_ops()), {"h", "t", "barrier"})

    # def test_parameterized_gates(self):
    #     """Test the Solovay-Kitaev transpiler pass on circuits with parameterized gates."""
    #     qc = QuantumCircuit(1)
    #     qc.x(0)
    #     qc.rz(Parameter("t"), 0)
    #     transpiled = SolovayKitaev()(qc)
    #     self.assertEqual(set(transpiled.count_ops()), {"h", "t", "rz"})

    # def test_control_flow_if(self):
    #     """Test the Solovay-Kitaev transpiler pass on circuits with control flow ops"""
    #     qr = QuantumRegister(1)
    #     cr = ClassicalRegister(1)
    #     qc = QuantumCircuit(qr, cr)

    #     with qc.if_test((cr[0], 0)) as else_:
    #         qc.y(0)
    #     with else_:
    #         qc.z(0)
    #     transpiled = SolovayKitaev()(qc)

    #     # check that we still have an if-else block and all the operations within
    #     # have been recursively synthesized
    #     self.assertEqual(transpiled[0].name, "if_else")
    #     for block in transpiled[0].operation.blocks:
    #         self.assertLessEqual(set(block.count_ops()), {"h", "t", "tdg"})

    # def test_no_to_matrix(self):
    #     """Test the Solovay-Kitaev transpiler pass ignores gates without to_matrix."""
    #     qc = QuantumCircuit(1)
    #     qc.initialize("0")

    #     transpiled = SolovayKitaev()(qc)
    #     self.assertEqual(set(transpiled.count_ops()), {"initialize"})


if __name__ == "__main__":
    unittest.main()
