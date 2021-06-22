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

"""Tests preset pass manager API"""
from test import combine
from ddt import ddt, data

import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.compiler import transpile, assemble
from qiskit.transpiler import CouplingMap, Layout
from qiskit.circuit.library import U2Gate, U3Gate
from qiskit.test import QiskitTestCase
from qiskit.test.mock import (
    FakeTenerife,
    FakeMelbourne,
    FakeJohannesburg,
    FakeRueschlikon,
    FakeTokyo,
    FakePoughkeepsie,
)
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import GraphState


def emptycircuit():
    """Empty circuit"""
    return QuantumCircuit()


def circuit_2532():
    """See https://github.com/Qiskit/qiskit-terra/issues/2532"""
    circuit = QuantumCircuit(5)
    circuit.cx(2, 4)
    return circuit


@ddt
class TestPresetPassManager(QiskitTestCase):
    """Test preset passmanagers work as expected."""

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_no_coupling_map(self, level):
        """Test that coupling_map can be None (level={level})"""
        q = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(q)
        circuit.cz(q[0], q[1])
        result = transpile(circuit, basis_gates=["u1", "u2", "u3", "cx"], optimization_level=level)
        self.assertIsInstance(result, QuantumCircuit)

    def test_layout_3239(self, level=3):
        """Test final layout after preset level3 passmanager does not include diagonal gates
        See: https://github.com/Qiskit/qiskit-terra/issues/3239
        """
        qc = QuantumCircuit(5, 5)
        qc.h(0)
        qc.cx(range(3), range(1, 4))
        qc.z(range(4))
        qc.measure(range(4), range(4))
        result = transpile(
            qc,
            basis_gates=["u1", "u2", "u3", "cx"],
            layout_method="trivial",
            optimization_level=level,
        )

        dag = circuit_to_dag(result)
        op_nodes = [node.name for node in dag.topological_op_nodes()]
        self.assertNotIn("u1", op_nodes)  # Check if the diagonal Z-Gates (u1) were removed

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_no_basis_gates(self, level):
        """Test that basis_gates can be None (level={level})"""
        q = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(q)
        circuit.h(q[0])
        circuit.cz(q[0], q[1])
        result = transpile(circuit, basis_gates=None, optimization_level=level)
        self.assertEqual(result, circuit)

    def test_level0_keeps_reset(self):
        """Test level 0 should keep the reset instructions"""
        q = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(q)
        circuit.reset(q[0])
        circuit.reset(q[0])
        result = transpile(circuit, basis_gates=None, optimization_level=0)
        self.assertEqual(result, circuit)

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_respect_basis(self, level):
        """Test that all levels respect basis"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.cp(np.pi / 8, 0, 1)
        qc.cp(np.pi / 4, 0, 2)
        basis_gates = ["id", "rz", "sx", "x", "cx"]
        result = transpile(
            qc, basis_gates=basis_gates, coupling_map=[[0, 1], [2, 1]], optimization_level=level
        )

        dag = circuit_to_dag(result)
        circuit_ops = {node.name for node in dag.topological_op_nodes()}
        self.assertEqual(circuit_ops.union(set(basis_gates)), set(basis_gates))


@ddt
class TestTranspileLevels(QiskitTestCase):
    """Test transpiler on fake backend"""

    @combine(
        circuit=[emptycircuit, circuit_2532],
        level=[0, 1, 2, 3],
        backend=[
            FakeTenerife(),
            FakeMelbourne(),
            FakeRueschlikon(),
            FakeTokyo(),
            FakePoughkeepsie(),
            None,
        ],
        dsc="Transpiler {circuit.__name__} on {backend} backend at level {level}",
        name="{circuit.__name__}_{backend}_level{level}",
    )
    def test(self, circuit, level, backend):
        """All the levels with all the backends"""
        result = transpile(circuit(), backend=backend, optimization_level=level, seed_transpiler=42)
        self.assertIsInstance(result, QuantumCircuit)


@ddt
class TestPassesInspection(QiskitTestCase):
    """Test run passes under different conditions"""

    def setUp(self):
        """Sets self.callback to set self.passes with the passes that have been executed"""
        super().setUp()
        self.passes = []

        def callback(**kwargs):
            self.passes.append(kwargs["pass_"].__class__.__name__)

        self.callback = callback

    @data(0, 1, 2, 3)
    def test_no_coupling_map(self, level):
        """Without coupling map, no layout selection nor swapper"""
        qr = QuantumRegister(3, "q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[2], qr[1])
        qc.cx(qr[2], qr[0])

        _ = transpile(qc, optimization_level=level, callback=self.callback)

        self.assertNotIn("SetLayout", self.passes)
        self.assertNotIn("TrivialLayout", self.passes)
        self.assertNotIn("ApplyLayout", self.passes)
        self.assertNotIn("StochasticSwap", self.passes)
        self.assertNotIn("CheckGateDirection", self.passes)

    @data(0, 1, 2, 3)
    def test_backend(self, level):
        """With backend a layout and a swapper is run"""
        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[2], qr[4])
        backend = FakeMelbourne()

        _ = transpile(qc, backend, optimization_level=level, callback=self.callback)

        self.assertIn("SetLayout", self.passes)
        self.assertIn("ApplyLayout", self.passes)
        self.assertIn("CheckGateDirection", self.passes)

    @data(0, 1, 2, 3)
    def test_5409(self, level):
        """The parameter layout_method='noise_adaptive' should be honored
        See: https://github.com/Qiskit/qiskit-terra/issues/5409
        """
        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[2], qr[4])
        backend = FakeMelbourne()

        _ = transpile(
            qc,
            backend,
            layout_method="noise_adaptive",
            optimization_level=level,
            callback=self.callback,
        )

        self.assertIn("SetLayout", self.passes)
        self.assertIn("ApplyLayout", self.passes)
        self.assertIn("NoiseAdaptiveLayout", self.passes)

    @data(0, 1, 2, 3)
    def test_symmetric_coupling_map(self, level):
        """Symmetric coupling map does not run CheckGateDirection"""
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[1])

        coupling_map = [[0, 1], [1, 0]]

        _ = transpile(
            qc,
            coupling_map=coupling_map,
            initial_layout=[0, 1],
            optimization_level=level,
            callback=self.callback,
        )

        self.assertIn("SetLayout", self.passes)
        self.assertIn("ApplyLayout", self.passes)
        self.assertNotIn("CheckGateDirection", self.passes)

    @data(0, 1, 2, 3)
    def test_initial_layout_fully_connected_cm(self, level):
        """Honor initial_layout when coupling_map=None
        See: https://github.com/Qiskit/qiskit-terra/issues/5345
        """
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])

        transpiled = transpile(
            qc, initial_layout=[0, 1], optimization_level=level, callback=self.callback
        )

        self.assertIn("SetLayout", self.passes)
        self.assertIn("ApplyLayout", self.passes)
        self.assertEqual(transpiled._layout, Layout.from_qubit_list([qr[0], qr[1]]))

    @data(0, 1, 2, 3)
    def test_partial_layout_fully_connected_cm(self, level):
        """Honor initial_layout (partially defined) when coupling_map=None
        See: https://github.com/Qiskit/qiskit-terra/issues/5345
        """
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])

        transpiled = transpile(
            qc, initial_layout=[4, 2], optimization_level=level, callback=self.callback
        )

        self.assertIn("SetLayout", self.passes)
        self.assertIn("ApplyLayout", self.passes)
        ancilla = QuantumRegister(3, "ancilla")
        self.assertEqual(
            transpiled._layout,
            Layout.from_qubit_list([ancilla[0], ancilla[1], qr[1], ancilla[2], qr[0]]),
        )


@ddt
class TestInitialLayouts(QiskitTestCase):
    """Test transpiling with different layouts"""

    @data(0, 1, 2, 3)
    def test_layout_1711(self, level):
        """Test that a user-given initial layout is respected,
        in the qobj.

        See: https://github.com/Qiskit/qiskit-terra/issues/1711
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3)
        ancilla = QuantumRegister(13, "ancilla")
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[2], qr[1])
        qc.cx(qr[2], qr[0])
        initial_layout = {0: qr[1], 2: qr[0], 15: qr[2]}
        final_layout = {
            0: qr[1],
            1: ancilla[0],
            2: qr[0],
            3: ancilla[1],
            4: ancilla[2],
            5: ancilla[3],
            6: ancilla[4],
            7: ancilla[5],
            8: ancilla[6],
            9: ancilla[7],
            10: ancilla[8],
            11: ancilla[9],
            12: ancilla[10],
            13: ancilla[11],
            14: ancilla[12],
            15: qr[2],
        }

        backend = FakeRueschlikon()

        qc_b = transpile(qc, backend, initial_layout=initial_layout, optimization_level=level)
        qobj = assemble(qc_b)

        self.assertEqual(qc_b._layout._p2v, final_layout)

        compiled_ops = qobj.experiments[0].instructions
        for operation in compiled_ops:
            if operation.name == "cx":
                self.assertIn(operation.qubits, backend.configuration().coupling_map)
                self.assertIn(operation.qubits, [[15, 0], [15, 2]])

    @data(0, 1, 2, 3)
    def test_layout_2532(self, level):
        """Test that a user-given initial layout is respected,
        in the transpiled circuit.

        See: https://github.com/Qiskit/qiskit-terra/issues/2532
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(5, "q")
        cr = ClassicalRegister(2)
        ancilla = QuantumRegister(9, "ancilla")
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[2], qr[4])
        initial_layout = {
            qr[2]: 11,
            qr[4]: 3,  # map to [11, 3] connection
            qr[0]: 1,
            qr[1]: 5,
            qr[3]: 9,
        }
        final_layout = {
            0: ancilla[0],
            1: qr[0],
            2: ancilla[1],
            3: qr[4],
            4: ancilla[2],
            5: qr[1],
            6: ancilla[3],
            7: ancilla[4],
            8: ancilla[5],
            9: qr[3],
            10: ancilla[6],
            11: qr[2],
            12: ancilla[7],
            13: ancilla[8],
        }
        backend = FakeMelbourne()

        qc_b = transpile(qc, backend, initial_layout=initial_layout, optimization_level=level)

        self.assertEqual(qc_b._layout._p2v, final_layout)

        output_qr = qc_b.qregs[0]
        for gate, qubits, _ in qc_b:
            if gate.name == "cx":
                for qubit in qubits:
                    self.assertIn(qubit, [output_qr[11], output_qr[3]])

    @data(0, 1, 2, 3)
    def test_layout_2503(self, level):
        """Test that a user-given initial layout is respected,
        even if cnots are not in the coupling map.

        See: https://github.com/Qiskit/qiskit-terra/issues/2503
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(2)
        ancilla = QuantumRegister(17, "ancilla")

        qc = QuantumCircuit(qr, cr)
        qc.append(U3Gate(0.1, 0.2, 0.3), [qr[0]])
        qc.append(U2Gate(0.4, 0.5), [qr[2]])
        qc.barrier()
        qc.cx(qr[0], qr[2])
        initial_layout = [6, 7, 12]

        final_layout = {
            0: ancilla[0],
            1: ancilla[1],
            2: ancilla[2],
            3: ancilla[3],
            4: ancilla[4],
            5: ancilla[5],
            6: qr[0],
            7: qr[1],
            8: ancilla[6],
            9: ancilla[7],
            10: ancilla[8],
            11: ancilla[9],
            12: qr[2],
            13: ancilla[10],
            14: ancilla[11],
            15: ancilla[12],
            16: ancilla[13],
            17: ancilla[14],
            18: ancilla[15],
            19: ancilla[16],
        }

        backend = FakePoughkeepsie()

        qc_b = transpile(qc, backend, initial_layout=initial_layout, optimization_level=level)

        self.assertEqual(qc_b._layout._p2v, final_layout)

        gate_0, qubits_0, _ = qc_b[0]
        gate_1, qubits_1, _ = qc_b[1]

        output_qr = qc_b.qregs[0]
        self.assertIsInstance(gate_0, U3Gate)
        self.assertEqual(qubits_0[0], output_qr[6])
        self.assertIsInstance(gate_1, U2Gate)
        self.assertEqual(qubits_1[0], output_qr[12])


@ddt
class TestFinalLayouts(QiskitTestCase):
    """Test final layouts after preset transpilation"""

    @data(0, 1, 2, 3)
    def test_layout_tokyo_2845(self, level):
        """Test that final layout in tokyo #2845
        See: https://github.com/Qiskit/qiskit-terra/issues/2845
        """
        qr1 = QuantumRegister(3, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qc = QuantumCircuit(qr1, qr2)
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr1[2])
        qc.cx(qr1[2], qr2[0])
        qc.cx(qr2[0], qr2[1])

        trivial_layout = {
            0: Qubit(QuantumRegister(3, "qr1"), 0),
            1: Qubit(QuantumRegister(3, "qr1"), 1),
            2: Qubit(QuantumRegister(3, "qr1"), 2),
            3: Qubit(QuantumRegister(2, "qr2"), 0),
            4: Qubit(QuantumRegister(2, "qr2"), 1),
            5: Qubit(QuantumRegister(15, "ancilla"), 0),
            6: Qubit(QuantumRegister(15, "ancilla"), 1),
            7: Qubit(QuantumRegister(15, "ancilla"), 2),
            8: Qubit(QuantumRegister(15, "ancilla"), 3),
            9: Qubit(QuantumRegister(15, "ancilla"), 4),
            10: Qubit(QuantumRegister(15, "ancilla"), 5),
            11: Qubit(QuantumRegister(15, "ancilla"), 6),
            12: Qubit(QuantumRegister(15, "ancilla"), 7),
            13: Qubit(QuantumRegister(15, "ancilla"), 8),
            14: Qubit(QuantumRegister(15, "ancilla"), 9),
            15: Qubit(QuantumRegister(15, "ancilla"), 10),
            16: Qubit(QuantumRegister(15, "ancilla"), 11),
            17: Qubit(QuantumRegister(15, "ancilla"), 12),
            18: Qubit(QuantumRegister(15, "ancilla"), 13),
            19: Qubit(QuantumRegister(15, "ancilla"), 14),
        }

        dense_layout = {
            2: Qubit(QuantumRegister(3, "qr1"), 0),
            6: Qubit(QuantumRegister(3, "qr1"), 1),
            1: Qubit(QuantumRegister(3, "qr1"), 2),
            5: Qubit(QuantumRegister(2, "qr2"), 0),
            0: Qubit(QuantumRegister(2, "qr2"), 1),
            3: Qubit(QuantumRegister(15, "ancilla"), 0),
            4: Qubit(QuantumRegister(15, "ancilla"), 1),
            7: Qubit(QuantumRegister(15, "ancilla"), 2),
            8: Qubit(QuantumRegister(15, "ancilla"), 3),
            9: Qubit(QuantumRegister(15, "ancilla"), 4),
            10: Qubit(QuantumRegister(15, "ancilla"), 5),
            11: Qubit(QuantumRegister(15, "ancilla"), 6),
            12: Qubit(QuantumRegister(15, "ancilla"), 7),
            13: Qubit(QuantumRegister(15, "ancilla"), 8),
            14: Qubit(QuantumRegister(15, "ancilla"), 9),
            15: Qubit(QuantumRegister(15, "ancilla"), 10),
            16: Qubit(QuantumRegister(15, "ancilla"), 11),
            17: Qubit(QuantumRegister(15, "ancilla"), 12),
            18: Qubit(QuantumRegister(15, "ancilla"), 13),
            19: Qubit(QuantumRegister(15, "ancilla"), 14),
        }

        csp_layout = {
            13: Qubit(QuantumRegister(3, "qr1"), 0),
            19: Qubit(QuantumRegister(3, "qr1"), 1),
            14: Qubit(QuantumRegister(3, "qr1"), 2),
            18: Qubit(QuantumRegister(2, "qr2"), 0),
            17: Qubit(QuantumRegister(2, "qr2"), 1),
            0: Qubit(QuantumRegister(15, "ancilla"), 0),
            1: Qubit(QuantumRegister(15, "ancilla"), 1),
            2: Qubit(QuantumRegister(15, "ancilla"), 2),
            3: Qubit(QuantumRegister(15, "ancilla"), 3),
            4: Qubit(QuantumRegister(15, "ancilla"), 4),
            5: Qubit(QuantumRegister(15, "ancilla"), 5),
            6: Qubit(QuantumRegister(15, "ancilla"), 6),
            7: Qubit(QuantumRegister(15, "ancilla"), 7),
            8: Qubit(QuantumRegister(15, "ancilla"), 8),
            9: Qubit(QuantumRegister(15, "ancilla"), 9),
            10: Qubit(QuantumRegister(15, "ancilla"), 10),
            11: Qubit(QuantumRegister(15, "ancilla"), 11),
            12: Qubit(QuantumRegister(15, "ancilla"), 12),
            15: Qubit(QuantumRegister(15, "ancilla"), 13),
            16: Qubit(QuantumRegister(15, "ancilla"), 14),
        }

        # Trivial layout
        expected_layout_level0 = trivial_layout
        # Dense layout
        expected_layout_level1 = dense_layout
        # CSP layout
        expected_layout_level2 = csp_layout
        expected_layout_level3 = csp_layout

        expected_layouts = [
            expected_layout_level0,
            expected_layout_level1,
            expected_layout_level2,
            expected_layout_level3,
        ]
        backend = FakeTokyo()
        result = transpile(qc, backend, optimization_level=level, seed_transpiler=42)
        self.assertEqual(result._layout._p2v, expected_layouts[level])

    @data(0, 1, 2, 3)
    def test_layout_tokyo_fully_connected_cx(self, level):
        """Test that final layout in tokyo in a fully connected circuit"""
        qr = QuantumRegister(5, "qr")
        qc = QuantumCircuit(qr)
        for qubit_target in qr:
            for qubit_control in qr:
                if qubit_control != qubit_target:
                    qc.cx(qubit_control, qubit_target)

        ancilla = QuantumRegister(15, "ancilla")

        trivial_layout = {
            0: qr[0],
            1: qr[1],
            2: qr[2],
            3: qr[3],
            4: qr[4],
            5: ancilla[0],
            6: ancilla[1],
            7: ancilla[2],
            8: ancilla[3],
            9: ancilla[4],
            10: ancilla[5],
            11: ancilla[6],
            12: ancilla[7],
            13: ancilla[8],
            14: ancilla[9],
            15: ancilla[10],
            16: ancilla[11],
            17: ancilla[12],
            18: ancilla[13],
            19: ancilla[14],
        }

        dense_layout = {
            2: qr[0],
            6: qr[1],
            1: qr[2],
            5: qr[3],
            0: qr[4],
            3: ancilla[0],
            4: ancilla[1],
            7: ancilla[2],
            8: ancilla[3],
            9: ancilla[4],
            10: ancilla[5],
            11: ancilla[6],
            12: ancilla[7],
            13: ancilla[8],
            14: ancilla[9],
            15: ancilla[10],
            16: ancilla[11],
            17: ancilla[12],
            18: ancilla[13],
            19: ancilla[14],
        }

        expected_layout_level0 = trivial_layout
        expected_layout_level1 = dense_layout
        expected_layout_level2 = dense_layout
        expected_layout_level3 = dense_layout

        expected_layouts = [
            expected_layout_level0,
            expected_layout_level1,
            expected_layout_level2,
            expected_layout_level3,
        ]
        backend = FakeTokyo()
        result = transpile(qc, backend, optimization_level=level, seed_transpiler=42)
        self.assertEqual(result._layout._p2v, expected_layouts[level])

    @data(0, 1, 2, 3)
    def test_all_levels_use_trivial_if_perfect(self, level):
        """Test that we always use trivial if it's a perfect match.

        See: https://github.com/Qiskit/qiskit-terra/issues/5694 for more
        details
        """
        backend = FakeTokyo()
        config = backend.configuration()

        rows = [x[0] for x in config.coupling_map]
        cols = [x[1] for x in config.coupling_map]

        adjacency_matrix = np.zeros((20, 20))
        adjacency_matrix[rows, cols] = 1
        qc = GraphState(adjacency_matrix)
        qc.measure_all()
        expected = {
            0: Qubit(QuantumRegister(20, "q"), 0),
            1: Qubit(QuantumRegister(20, "q"), 1),
            2: Qubit(QuantumRegister(20, "q"), 2),
            3: Qubit(QuantumRegister(20, "q"), 3),
            4: Qubit(QuantumRegister(20, "q"), 4),
            5: Qubit(QuantumRegister(20, "q"), 5),
            6: Qubit(QuantumRegister(20, "q"), 6),
            7: Qubit(QuantumRegister(20, "q"), 7),
            8: Qubit(QuantumRegister(20, "q"), 8),
            9: Qubit(QuantumRegister(20, "q"), 9),
            10: Qubit(QuantumRegister(20, "q"), 10),
            11: Qubit(QuantumRegister(20, "q"), 11),
            12: Qubit(QuantumRegister(20, "q"), 12),
            13: Qubit(QuantumRegister(20, "q"), 13),
            14: Qubit(QuantumRegister(20, "q"), 14),
            15: Qubit(QuantumRegister(20, "q"), 15),
            16: Qubit(QuantumRegister(20, "q"), 16),
            17: Qubit(QuantumRegister(20, "q"), 17),
            18: Qubit(QuantumRegister(20, "q"), 18),
            19: Qubit(QuantumRegister(20, "q"), 19),
        }
        trans_qc = transpile(qc, backend, optimization_level=level)
        self.assertEqual(trans_qc._layout._p2v, expected)

    @data(0, 1)
    def test_trivial_layout(self, level):
        """Test that trivial layout is preferred in level 0 and 1
        See: https://github.com/Qiskit/qiskit-terra/pull/3657#pullrequestreview-342012465
        """
        qr = QuantumRegister(10, "qr")
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[2], qr[6])
        qc.cx(qr[3], qr[8])
        qc.cx(qr[4], qr[9])
        qc.cx(qr[9], qr[8])
        qc.cx(qr[8], qr[7])
        qc.cx(qr[7], qr[6])
        qc.cx(qr[6], qr[5])
        qc.cx(qr[5], qr[0])

        ancilla = QuantumRegister(10, "ancilla")
        trivial_layout = {
            0: qr[0],
            1: qr[1],
            2: qr[2],
            3: qr[3],
            4: qr[4],
            5: qr[5],
            6: qr[6],
            7: qr[7],
            8: qr[8],
            9: qr[9],
            10: ancilla[0],
            11: ancilla[1],
            12: ancilla[2],
            13: ancilla[3],
            14: ancilla[4],
            15: ancilla[5],
            16: ancilla[6],
            17: ancilla[7],
            18: ancilla[8],
            19: ancilla[9],
        }

        expected_layouts = [trivial_layout, trivial_layout]

        backend = FakeTokyo()
        result = transpile(qc, backend, optimization_level=level, seed_transpiler=42)
        self.assertEqual(result._layout._p2v, expected_layouts[level])

    @data(0, 1, 2, 3)
    def test_initial_layout(self, level):
        """When a user provides a layout (initial_layout), it should be used."""
        qr = QuantumRegister(10, "qr")
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[3], qr[9])
        qc.cx(qr[4], qr[9])
        qc.cx(qr[9], qr[8])
        qc.cx(qr[8], qr[7])
        qc.cx(qr[7], qr[6])
        qc.cx(qr[6], qr[5])
        qc.cx(qr[5], qr[0])

        initial_layout = {
            0: qr[0],
            2: qr[1],
            4: qr[2],
            6: qr[3],
            8: qr[4],
            10: qr[5],
            12: qr[6],
            14: qr[7],
            16: qr[8],
            18: qr[9],
        }

        backend = FakeTokyo()
        result = transpile(
            qc, backend, optimization_level=level, initial_layout=initial_layout, seed_transpiler=42
        )

        for physical, virtual in initial_layout.items():
            self.assertEqual(result._layout._p2v[physical], virtual)


@ddt
class TestTranspileLevelsSwap(QiskitTestCase):
    """Test if swap is in the basis, do not unroll
    See https://github.com/Qiskit/qiskit-terra/pull/3963
    The circuit in combine should require a swap and that swap should exit at the end
    for the transpilation"""

    @combine(
        circuit=[circuit_2532],
        level=[0, 1, 2, 3],
        dsc="circuit: {circuit.__name__}, level: {level}",
        name="{circuit.__name__}_level{level}",
    )
    def test_1(self, circuit, level):
        """Simple coupling map (linear 5 qubits)."""
        basis = ["u1", "u2", "cx", "swap"]
        coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])
        result = transpile(
            circuit(),
            optimization_level=level,
            basis_gates=basis,
            coupling_map=coupling_map,
            seed_transpiler=42,
            initial_layout=[0, 1, 2, 3, 4],
        )
        self.assertIsInstance(result, QuantumCircuit)
        resulting_basis = {node.name for node in circuit_to_dag(result).op_nodes()}
        self.assertIn("swap", resulting_basis)

    @combine(
        level=[0, 1, 2, 3],
        dsc="If swap in basis, do not decompose it. level: {level}",
        name="level{level}",
    )
    def test_2(self, level):
        """Simple coupling map (linear 5 qubits).
        The circuit requires a swap and that swap should exit at the end
        for the transpilation"""
        basis = ["u1", "u2", "cx", "swap"]
        circuit = QuantumCircuit(5)
        circuit.cx(0, 4)
        circuit.cx(1, 4)
        circuit.cx(2, 4)
        circuit.cx(3, 4)
        coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])
        result = transpile(
            circuit,
            optimization_level=level,
            basis_gates=basis,
            coupling_map=coupling_map,
            seed_transpiler=42,
        )
        self.assertIsInstance(result, QuantumCircuit)
        resulting_basis = {node.name for node in circuit_to_dag(result).op_nodes()}
        self.assertIn("swap", resulting_basis)


@ddt
class TestOptimizationWithCondition(QiskitTestCase):
    """Test optimization levels with condition in the circuit"""

    @data(0, 1, 2, 3)
    def test_optimization_condition(self, level):
        """Test optimization levels with condition in the circuit"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        qc.cx(0, 1).c_if(cr, 1)
        backend = FakeJohannesburg()
        circ = transpile(qc, backend, optimization_level=level)
        self.assertIsInstance(circ, QuantumCircuit)

    def test_input_dag_copy(self):
        """Test substitute_node_with_dag input_dag copy on condition"""
        qc = QuantumCircuit(2, 1)
        qc.cx(0, 1).c_if(qc.cregs[0], 1)
        qc.cx(1, 0)
        circ = transpile(qc, basis_gates=["u3", "cz"])
        self.assertIsInstance(circ, QuantumCircuit)
