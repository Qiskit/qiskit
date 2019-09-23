# -*- coding: utf-8 -*-

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

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.compiler import transpile, assemble
from qiskit.extensions.standard import U2Gate, U3Gate
from qiskit.test import QiskitTestCase
from qiskit.test.mock import (FakeTenerife, FakeMelbourne,
                              FakeRueschlikon, FakeTokyo, FakePoughkeepsie)


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

    @combine(level=[0, 1, 2, 3],
             dsc='Test that coupling_map can be None (level={level})',
             name='coupling_map_none_level{level}')
    def test_no_coupling_map(self, level):
        """Test that coupling_map can be None"""
        q = QuantumRegister(2, name='q')
        circuit = QuantumCircuit(q)
        circuit.cz(q[0], q[1])
        result = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=level)
        self.assertIsInstance(result, QuantumCircuit)


@ddt
class TestTranspileLevels(QiskitTestCase):
    """Test transpiler on fake backend"""

    @combine(circuit=[emptycircuit, circuit_2532],
             level=[0, 1, 2, 3],
             backend=[FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo(),
                      FakePoughkeepsie(), None],
             dsc='Transpiler {circuit.__name__} on {backend} backend at level {level}',
             name='{circuit.__name__}_{backend}_level{level}')
    def test(self, circuit, level, backend):
        """All the levels with all the backends"""
        result = transpile(circuit(), backend=backend, optimization_level=level, seed_transpiler=42)
        self.assertIsInstance(result, QuantumCircuit)


@ddt
class TestPassesInspection(QiskitTestCase):
    """Test run passes under different conditions"""

    def setUp(self):
        """Sets self.callback to set self.passes with the passes that have been executed"""
        self.passes = []

        def callback(**kwargs):
            self.passes.append(kwargs['pass_'].__class__.__name__)

        self.callback = callback

    @data(0, 1, 2, 3)
    def test_no_coupling_map(self, level):
        """Without coupling map, no layout selection nor swapper"""
        qr = QuantumRegister(3, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[2], qr[1])
        qc.cx(qr[2], qr[0])

        _ = transpile(qc, optimization_level=level, callback=self.callback)

        self.assertNotIn('SetLayout', self.passes)
        self.assertNotIn('TrivialLayout', self.passes)
        self.assertNotIn('ApplyLayout', self.passes)
        self.assertNotIn('StochasticSwap', self.passes)
        self.assertNotIn('CheckCXDirection', self.passes)

    @data(0, 1, 2, 3)
    def test_backend(self, level):
        """With backend a layout and a swapper is run
        """
        qr = QuantumRegister(5, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[2], qr[4])
        backend = FakeMelbourne()

        _ = transpile(qc, backend, optimization_level=level, callback=self.callback)

        self.assertIn('SetLayout', self.passes)
        self.assertIn('ApplyLayout', self.passes)
        self.assertIn('CheckCXDirection', self.passes)

    @data(0, 1, 2, 3)
    def test_symmetric_coupling_map(self, level):
        """Symmetric coupling map does not run CheckCXDirection
        """
        qr = QuantumRegister(2, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[1])

        coupling_map = [[0, 1], [1, 0]]

        _ = transpile(qc,
                      coupling_map=coupling_map,
                      initial_layout=[0, 1],
                      optimization_level=level,
                      callback=self.callback)

        self.assertIn('SetLayout', self.passes)
        self.assertIn('ApplyLayout', self.passes)
        self.assertNotIn('CheckCXDirection', self.passes)


@ddt
class TestInitialLayouts(QiskitTestCase):
    """Test transpiing with different layouts"""

    @data(0, 1, 2, 3)
    def test_layout_1711(self, level):
        """Test that a user-given initial layout is respected,
        in the qobj.

        See: https://github.com/Qiskit/qiskit-terra/issues/1711
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3)
        ancilla = QuantumRegister(13, 'ancilla')
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[2], qr[1])
        qc.cx(qr[2], qr[0])
        initial_layout = {0: qr[1], 2: qr[0], 15: qr[2]}
        final_layout = {0: qr[1], 1: ancilla[0], 2: qr[0], 3: ancilla[1], 4: ancilla[2],
                        5: ancilla[3], 6: ancilla[4], 7: ancilla[5], 8: ancilla[6],
                        9: ancilla[7], 10: ancilla[8], 11: ancilla[9], 12: ancilla[10],
                        13: ancilla[11], 14: ancilla[12], 15: qr[2]}

        backend = FakeRueschlikon()

        qc_b = transpile(qc, backend, initial_layout=initial_layout, optimization_level=level)
        qobj = assemble(qc_b)

        self.assertEqual(qc_b._layout._p2v, final_layout)

        compiled_ops = qobj.experiments[0].instructions
        for operation in compiled_ops:
            if operation.name == 'cx':
                self.assertIn(operation.qubits, backend.configuration().coupling_map)
                self.assertIn(operation.qubits, [[15, 0], [15, 2]])

    @data(0, 1, 2, 3)
    def test_layout_2532(self, level):
        """Test that a user-given initial layout is respected,
        in the transpiled circuit.

        See: https://github.com/Qiskit/qiskit-terra/issues/2532
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(5, 'q')
        cr = ClassicalRegister(2)
        ancilla = QuantumRegister(9, 'ancilla')
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[2], qr[4])
        initial_layout = {qr[2]: 11, qr[4]: 3,  # map to [11, 3] connection
                          qr[0]: 1, qr[1]: 5, qr[3]: 9}
        final_layout = {0: ancilla[0], 1: qr[0], 2: ancilla[1], 3: qr[4], 4: ancilla[2], 5: qr[1],
                        6: ancilla[3], 7: ancilla[4], 8: ancilla[5], 9: qr[3], 10: ancilla[6],
                        11: qr[2], 12: ancilla[7], 13: ancilla[8]}
        backend = FakeMelbourne()

        qc_b = transpile(qc, backend, initial_layout=initial_layout, optimization_level=level)

        self.assertEqual(qc_b._layout._p2v, final_layout)

        for gate, qubits, _ in qc_b:
            if gate.name == 'cx':
                for qubit in qubits:
                    self.assertIn(qubit.index, [11, 3])

    @data(0, 1, 2, 3)
    def test_layout_2503(self, level):
        """Test that a user-given initial layout is respected,
        even if cnots are not in the coupling map.

        See: https://github.com/Qiskit/qiskit-terra/issues/2503
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(2)
        ancilla = QuantumRegister(17, 'ancilla')

        qc = QuantumCircuit(qr, cr)
        qc.u3(0.1, 0.2, 0.3, qr[0])
        qc.u2(0.4, 0.5, qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[2])
        initial_layout = [6, 7, 12]

        final_layout = {0: ancilla[0], 1: ancilla[1], 2: ancilla[2], 3: ancilla[3], 4: ancilla[4],
                        5: ancilla[5], 6: qr[0], 7: qr[1], 8: ancilla[6], 9: ancilla[7],
                        10: ancilla[8], 11: ancilla[9], 12: qr[2], 13: ancilla[10], 14: ancilla[11],
                        15: ancilla[12], 16: ancilla[13], 17: ancilla[14], 18: ancilla[15],
                        19: ancilla[16]}

        backend = FakePoughkeepsie()

        qc_b = transpile(qc, backend, initial_layout=initial_layout, optimization_level=level)

        self.assertEqual(qc_b._layout._p2v, final_layout)

        gate_0, qubits_0, _ = qc_b[0]
        gate_1, qubits_1, _ = qc_b[1]

        self.assertIsInstance(gate_0, U3Gate)
        self.assertEqual(qubits_0[0].index, 6)
        self.assertIsInstance(gate_1, U2Gate)
        self.assertEqual(qubits_1[0].index, 12)


@ddt
class TestFinalLayouts(QiskitTestCase):
    """Test final layouts after preset transpilation"""

    @data(0, 1, 2, 3)
    def test_layout_tokyo_2845(self, level):
        """Test that final layout in tokyo #2845
        See: https://github.com/Qiskit/qiskit-terra/issues/2845
        """
        qr1 = QuantumRegister(3, 'qr1')
        qr2 = QuantumRegister(2, 'qr2')
        qc = QuantumCircuit(qr1, qr2)
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr1[2])
        qc.cx(qr1[2], qr2[0])
        qc.cx(qr2[0], qr2[1])

        ancilla = QuantumRegister(15, 'ancilla')
        trivial_layout = {0: qr1[0], 1: qr1[1], 2: qr1[2], 3: qr2[0], 4: qr2[1],
                          5: ancilla[0], 6: ancilla[1], 7: ancilla[2], 8: ancilla[3],
                          9: ancilla[4], 10: ancilla[5], 11: ancilla[6], 12: ancilla[7],
                          13: ancilla[8], 14: ancilla[9], 15: ancilla[10], 16: ancilla[11],
                          17: ancilla[12], 18: ancilla[13], 19: ancilla[14]}

        noise_adaptive_layout = {6: qr1[0], 11: qr1[1], 5: qr1[2],
                                 0: qr2[0], 1: qr2[1],
                                 2: ancilla[0], 3: ancilla[1], 4: ancilla[2],
                                 7: ancilla[3], 8: ancilla[4], 9: ancilla[5],
                                 10: ancilla[6], 12: ancilla[7], 13: ancilla[8],
                                 14: ancilla[9], 15: ancilla[10], 16: ancilla[11],
                                 17: ancilla[12], 18: ancilla[13], 19: ancilla[14]}

        expected_layout_level0 = trivial_layout
        expected_layout_level1 = trivial_layout
        expected_layout_level2 = noise_adaptive_layout
        expected_layout_level3 = noise_adaptive_layout
        expected_layouts = [expected_layout_level0,
                            expected_layout_level1,
                            expected_layout_level2,
                            expected_layout_level3]
        backend = FakeTokyo()
        result = transpile(qc, backend, optimization_level=level, seed_transpiler=42)
        self.assertEqual(result._layout._p2v, expected_layouts[level])
