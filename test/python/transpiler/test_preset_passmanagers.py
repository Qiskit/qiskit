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

"""Tests preset pass manager functionalities"""

from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.extensions.standard import U2Gate, U3Gate
from qiskit.test.mock import (FakeTenerife, FakeMelbourne,
                              FakeRueschlikon, FakeTokyo, FakePoughkeepsie)


class TestPresetPassManager(QiskitTestCase):
    """Test preset passmanagers work as expected."""

    def test_no_coupling_map(self):
        """Test that coupling_map can be None"""
        q = QuantumRegister(2, name='q')
        test = QuantumCircuit(q)
        test.cz(q[0], q[1])
        for level in [0, 1, 2, 3]:
            with self.subTest(level=level):
                test2 = transpile(test, basis_gates=['u1', 'u2', 'u3', 'cx'],
                                  optimization_level=level)
                self.assertIsInstance(test2, QuantumCircuit)


class TestFakeBackendTranspiling(QiskitTestCase):
    """Test transpiling on mock backends work properly"""

    def setUp(self):

        q = QuantumRegister(2)
        c = ClassicalRegister(2)

        self._circuit = QuantumCircuit(q, c)
        self._circuit.h(q[0])
        self._circuit.cx(q[0], q[1])
        self._circuit.measure(q, c)

    def test_optimization_level(self):
        """Test several backends with all optimization levels"""
        for backend in [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(),
                        FakeTokyo(), FakePoughkeepsie()]:
            for optimization_level in range(4):
                result = transpile(
                    [self._circuit],
                    backend=backend,
                    optimization_level=optimization_level
                )
                self.assertIsInstance(result, QuantumCircuit)

    # TODO: make these tests more compact with ddt
    def test_initial_layout_1(self):
        """Test that a user-given initial layout is respected,
        in the qobj.

        See: https://github.com/Qiskit/qiskit-terra/issues/1711
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[2], qr[1])
        qc.cx(qr[2], qr[0])
        initial_layout = {0: qr[1], 2: qr[0], 15: qr[2]}
        backend = FakeRueschlikon()

        for optimization_level in range(4):
            qc_b = transpile(qc, backend, initial_layout=initial_layout,
                             optimization_level=optimization_level)
            qobj = assemble(qc_b)

            compiled_ops = qobj.experiments[0].instructions
            for operation in compiled_ops:
                if operation.name == 'cx':
                    self.assertIn(operation.qubits, backend.configuration().coupling_map)
                    self.assertIn(operation.qubits, [[15, 0], [15, 2]])

    # TODO: make these tests more compact with ddt
    def test_initial_layout_2(self):
        """Test that a user-given initial layout is respected,
        in the transpiled circuit.

        See: https://github.com/Qiskit/qiskit-terra/issues/2532
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(5)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[2], qr[4])
        initial_layout = {qr[2]: 11, qr[4]: 3,  # map to [11, 3] connection
                          qr[0]: 1, qr[1]: 5, qr[3]: 9}
        backend = FakeMelbourne()

        for optimization_level in range(4):
            qc_b = transpile(qc, backend, initial_layout=initial_layout,
                             optimization_level=optimization_level)

            for gate, qubits, _ in qc_b:
                if gate.name == 'cx':
                    for qubit in qubits:
                        self.assertIn(qubit.index, [11, 3])

    # TODO: make these tests more compact with ddt
    def test_initial_layout_3(self):
        """Test that a user-given initial layout is respected,
        even if cnots are not in the coupling map.

        See: https://github.com/Qiskit/qiskit-terra/issues/2503
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(3)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.u3(0.1, 0.2, 0.3, qr[0])
        qc.u2(0.4, 0.5, qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[2])
        initial_layout = [6, 7, 12]
        backend = FakePoughkeepsie()

        for optimization_level in range(4):
            qc_b = transpile(qc, backend, initial_layout=initial_layout,
                             optimization_level=optimization_level)

            gate_0, qubits_0, _ = qc_b[0]
            gate_1, qubits_1, _ = qc_b[1]

            self.assertIsInstance(gate_0, U3Gate)
            self.assertEqual(qubits_0[0].index, 6)
            self.assertIsInstance(gate_1, U2Gate)
            self.assertEqual(qubits_1[0].index, 12)
