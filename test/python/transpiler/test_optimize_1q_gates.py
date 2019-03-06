# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the optimize-1q-gate pass"""

import unittest
import sympy
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeRueschlikon


class TestOptimize1qGates(QiskitTestCase):
    """ Test for 1q gate optimizations. """

    def test_optimize_id(self):
        """ qr0:--[id]-- == qr0:------ """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.iden(qr)
        circuit.iden(qr)
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)

        pass_ = Optimize1qGates()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


class TestOptimize1qGatesTranspiler(QiskitTestCase):
    """ Test for 1q gate optimizations as part of the transpiler, with a PassManager """

    def test_optimize_h_gates(self):
        """ Transpile: qr:--[H]-[H]-[H]-- == qr:--[u2]-- """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])

        expected = QuantumCircuit(qr)
        expected.u2(0, np.pi, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates())
        result = transpile(circuit, FakeRueschlikon(), pass_manager=passmanager)
        self.assertEqual(expected, result)


class TestMovedFromMapper(QiskitTestCase):
    """ This tests are moved from test_mapper.py"""

    def test_optimize_1q_gates_collapse_identity(self):
        """test optimize_1q_gates removes u1(2*pi) rotations.

        See: https://github.com/Qiskit/qiskit-terra/issues/159
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[1], qr[0])
        qc.u1(2 * sympy.pi, qr[0])
        qc.cx(qr[1], qr[0])
        qc.u1(sympy.pi / 2, qr[0])  # these three should combine
        qc.u1(sympy.pi, qr[0])      # to identity then
        qc.u1(sympy.pi / 2, qr[0])  # optimized away.
        qc.cx(qr[1], qr[0])
        qc.u1(np.pi, qr[1])
        qc.u1(np.pi, qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        dag = circuit_to_dag(qc)
        simplified_dag = Optimize1qGates().run(dag)
        num_u1_gates_remaining = len(simplified_dag.get_named_nodes('u1'))
        self.assertEqual(num_u1_gates_remaining, 0)

    def test_optimize_1q_gates_symbolic(self):
        """optimizes single qubit gate sequences with symbolic params.

        See: https://github.com/Qiskit/qiskit-terra/issues/172
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        # unary
        circ.u1(-sympy.pi, qr[0])
        circ.u1(-sympy.pi / 2, qr[0])
        # binary
        circ.u1(0.2 * sympy.pi + 0.3 * sympy.pi, qr[1])
        circ.u1(1.3 - 0.3, qr[1])
        circ.u1(0.1 * sympy.pi / 2, qr[1])
        # extern
        circ.u1(sympy.sin(0.2 + 0.3 - sympy.pi), qr[2])
        # power
        circ.u1(sympy.pi, qr[3])
        circ.u1(0.3 + (-sympy.pi) ** 2, qr[3])

        dag = circuit_to_dag(circ)
        simplified_dag = Optimize1qGates().run(dag)

        params = set()
        for n in simplified_dag.multi_graph.nodes:
            node = simplified_dag.multi_graph.node[n]
            if node['name'] == 'u1':
                params.add(node['op'].params[0])

        expected_params = {sympy.Number(-3 * np.pi / 2),
                           sympy.Number(1.0 + 0.55 * np.pi),
                           sympy.Number(-0.479425538604203),
                           sympy.Number(0.3 + np.pi + np.pi ** 2)}

        self.assertEqual(params, expected_params)


if __name__ == '__main__':
    unittest.main()
