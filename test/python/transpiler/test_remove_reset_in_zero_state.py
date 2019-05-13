# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test RemoveResetInZeroState pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.compiler import transpile
from qiskit.transpiler.passes import RemoveResetInZeroState, DAGFixedPoint
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestRemoveResetInZeroState(QiskitTestCase):
    """ Test swap-followed-by-measure optimizations. """

    def test_optimize_single_reset(self):
        """ Remove a single reset
            qr0:--|0>--   ==>    qr0:----
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.reset(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)

        pass_ = RemoveResetInZeroState()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_dont_optimize_non_zero_state(self):
        """ Do not remove reset if not in a zero state
            qr0:--[H]--|0>--   ==>    qr0:--[H]--|0>--
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.reset(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr)
        expected.reset(qr)

        pass_ = RemoveResetInZeroState()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_single_reset_in_diff_qubits(self):
        """ Remove a single reset in different qubits
            qr0:--|0>--          qr0:----
                          ==>
            qr1:--|0>--          qr1:----
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.reset(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)

        pass_ = RemoveResetInZeroState()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


class TestRemoveResetInZeroStateFixedPoint(QiskitTestCase):
    """ Test RemoveResetInZeroState in a transpiler, using fixed point. """

    def test_two_resets(self):
        """ Remove two initial resets
            qr0:--|0>-|0>--   ==>    qr0:----
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.reset(qr[0])
        circuit.reset(qr[0])

        expected = QuantumCircuit(qr)

        pass_manager = PassManager()
        pass_manager.append(
            [RemoveResetInZeroState(), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = transpile(circuit, pass_manager=pass_manager)

        self.assertEqual(expected, after)


if __name__ == '__main__':
    unittest.main()
