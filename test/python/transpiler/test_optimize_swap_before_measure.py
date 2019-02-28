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
from qiskit.transpiler.passes import OptimizeSwapBeforeMeasure
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeRueschlikon


class TestOptimizeSwapBeforeMeasure(QiskitTestCase):
    """ Test swap-followed-by-measure optimizations. """

    def test_optimize_1swap_1measure(self):
        """ Remove a single swap
            qr0:--X--m--       qr0:----
                  |  |
            qr1:--X--|--  ==>  qr1:--m-
                     |               |
            cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[1], cr[0])
        print()
        print(circuit)
        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        # self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_undone_swap(self):
        """ Remove redundant swap
            qr0:--X--X--m--       qr0:--m-
                  |  |  |               |
            qr1:--X--X--|--  ==>  qr1:--|-
                        |               |
            cr0:--------.--       cr0:--.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        print()
        print(circuit)
        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        # self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_overlap_swap(self):
        """ Remove two swaps that overlap
            qr0:--X--------       qr0:--m--
                  |                     |
            qr1:--X--X-----       qr1:--|--
                     |       ==>        |
            qr2:-----X--m--       qr2:--|--
                        |               |
            cr0:--------.--       cr0:--.--
        """
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.measure(qr[2], cr[0])
        dag = circuit_to_dag(circuit)

        print()
        print(circuit)
        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        # self.assertEqual(circuit_to_dag(expected), after)

if __name__ == '__main__':
    unittest.main()
