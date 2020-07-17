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

"""Test OptimizeSwapBeforeMeasure pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import OptimizeSwapBeforeMeasure, DAGFixedPoint
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


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

        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1swap_2measure(self):
        """ Remove a single swap affecting two measurements
            qr0:--X--m--         qr0:--m----
                  |  |                 |
            qr1:--X--|--m   ==>  qr1:--|--m-
                     |  |              |  |
            cr0:-----.--|--      cr0:--|--.-
            cr1:--------.--      cr1:--.----
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[1], cr[0])
        expected.measure(qr[0], cr[1])

        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_cannot_optimize(self):
        """ Cannot optimize when swap is not at the end in all of the successors
            qr0:--X-----m--
                  |     |
            qr1:--X-[H]-|--
                        |
            cr0:--------.--
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.h(qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(circuit), after)


class TestOptimizeSwapBeforeMeasureFixedPoint(QiskitTestCase):
    """ Test swap-followed-by-measure optimizations in a transpiler, using fixed point. """

    def test_optimize_undone_swap(self):
        """ Remove redundant swap
            qr0:--X--X--m--       qr0:--m---
                  |  |  |               |
            qr1:--X--X--|--  ==>  qr1:--|--
                        |               |
            cr0:--------.--       cr0:--.--
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)

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

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)


if __name__ == '__main__':
    unittest.main()
