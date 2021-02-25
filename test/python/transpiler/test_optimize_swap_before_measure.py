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

    def test_all_measurement(self):
        """OptimizeSwapBeforeMeasure(all_measurement=True) on total measurment
            qr0:--X-----m--
                  |     |
            qr1:--X--m--|--
                     |  |
            cr :-----0--1--
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr[1], cr[0])
        circuit.measure(qr[0], cr[1])

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[1])

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(all_measurement=True), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)

    def test_optimize_nswap_nmeasure(self):
        """ Remove several swap affecting multiple measurements
                            ┌─┐                                                   ┌─┐
        q_0: ─X──X─────X────┤M├─────────────────────────────────       q_0: ──────┤M├───────────────
              │  │     │    └╥┘         ┌─┐                                    ┌─┐└╥┘
        q_1: ─X──X──X──X──X──╫─────X────┤M├─────────────────────       q_1: ───┤M├─╫────────────────
                    │     │  ║     │    └╥┘      ┌─┐                        ┌─┐└╥┘ ║
        q_2: ───────X──X──X──╫──X──X─────╫──X────┤M├────────────       q_2: ┤M├─╫──╫────────────────
                       │     ║  │        ║  │    └╥┘┌─┐                     └╥┘ ║  ║    ┌─┐
        q_3: ─X─────X──X─────╫──X──X──X──╫──X─────╫─┤M├─────────       q_3: ─╫──╫──╫────┤M├─────────
              │     │        ║     │  │  ║        ║ └╥┘┌─┐                   ║  ║  ║    └╥┘      ┌─┐
        q_4: ─X──X──X──X─────╫──X──X──X──╫──X─────╫──╫─┤M├──────  ==>  q_4: ─╫──╫──╫─────╫───────┤M├
                 │     │     ║  │        ║  │     ║  ║ └╥┘┌─┐                ║  ║  ║ ┌─┐ ║       └╥┘
        q_5: ────X──X──X──X──╫──X──X─────╫──X──X──╫──╫──╫─┤M├───       q_5: ─╫──╫──╫─┤M├─╫────────╫─
                    │     │  ║     │     ║     │  ║  ║  ║ └╥┘┌─┐             ║  ║  ║ └╥┘ ║ ┌─┐    ║
        q_6: ─X──X──X──X──X──╫──X──X─────╫─────X──╫──╫──╫──╫─┤M├       q_6: ─╫──╫──╫──╫──╫─┤M├────╫─
              │  │     │     ║  │ ┌─┐    ║        ║  ║  ║  ║ └╥┘             ║  ║  ║  ║  ║ └╥┘┌─┐ ║
        q_7: ─X──X─────X─────╫──X─┤M├────╫────────╫──╫──╫──╫──╫─       q_7: ─╫──╫──╫──╫──╫──╫─┤M├─╫─
                             ║    └╥┘    ║        ║  ║  ║  ║  ║              ║  ║  ║  ║  ║  ║ └╥┘ ║
        c: 8/════════════════╩═════╩═════╩════════╩══╩══╩══╩══╩═       c: 8/═╩══╩══╩══╩══╩══╩══╩══╩═
                             0     7     1        2  3  4  5  6              0  1  2  3  4  5  6  7
        """
        circuit = QuantumCircuit(8, 8)
        circuit.swap(3, 4)
        circuit.swap(6, 7)
        circuit.swap(0, 1)
        circuit.swap(6, 7)
        circuit.swap(4, 5)
        circuit.swap(0, 1)
        circuit.swap(5, 6)
        circuit.swap(3, 4)
        circuit.swap(1, 2)
        circuit.swap(6, 7)
        circuit.swap(4, 5)
        circuit.swap(2, 3)
        circuit.swap(0, 1)
        circuit.swap(5, 6)
        circuit.swap(1, 2)
        circuit.swap(6, 7)
        circuit.swap(4, 5)
        circuit.swap(2, 3)
        circuit.swap(3, 4)
        circuit.swap(3, 4)
        circuit.swap(5, 6)
        circuit.swap(1, 2)
        circuit.swap(4, 5)
        circuit.swap(2, 3)
        circuit.swap(5, 6)
        circuit.measure(range(8), range(8))

        expected = QuantumCircuit(8, 8)
        expected.measure(0, 2)
        expected.measure(1, 1)
        expected.measure(2, 0)
        expected.measure(3, 4)
        expected.measure(4, 7)
        expected.measure(5, 3)
        expected.measure(6, 5)
        expected.measure(7, 6)

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)

    def test_all_measurement_skip(self):
        """OptimizeSwapBeforeMeasure(all_measurement=True) on no total measurements
            qr0:--X-----
                  |
            qr1:--X--m--
                     |
            cr0:-----.--
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr[1], cr[0])

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(all_measurement=True), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(circuit, after)

    def test_all_measurement_mixed(self):
        """OptimizeSwapBeforeMeasure(all_measurement=True) on mixed measurement
            qr0:--X-----------
                  |
            qr1:--X--X-----m--
                     |     |
            qr2:-----X--m--|--
                        |  |
            cr :--------0--1--
        """
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.measure(qr[2], cr[0])
        circuit.measure(qr[1], cr[1])

        expected = QuantumCircuit(qr, cr)
        expected.swap(qr[0], qr[1])
        expected.measure(qr[1], cr[0])
        expected.measure(qr[2], cr[1])

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(all_measurement=True), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)


class TestOptimizeSwapBeforeMeasureMidMeasure(QiskitTestCase):
    """ Test swap-followed-by-measure optimizations, with mid-circuit measurement."""

    def test_mid_circuit(self):
        """Test mid-circuit measurement"""
        qr1 = QuantumRegister(1, 'qr1')
        qr2 = QuantumRegister(2, 'qr2')
        cr = ClassicalRegister(3, 'cr')
        circuit = QuantumCircuit(qr1, qr2, cr)
        circuit.h(qr1[0])
        circuit.h(qr2[1])
        circuit.swap(qr1[0], qr2[0])
        circuit.measure(qr1[0], cr[0])
        circuit.measure(qr2[0], cr[1])
        circuit.cx(qr1[0], qr2[1])
        circuit.swap(qr1[0], qr2[0])
        circuit.measure(qr1[0], cr[0])
        circuit.measure(qr2[0], cr[1])

        expected = QuantumCircuit(qr1, qr2, cr)
        expected.h(qr1[0])
        expected.h(qr2[1])
        expected.swap(qr1[0], qr2[0])
        expected.measure(qr1[0], cr[0])
        expected.measure(qr2[0], cr[1])
        expected.cx(qr1[0], qr2[1])
        expected.measure(qr2[0], cr[0])
        expected.measure(qr1[0], cr[1])

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)

    def test_all_measurement_remove_one(self):
        """OptimizeSwapBeforeMeasure(all_measurement=True) with mid-circ measurements, remove one
            qr0:--X-----------H-----------
                  |
            qr1:--X--X-----m--X--X--m-----
                     |     |     |  |
            qr2:-----X--m--|--H--X--|--m--
                        |  |        |  |
            cr :--------0--1--------1--0--

        Only the last swap should be removed
        """
        circuit = QuantumCircuit(3, 2)
        circuit.swap(0, 1)
        circuit.swap(1, 2)
        circuit.measure(2, 0)
        circuit.measure(1, 1)
        circuit.h(0)
        circuit.x(1)
        circuit.h(2)
        circuit.swap(1, 2)
        circuit.measure(1, 1)
        circuit.measure(2, 0)

        expected = QuantumCircuit(3, 2)
        expected.swap(0, 1)
        expected.swap(1, 2)
        expected.measure(2, 0)
        expected.measure(1, 1)
        expected.h(0)
        expected.x(1)
        expected.h(2)
        expected.measure(2, 1)
        expected.measure(1, 0)

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(all_measurement=True), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)

    def test_all_measurement_remove_none(self):
        """OptimizeSwapBeforeMeasure(all_measurement=True) with mid-circ measurements, remove none
            qr0:--X-----------H--------
                  |
            qr1:--X--X-----m--X--X--m--
                     |     |     |  |
            qr2:-----X--m--|--H--X--|--
                        |  |        |
            cr :--------0--1--------1--

        Last swap should stay, because is partially measured
        """
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.measure(qr[2], cr[0])
        circuit.measure(qr[1], cr[1])
        circuit.h(qr[0])
        circuit.x(qr[1])
        circuit.h(qr[2])
        circuit.swap(qr[1], qr[2])
        circuit.measure(qr[1], cr[1])

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(all_measurement=True), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(circuit, after)

    def test_move_swap(self):
        """OptimizeSwapBeforeMeasure(move_swap=True) with mid-circ measurements
            qr0:--X-----------H-----------
                  |
            qr1:--X--X-----m--X--X--m-----
                     |     |     |  |
            qr2:-----X--m--|--H--X--|--m--
                        |  |        |  |
            cr :--------0--1--------1--0--

        Only the last swap should be removed
        """
        circuit = QuantumCircuit(3, 2)
        circuit.swap(0, 1)
        circuit.swap(1, 2)
        circuit.measure(2, 0)
        circuit.measure(1, 1)
        circuit.h(0)
        circuit.x(1)
        circuit.h(2)
        circuit.swap(1, 2)
        circuit.measure(1, 1)
        circuit.measure(2, 0)

        expected = QuantumCircuit(3, 2)
        expected.swap(0, 1)
        expected.measure(1, 0)
        expected.measure(2, 1)
        expected.swap(1, 2)
        expected.h(0)
        expected.x(1)
        expected.h(2)
        expected.measure(2, 1)
        expected.measure(1, 0)
        expected.swap(1, 2)

        pass_manager = PassManager()
        pass_manager.append(
            [OptimizeSwapBeforeMeasure(move_swap=True), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)


if __name__ == '__main__':
    unittest.main()
