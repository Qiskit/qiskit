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

"""Test RemoveDiagonalGatesBeforeMeasure pass"""

import unittest
from copy import deepcopy

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure, DAGFixedPoint
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TesRemoveDiagonalGatesBeforeMeasure(QiskitTestCase):
    """ Test remove_diagonal_gates_before_measure optimizations. """

    def test_optimize_1rz_1measure(self):
        """ Remove a single RZGate
            qr0:-RZ--m--       qr0:--m-
                     |               |
            qr1:-----|--  ==>  qr1:--|-
                     |               |
            cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1z_1measure(self):
        """ Remove a single ZGate
            qr0:--Z--m--       qr0:--m-
                     |               |
            qr1:-----|--  ==>  qr1:--|-
                     |               |
            cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.z(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1t_1measure(self):
        """ Remove a single TGate, SGate, TdgGate, SdgGate, U1Gate
            qr0:--T--m--       qr0:--m-
                     |               |
            qr1:-----|--  ==>  qr1:--|-
                     |               |
            cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.t(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1s_1measure(self):
        """ Remove a single SGate
            qr0:--S--m--       qr0:--m-
                     |               |
            qr1:-----|--  ==>  qr1:--|-
                     |               |
            cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.s(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1tdg_1measure(self):
        """ Remove a single TdgGate
            qr0:-Tdg-m--       qr0:--m-
                     |               |
            qr1:-----|--  ==>  qr1:--|-
                     |               |
            cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.tdg(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1sdg_1measure(self):
        """ Remove a single SdgGate
            qr0:-Sdg--m--       qr0:--m-
                      |               |
            qr1:------|--  ==>  qr1:--|-
                      |               |
            cr0:------.--       cr0:--.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.sdg(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1u1_1measure(self):
        """ Remove a single U1Gate
            qr0:--U1-m--       qr0:--m-
                     |               |
            qr1:-----|--  ==>  qr1:--|-
                     |               |
            cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.u1(0.1, qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1rz_1z_1measure(self):
        """ Remove a single RZ and leave the other Z
            qr0:-RZ--m--       qr0:----m-
                     |                 |
            qr1:--Z--|--  ==>  qr1:--Z-|-
                     |                 |
            cr0:-----.--       cr0:----.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.z(qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.z(qr[1])
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


class TesRemoveDiagonalControlGatesBeforeMeasure(QiskitTestCase):
    """ Test remove diagonal control gates before measure. """

    def test_optimize_1cz_2measure(self):
        """ Remove a single CZGate
            qr0:--Z--m---       qr0:--m---
                  |  |                |
            qr1:--.--|-m-  ==>  qr1:--|-m-
                     | |              | |
            cr0:-----.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1crz_2measure(self):
        """ Remove a single CRZGate
            qr0:-RZ--m---       qr0:--m---
                  |  |                |
            qr1:--.--|-m-  ==>  qr1:--|-m-
                     | |              | |
            cr0:-----.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.crz(0.1, qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1cu1_2measure(self):
        """ Remove a single CU1Gate
            qr0:-CU1-m---       qr0:--m---
                  |  |                |
            qr1:--.--|-m-  ==>  qr1:--|-m-
                     | |              | |
            cr0:-----.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.cu1(0.1, qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1rzz_2measure(self):
        """ Remove a single RZZGate
            qr0:--.----m---       qr0:--m---
                  |zz  |                |
            qr1:--.----|-m-  ==>  qr1:--|-m-
                       | |              | |
            cr0:-------.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rzz(0.1, qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


class TestRemoveDiagonalGatesBeforeMeasureOveroptimizations(QiskitTestCase):
    """ Test situations where remove_diagonal_gates_before_measure should not optimize """

    def test_optimize_1cz_1measure(self):
        """ Do not remove a CZGate because measure happens on only one of the wires
        Compare with test_optimize_1cz_2measure.

            qr0:--Z--m---
                  |  |
            qr1:--.--|---
                     |
            cr0:-----.---
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = deepcopy(dag)

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(expected, after)

    def test_do_not_optimize_with_conditional(self):
        """ Diagonal gates with conditionals on a measurement target.
        See https://github.com/Qiskit/qiskit-terra/pull/2208#issuecomment-487238819
                                 ░ ┌───┐┌─┐
            qr_0: |0>────────────░─┤ H ├┤M├
                     ┌─────────┐ ░ └───┘└╥┘
            qr_1: |0>┤ Rz(0.1) ├─░───────╫─
                     └─┬──┴──┬─┘ ░       ║
             cr_0: 0 ══╡ = 1 ╞═══════════╩═
                       └─────┘
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[1]).c_if(cr, 1)
        circuit.barrier()
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = deepcopy(dag)

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(expected, after)


class TestRemoveDiagonalGatesBeforeMeasureFixedPoint(QiskitTestCase):
    """ Test remove_diagonal_gates_before_measure optimizations in
        a transpiler, using fixed point. """

    def test_optimize_rz_z(self):
        """ Remove two swaps that overlap
            qr0:--RZ-Z--m--       qr0:--m--
                        |               |
            cr0:--------.--       cr0:--.--
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.z(qr[0])
        circuit.measure(qr[0], cr[0])

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_manager = PassManager()
        pass_manager.append(
            [RemoveDiagonalGatesBeforeMeasure(), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)


if __name__ == '__main__':
    unittest.main()
