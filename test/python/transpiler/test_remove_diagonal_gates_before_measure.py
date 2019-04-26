# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test RemoveDiagonalGatesBeforeMeasure pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import PassManager, transpile
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
        after = transpile(circuit, pass_manager=pass_manager)

        self.assertEqual(expected, after)


if __name__ == '__main__':
    unittest.main()
