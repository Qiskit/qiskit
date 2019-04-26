# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test RemoveRZandZbeforeMeasure pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import RemoveRZandZbeforeMeasure, DAGFixedPoint
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestRemoveRZandZbeforeMeasure(QiskitTestCase):
    """ Test remove_rz_and_z_before_measure optimizations. """

    def test_optimize_1rz_1measure(self):
        """ Remove a single RZ
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

        pass_ = RemoveRZandZbeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1z_1measure(self):
        """ Remove a single Z
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

        pass_ = RemoveRZandZbeforeMeasure()
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

        pass_ = RemoveRZandZbeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

class TestRemoveRZandZbeforeMeasureFixedPoint(QiskitTestCase):
    """ Test remove_rz_and_z_before_measure optimizations in a transpiler, using fixed point. """

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
            [RemoveRZandZbeforeMeasure(), DAGFixedPoint()],
            do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = transpile(circuit, pass_manager=pass_manager)

        self.assertEqual(expected, after)


if __name__ == '__main__':
    unittest.main()
