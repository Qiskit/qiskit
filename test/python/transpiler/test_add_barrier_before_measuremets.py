# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the Add Barrier Before Measurments pass"""

import unittest
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from ..common import QiskitTestCase


class TestAddBarrierBeforeMeasuremets(QiskitTestCase):
    """ Tests the BarrierBeforeFinalMeasurements pass."""

    def test_single_measure(self):
        """ A single measurement at the end
                             |
         q0:--[m]--     q0:--|-[m]---
               |    ->       |  |
         c1:---.---     c1:-----.---
        """
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)

        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_ignore_single_measure(self):
        """ Ignore single measurement because is not at the end
         q0:--[m]-[H]-      q0:--[m]-[H]-
               |        ->        |
         c1:---.------      c1:---.------
        """
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.h(qr[0])

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr, cr)
        expected.h(qr[0])

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))


if __name__ == '__main__':
    unittest.main()
