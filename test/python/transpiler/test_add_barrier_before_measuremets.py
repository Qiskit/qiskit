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
from qiskit.test import QiskitTestCase


class TestAddBarrierBeforeMeasuremets(QiskitTestCase):
    """ Tests the BarrierBeforeFinalMeasurements pass."""

    def test_single_measure(self):
        """ A single measurement at the end
                           |
         q:--[m]--     q:--|-[m]---
              |    ->      |  |
         c:---.---     c:-----.---
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
         q:--[m]-[H]-      q:--[m]-[H]-
              |        ->       |
         c:---.------      c:---.------
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

    def test_single_measure_mix(self):
        """ Two measurements, but only one is at the end
                                                 |
         q0:--[m]--[H]--[m]--     q0:--[m]--[H]--|-[m]---
               |         |    ->        |        |  |
          c:---.---------.---      c:---.-----------.---
        """
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.h(qr)
        circuit.measure(qr, cr)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr, cr)
        expected.h(qr)
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_two_qregs(self):
        """ Two measurements in different qregs to different cregs
                                           |
         q0:--[H]--[m]------     q0:--[H]--|--[m]------
                    |                      |   |
         q1:--------|--[m]--  -> q1:-------|---|--[m]--
                    |   |                  |   |   |
         c0:--------.---|---      c0:----------.---|---
         c1:------------.---      c0:--------------.---
        """
        qr0 = QuantumRegister(1, 'q0')
        qr1 = QuantumRegister(1, 'q1')
        cr0 = ClassicalRegister(1, 'c0')
        cr1 = ClassicalRegister(1, 'c1')

        circuit = QuantumCircuit(qr0, qr1, cr0, cr1)
        circuit.h(qr0)
        circuit.measure(qr0, cr0)
        circuit.measure(qr1, cr1)

        expected = QuantumCircuit(qr0, qr1, cr0, cr1)
        expected.h(qr0)
        expected.barrier(qr0, qr1)
        expected.measure(qr0, cr0)
        expected.measure(qr1, cr1)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_two_qregs_to_a_single_creg(self):
        """ Two measurements in different qregs to the same creg
                                           |
         q0:--[H]--[m]------     q0:--[H]--|--[m]------
                    |                      |   |
         q1:--------|--[m]--  -> q1:-------|---|--[m]--
                    |   |                  |   |   |
         c0:--------.---.---      c0:----------.---.---
        """
        qr0 = QuantumRegister(1, 'q0')
        qr1 = QuantumRegister(1, 'q1')
        cr0 = ClassicalRegister(1, 'c0')

        circuit = QuantumCircuit(qr0, qr1, cr0)
        circuit.h(qr0)
        circuit.measure(qr0, cr0)
        circuit.measure(qr1, cr0)

        expected = QuantumCircuit(qr0, qr1, cr0)
        expected.h(qr0)
        expected.barrier(qr0, qr1)
        expected.measure(qr0, cr0)
        expected.measure(qr1, cr0)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))


class TestAddBarrierBeforeMeasuremetsWhenABarrierIsAlreadyThere(QiskitTestCase):
    """ Tests the BarrierBeforeFinalMeasurements pass when there is a barrier already"""

    def test_handle_redundancy(self):
        """ The pass is idempotent
             |                |
         q:--|-[m]--      q:--|-[m]---
             |  |     ->      |  |
         c:-----.---      c:-----.---
        """
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')

        circuit = QuantumCircuit(qr, cr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)

        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_remove_barrier_in_different_qregs(self):
        """ Two measurements in different qregs to the same creg
                                     |
         q0:--|--[m]------     q0:---|--[m]------
                  |                  |   |
         q1:--|---|--[m]--  -> q1:---|---|--[m]--
                  |   |              |   |   |
         c0:------.---.---     c0:-------.---.---
        """
        qr0 = QuantumRegister(1, 'q0')
        qr1 = QuantumRegister(1, 'q1')
        cr0 = ClassicalRegister(1, 'c0')

        circuit = QuantumCircuit(qr0, qr1, cr0)
        circuit.barrier(qr0)
        circuit.barrier(qr1)
        circuit.measure(qr0, cr0)
        circuit.measure(qr1, cr0)

        expected = QuantumCircuit(qr0, qr1, cr0)
        expected.barrier(qr0, qr1)
        expected.measure(qr0, cr0)
        expected.measure(qr1, cr0)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))


if __name__ == '__main__':
    unittest.main()
