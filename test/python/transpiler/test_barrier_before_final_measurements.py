# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the BarrierBeforeFinalMeasurements pass"""

import unittest
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.test import QiskitTestCase


class TestBarrierBeforeFinalMeasurements(QiskitTestCase):
    """Tests the BarrierBeforeFinalMeasurements pass."""

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
        """Ignore single measurement because it is not at the end
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
        """Two measurements, but only one is at the end
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
        """Two measurements in different qregs to different cregs
                                           |
         q0:--[H]--[m]------     q0:--[H]--|--[m]------
                    |                      |   |
         q1:--------|--[m]--  -> q1:-------|---|--[m]--
                    |   |                  |   |   |
         c0:--------.---|---      c0:----------.---|---
                        |                          |
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
        """Two measurements in different qregs to the same creg
                                           |
         q0:--[H]--[m]------     q0:--[H]--|--[m]------
                    |                      |   |
         q1:--------|--[m]--  -> q1:-------|---|--[m]--
                    |   |                  |   |   |
         c0:--------.---|---     c0:-----------.---|---
            ------------.---        ---------------.---
        """
        qr0 = QuantumRegister(1, 'q0')
        qr1 = QuantumRegister(1, 'q1')
        cr0 = ClassicalRegister(2, 'c0')

        circuit = QuantumCircuit(qr0, qr1, cr0)
        circuit.h(qr0)
        circuit.measure(qr0, cr0[0])
        circuit.measure(qr1, cr0[1])

        expected = QuantumCircuit(qr0, qr1, cr0)
        expected.h(qr0)
        expected.barrier(qr0, qr1)
        expected.measure(qr0, cr0[0])
        expected.measure(qr1, cr0[1])

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_preserve_measure_for_conditional(self):
        """Test barrier is inserted after any measurements used for conditionals

         q0:--[H]--[m]------------     q0:--[H]--[m]-------|-------
                    |                             |        |
         q1:--------|--[ z]--[m]--  -> q1:--------|--[ z]--|--[m]--
                    |    |    |                   |    |       |
         c0:--------.--[=1]---|---     c0:--------.--[=1]------|---
                              |                                |
         c1:------------------.---     c1:---------------------.---
        """
        qr0 = QuantumRegister(1, 'q0')
        qr1 = QuantumRegister(1, 'q1')
        cr0 = ClassicalRegister(1, 'c0')
        cr1 = ClassicalRegister(1, 'c1')
        circuit = QuantumCircuit(qr0, qr1, cr0, cr1)

        circuit.h(qr0)
        circuit.measure(qr0, cr0)
        circuit.z(qr1).c_if(cr0, 1)
        circuit.measure(qr1, cr1)

        expected = QuantumCircuit(qr0, qr1, cr0, cr1)
        expected.h(qr0)
        expected.measure(qr0, cr0)
        expected.z(qr1).c_if(cr0, 1)
        expected.barrier(qr0, qr1)
        expected.measure(qr1, cr1)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))


class TestBarrierBeforeMeasurementsWhenABarrierIsAlreadyThere(QiskitTestCase):
    """Tests the BarrierBeforeFinalMeasurements pass when there is a barrier already"""

    def test_handle_redundancy(self):
        """The pass is idempotent
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

    def test_preserve_barriers_for_measurement_ordering(self):
        """If the circuit has a barrier to enforce a measurement order,
        preserve it in the output.

         q:---[m]--|-------     q:---|--[m]--|-------
           ----|---|--[m]--  ->   ---|---|---|--[m]--
               |       |                 |       |
         c:----.-------|---     c:-------.-------|---
           ------------.---       ---------------.---
        """
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])

        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr[0], cr[0])
        expected.barrier(qr)
        expected.measure(qr[1], cr[1])

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_measures_followed_by_barriers_should_be_final(self):
        """If a measurement is followed only by a barrier,
        insert the barrier before it.

         q:---[H]--|--[m]--|-------     q:---[H]--|--[m]-|-------
           ---[H]--|---|---|--[m]--  ->   ---[H]--|---|--|--[m]--
                       |       |                      |      |
         c:------------.-------|---     c:------------.------|---
           --------------------.---       -------------------.---
        """
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')

        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])

        expected = QuantumCircuit(qr, cr)
        expected.h(qr)
        expected.barrier(qr)
        expected.measure(qr[0], cr[0])
        expected.barrier(qr)
        expected.measure(qr[1], cr[1])

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_should_merge_with_smaller_duplicate_barrier(self):
        """If an equivalent barrier exists covering a subset of the qubits
        covered by the new barrier, it should be replaced.

         q:---|--[m]-------------     q:---|--[m]-------------
           ---|---|---[m]--------  ->   ---|---|---[m]--------
           -------|----|---[m]---       ---|---|----|---[m]---
                  |    |    |                  |    |    |
         c:-------.----|----|----     c:-------.----|----|----
           ------------.----|----       ------------.----|----
           -----------------.----       -----------------.----
        """
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')

        circuit = QuantumCircuit(qr, cr)
        circuit.barrier(qr[0], qr[1])
        circuit.measure(qr, cr)

        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_should_merge_with_larger_duplicate_barrier(self):
        """If a barrier exists and is stronger than the barrier to be inserted,
        preserve the existing barrier and do not insert a new barrier.

         q:---|--[m]--|-------     q:---|--[m]-|-------
           ---|---|---|--[m]--  ->   ---|---|--|--[m]--
           ---|---|---|---|---       ---|---|--|---|---
                  |       |                 |      |
         c:-------.-------|---     c:-------.------|---
           ---------------.---       --------------.---
           -------------------       ------------------
        """
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')

        circuit = QuantumCircuit(qr, cr)
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])

        expected = circuit

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barrier_doesnt_reorder_gates(self):
        """ A barrier should not allow the reordering of gates, as pointed out in #2102

         q:--[u1(0)]-----------[m]---------      q:--[u1(0)]------------|--[m]---------
           --[u1(1)]------------|-[m]------  ->    --[u1(1)]------------|---|-[m]------
           --[u1(2)]-|----------|--|-[m]----       --[u1(2)]-|----------|---|--|-[m]----
           ----------|-[u1(03)]-|--|--|-[m]-       ----------|-[u1(03)]-|---|--|--|-[m]-
                                |  |  |  |                                  |  |  |  |
         c:---------------------.--|--|--|-     c:--------------------------.--|--|--|-
           ------------------------.--|--|-       -----------------------------.--|--|-
           ---------------------------.--|-       --------------------------------.--|-
           ------------------------------.-       -----------------------------------.-

        """

        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circuit = QuantumCircuit(qr, cr)

        circuit.u1(0, qr[0])
        circuit.u1(1, qr[1])
        circuit.u1(2, qr[2])
        circuit.barrier(qr[2], qr[3])
        circuit.u1(3, qr[3])

        test_circuit = circuit.copy()
        test_circuit.measure(qr, cr)

        # expected circuit is the same, just with a barrier before the measurements
        expected = circuit.copy()
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(test_circuit))

        self.assertEqual(result, circuit_to_dag(expected))


if __name__ == '__main__':
    unittest.main()
