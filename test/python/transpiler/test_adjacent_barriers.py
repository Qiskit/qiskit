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

"""Test the MergeAdjacentBarriers pass"""

import unittest
from qiskit.transpiler.passes import MergeAdjacentBarriers
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestMergeAdjacentBarriers(QiskitTestCase):
    """Test the MergeAdjacentBarriers pass"""

    def test_two_identical_barriers(self):
        """ Merges two barriers that are identical into one
                     ░  ░                  ░
            q_0: |0>─░──░─   ->   q_0: |0>─░─
                     ░  ░                  ░
        """
        qr = QuantumRegister(1, 'q')

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_numerous_identical_barriers(self):
        """ Merges 5 identical barriers in a row into one
                 ░  ░  ░  ░  ░  ░                     ░
        q_0: |0>─░──░──░──░──░──░─    ->     q_0: |0>─░─
                 ░  ░  ░  ░  ░  ░                     ░
        """
        qr = QuantumRegister(1, 'q')

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_of_different_sizes(self):
        """ Test two barriers of different sizes are merged into one
                     ░  ░                     ░
            q_0: |0>─░──░─           q_0: |0>─░─
                     ░  ░     ->              ░
            q_1: |0>────░─           q_1: |0>─░─
                        ░                     ░
        """
        qr = QuantumRegister(2, 'q')

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_not_overlapping_barriers(self):
        """ Test two barriers with no overlap are not merged
            (NB in these pictures they look like 1 barrier but they are
                actually 2 distinct barriers, this is just how the text
                drawer draws them)
                     ░                     ░
            q_0: |0>─░─           q_0: |0>─░─
                     ░     ->              ░
            q_1: |0>─░─           q_1: |0>─░─
                     ░                     ░
        """
        qr = QuantumRegister(2, 'q')

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.barrier(qr[1])

        expected = QuantumCircuit(qr)
        expected.barrier(qr[0])
        expected.barrier(qr[1])

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_obstacle_before(self):
        """ Test with an obstacle before the larger barrier
                  ░   ░                          ░
        q_0: |0>──░───░─           q_0: |0>──────░─
                ┌───┐ ░     ->             ┌───┐ ░
        q_1: |0>┤ H ├─░─           q_1: |0>┤ H ├─░─
                └───┘ ░                    └───┘ ░
        """
        qr = QuantumRegister(2, 'q')

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.h(qr[1])
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.h(qr[1])
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_obstacle_after(self):
        """ Test with an obstacle after the larger barrier
                 ░   ░                      ░
        q_0: |0>─░───░──           q_0: |0>─░──────
                 ░ ┌───┐    ->              ░ ┌───┐
        q_1: |0>─░─┤ H ├           q_1: |0>─░─┤ H ├
                 ░ └───┘                    ░ └───┘
        """
        qr = QuantumRegister(2, 'q')

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.barrier(qr[0])
        circuit.h(qr[1])

        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        expected.h(qr[1])

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle(self):
        """ Test that barriers don't merge if there is an obstacle that
            is blocking
                     ░ ┌───┐ ░                     ░ ┌───┐ ░
            q_0: |0>─░─┤ H ├─░─    ->     q_0: |0>─░─┤ H ├─░─
                     ░ └───┘ ░                     ░ └───┘ ░
        """
        qr = QuantumRegister(1, 'q')

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.h(qr)
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        expected.h(qr)
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle_long(self):
        """ Test that barriers don't merge if there is an obstacle that
            is blocking
                 ░ ┌───┐ ░                     ░ ┌───┐ ░
        q_0: |0>─░─┤ H ├─░─           q_0: |0>─░─┤ H ├─░─
                 ░ └───┘ ░     ->              ░ └───┘ ░
        q_1: |0>─────────░─           q_1: |0>─────────░─
                         ░                             ░
        """
        qr = QuantumRegister(2, 'q')

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.h(qr[0])
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr[0])
        expected.h(qr[0])
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))


if __name__ == '__main__':
    unittest.main()
