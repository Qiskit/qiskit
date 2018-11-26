# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the Swap Mapper pass"""

import unittest

from qiskit.transpiler.passes import SwapMapper
from qiskit.mapper import Coupling
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister, QuantumCircuit
from ..common import QiskitTestCase


class TestSwapMapper(QiskitTestCase):
    """ Tests the SwapMapper pass."""

    def assertEndswith(self, long_str, endlist):
        """
        Asserts that the list of lines defined in `endlist` is at the end of `long_str`.
        Args:
            long_str (str): A long string.
            endlist (list): A list of lines.
        """
        end_string = '\n'.join(['}'] + endlist) + '\n'
        self.assertEqual(long_str[-len(end_string):], end_string)

    def test_trivial_case(self):
        """No need to have any swap, the CX are distance 1 to each other
         q0:--(+)-[U]-(+)-
               |       |
         q1:---.-------|--
                       |
         q2:-----------.--

         Coupling map: [1]--[0]--[2]
        """
        coupling = Coupling({0: [1, 2]})

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])

        dag = DAGCircuit.fromQuantumCircuit(circuit)
        before = dag.qasm()
        pass_ = SwapMapper(coupling)
        after_dag = pass_.run(dag)

        self.assertEqual(before, after_dag.qasm())

    def test_trivial_in_same_layer(self):
        """ No need to have any swap, two CXs distance 1 to each other, in the same layer
         q0:--(+)--
               |
         q1:---.---

         q2:--(+)--
               |
         q3:---.---

         Coupling map: [0]--[1]--[2]--[3]
        """
        coupling = Coupling({0: [1], 1: [2], 2: [3]})

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])

        dag = DAGCircuit.fromQuantumCircuit(circuit)
        before = dag.qasm()
        pass_ = SwapMapper(coupling)
        after_dag = pass_.run(dag)

        self.assertEqual(before, after_dag.qasm())

    def test_a_single_swap(self):
        """ Adding a swap
         q0:-------

         q1:--(+)--
               |
         q2:---.---

         Coupling map: [1]--[0]--[2]

         q0:--X--.---
              |  |
         q1:--|-(+)--
              |
         q2:--X------

        """
        coupling = Coupling({0: [1, 2]})

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        expected = ['swap q[0],q[2];',
                    'cx q[1],q[0];']

        pass_ = SwapMapper(coupling)
        after_dag = pass_.run(dag)

        self.assertEndswith(after_dag.qasm(), expected)

    def test_keep_layout(self):
        """After a swap, the following gates also change the wires.
         qr0:---.--------
                |
         qr1:---|--------
                |
         qr2:--(+)--[H]--

         Coupling map: [0]--[1]--[2]

         qr0:------.---------
                   |
         qr1:--X--(+)--[H]--
               |
         qr2:--X-------------
        """
        coupling = Coupling({1: [0, 2]})

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.h(qr[2])
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        expected = ['swap q[1],q[2];',
                    'cx q[0],q[1];',
                    'h q[1];']

        pass_ = SwapMapper(coupling)
        after_dag = pass_.run(dag)

        self.assertEndswith(after_dag.qasm(), expected)

    def test_far_swap(self):
        """ A far swap that affects coming CXs.
         qr0:--(+)---.--
                |    |
         qr1:---|----|--
                |    |
         qr2:---|----|--
                |    |
         qr3:---.---(+)-

         Coupling map: [0]--[1]--[2]--[3]

         qr0:----(+)---.--
                  |    |
         qr1:--X--.---(+)-
               |
         qr2:--|----------
               |
         qr3:--X----------

        """
        coupling = Coupling({0: [1], 1: [2], 2: [3]})

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = DAGCircuit.fromQuantumCircuit(circuit)
        expected = ['swap q[1],q[3];',
                    'cx q[0],q[1];',
                    'cx q[1],q[0];']

        pass_ = SwapMapper(coupling)
        after_dag = pass_.run(dag)

        self.assertEndswith(after_dag.qasm(), expected)

    # def test_swap_between_qregs (self):
    #     """ Adding a swap affecting different qregs
    #      qr0_0:-------
    #
    #      qr1_0:--(+)--
    #               |
    #      qr1_1:---.---
    #
    #      Coupling map: [1]--[0]--[2]
    #
    #      qr0_0:--X--.---
    #              |  |
    #      qr1_0:--|-(+)--
    #              |
    #      qr1_1:--X------
    #
    #     """
    #     coupling = Coupling({0: [1, 2]})
    #
    #     qr0 = QuantumRegister(1, 'qr0')
    #     qr1 = QuantumRegister(2, 'qr1')
    #
    #     circuit = QuantumCircuit(qr0,qr1)
    #     circuit.cx(qr1[0], qr1[1])
    #     dag = DAGCircuit.fromQuantumCircuit(circuit)
    #
    #     expected = ['swap qr0[0],qr1[1];',
    #                 'cx qr1[0],qr0[0];']
    #
    #     pass_ = SwapMapper(coupling)
    #     after_dag = pass_.run(dag)
    #
    #     self.assertEndswith(after_dag.qasm(), expected)


if __name__ == '__main__':
    unittest.main()
