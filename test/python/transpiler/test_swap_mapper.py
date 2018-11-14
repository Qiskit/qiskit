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
from qiskit import QuantumRegister
from ..common import QiskitTestCase


def _create_dag(instructions):
    dag = DAGCircuit()
    qregs = {}
    basic_elements = {}

    for instruction in instructions:
        basic_elements[instruction[0]] = len(instruction[1])

        for arg in instruction[1]:
            if qregs.get(arg[0]) is None:
                qregs[arg[0]] = []
            qregs[arg[0]].append(arg[1])

    # dag.add_qreg
    for name, regs in qregs.items():
        dag.add_qreg(QuantumRegister(max(regs) + 1, name))

    # dag.add_basis_element
    for name, amount in basic_elements.items():
        dag.add_basis_element(name, amount)

    # apply_operation_back
    for instruction, args in instructions:
        dag.apply_operation_back(instruction, args)

    return dag


class TestSwapMapper(QiskitTestCase):
    """ Tests the SwapMapper pass."""

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
        dag = _create_dag([('CX', [('q', 0), ('q', 1)]),
                           ('U', [('q', 0)]),
                           ('CX', [('q', 0), ('q', 2)])])

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
        dag = _create_dag([('CX', [('q', 2), ('q', 3)]),
                           ('CX', [('q', 0), ('q', 1)])])

        before = dag.qasm()
        pass_ = SwapMapper(coupling)
        after_dag = pass_.run(dag)

        self.assertEqual(before, after_dag.qasm())

    def test_a_single_swap(self):
        """ Adding a swap
         q0:--(+)------
               |
         q1:---.--(+)--
                   |
         q2:-------.---

         Coupling map: [1]--[0]--[2]
        """
        coupling = Coupling({0: [1, 2]})
        dag = _create_dag([('CX', [('q', 0), ('q', 1)]),
                           ('CX', [('q', 1), ('q', 2)])])
        expected = '\n'.join(["OPENQASM 2.0;",
                              "qreg q[3];",
                              "opaque swap a,b;",
                              "CX q[0],q[1];",
                              "swap q[0],q[2];",
                              "CX q[1],q[0];"]) + '\n'
        pass_ = SwapMapper(coupling)
        after_dag = pass_.run(dag)

        self.assertEqual(expected, after_dag.qasm())


if __name__ == '__main__':
    unittest.main()
