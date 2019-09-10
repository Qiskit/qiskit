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

# pylint: disable=unused-import

"""Test the unrolling of cx into cz"""


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler.passes import Unroller
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError


class TestUnrollCX(QiskitTestCase):
    """Tests the unrolling of CX into CZ."""

    def test_cx_unroll(self):
        """Test decompose a CX into CZ and Hadamards.
        q0:-----.-----      q0:---------.---------
                |                       |
        q1:----(+)----   =  q1:---[H]--[z]--[H]---
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(['cz', 'h'])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 3)
        self.assertEqual(op_nodes[0].name, 'h')
        self.assertEqual(op_nodes[1].name, 'cz')
        self.assertEqual(op_nodes[2].name, 'h')
