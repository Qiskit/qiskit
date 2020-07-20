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

"""Test the LayoutTransformation pass"""

import unittest

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import LayoutTransformation


class TestLayoutTransformation(QiskitTestCase):
    """
    Tests the LayoutTransformation pass.
    """

    def test_three_qubit(self):
        """Test if the permutation {0->2,1->0,2->1} is implemented correctly."""
        np.random.seed(0)
        v = QuantumRegister(3, 'v')  # virtual qubits
        coupling = CouplingMap([[0, 1], [1, 2]])
        from_layout = Layout({v[0]: 0, v[1]: 1, v[2]: 2})
        to_layout = Layout({v[0]: 2, v[1]: 0, v[2]: 1})
        ltpass = LayoutTransformation(coupling_map=coupling,
                                      from_layout=from_layout,
                                      to_layout=to_layout)
        qc = QuantumCircuit(3)  # input (empty) physical circuit
        dag = circuit_to_dag(qc)
        q = dag.qubits
        output_dag = ltpass.run(dag)
        # output_dag.draw()
        # Check that only two swaps were performed
        self.assertCountEqual(["swap"] * 2, [op.name for op in output_dag.topological_op_nodes()])
        # And check that the swaps were first performed on {q0,q1} then on {q1,q2}.
        self.assertEqual([frozenset([q[0], q[1]]), frozenset([q[1], q[2]])],
                         [frozenset(op.qargs) for op in output_dag.topological_op_nodes()])


if __name__ == '__main__':
    unittest.main()
