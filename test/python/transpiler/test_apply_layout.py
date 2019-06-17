# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the ApplyLayout pass"""

import unittest

from qiskit.circuit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import ApplyLayout


class TestApplyLayout(QiskitTestCase):
    """ Tests the ApplyLayout pass."""

    def test_circuit_with_swap_gate(self):
        """Test if the circuit with virtual qubits is transformed into
        the circuit with physical qubits.

        [Circuit with virtual qubits]
          v0:--X---.---M(v0->c0)
               |   |
          v1:--X---|---M(v1->c1)
                   |
          v2:-----(+)--M(v2->c2)

         Initial layout: {v[0]: 2, v[1]: 1, v[2]: 0}
         CouplingMap map: [1]--[0]--[2]

        [Circuit with physical qubits]
          q2:--X---.---M(q2->c0)
               |   |
          q1:--X---|---M(q1->c1)
                   |
          q0:-----(+)--M(q0->c2)
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        v = QuantumRegister(3, 'v')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(v, cr)
        circuit.swap(v[0], v[1])
        circuit.cx(v[0], v[2])
        circuit.measure(v[0], cr[0])
        circuit.measure(v[1], cr[1])
        circuit.measure(v[2], cr[2])

        initial_layout = Layout({v[0]: 2, v[1]: 1, v[2]: 0})

        q = QuantumRegister(3, 'q')
        expected = QuantumCircuit(q, cr)
        expected.swap(q[2], q[1])
        expected.cx(q[2], q[0])
        expected.measure(q[2], cr[0])
        expected.measure(q[1], cr[1])
        expected.measure(q[0], cr[2])

        dag = circuit_to_dag(circuit)
        pass_ = ApplyLayout(coupling=coupling, initial_layout=initial_layout)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


if __name__ == '__main__':
    unittest.main()
