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

"""Test the Stochastic Swap pass"""

import unittest
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestSabreSwap(QiskitTestCase):
    """Tests the SabreSwap pass."""

    def test_trivial_case(self):
        """
         q0:--(+)-[H]-(+)-
               |       |
         q1:---.-------|--
                       |
         q2:-----------.--
         Coupling map: [1]--[0]--[2]
        """
        coupling = CouplingMap.from_line(5)

        qr = QuantumRegister(5, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)      # free
        qc.cx(0, 4)  # F
        qc.cx(1, 2)  # free
        qc.cx(1, 3)  # F
        qc.cx(2, 3)  # E
        qc.cx(1, 3)  # E

        dag = circuit_to_dag(circuit)
        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)


if __name__ == '__main__':
    unittest.main()
