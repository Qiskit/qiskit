# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for the converter dag dependency to dag circuit and
dag circuit to dag dependency."""

import unittest

from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_dagdependency import dag_to_dagdependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestCircuitToDagDependency(QiskitTestCase):
    """Test DAGCircuit to DAGDependency."""

    def test_circuit_and_dag_dependency(self):
        """Check convert to dag dependency and back"""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit_in = QuantumCircuit(qr, cr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[1])
        circuit_in.measure(qr[0], cr[0])
        circuit_in.measure(qr[1], cr[1])
        circuit_in.x(qr[0]).c_if(cr, 0x3)
        circuit_in.measure(qr[0], cr[0])
        circuit_in.measure(qr[1], cr[1])
        circuit_in.measure(qr[2], cr[2])
        dag_in = circuit_to_dag(circuit_in)

        dag_dependency = dag_to_dagdependency(dag_in)
        dag_out = dagdependency_to_dag(dag_dependency)

        self.assertEqual(dag_out, dag_in)


if __name__ == '__main__':
    unittest.main(verbosity=2)
