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
from qiskit.converters.dag_to_dagdependency import (
    dag_to_dagdependency,
    dag_to_dagdependency_with_data,
)
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from test import QiskitTestCase  # pylint: disable=wrong-import-order


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

    def test_circuit_and_dag_dependency2(self):
        """Check convert to dag dependency and back
        also when the option ``create_preds_and_succs`` is False."""
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

        dag_dependency = dag_to_dagdependency(dag_in, create_preds_and_succs=False)
        dag_out = dagdependency_to_dag(dag_dependency)

        self.assertEqual(dag_out, dag_in)

    def test_metadata(self):
        """Test circuit metadata is preservered through conversion."""
        meta_dict = {"experiment_id": "1234", "execution_number": 4}
        qr = QuantumRegister(2)
        circuit_in = QuantumCircuit(qr, metadata=meta_dict)
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.measure_all()
        dag = circuit_to_dag(circuit_in)
        self.assertEqual(dag.metadata, meta_dict)
        dag_dependency = dag_to_dagdependency(dag)
        self.assertEqual(dag_dependency.metadata, meta_dict)
        dag_out = dagdependency_to_dag(dag_dependency)
        self.assertEqual(dag_out.metadata, meta_dict)

    def test_conversion_data(self):
        """Test that conversion data is created and is correct when
        ``create_conversion_data`` is ``True``.
        """
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

        dag_dependency, conversion_data = dag_to_dagdependency_with_data(
            dag_in, create_conversion_data=True
        )
        self.assertIsNotNone(conversion_data)

        # Check that mapping an op_node first forward and then backward gives back the same node.
        for in_node in dag_in.op_nodes():
            out_node = conversion_data.forward_map(in_node)
            self.assertIsNotNone(out_node)
            in_out_node = conversion_data.backward_map(out_node)
            self.assertEqual(in_node, in_out_node)

    def test_conversion_data_not_created_by_default(self):
        """Test that conversion data is not created by default."""
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

        dag_dependency, conversion_data = dag_to_dagdependency_with_data(dag_in)
        self.assertIsNone(conversion_data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
