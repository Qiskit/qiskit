# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test RemoveFinalMeasurements pass"""

import unittest

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.classicalregister import Clbit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveFinalMeasurements
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestRemoveFinalMeasurements(QiskitTestCase):
    """Test removing final measurements."""

    def test_remove_reg_when_remove_all_clbits(self):
        """Remove register when all clbits removed."""
        rq = QuantumRegister(4)
        rc1 = ClassicalRegister(2)
        rc2 = ClassicalRegister(2)
        c = QuantumCircuit(rq, rc1, rc2)

        # measure into all clbits of rc1
        c.measure(0, 0)
        c.measure(1, 1)

        # measure into single bit of rc2
        c.measure(2, 2)

        dag = circuit_to_dag(c)
        remove_final_meas = RemoveFinalMeasurements()
        remove_final_meas.run(dag)

        # only clbits of rc2 expected, since rc1 and corresponding clbits
        # should have been removed.
        expected_clbits = [b for b in c.clbits if b.register == rc2]

        self.assertEqual(set(dag.cregs.values()), { rc2 }, "Expected only rc2 remains.")
        self.assertEqual(dag.clbits, expected_clbits, "Expected only rc2 bits remain.")

    def test_shared_register_bits(self):
        """"""
        rq = QuantumRegister(2)
        rc1 = ClassicalRegister(2)
        c = QuantumCircuit(rq, rc1)

        # Create reg with shared bits (same as rc1)
        rc2 = ClassicalRegister(bits=c.clbits)
        c.add_register(rc2)

        # measure into all clbits of rc1
        c.measure(0, 0)
        c.measure(1, 1)

        dag = circuit_to_dag(c)
        remove_final_meas = RemoveFinalMeasurements()
        remove_final_meas.run(dag)

        self.assertFalse(dag.cregs, "Expected no cregs remain.")
        self.assertFalse(dag.clbits, "Expected no clbits remain.")

    def test_final_measures_share_dest(self):
        """Multiple final measurements use the same clbit."""
        rq = QuantumRegister(2)
        rc = ClassicalRegister(1)
        c = QuantumCircuit(rq, rc)

        # Measure to same clbit
        c.measure(0, 0)
        c.measure(1, 0)

        dag = circuit_to_dag(c)
        measure_nodes = sum(1 for n in dag.op_nodes() if n.op.name == "measure")
        #measure_nodes = [next(dag.predecessors(dag.output_map[q])).op.name for q in dag.qubits]
        self.assertEqual(measure_nodes, 2)

        remove_final_meas = RemoveFinalMeasurements()
        remove_final_meas.run(dag)

        measure_nodes = sum(1 for n in dag.op_nodes() if n.op.name == "measure")
        self.assertEqual(measure_nodes, 0)
        self.assertFalse(dag.cregs, "Expected no cregs remain.")
        self.assertFalse(dag.clbits, "Expected no clbits remain.")

    def test_remove_chained_final_measurements(self):
        """Remove successive final measurements."""
        qr = QuantumRegister(1)
        qr2 = QuantumRegister(1)
        cr = ClassicalRegister(1)
        cr2 = ClassicalRegister(1)
        c = QuantumCircuit(qr, cr, qr2, cr2)

        c.measure(0, 0)
        c.measure(0, 0)
        c.barrier()
        c.h(1)
        c.measure(1, 1)
        c.measure(0, 1)

        dag = circuit_to_dag(c)
        pred = list(dag.predecessors(dag.output_map[c.qubits[0]]))[0]
        self.assertEqual(list(dag.predecessors(pred)), [])
        dag.draw()
        measure_nodes = sum(1 for n in dag.op_nodes() if n.op.name == "measure")
        #measure_nodes = [next(dag.predecessors(dag.output_map[q])).op.name for q in dag.qubits]
        self.assertEqual(measure_nodes, 2)

        remove_final_meas = RemoveFinalMeasurements()
        remove_final_meas.run(dag)

        measure_nodes = [n.op.name for n in dag.op_nodes()].count("measure")
        self.assertEqual(measure_nodes, 0)

    def test_remove_clbits_without_register(self):
        """clbits of final measurements not in a register are removed."""
        qr = QuantumRegister(1)
        c = QuantumCircuit(qr)

        # Add clbit without adding register
        cr = ClassicalRegister(1)
        clbit = Clbit(cr, 0)
        c.add_bits([clbit])

        self.assertFalse(c.cregs, "Expected no cregs.")

        # Measure to regless clbit
        c.measure(0, 0)

        dag = circuit_to_dag(c)
        RemoveFinalMeasurements().run(dag)

        self.assertFalse(dag.cregs, "Expected no cregs remain.")
        self.assertFalse(dag.clbits, "Expected no clbits remain.")

if __name__ == "__main__":
    unittest.main()
