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
from qiskit.transpiler.passes import RemoveFinalMeasurements
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestRemoveFinalMeasurements(QiskitTestCase):
    """Test removing final measurements."""

    def test_multi_bit_register_removed_with_clbits(self):
        """Remove register when all clbits removed."""
        qc = QuantumCircuit(2, 2)

        # measure into all clbits of c0
        qc.measure(0, 0)
        qc.measure(1, 1)

        dag = circuit_to_dag(qc)
        RemoveFinalMeasurements().run(dag)

        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)

    def test_register_kept_if_measured_clbit_busy(self):
        """
        A register is kept if the measure destination bit is still
        busy after measure removal.
        """
        c0 = ClassicalRegister(1)
        qc = QuantumCircuit(QuantumRegister(1), c0)

        # make c0 busy
        qc.x(0).c_if(c0, 0)

        # measure into c0
        qc.measure(0, c0)

        dag = circuit_to_dag(qc)
        RemoveFinalMeasurements().run(dag)

        self.assertSetEqual(set(dag.cregs.values()), {c0})
        self.assertSetEqual(set(dag.clbits), set(c0))

    def test_multi_bit_register_kept_if_not_measured_clbit_busy(self):
        """
        A multi-bit register is kept if it contains a busy bit even if
        the measure destination bit itself is idle.
        """
        c0 = ClassicalRegister(2)
        qc = QuantumCircuit(QuantumRegister(1), c0)

        # make c0[0] busy
        qc.x(0).c_if(c0[0], 0)

        # measure into not busy c0[1]
        qc.measure(0, c0[1])

        dag = circuit_to_dag(qc)
        RemoveFinalMeasurements().run(dag)

        # c0 should not be removed because it has busy bit c0[0]
        self.assertSetEqual(set(dag.cregs.values()), {c0})

        # note: c0[1] should not be removed even though it is now idle
        # because it is referenced by creg c0.
        self.assertSetEqual(set(dag.clbits), set(c0))

    def test_multi_bit_register_removed_if_all_bits_idle(self):
        """A multibit register is removed when all bits are idle."""
        qc = QuantumCircuit(1, 2)

        # measure into single bit c0[0] of c0
        qc.measure(0, 0)

        dag = circuit_to_dag(qc)
        RemoveFinalMeasurements().run(dag)

        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)

    def test_multi_reg_shared_bits_removed(self):
        """All registers sharing removed bits should be removed."""
        q0 = QuantumRegister(2)
        c0 = ClassicalRegister(2)
        qc = QuantumCircuit(q0, c0)

        # Create reg with shared bits (same as c0)
        c1 = ClassicalRegister(bits=qc.clbits)
        qc.add_register(c1)

        # measure into all clbits of c0
        qc.measure(0, c0[0])
        qc.measure(1, c0[1])

        dag = circuit_to_dag(qc)
        remove_final_meas = RemoveFinalMeasurements()
        remove_final_meas.run(dag)

        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)

    def test_final_measures_share_dest(self):
        """Multiple final measurements use the same clbit."""
        def expected_dag():
            qc = QuantumCircuit(QuantumRegister(2, "q0"))
            return circuit_to_dag(qc)

        rq = QuantumRegister(2, "q0")
        rc = ClassicalRegister(1, "c0")
        qc = QuantumCircuit(rq, rc)

        qc.measure(0, 0)
        qc.measure(1, 0)

        dag = circuit_to_dag(qc)
        RemoveFinalMeasurements().run(dag)

        self.assertEqual(dag, expected_dag())

    def test_remove_chained_final_measurements(self):
        """Remove successive final measurements."""
        def expected_dag():
            q0 = QuantumRegister(1, "q0")
            q1 = QuantumRegister(1, "q1")
            c0 = ClassicalRegister(1, "c0")
            qc = QuantumCircuit(q0, c0, q1)

            qc.measure(q0, c0)
            qc.measure(q0, c0)
            qc.barrier()
            qc.h(q1)

            return circuit_to_dag(qc)

        q0 = QuantumRegister(1, "q0")
        q1 = QuantumRegister(1, "q1")
        c0 = ClassicalRegister(1, "c0")
        c1 = ClassicalRegister(1, "c1")
        qc = QuantumCircuit(q0, c0, q1, c1)

        qc.measure(q0, c0)
        qc.measure(q0, c0)
        qc.barrier()
        qc.h(q1)
        qc.measure(q1, c1)
        qc.measure(q0, c1)

        dag = circuit_to_dag(qc)
        RemoveFinalMeasurements().run(dag)

        self.assertEqual(dag, expected_dag())

    def test_remove_clbits_without_register(self):
        """clbits of final measurements not in a register are removed."""
        qc = QuantumCircuit(1)

        # Add clbit without adding register
        c0 = ClassicalRegister(1)
        clbit = Clbit(c0, 0)
        qc.add_bits([clbit])

        self.assertFalse(qc.cregs)

        # Measure to regless clbit
        qc.measure(0, 0)

        dag = circuit_to_dag(qc)
        RemoveFinalMeasurements().run(dag)

        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)

    def test_final_barriers_and_measures_complex(self):
        """Test complex final barrier and measure removal."""
        def expected_dag():
            q0 = QuantumRegister(5, "q0")
            c1 = ClassicalRegister(1, "c1")
            qc = QuantumCircuit(q0, c1)
            qc.h(q0[0])
            return circuit_to_dag(qc)

        q0 = QuantumRegister(5, "q0")
        c0 = ClassicalRegister(1, "c0")
        c1 = ClassicalRegister(1, "c1")
        qc = QuantumCircuit(q0, c0, c1)

        qc.measure(q0[1], c0)
        qc.h(q0[0])
        qc.measure(q0[0], c0[0])
        qc.barrier()
        qc.barrier(q0[2], q0[3])
        qc.measure_all()
        qc.barrier(q0[4])

        dag = circuit_to_dag(qc)
        RemoveFinalMeasurements().run(dag)

        self.assertEqual(dag, expected_dag())


if __name__ == "__main__":
    unittest.main()
