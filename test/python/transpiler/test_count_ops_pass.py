# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Depth pass testing"""

import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import CountOps, GateCount
from test import QiskitTestCase


class TestCountOpsPass(QiskitTestCase):
    """Tests for CountOps analysis methods."""

    def test_empty_dag(self):
        """Empty DAG has empty counts."""
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)

        pass_ = CountOps()
        _ = pass_.run(dag)

        self.assertDictEqual(pass_.property_set["count_ops"], {})

    def test_just_qubits(self):
        """A dag with 8 operations (6 CXs and 2 Hs)"""

        #       ┌───┐                    ┌───┐┌───┐
        # q0_0: ┤ H ├──■────■────■────■──┤ X ├┤ X ├
        #       ├───┤┌─┴─┐┌─┴─┐┌─┴─┐┌─┴─┐└─┬─┘└─┬─┘
        # q0_1: ┤ H ├┤ X ├┤ X ├┤ X ├┤ X ├──■────■──
        #       └───┘└───┘└───┘└───┘└───┘
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = CountOps()
        _ = pass_.run(dag)

        self.assertDictEqual(pass_.property_set["count_ops"], {"cx": 6, "h": 2})


class TestGateCountPass(QiskitTestCase):
    """Tests for GateCount analysis pass."""

    def test_count_t_and_tdg(self):
        """Count T and Tdg gates."""
        circuit = QuantumCircuit(1)
        circuit.t(0)
        circuit.t(0)
        circuit.tdg(0)
        dag = circuit_to_dag(circuit)

        pass_ = GateCount(gates=["t", "tdg"], key="t_count")
        pass_.run(dag)

        self.assertEqual(pass_.property_set["t_count"], 3)

    def test_count_rz(self):
        """Count Rz gates."""
        circuit = QuantumCircuit(2)
        circuit.rz(0.5, 0)
        circuit.rz(1.0, 1)
        circuit.h(0)
        dag = circuit_to_dag(circuit)

        pass_ = GateCount(gates=["rz"], key="rz_count")
        pass_.run(dag)

        self.assertEqual(pass_.property_set["rz_count"], 2)

    def test_no_matching_gates(self):
        """Returns 0 when none of the specified gates are in the circuit."""
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.x(0)
        dag = circuit_to_dag(circuit)

        pass_ = GateCount(gates=["t"], key="t_count")
        pass_.run(dag)

        self.assertEqual(pass_.property_set["t_count"], 0)

    def test_invalid_gate_name_ignored(self):
        """Invalid gate names are silently ignored."""
        circuit = QuantumCircuit(1)
        circuit.t(0)
        dag = circuit_to_dag(circuit)

        pass_ = GateCount(gates=["t", "not_a_gate"], key="t_count")
        pass_.run(dag)

        self.assertEqual(pass_.property_set["t_count"], 1)

    def test_control_flow_recurse(self):
        """Test recursing into control flow."""
        circuit = QuantumCircuit(1)
        circuit.t(0)

        # Note: this is counted as 1 T gate, not 3! We generally cannot know how often
        # a loop is executed (while loop with some dynamic condition for example) and the count
        # here is referring to the counts in the compiled circuit, not the executed count at
        # runtime.
        with circuit.for_loop(range(3)):
            circuit.t(0)
        dag = circuit_to_dag(circuit)

        pass_recurse = GateCount(gates=["t"], key="t_count", recurse=True)
        pass_recurse.run(dag)
        self.assertEqual(pass_recurse.property_set["t_count"], 2)

        pass_no_recurse = GateCount(gates=["t"], key="t_count", recurse=False)
        pass_no_recurse.run(dag)
        self.assertEqual(pass_no_recurse.property_set["t_count"], 1)


if __name__ == "__main__":
    unittest.main()
