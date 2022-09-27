"""Test PermRowColSynthesis"""

import unittest
import numpy as np
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.synthesis.perm_row_col_synthesis import PermRowColSynthesis
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap


class TestPermRowColSynthesis(QiskitTestCase):
    """Test PermRowColSynthesis"""

    def test_run_returns_a_dag(self):
        coupling = CouplingMap()
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)
        synthesis = PermRowColSynthesis(coupling)

        instance = synthesis.run(dag)

        self.assertIsInstance(instance, DAGCircuit)

    def test_perm_row_col_returns_circuit(self):
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.identity(3)

        instance = synthesis.perm_row_col(parity_mat, coupling)

        self.assertIsInstance(instance, QuantumCircuit)


if __name__ == "__main__":
    unittest.main()
