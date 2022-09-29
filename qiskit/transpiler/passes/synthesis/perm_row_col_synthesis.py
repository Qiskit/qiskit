import numpy as np
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis
from qiskit.circuit.library.generalized_gates.linear_function import LinearFunction
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumRegister, QuantumCircuit


class PermRowColSynthesis(HighLevelSynthesis):
    """Synthesize high-level objects by using permrowcol algorithm"""

    def __init__(self, coupling_map: CouplingMap):
        self._coupling_map = coupling_map

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Perform the synthesization and return the synthesized
        circuit as a dag

        Args:
            dag (DAGCircuit): dag circuit to re-synthesize

        Returns:
            DAGCircuit: re-synthesized dag circuit
        """
        for node in dag.named_nodes("cx"):
            # TODO: do something to the nodes
            pass

        # alt parity matrix of dag circuit, dtype=bool
        # parity_mat = LinearFunction(dag_to_circuit(dag)).linear

        parity_mat = np.identity(3)
        res_circuit = self.perm_row_col(parity_mat, self._coupling_map)
        return circuit_to_dag(res_circuit)

    def perm_row_col(self, parity_mat: np.ndarray, coupling_map: CouplingMap) -> QuantumCircuit:
        """Run permrowcol algorithm on the given parity matrix

        Args:
            parity_mat (np.ndarray): parity matrix representing a circuit
            coupling_map (CouplingMap): topology constraint

        Returns:
            QuantumCircuit: synthesized circuit
        """
        circuit = QuantumCircuit(QuantumRegister(6, "q"))
        return circuit
