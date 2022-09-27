import numpy as np
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit


class PermRowColSynthesis(HighLevelSynthesis):
    """Synthesize high-level objects by using permrowcol algorithm"""

    def __init__(self, coupling_map: CouplingMap):
        self._coupling_map = coupling_map

    def run(self, dag: DAGCircuit) -> DAGCircuit:

        # this parity matrix is extracted from the coupling map
        parity_mat = np.identity(3)
        res_circuit = self.perm_row_col(parity_mat, self._coupling_map)
        #return circuit_to_dag(res_circuit)
        return dag

    def perm_row_col(self, parity_mat: np.ndarray, coupling_map: CouplingMap) -> QuantumCircuit:
        # use the parity matrix to synthesize new circuit
        circuit = QuantumCircuit(QuantumRegister(6, 'q'))
        return circuit