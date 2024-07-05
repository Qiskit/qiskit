import numpy as np
from qiskit.dagcircuit import DAGOpNode

from qiskit.synthesis.two_qubit.local_invariance import two_qubit_local_invariants
from qiskit.transpiler import AnalysisPass
from qiskit.transpiler.passes.utils import _block_to_matrix
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate

class CheckSimplify2Q(AnalysisPass):
    """Check if the `dag` contains two-qubit gates that can be decomposed into single-qubit gates.

    """

    def _block_qargs_to_indices(self, dag, block_qargs):
        """Map each qubit in block_qargs to its wire position among the block's wires.
        Args:
            block_qargs (list): list of qubits that a block acts on
            global_index_map (dict): mapping from each qubit in the
                circuit to its wire position within that circuit
        Returns:
            dict: mapping from qarg to position in block
        """
        block_indices = [dag.find_bit(q).index for q in block_qargs]
        ordered_block_indices = {bit: index for index, bit in enumerate(sorted(block_indices))}
        block_positions = {q: ordered_block_indices[dag.find_bit(q).index] for q in block_qargs}
        return block_positions

    def run(self, dag):
        """Run the CheckSimplify2Q pass on `dag`."""

        blocks = self.property_set["block_list"] or []
        self.property_set['removable_2q'] = 0
        for block in blocks:
            block_qargs = set()
            block_cargs = set()
            for nd in block:
                block_qargs |= set(nd.qargs)
                if isinstance(nd, DAGOpNode) and getattr(nd.op, "condition", None):
                    block_cargs |= set(getattr(nd.op, "condition", None)[0])

            # skip blocks with conditional gates so far
            if len(block_cargs) > 0 or len(block_qargs) != 2:
                continue

            block_index_map = self._block_qargs_to_indices(dag, block_qargs)

            matrix = _block_to_matrix(block, block_index_map)

            if np.all(two_qubit_local_invariants(matrix) == [1, 0, 3]):
                self.property_set['removable_2q'] += 1

        removable_2q = [node for node in dag.topological_op_nodes() if
                        len(node.cargs) == 0 and len(node.qargs) == 2 and np.all(
                            two_qubit_local_invariants(node.op) == [1, 0, 3])]
        print("len_remov", len(removable_2q))
        return dag
