# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Replace each sequence of CX and SWAP gates by a single LinearFunction gate."""

from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumCircuit


class CollectLinearFunctions(TransformationPass):
    """Collect blocks of linear gates (:class:`.CXGate` and :class:`.SwapGate` gates)
    and replaces them by linear functions (:class:`.LinearFunction`)."""

    @staticmethod
    def _is_linear_gate(op):
        return op.name in ("cx", "swap") and op.condition is None

    # Called when reached the end of the linear block (either the next gate
    # is not linear, or no more nodes)
    @staticmethod
    def _finalize_processing_block(cur_nodes, blocks):
        # Collect only blocks comprising at least 2 gates.
        if len(cur_nodes) >= 2:
            blocks.append(cur_nodes)

    # For now, we implement the naive greedy algorithm that makes a linear sweep over
    # nodes in DAG (using the topological order) and collects blocks of linear gates.
    def run(self, dag):
        """Run the CollectLinearFunctions pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        blocks = []

        cur_nodes = []
        for node in dag.topological_op_nodes():
            # If the current gate is not linear, we are done processing the current block
            if not self._is_linear_gate(node.op):
                self._finalize_processing_block(cur_nodes, blocks)
                cur_nodes = []

            else:
                # This is a linear gate, we add the node and its qubits
                cur_nodes.append(node)

        # Last block
        self._finalize_processing_block(cur_nodes, blocks)

        # Replace every discovered block by a linear function
        global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        for cur_nodes in blocks:
            # Find the set of all qubits used in this block
            cur_qubits = set()
            for node in cur_nodes:
                cur_qubits.update(node.qargs)

            # For reproducibility, order these qubits compatibly with the global order
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])
            wire_pos_map = dict((qb, ix) for ix, qb in enumerate(sorted_qubits))

            # Construct a linear circuit
            qc = QuantumCircuit(len(cur_qubits))
            for node in cur_nodes:
                if node.op.name == "cx":
                    qc.cx(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])
                elif node.op.name == "swap":
                    qc.swap(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])

            # Create a linear function from this quantum circuit
            op = LinearFunction(qc)

            # Replace the block by the constructed circuit
            dag.replace_block_with_op(cur_nodes, op, wire_pos_map, cycle_check=False)

        return dag
