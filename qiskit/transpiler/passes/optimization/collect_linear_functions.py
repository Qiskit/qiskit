# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Replace each SWAP-CX-SWAP sequence by a single Bridge gate."""

from qiskit.circuit.library.standard_gates import CXGate, SwapGate
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumCircuit


class CollectLinearFunctions(TransformationPass):
    """Collect blocks of linear gates (CX and SWAP) and replace them by linear functions."""

    @staticmethod
    def _is_linear_gate(op):
        return isinstance(op, (CXGate, SwapGate)) and op.condition is None

    # Called when reached the end of the linear block (either the next gate
    # is not linear, or no more nodes)
    @staticmethod
    def _finalize_processing_block(cur_nodes, cur_qubits, blocks):
        # Collect only blocks comprising at least 2 gates.
        # ToDo: possibly make the minimum number of gates as a parameter
        if len(cur_nodes) >= 2:
            cur_wire_pos_map = dict((qb, ix) for ix, qb in enumerate(cur_qubits))
            blocks.append((cur_nodes, cur_wire_pos_map))

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
        cur_qubits = set(())

        for node in dag.topological_op_nodes():
            # If the current gate is not linear, we are done processing the current block
            if not self._is_linear_gate(node.op):
                self._finalize_processing_block(cur_nodes, cur_qubits, blocks)
                cur_nodes = []
                cur_qubits = set(())

            else:
                # This is a linear gate, we add the node and its qubits
                cur_nodes.append(node)
                cur_qubits.update(node.qargs)

        # Last block
        self._finalize_processing_block(cur_nodes, cur_qubits, blocks)

        for block, wire_pos_map in blocks:
            qc = QuantumCircuit(len(wire_pos_map))
            for node in block:
                if isinstance(node.op, CXGate):
                    qc.cx(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])
                elif isinstance(node.op, SwapGate):
                    qc.swap(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])

            lf = LinearFunction(len(wire_pos_map), qc)
            op = lf.to_instruction()
            op.condition = None
            dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)

        return dag
