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
from qiskit.circuit import QuantumCircuit
from .collapse_chains import CollapseChains


class CollectLinearFunctions(CollapseChains):
    """Collect blocks of linear gates (:class:`.CXGate` and :class:`.SwapGate` gates)
    and replaces them by linear functions (:class:`.LinearFunction`)."""

    def filter_function(self, node):
        """Specifies which nodes to collect into blocks."""
        return node.op.name in ("cx", "swap") and node.op.condition is None

    def collapse_function(self, blocks, dag):
        """Specifies what to do with collected blocks.
        """
        # Replace every discovered block by a linear function
        global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        for block in blocks:
            # Find the set of all qubits used in this block
            cur_qubits = set()
            for node in block:
                cur_qubits.update(node.qargs)

            # For reproducibility, order these qubits compatibly with the global order
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])
            wire_pos_map = dict((qb, ix) for ix, qb in enumerate(sorted_qubits))

            # Construct a linear circuit
            qc = QuantumCircuit(len(cur_qubits))
            for node in block:
                if node.op.name == "cx":
                    qc.cx(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])
                elif node.op.name == "swap":
                    qc.swap(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])

            # Create a linear function from this quantum circuit
            op = LinearFunction(qc)

            # Replace the block by the constructed circuit
            dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)

        return dag
