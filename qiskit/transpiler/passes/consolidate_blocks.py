# -*- coding: utf-8 -*-

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

# pylint: disable=cell-var-from-loop

"""
Replace each block of consecutive gates by a single Unitary node.
The blocks are collected by a previous pass, such as Collect2qBlocks.
"""

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate
from qiskit.transpiler.basepasses import TransformationPass


class ConsolidateBlocks(TransformationPass):
    """
    Pass to consolidate sequences of uninterrupted gates acting on
    the same qubits into a Unitary node, to be resynthesized later,
    to a potentially more optimal subcircuit.
    Important note: this pass assumes that the 'blocks_list' property that
    it reads is given such that blocks are in topological order.
    """
    def run(self, dag):
        """iterate over each block and replace it with an equivalent Unitary
        on the same wires.
        """
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # compute ordered indices for the global circuit wires
        global_index_map = {}
        for wire in dag.wires:
            if not isinstance(wire[0], QuantumRegister):
                continue
            global_qregs = list(dag.qregs.values())
            global_index_map[wire] = global_qregs.index(wire[0]) + wire[1]

        blocks = self.property_set['block_list']
        nodes_seen = set()

        for node in dag.topological_op_nodes():
            # skip already-visited nodes or input/output nodes
            if node in nodes_seen or node.type == 'in' or node.type == 'out':
                continue
            # check if the node belongs to the next block
            if blocks and node in blocks[0]:
                block = blocks[0]
                # find the qubits involved in this block
                block_qargs = set()
                for nd in block:
                    block_qargs |= set(nd.qargs)
                # convert block to a sub-circuit, then simulate unitary and add
                block_width = len(block_qargs)
                q = QuantumRegister(block_width)
                subcirc = QuantumCircuit(q)
                block_index_map = self._block_qargs_to_indices(block_qargs,
                                                               global_index_map)
                for nd in block:
                    nodes_seen.add(nd)
                    subcirc.append(nd.op, [q[block_index_map[i]] for i in nd.qargs])
                unitary = UnitaryGate(Operator(subcirc))  # simulates the circuit
                new_dag.apply_operation_back(
                    unitary, sorted(block_qargs, key=lambda x: block_index_map[x]))
                del blocks[0]
            else:
                # the node could belong to some future block, but in that case
                # we simply skip it. It is guaranteed that we will revisit that
                # future block, via its other nodes
                for block in blocks[1:]:
                    if node in block:
                        break
                # freestanding nodes can just be added
                else:
                    nodes_seen.add(node)
                    new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag

    def _block_qargs_to_indices(self, block_qargs, global_index_map):
        """
        Map each qubit in block_qargs to its wire position among the block's wires.
        Args:
            block_qargs (list): list of qubits that a block acts on
            global_index_map (dict): mapping from each qubit in the
                circuit to its wire position within that circuit
        Returns:
            dict: mapping from qarg to position in block
        """
        block_indices = [global_index_map[q] for q in block_qargs]
        ordered_block_indices = sorted(block_indices)
        block_positions = {q: ordered_block_indices.index(global_index_map[q])
                           for q in block_qargs}
        return block_positions
