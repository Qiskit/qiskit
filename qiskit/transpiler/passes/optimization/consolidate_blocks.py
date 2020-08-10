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

"""Replace each block of consecutive gates by a single Unitary node."""


from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis import TwoQubitBasisDecomposer
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.synthesis import unitary_synthesis


class ConsolidateBlocks(TransformationPass):
    """Replace each block of consecutive gates by a single Unitary node.

    Pass to consolidate sequences of uninterrupted gates acting on
    the same qubits into a Unitary node, to be resynthesized later,
    to a potentially more optimal subcircuit.

    Notes:
        This pass assumes that the 'blocks_list' property that it reads is
        given such that blocks are in topological order. The blocks are
        collected by a previous pass, such as `Collect2qBlocks`.
    """

    def __init__(self,
                 kak_basis_gate=None,
                 force_consolidate=False,
                 basis_gates=None):
        """ConsolidateBlocks initializer.

        Args:
            kak_basis_gate (Gate): Basis gate for KAK decomposition.
            force_consolidate (bool): Force block consolidation
            basis_gates (List(str)): Basis gates from which to choose a KAK gate.
        """
        super().__init__()
        self.basis_gates = basis_gates
        self.force_consolidate = force_consolidate

        if kak_basis_gate is not None:
            self.decomposer = TwoQubitBasisDecomposer(kak_basis_gate)
        elif basis_gates is not None:
            kak_basis_gate = unitary_synthesis._choose_kak_gate(basis_gates)
            if kak_basis_gate is not None:
                self.decomposer = TwoQubitBasisDecomposer(kak_basis_gate)
            else:
                self.decomposer = None
        else:
            self.decomposer = TwoQubitBasisDecomposer(CXGate())

    def run(self, dag):
        """Run the ConsolidateBlocks pass on `dag`.

        Iterate over each block and replace it with an equivalent Unitary
        on the same wires.
        """

        if self.decomposer is None:
            return dag

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # compute ordered indices for the global circuit wires
        global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}

        blocks = self.property_set['block_list']
        # just to make checking if a node is in any block easier
        all_block_nodes = {nd for bl in blocks for nd in bl}

        for node in dag.topological_op_nodes():
            if node not in all_block_nodes:
                # need to add this node to find out where in the list it goes
                preds = [nd for nd in dag.predecessors(node) if nd.type == 'op']

                block_count = 0
                while preds:
                    if block_count < len(blocks):
                        block = blocks[block_count]

                        # if any of the predecessors are in the block, remove them
                        preds = [p for p in preds if p not in block]
                    else:
                        # should never occur as this would mean not all
                        # nodes before this one topologically had been added
                        # so not all predecessors were removed
                        raise TranspilerError("Not all predecessors removed due to error"
                                              " in topological order")

                    block_count += 1

                # we have now seen all predecessors
                # so update the blocks list to include this block
                blocks = blocks[:block_count] + [[node]] + blocks[block_count:]

        # create the dag from the updated list of blocks
        basis_gate_name = self.decomposer.gate.name
        for block in blocks:
            if len(block) == 1 and (block[0].name != basis_gate_name
                                    or block[0].op.is_parameterized()):
                # an intermediate node that was added into the overall list
                new_dag.apply_operation_back(block[0].op, block[0].qargs,
                                             block[0].cargs)
            else:
                # find the qubits involved in this block
                block_qargs = set()
                block_cargs = set()
                for nd in block:
                    block_qargs |= set(nd.qargs)
                    if nd.condition:
                        block_cargs |= set(nd.condition[0])
                # convert block to a sub-circuit, then simulate unitary and add
                q = QuantumRegister(len(block_qargs))
                # if condition in node, add clbits to circuit
                if len(block_cargs) > 0:
                    c = ClassicalRegister(len(block_cargs))
                    subcirc = QuantumCircuit(q, c)
                else:
                    subcirc = QuantumCircuit(q)
                block_index_map = self._block_qargs_to_indices(block_qargs,
                                                               global_index_map)
                basis_count = 0
                for nd in block:
                    if nd.op.name == basis_gate_name:
                        basis_count += 1
                    subcirc.append(nd.op, [q[block_index_map[i]] for i in nd.qargs])
                unitary = UnitaryGate(Operator(subcirc))  # simulates the circuit

                max_2q_depth = 20  # If depth > 20, there will be 1q gates to consolidate.
                if (  # pylint: disable=too-many-boolean-expressions
                        self.force_consolidate
                        or unitary.num_qubits > 2
                        or self.decomposer.num_basis_gates(unitary) < basis_count
                        or len(subcirc) > max_2q_depth
                        or (self.basis_gates is not None
                            and not set(subcirc.count_ops()).issubset(self.basis_gates))
                ):
                    new_dag.apply_operation_back(
                        UnitaryGate(unitary),
                        sorted(block_qargs, key=lambda x: block_index_map[x]))
                else:
                    for nd in block:
                        new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)

        return new_dag

    def _block_qargs_to_indices(self, block_qargs, global_index_map):
        """Map each qubit in block_qargs to its wire position among the block's wires.

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
