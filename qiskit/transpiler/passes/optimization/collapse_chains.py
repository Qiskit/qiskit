# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Provides an abstract interface for walking over a circuit, collecting blocks of gates
that match a given filter function, and consolidating these blocks into objects of
a given type."""

from typing import Union

from abc import abstractmethod

from qiskit import QuantumCircuit
from qiskit.circuit import Operation, CircuitInstruction
from qiskit.transpiler import TransformationPass
from qiskit.converters import dag_to_dagdependency, dagdependency_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGDepNode
from qiskit.dagcircuit.collect_blocks import BlockCollector, BlockSplitter
from qiskit.transpiler.passes.utils import control_flow

# pylint: disable=unused-import
from qiskit.dagcircuit import DAGCircuit


class CollapseChains(TransformationPass):
    """Provides an API for collecting and collapsing blocks of nodes.

    A derived class is required to implement two functions:
    a :func:`~.filter_function` that specifies which nodes should be collected into blocks,
    and a :func:`~.collapse_function` that specifies how to collapse blocks.
    """

    def __init__(self, do_commutative_analysis=False, split_blocks=True, min_block_size=2):
        """CollapseChains is a transformation pass that greedily collects maximal blocks
        of nodes matching a given ``filter_fn`` and combines the nodes in each block
        as specified by a given ``collapse_function``.

        Args:
            do_commutative_analysis (bool): if True, exploits commutativity relations
                between nodes.
            split_blocks (bool): if True, splits collected blocks into sub-blocks over
                disjoint qubit subsets.
            min_block_size (int): specifies the minimum size of collapsable blocks.
        """
        self.do_commutative_analysis = do_commutative_analysis
        self.split_blocks = split_blocks
        self.min_block_size = min_block_size

        super().__init__()

    @abstractmethod
    def filter_function(self, node: Union[DAGOpNode, DAGDepNode]) -> bool:
        """Specifies which nodes to collect into blocks.

        Args:
            node (Union[DAGOpNode, DAGDepNode]): the node being examined.

        Returns:
            bool: whether the node should be collected.
        """
        raise NotImplementedError

    @abstractmethod
    def collapse_function(
        self,
        circuit: QuantumCircuit,
    ) -> Operation:
        """Specifies what to do with the collected blocks.

        Args:
            circuit (QuantumCircuit): quantum circuit containing the block of nodes
                to be collapsed (combined).

        Returns:
            Operation: the result of combining nodes in ``circuit``.
        """
        raise NotImplementedError

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the CollectLinearFunctions pass on `dag`.
        Args:
            dag (DAGCircuit): the DAG to be optimized.
        Returns:
            DAGCircuit: the optimized DAG.
        """

        # If the option commutative_analysis is set, construct DAGDependency from the given DAGCircuit.
        if self.do_commutative_analysis:
            dag = dag_to_dagdependency(dag)

        # Collect blocks of consecutive gates matching filter_function
        # (filter_function must be defined in a derived class).
        blocks = BlockCollector(dag).collect_all_matching_blocks(filter_fn=self.filter_function)

        # If the option split_blocks is set, refine blocks by splitting them into sub-blocks over
        # disconnected qubit subsets.
        if self.split_blocks:
            split_blocks = []
            for block in blocks:
                split_blocks.extend(BlockSplitter().run(block))
            blocks = split_blocks

        # Keep only blocks with at least min_block_sizes.
        blocks = [block for block in blocks if len(block) >= self.min_block_size]

        # For each block, construct a quantum circuit containing instructions in the block,
        # then use collapse_function to collapse this circuit into a single operation
        # (collapse_function must be defined in a derived class).
        global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        for block in blocks:
            # Find the set of qubits used in this block (which might be much smaller than
            # the set of all qubits).
            cur_qubits = set()
            for node in block:
                cur_qubits.update(node.qargs)

            # For reproducibility, order these qubits compatibly with the global order.
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])

            # Construct a quantum circuit from the nodes in the block, remapping the qubits.
            wire_pos_map = dict((qb, ix) for ix, qb in enumerate(sorted_qubits))
            qc = QuantumCircuit(len(cur_qubits))
            for node in block:
                remapped_qubits = [wire_pos_map[qarg] for qarg in node.qargs]
                qc.append(CircuitInstruction(node.op, remapped_qubits, node.cargs))

            # Collapse this quantum circuit into an operation.
            op = self.collapse_function(qc)

            # Replace the block of nodes in the DAG by the constructed operation
            # (the function replace_block_with_op is implemented both in DAGCircuit and DAGDependency).
            dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)

        # If the option commutative_analysis is set, construct back DAGCircuit from DAGDependency.
        if self.do_commutative_analysis:
            dag = dagdependency_to_dag(dag)

        return dag
