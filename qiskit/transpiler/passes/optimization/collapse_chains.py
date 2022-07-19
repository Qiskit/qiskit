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


"""Collect and collapse chains of matching nodes."""
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from qiskit.transpiler import TransformationPass
from qiskit.converters import dag_to_dagdependency, dagdependency_to_dag
from qiskit.dagcircuit import DAGNode, DAGDepNode
from .collect_blocks import BlockCollector, BlockSplitter

# ToDo: fix inheritance from ABC


class CollapseChains(TransformationPass):
    """Provides an API for collecting and collapsing blocks of nodes.

    A derived class is required to implement two functions:
        filter_function: specifying which nodes to collect into blocks
        collapse_function: specifying what to do with collected blocks
    """

    def __init__(self,
                 do_commutative_analysis=True,
                 split_blocks=True,
                 min_nodes_per_block=2):
        """CollapseChains is a transformation pass that collect maximal blocks
        of nodes matching a given filter function and processes these blocks of
        nodes as specified by collapse_function.

        Args:
            do_commutative_analysis (bool): if True, exploits commutativity relations.
            split_blocks (bool): if True, splits detected blocks into sub-blocks over disjoint qubit sets.
            min_nodes_per_block (int): specifies the minimum number of nodes in a block on
                which to apply collapse_function.
        """
        self.do_commutative_analysis = do_commutative_analysis
        self.split_blocks = split_blocks
        self.min_nodes_per_block = min_nodes_per_block

        super().__init__()

    def run(self, dag):
        """Run the CollapseChainsPass pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        if self.do_commutative_analysis:
            processed_dag = dag_to_dagdependency(dag)
        else:
            processed_dag = dag

        matching_blocks = BlockCollector(processed_dag).collect_all_matching_blocks(filter_fn=self.filter_function)

        if self.split_blocks:
            blocks = []
            for block in matching_blocks:
                blocks.extend(BlockSplitter().run(block))
        else:
            blocks = matching_blocks

        filtered_blocks = [block for block in blocks if len(block) >= self.min_nodes_per_block]

        collapsed_dep_dag = self.collapse_function(filtered_blocks, processed_dag)

        if self.do_commutative_analysis:
            collapsed_dag = dagdependency_to_dag(collapsed_dep_dag)
        else:
            collapsed_dag = collapsed_dep_dag

        return collapsed_dag

    @abstractmethod
    def filter_function(self, node: Union[DAGNode, DAGDepNode]) -> bool:
        """Specifies which nodes to collect into blocks.
        """
        pass

    @abstractmethod
    def collapse_function(self, blocks: Union[List[List[DAGNode]], List[List[DAGDepNode]]], dag):
        """Specifies what to do with collected blocks.
        """
        pass
