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

"""Test functionality to collect, split and consolidate blocks from DAGCircuits."""


import unittest

from qiskit.converters import circuit_to_dag, circuit_to_dagdependency
from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit.collect_blocks import BlockCollector, BlockSplitter, BlockCollapser


class TestCollectBlocks(QiskitTestCase):
    """Tests to verify correctness of collecting, splitting, and consolidating blocks
    from DAGCircuit and DAGDependency. Additional tests appear as a part of
    CollectLinearFunctions and CollectCliffords passes.
    """

    def test_collect_gates_from_dagcircuit_1(self):
        """Test collecting CX gates from DAGCircuits."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(0, 3)
        qc.cx(0, 4)

        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name == "cx")

        # The middle z-gate leads to two blocks of size 2 each
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 2)
        self.assertEqual(len(blocks[1]), 2)

    def test_collect_gates_from_dagcircuit_2(self):
        """Test collecting both CX and Z gates from DAGCircuits."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(0, 3)
        qc.cx(0, 4)

        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "z"]
        )

        # All of the gates are part of a single block
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

    def test_collect_gates_from_dagdependency_1(self):
        """Test collecting CX gates from DAGDependency."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(0, 3)
        qc.cx(0, 4)

        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name == "cx")

        # The middle z-gate commutes with CX-gates, which leads to a single block of length 4
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 4)

    def test_collect_gates_from_dagdependency_2(self):
        """Test collecting both CX and Z gates from DAGDependency."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(0, 3)
        qc.cx(0, 4)

        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "z"]
        )

        # All the gates are part of a single block
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

    def test_collect_and_split_gates_from_dagcircuit(self):
        """Test collecting and splitting blocks from DAGCircuit."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(3, 5)
        qc.cx(2, 4)
        qc.swap(1, 0)
        qc.cz(5, 3)

        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: True)

        # All the gates are part of a single block
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

        # Split the first block into sub-blocks over disjoint qubit sets
        # We should get 3 sub-blocks
        split_blocks = BlockSplitter().run(blocks[0])
        self.assertEqual(len(split_blocks), 3)

    def test_collect_and_split_gates_from_dagdependency(self):
        """Test collecting and splitting blocks from DAGDependecy."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(3, 5)
        qc.cx(2, 4)
        qc.swap(1, 0)
        qc.cz(5, 3)

        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: True)

        # All the gates are part of a single block
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

        # Split the first block into sub-blocks over disjoint qubit sets
        # We should get 3 sub-blocks
        split_blocks = BlockSplitter().run(blocks[0])
        self.assertEqual(len(split_blocks), 3)


if __name__ == "__main__":
    unittest.main()
