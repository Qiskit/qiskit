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
from qiskit.dagcircuit.collect_blocks import BlockCollector, BlockSplitter


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
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name == "cx",
            split_blocks=False,
            min_block_size=2,
        )

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
            lambda node: node.op.name in ["cx", "z"],
            split_blocks=False,
            min_block_size=1,
        )

        # All of the gates are part of a single block
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

    def test_collect_gates_from_dagcircuit_3(self):
        """Test collecting CX gates from DAGCircuits."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(1, 3)
        qc.cx(0, 3)
        qc.cx(0, 4)

        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx"],
            split_blocks=False,
            min_block_size=1,
        )

        # We should end up with two CX blocks: even though there is "a path
        # around z(0)", without commutativity analysis z(0) prevents from
        # including all CX-gates into the same block
        self.assertEqual(len(blocks), 2)

    def test_collect_gates_from_dagdependency_1(self):
        """Test collecting CX gates from DAGDependency."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(0, 3)
        qc.cx(0, 4)

        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name == "cx",
            split_blocks=False,
            min_block_size=1,
        )

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
            lambda node: node.op.name in ["cx", "z"],
            split_blocks=False,
            min_block_size=1,
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
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: True,
            split_blocks=False,
            min_block_size=1,
        )

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
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: True,
            split_blocks=False,
            min_block_size=1,
        )

        # All the gates are part of a single block
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

        # Split the first block into sub-blocks over disjoint qubit sets
        # We should get 3 sub-blocks
        split_blocks = BlockSplitter().run(blocks[0])
        self.assertEqual(len(split_blocks), 3)

    def test_circuit_has_measure(self):
        """Test that block collection works properly when there is a measure in the
        middle of the circuit."""

        qc = QuantumCircuit(2, 1)
        qc.cx(1, 0)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        qc.x(0)
        qc.cx(1, 0)

        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["x", "cx"],
            split_blocks=False,
            min_block_size=1,
        )

        # measure prevents combining the two blocks
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 3)
        self.assertEqual(len(blocks[1]), 2)

    def test_circuit_has_measure_dagdependency(self):
        """Test that block collection works properly when there is a measure in the
        middle of the circuit."""

        qc = QuantumCircuit(2, 1)
        qc.cx(1, 0)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        qc.x(0)
        qc.cx(1, 0)

        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["x", "cx"],
            split_blocks=False,
            min_block_size=1,
        )

        # measure prevents combining the two blocks
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 3)
        self.assertEqual(len(blocks[1]), 2)

    def test_circuit_has_conditional_gates(self):
        """Test that block collection works properly when there the circuit
        contains conditional gates."""

        qc = QuantumCircuit(2, 1)
        qc.x(0)
        qc.x(1)
        qc.cx(1, 0)
        qc.x(1).c_if(0, 1)
        qc.x(0)
        qc.x(1)
        qc.cx(0, 1)

        # If the filter_function does not look at conditions, we should collect all
        # gates into the block.
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["x", "cx"],
            split_blocks=False,
            min_block_size=1,
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 7)

        # If the filter_function does look at conditions, we should not collect the middle
        # conditional gate (note that x(1) following the measure is collected into the first
        # block).
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["x", "cx"] and not getattr(node.op, "condition", None),
            split_blocks=False,
            min_block_size=1,
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 4)
        self.assertEqual(len(blocks[1]), 2)

    def test_circuit_has_conditional_gates_dagdependency(self):
        """Test that block collection works properly when there the circuit
        contains conditional gates."""

        qc = QuantumCircuit(2, 1)
        qc.x(0)
        qc.x(1)
        qc.cx(1, 0)
        qc.x(1).c_if(0, 1)
        qc.x(0)
        qc.x(1)
        qc.cx(0, 1)

        # If the filter_function does not look at conditions, we should collect all
        # gates into the block.
        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["x", "cx"],
            split_blocks=False,
            min_block_size=1,
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 7)

        # If the filter_function does look at conditions, we should not collect the middle
        # conditional gate (note that x(1) following the measure is collected into the first
        # block).
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["x", "cx"] and not getattr(node.op, "condition", None),
            split_blocks=False,
            min_block_size=1,
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 4)
        self.assertEqual(len(blocks[1]), 2)

    def test_multiple_collection_methods(self):
        """Test that block collection allows to collect blocks using several different
        filter functions."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.swap(1, 4)
        qc.swap(4, 3)
        qc.z(0)
        qc.z(1)
        qc.z(2)
        qc.z(3)
        qc.z(4)
        qc.swap(3, 4)
        qc.cx(0, 3)
        qc.cx(0, 4)

        block_collector = BlockCollector(circuit_to_dag(qc))
        linear_blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "swap"],
            split_blocks=False,
            min_block_size=1,
        )
        cx_blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx"],
            split_blocks=False,
            min_block_size=1,
        )
        swapz_blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["swap", "z"],
            split_blocks=False,
            min_block_size=1,
        )

        # We should end up with two linear blocks
        self.assertEqual(len(linear_blocks), 2)
        self.assertEqual(len(linear_blocks[0]), 4)
        self.assertEqual(len(linear_blocks[1]), 3)

        # We should end up with two cx blocks
        self.assertEqual(len(cx_blocks), 2)
        self.assertEqual(len(cx_blocks[0]), 2)
        self.assertEqual(len(cx_blocks[1]), 2)

        # We should end up with one swap,z blocks
        self.assertEqual(len(swapz_blocks), 1)
        self.assertEqual(len(swapz_blocks[0]), 8)

    def test_min_block_size(self):
        """Test that the option min_block_size for collecting blocks works correctly."""

        # original circuit
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)

        block_collector = BlockCollector(circuit_to_dag(circuit))

        # When min_block_size = 1, we should obtain 3 linear blocks
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "swap"],
            split_blocks=False,
            min_block_size=1,
        )
        self.assertEqual(len(blocks), 3)

        # When min_block_size = 2, we should obtain 2 linear blocks
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "swap"],
            split_blocks=False,
            min_block_size=2,
        )
        self.assertEqual(len(blocks), 2)

        # When min_block_size = 3, we should obtain 1 linear block
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "swap"],
            split_blocks=False,
            min_block_size=3,
        )
        self.assertEqual(len(blocks), 1)

        # When min_block_size = 4, we should obtain no linear blocks
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "swap"],
            split_blocks=False,
            min_block_size=4,
        )
        self.assertEqual(len(blocks), 0)

    def test_split_blocks(self):
        """Test that splitting blocks of nodes into sub-blocks works correctly."""

        # original circuit is linear
        circuit = QuantumCircuit(5)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.cx(2, 0)
        circuit.cx(0, 3)
        circuit.swap(3, 2)
        circuit.swap(4, 1)

        block_collector = BlockCollector(circuit_to_dag(circuit))

        # If we do not split block into sub-blocks, we expect a single linear block.
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "swap"],
            split_blocks=False,
            min_block_size=2,
        )
        self.assertEqual(len(blocks), 1)

        # If we do split block into sub-blocks, we expect two linear blocks:
        # one over qubits {0, 2, 3}, and another over qubits {1, 4}.
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "swap"],
            split_blocks=True,
            min_block_size=2,
        )
        self.assertEqual(len(blocks), 2)

    def test_do_not_split_blocks(self):
        """Test that splitting blocks of nodes into sub-blocks works correctly."""

        # original circuit is linear
        circuit = QuantumCircuit(5)
        circuit.cx(0, 3)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.swap(4, 2)

        block_collector = BlockCollector(circuit_to_dagdependency(circuit))

        # Check that we have a single linear block
        blocks = block_collector.collect_all_matching_blocks(
            lambda node: node.op.name in ["cx", "swap"],
            split_blocks=True,
            min_block_size=1,
        )
        self.assertEqual(len(blocks), 1)


if __name__ == "__main__":
    unittest.main()
