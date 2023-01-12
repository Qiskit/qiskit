# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=redefined-builtin

"""Objects to represent the information at a node in the DAGCircuit."""

import warnings
from typing import Iterable

from qiskit.circuit import Qubit, Clbit


def _condition_as_indices(operation, bit_indices):
    cond = getattr(operation, "condition", None)
    if cond is None:
        return None
    bits, value = cond
    indices = [bit_indices[bits]] if isinstance(bits, Clbit) else [bit_indices[x] for x in bits]
    return indices, value


class DAGNode:
    """Parent class for DAGOpNode, DAGInNode, and DAGOutNode."""

    __slots__ = ["_node_id"]

    def __init__(self, nid=-1):
        """Create a node"""
        self._node_id = nid

    def __lt__(self, other):
        return self._node_id < other._node_id

    def __gt__(self, other):
        return self._node_id > other._node_id

    def __str__(self):
        # TODO is this used anywhere other than in DAG drawing?
        # needs to be unique as it is what pydot uses to distinguish nodes
        return str(id(self))

    @staticmethod
    def semantic_eq(node1, node2, bit_indices1=None, bit_indices2=None):
        """
        Check if DAG nodes are considered equivalent, e.g., as a node_match for nx.is_isomorphic.

        Args:
            node1 (DAGOpNode, DAGInNode, DAGOutNode): A node to compare.
            node2 (DAGOpNode, DAGInNode, DAGOutNode): The other node to compare.
            bit_indices1 (dict): Dictionary mapping Bit instances to their index
                within the circuit containing node1
            bit_indices2 (dict): Dictionary mapping Bit instances to their index
                within the circuit containing node2

        Return:
            Bool: If node1 == node2
        """
        if bit_indices1 is None or bit_indices2 is None:
            warnings.warn(
                "DAGNode.semantic_eq now expects two bit-to-circuit index "
                "mappings as arguments. To ease the transition, these will be "
                "pre-populated based on the values found in Bit.index and "
                "Bit.register. However, this behavior is deprecated and a future "
                "release will require the mappings to be provided as arguments.",
                DeprecationWarning,
            )

            bit_indices1 = {arg: arg for arg in node1.qargs + node1.cargs}
            bit_indices2 = {arg: arg for arg in node2.qargs + node2.cargs}

        if isinstance(node1, DAGOpNode) and isinstance(node2, DAGOpNode):
            node1_qargs = [bit_indices1[qarg] for qarg in node1.qargs]
            node1_cargs = [bit_indices1[carg] for carg in node1.cargs]

            node2_qargs = [bit_indices2[qarg] for qarg in node2.qargs]
            node2_cargs = [bit_indices2[carg] for carg in node2.cargs]

            # For barriers, qarg order is not significant so compare as sets
            if node1.op.name == node2.op.name and node1.name in {"barrier", "swap"}:
                return set(node1_qargs) == set(node2_qargs)

            return (
                node1_qargs == node2_qargs
                and node1_cargs == node2_cargs
                and (
                    _condition_as_indices(node1.op, bit_indices1)
                    == _condition_as_indices(node2.op, bit_indices2)
                )
                and node1.op == node2.op
            )
        if (isinstance(node1, DAGInNode) and isinstance(node2, DAGInNode)) or (
            isinstance(node1, DAGOutNode) and isinstance(node2, DAGOutNode)
        ):
            return bit_indices1.get(node1.wire, None) == bit_indices2.get(node2.wire, None)

        return False


class DAGOpNode(DAGNode):
    """Object to represent an Instruction at a node in the DAGCircuit."""

    __slots__ = ["op", "qargs", "cargs", "sort_key"]

    def __init__(self, op, qargs: Iterable[Qubit] = (), cargs: Iterable[Clbit] = ()):
        """Create an Instruction node"""
        super().__init__()
        self.op = op
        self.qargs = tuple(qargs)
        self.cargs = tuple(cargs)
        self.sort_key = str(self.qargs)

    @property
    def name(self):
        """Returns the Instruction name corresponding to the op for this node"""
        return self.op.name

    @name.setter
    def name(self, new_name):
        """Sets the Instruction name corresponding to the op for this node"""
        self.op.name = new_name

    def __repr__(self):
        """Returns a representation of the DAGOpNode"""
        return f"DAGOpNode(op={self.op}, qargs={self.qargs}, cargs={self.cargs})"


class DAGInNode(DAGNode):
    """Object to represent an incoming wire node in the DAGCircuit."""

    __slots__ = ["wire", "sort_key"]

    def __init__(self, wire):
        """Create an incoming node"""
        super().__init__()
        self.wire = wire
        # TODO sort_key which is used in dagcircuit.topological_nodes
        # only works as str([]) for DAGInNodes. Need to figure out why.
        self.sort_key = str([])

    def __repr__(self):
        """Returns a representation of the DAGInNode"""
        return f"DAGInNode(wire={self.wire})"


class DAGOutNode(DAGNode):
    """Object to represent an outgoing wire node in the DAGCircuit."""

    __slots__ = ["wire", "sort_key"]

    def __init__(self, wire):
        """Create an outgoing node"""
        super().__init__()
        self.wire = wire
        # TODO sort_key which is used in dagcircuit.topological_nodes
        # only works as str([]) for DAGOutNodes. Need to figure out why.
        self.sort_key = str([])

    def __repr__(self):
        """Returns a representation of the DAGOutNode"""
        return f"DAGOutNode(wire={self.wire})"
