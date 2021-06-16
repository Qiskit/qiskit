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

from qiskit.exceptions import QiskitError


class DAGNode:
    """DEPRECATED as of 0.18.0. Object to represent the information at a node in
    the DAGCircuit. It is used as the return value from `*_nodes()` functions and can
    be supplied to functions that take a node.
    """

    __slots__ = ["type", "op", "_qargs", "cargs", "wire", "sort_key", "_node_id"]

    def __init__(self, type=None, op=None, name=None, qargs=None, cargs=None, wire=None, nid=-1):
        """Create a node"""
        if type is not None:
            warnings.warn(
                "The DAGNode 'type' attribute is deprecated as of 0.18.0 and "
                "will be removed no earlier than 3 months after the release date. "
                "Use OpNode, InNode, or OutNode instead of DAGNode.",
                DeprecationWarning,
                2,
            )
            self.type = type
        if op is not None:
            warnings.warn(
                "The DAGNode 'op' attribute is deprecated as of 0.18.0 and "
                "will be removed no earlier than 3 months after the release date."
                "Use OpNode instead of DAGNode.",
                DeprecationWarning,
                2,
            )
            self.op = op
        if name is not None:
            warnings.warn(
                "The DAGNode 'name' attribute is deprecated as of 0.18.0 and "
                "will be removed no earlier than 3 months after the release date. "
                "You can use 'DAGNode.op.name' if the DAGNode is of type 'op'.",
                DeprecationWarning,
                2,
            )
        self._qargs = qargs if qargs is not None else []
        self.cargs = cargs if cargs is not None else []
        if wire is not None:
            warnings.warn(
                "The DAGNode 'wire' attribute is deprecated as of 0.18.0 and "
                "will be removed no earlier than 3 months after the release date."
                "Use InNode or OutNode instead of DAGNode.",
                DeprecationWarning,
                2,
            )
            self.wire = wire
        self._node_id = nid
        self.sort_key = str(self._qargs)

    @property
    def name(self):
        """Returns the Instruction name corresponding to the op for this node"""
        if self.type and self.type == "op":
            return self.op.name
        return None

    @name.setter
    def name(self, name):
        """Sets the Instruction name corresponding to the op for this node"""
        if self.type and self.type == "op":
            self.op.name = name

    @property
    def condition(self):
        """Returns the condition of the node.op"""
        if not self.type or self.type != "op":
            raise QiskitError("The node %s is not an op node" % (str(self)))
        warnings.warn(
            "The DAGNode 'condition' attribute is deprecated as of 0.18.0 and "
            "will be removed no earlier than 3 months after the release date. "
            "You can use 'DAGNode.op.condition' if the DAGNode is of type 'op'.",
            DeprecationWarning,
            2,
        )
        return self._op.condition

    @condition.setter
    def condition(self, new_condition):
        """Sets the node.condition which sets the node.op.condition."""
        if not self.type or self.type != "op":
            raise QiskitError("The node %s is not an op node" % (str(self)))
        warnings.warn(
            "The DAGNode 'condition' attribute is deprecated as of 0.18.0 and "
            "will be removed no earlier than 3 months after the release date. "
            "You can use 'DAGNode.op.condition' if the DAGNode is of type 'op'.",
            DeprecationWarning,
            2,
        )
        self._op.condition = new_condition

    @property
    def qargs(self):
        """
        Returns list of Qubit, else an empty list.
        """
        return self._qargs

    @qargs.setter
    def qargs(self, new_qargs):
        """Sets the qargs to be the given list of qargs."""
        self._qargs = new_qargs
        self.sort_key = str(new_qargs)

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
            node1 (OpNode, InNode, OutNode): A node to compare.
            node2 (OpNode, InNode, OutNode): The other node to compare.
            bit_indices1 (dict): Dictionary mapping Bit instances to their index
                within the circuit containing node1
            bit_indices2 (dict): Dictionary mapping Bit instances to their index
                within the circuit containing node2

        Return:
            Bool: If node1 == node2
        """
        if bit_indices1 is None or bit_indices2 is None:
            warnings.warn(
                "DAGNodeP.semantic_eq now expects two bit-to-circuit index "
                "mappings as arguments. To ease the transition, these will be "
                "pre-populated based on the values found in Bit.index and "
                "Bit.register. However, this behavior is deprecated and a future "
                "release will require the mappings to be provided as arguments.",
                DeprecationWarning,
            )

            bit_indices1 = {arg: arg for arg in node1.qargs + node1.cargs}
            bit_indices2 = {arg: arg for arg in node2.qargs + node2.cargs}

        if isinstance(node1, OpNode) and isinstance(node2, OpNode):
            node1_qargs = [bit_indices1[qarg] for qarg in node1.qargs]
            node1_cargs = [bit_indices1[carg] for carg in node1.cargs]

            node2_qargs = [bit_indices2[qarg] for qarg in node2.qargs]
            node2_cargs = [bit_indices2[carg] for carg in node2.cargs]

            # For barriers, qarg order is not significant so compare as sets
            if "barrier" == node1.op.name == node2.op.name:
                return set(node1_qargs) == set(node2_qargs)

            if node1.op.name == node2.op.name:
                if node1_qargs == node2_qargs:
                    if node1_cargs == node2_cargs:
                        if node1.op.condition == node2.op.condition:
                            return True
        elif (isinstance(node1, InNode) and isinstance(node2, InNode)) or (
            isinstance(node1, OutNode) and isinstance(node2, OutNode)
        ):
            if bit_indices1.get(node1.wire, None) == bit_indices2.get(node2.wire, None):
                return True

        return False


class OpNode(DAGNode):
    """Object to represent an Instruction at a node in the DAGCircuit."""

    __slots__ = ["op"]

    def __init__(self, op, qargs=None, cargs=None):
        """Create an Instruction node"""
        self.op = op
        super().__init__(qargs=qargs, cargs=cargs)

    @property
    def name(self):
        """Returns the Instruction name corresponding to the op for this node"""
        return self.op.name

    @name.setter
    def name(self, new_name):
        """Sets the Instruction name corresponding to the op for this node"""
        self.op.name = new_name


class InNode(DAGNode):
    """Object to represent an incoming wire node in the DAGCircuit."""

    __slots__ = ["wire"]

    def __init__(self, wire):
        """Create an incoming node"""
        self.wire = wire
        super().__init__()


class OutNode(DAGNode):
    """Object to represent an outgoing wire node in the DAGCircuit."""

    __slots__ = ["wire"]

    def __init__(self, wire):
        """Create an outgoing node"""
        self.wire = wire
        super().__init__()
