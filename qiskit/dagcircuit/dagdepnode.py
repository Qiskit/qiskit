# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=redefined-builtin

"""Object to represent the information at a node in the DAGCircuit."""

from qiskit.exceptions import QiskitError


class DAGDepNode:
    """Object to represent the information at a node in the DAGDependency().

    It is used as the return value from `*_nodes()` functions and can
    be supplied to functions that take a node.
    """

    __slots__ = [
        "_op",
        "name",
        "_qargs",
        "cargs",
        "sort_key",
        "node_id",
        "reachable",
        "matchedwith",
        "isblocked",
        "successorstovisit",
        "qindices",
        "cindices",
    ]

    def __init__(
        self,
        op=None,
        name=None,
        qargs=(),
        cargs=(),
        reachable=None,
        matchedwith=None,
        successorstovisit=None,
        isblocked=None,
        qindices=None,
        cindices=None,
        nid=-1,
    ):

        self._op = op
        self.name = name
        self._qargs = tuple(qargs) if qargs is not None else ()
        self.cargs = tuple(cargs) if cargs is not None else ()
        self.node_id = nid
        self.sort_key = str(self._qargs)
        self.reachable = reachable
        self.matchedwith = matchedwith if matchedwith is not None else []
        self.isblocked = isblocked
        self.successorstovisit = successorstovisit if successorstovisit is not None else []
        self.qindices = qindices if qindices is not None else []
        self.cindices = cindices if cindices is not None else []

    @property
    def op(self):
        """Returns the Instruction object corresponding to the op for the node."""
        return self._op

    @op.setter
    def op(self, data):
        self._op = data

    @property
    def qargs(self):
        """
        Returns list of Qubit, else an empty list.
        """
        return self._qargs

    @qargs.setter
    def qargs(self, new_qargs):
        """Sets the qargs to be the given list of qargs."""
        self._qargs = tuple(new_qargs)
        self.sort_key = str(new_qargs)

    @staticmethod
    def semantic_eq(node1, node2):
        """
        Check if DAG nodes are considered equivalent, e.g., as a node_match for nx.is_isomorphic.

        Args:
            node1 (DAGDepNode): A node to compare.
            node2 (DAGDepNode): The other node to compare.

        Return:
            Bool: If node1 == node2
        """
        # For barriers, qarg order is not significant so compare as sets
        if "barrier" == node1.name == node2.name:
            return set(node1._qargs) == set(node2._qargs)

        if node1._op == node2._op:
            if node1.name == node2.name:
                if node1._qargs == node2._qargs:
                    if node1.cargs == node2.cargs:
                        if getattr(node1._op, "condition", None) != getattr(
                            node2._op, "condition", None
                        ):
                            return False
                    return True
        return False

    def copy(self):
        """
        Function to copy a DAGDepNode object.
        Returns:
            DAGDepNode: a copy of a DAGDepNode object.
        """

        dagdepnode = DAGDepNode()

        dagdepnode._op = self.op
        dagdepnode.name = self.name
        dagdepnode._qargs = self._qargs
        dagdepnode.cargs = self.cargs
        dagdepnode.node_id = self.node_id
        dagdepnode.sort_key = self.sort_key
        dagdepnode.reachable = self.reachable
        dagdepnode.isblocked = self.isblocked
        dagdepnode.successorstovisit = self.successorstovisit
        dagdepnode.qindices = self.qindices
        dagdepnode.cindices = self.cindices
        dagdepnode.matchedwith = self.matchedwith.copy()

        return dagdepnode
