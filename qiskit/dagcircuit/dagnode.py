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

# pylint: disable=redefined-builtin

"""Object to represent the information at a node in the DAGCircuit."""

import warnings

from qiskit.exceptions import QiskitError


class DAGNode:
    """Object to represent the information at a node in the DAGCircuit.

    It is used as the return value from `*_nodes()` functions and can
    be supplied to functions that take a node.
    """

    __slots__ = ['type', '_op', 'name', '_qargs', 'cargs', 'condition', '_wire',
                 'sort_key', '_node_id']

    def __init__(self, type=None, op=None, name=None, qargs=None, cargs=None,
                 condition=None, wire=None, nid=-1):
        """Create a node """
        if isinstance(type, dict):
            warnings.warn("Using data_dict as a a parameter to create a "
                          "DAGNode object is deprecated. Instead pass each "
                          "field in the dictionary as a kwarg",
                          DeprecationWarning, stacklevel=2)
            data_dict = type
            self.type = data_dict.get('type')
            self._op = data_dict.get('op')
            self.name = data_dict.get('name')
            self._qargs = data_dict.get('qargs')
            self.cargs = data_dict.get('cargs')
            if data_dict.get('condition'):
                warnings.warn("Use of condition arg is deprecated, set condition in instruction",
                              DeprecationWarning)
            if self._op:
                self._op.condition = (data_dict.get('condition') if self._op.condition is None
                                      else self._op.condition)
            self.condition = self._op.condition if self._op is not None else None
            self._wire = data_dict.get('wire')
        else:
            self.type = type
            self._op = op
            self.name = name
            self._qargs = qargs if qargs is not None else []
            self.cargs = cargs if cargs is not None else []
            if condition:
                warnings.warn("Use of condition arg is deprecated, set condition in instruction.",
                              DeprecationWarning)
            if self._op:
                self._op.condition = condition if self._op.condition is None else self._op.condition
            self.condition = self._op.condition if self._op is not None else None
            self._wire = wire
        self._node_id = nid
        self.sort_key = str(self._qargs)

    @property
    def op(self):
        """Returns the Instruction object corresponding to the op for the node, else None"""
        if not self.type or self.type != 'op':
            raise QiskitError("The node %s is not an op node" % (str(self)))
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
        self._qargs = new_qargs
        self.sort_key = str(new_qargs)

    @property
    def wire(self):
        """
        Returns the Bit object, else None.
        """
        if self.type not in ['in', 'out']:
            raise QiskitError('The node %s is not an input/output node' % str(self))
        return self._wire

    @wire.setter
    def wire(self, data):
        self._wire = data

    def __lt__(self, other):
        return self._node_id < other._node_id

    def __gt__(self, other):
        return self._node_id > other._node_id

    def __hash__(self):
        """Needed for ancestors function, which returns a set.
        To be in a set requires the object to be hashable
        """
        return hash(id(self))

    def __str__(self):
        # TODO is this used anywhere other than in DAG drawing?
        # needs to be unique as it is what pydot uses to distinguish nodes
        return str(id(self))

    @staticmethod
    def semantic_eq(node1, node2):
        """
        Check if DAG nodes are considered equivalent, e.g., as a node_match for nx.is_isomorphic.

        Args:
            node1 (DAGNode): A node to compare.
            node2 (DAGNode): The other node to compare.

        Return:
            Bool: If node1 == node2
        """
        # For barriers, qarg order is not significant so compare as sets
        if 'barrier' == node1.name == node2.name:
            return set(node1._qargs) == set(node2._qargs)
        result = False
        if node1.type == node2.type:
            if node1._op == node2._op:
                if node1.name == node2.name:
                    if node1._qargs == node2._qargs:
                        if node1.cargs == node2.cargs:
                            if node1.condition == node2.condition:
                                if node1._wire == node2._wire:
                                    result = True
        return result
