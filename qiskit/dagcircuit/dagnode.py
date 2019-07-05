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

"""Object to represent the information at a node in the DAGCircuit
"""

from qiskit.exceptions import QiskitError


class DAGNode:
    """
    Object to represent the information at a node in the DAGCircuit

    It is used as the return value from *_nodes() functions and can
    be supplied to functions that take a node.
    """

    def __init__(self, data_dict, nid=-1):
        """Create a node """
        self._node_id = nid
        self.data_dict = data_dict

    @property
    def type(self):
        """Returns a str which is the type of the node else None"""
        return self.data_dict.get('type')

    @property
    def op(self):
        """Returns the Instruction object corresponding to the op for the node else None"""
        if 'type' not in self.data_dict or self.data_dict['type'] != 'op':
            raise QiskitError("The node %s is not an op node" % (str(self)))
        return self.data_dict.get('op')

    @property
    def name(self):
        """Returns a str which is the name of the node else None"""
        return self.data_dict.get('name')

    @name.setter
    def name(self, new_name):
        """Sets the name of the node to be the given value"""
        self.data_dict['name'] = new_name

    @property
    def qargs(self):
        """
        Returns list of (QuantumRegister, int) tuples where the int is the index
        of the qubit else an empty list
        """
        return self.data_dict.get('qargs', [])

    @qargs.setter
    def qargs(self, new_qargs):
        """Sets the qargs to be the given list of qargs"""
        self.data_dict['qargs'] = new_qargs

    @property
    def cargs(self):
        """
        Returns list of (ClassicalRegister, int) tuples where the int is the index
        of the cbit else an empty list
        """
        return self.data_dict.get('cargs', [])

    @property
    def condition(self):
        """
        Returns a tuple (ClassicalRegister, int) where the int is the
        value of the condition else None
        """
        return self.data_dict.get('condition')

    @property
    def wire(self):
        """
        Returns (Register, int) tuple where the int is the index of
        the wire else None
        """
        if self.data_dict['type'] not in ['in', 'out']:
            raise QiskitError('The node %s is not an input/output node' % str(self))
        return self.data_dict.get('wire')

    def __lt__(self, other):
        return self._node_id < other._node_id

    def __gt__(self, other):
        return self._node_id > other._node_id

    def __hash__(self):
        """Needed for ancestors function, which returns a set
        to be in a set requires the object to be hashable
        """
        return hash(id(self))

    def __str__(self):
        # TODO is this used anywhere other than in DAG drawing?
        # needs to be unique as it is what pydot uses to distinguish nodes
        return str(id(self))

    def pop(self, val):
        """Remove the provided value from the dictionary"""
        del self.data_dict[val]

    @staticmethod
    def semantic_eq(node1, node2):
        """
        Check if DAG nodes are considered equivalent, e.g. as a node_match for nx.is_isomorphic.

        Args:
            node1 (DAGNode): A node to compare.
            node2 (DAGNode): The other node to compare.

        Return:
            Bool: If node1 == node2
        """
        # For barriers, qarg order is not significant so compare as sets
        if 'barrier' == node1.name == node2.name:
            return set(node1.qargs) == set(node2.qargs)
        return node1.data_dict == node2.data_dict
