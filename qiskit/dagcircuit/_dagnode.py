# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
   Object to represent the information at a node in the DAGCircuit
"""


class DAGNode:
    """
    Object to represent the information at a node in the DAGCircuit

    It is used as the return value from *_nodes() functions and can
    be supplied to functions that take a node.

    """
    def __init__(self, data_dict, node_id=None):
        """ Create a node """
        self.data_dict = data_dict
        self.node_id = node_id

    @property
    def type(self):
        """ Returns a str which is the type of the node else None"""
        return self.data_dict.get('type')

    @property
    def op(self):
        """ Returns the Gate object corresponding to the op for the node else None"""
        return self.data_dict.get('op')

    @property
    def name(self):
        """ Returns a str which is the name of the node else None"""
        return self.data_dict.get('name')

    @property
    def qargs(self):
        """
        Returns list of (QuantumRegister, int) tuples where the int is the index
        of the qubit else an empty list
        """
        return self.data_dict.get('qargs', [])

    @property
    def cargs(self):
        """
        Returns list of (ClassicalRegister, int) tuples where the int is the index
        of the cbit else an empty list
        """
        return self.data_dict.get('cargs', [])

    @property
    def condition(self):
        """ Returns a tuple (ClassicalRegister, int) where the int is the
        value of the condition else None"""
        return self.data_dict.get('condition')

    @property
    def wire(self):
        """
        Returns (Register, int) tuple where the int is the index of
        the wire else None
        """
        return self.data_dict.get('wire')

    def __eq__(self, other):

        # For barriers, qarg order is not significant so compare as sets
        if 'barrier' == self.name == other.name:
            node1_qargs = set(self.qargs)
            node2_qargs = set(other.qargs)

            if node1_qargs != node2_qargs:
                return False

            # qargs must be equal, so remove them from the dict then compare
            copy_self = {k: v for (k, v) in self.data_dict.items() if k != 'qargs'}
            copy_other = {k: v for (k, v) in other.data_dict.items() if k != 'qargs'}

            return copy_self == copy_other

        return self.data_dict == other.data_dict

    def to_tuple(self):
        """
        Return a tuple of the node_id and data_dict
        Required when adding nodes to a multigraph
        """
        return self.node_id, self.data_dict

    def __hash__(self):
        """Needed for ancestors function, which returns a set
        to be in a set requires the object to be hashable
        """
        return hash((self.node_id,
                     self.type,
                     str(self.op),
                     self.name,
                     tuple(self.qargs),
                     tuple(self.cargs),
                     self.condition,
                     self.wire))

    def __str__(self):
        return str(self.data_dict)
