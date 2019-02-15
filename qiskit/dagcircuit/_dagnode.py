# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
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
    def __init__(self, node_id=None, data_dict=None):
        """ Create a node """
        self.node_id = node_id

        self.data_dict = data_dict

    @property
    def type(self):
        """ Returns the type of the node else None"""
        return self.data_dict['type'] if self.data_dict and'type' in self.data_dict else None

    @property
    def op(self):
        """ Returns the op for the node else None"""
        return self.data_dict['op'] if self.data_dict and 'op' in self.data_dict else None

    @property
    def name(self):
        """ Returns the name of the node else None"""
        return self.data_dict['name'] if self.data_dict and 'name' in self.data_dict else None

    @property
    def qargs(self):
        """ Returns the qargs for the node else an empty list"""
        return self.data_dict['qargs'] if self.data_dict and 'qargs' in self.data_dict else []

    @property
    def cargs(self):
        """ Returns the cargs for the node else an empty list"""
        return self.data_dict['cargs'] if self.data_dict and 'cargs' in self.data_dict else []

    @property
    def condition(self):
        """ Returns the condition of the node else None"""
        return self.data_dict['condition'] if self.data_dict and 'condition' in self.data_dict \
            else None

    @property
    def wire(self):
        """ Returns the wire of the node else None"""
        return self.data_dict['wire'] if self.data_dict and 'wire' in self.data_dict else None

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
