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
        self.node_id = node_id or None

        self.type = data_dict['type'] or None
        self.op = data_dict['op'] or None
        self.name = data_dict['name'] or None
        self.qargs = data_dict['qargs'] or []
        self.cargs = data_dict['cargs'] or []
        self.condition = data_dict['condition'] or None

        self.data_dict = {
            'type': self.type,
            'op': self.op,
            'name': self.name,
            'qargs': self.qargs,
            'cargs': self.cargs,
            'condition': self.condition
        }

    def to_tuple(self):
        """ Return a tuple of the node_id and the node data"""
        return self.node_id, self.data_dict
