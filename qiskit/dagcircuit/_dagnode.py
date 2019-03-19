# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
   Object to represent the information at a node in the DAGCircuit
"""
from qiskit.exceptions import QiskitError


class DAGNode:
    """
    Object to represent the information at a node in the DAGCircuit

    It is used as the return value from *_nodes() functions and can
    be supplied to functions that take a node.

    """
    def __init__(self, data_dict, nid=-1):
        """ Create a node """
        self._node_id = nid
        self.data_dict = data_dict

    @property
    def type(self):
        """ Returns a str which is the type of the node else None"""
        return self.data_dict.get('type')

    @property
    def op(self):
        """ Returns the Instruction object corresponding to the op for the node else None"""
        if 'type' not in self.data_dict or self.data_dict['type'] != 'op':
            raise QiskitError("The node %s is not an op node" % (str(self)))
        return self.data_dict.get('op')

    @property
    def name(self):
        """ Returns a str which is the name of the node else None"""
        return self.data_dict.get('name')

    @name.setter
    def name(self, new_name):
        """ Sets the name of the node to be the given value"""
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
        """ Sets the qargs to be the given list of qargs"""
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
        """ Returns a tuple (ClassicalRegister, int) where the int is the
        value of the condition else None"""
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

    def __eq__(self, other):

        # if other is None
        if not other:
            return False

        if isinstance(other, int):
            return other == self._node_id

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
