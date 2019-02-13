# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


class DAGNode:

    def __init__(self, node_id=None, data_dict=None):
        """ Create a node """
        self.node_id = node_id or None

        self.data_dict = {
            'type': self.type,
            'op': self.op,
            'name': self.name,
            'qargs': self.qargs,
            'cargs': self.cargs,
            'condtion': self.condition
        }

        if data_dict:
            self.type = data_dict['type'] or None
            self.op = data_dict['op'] or None
            self.name = data_dict['name'] or None
            self.qargs = data_dict['qargs'] or None
            self.cargs =  data_dict['cargs'] or None
            self.condition = data_dict['condition'] or None

    def __getitem__(self, key):
        return self.data_dict[key]
