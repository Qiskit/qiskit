# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Fenced objects are Object Proxies for raising  TranspilerAccessError when they are modified."""

import wrapt

class FencedPropertySet(wrapt.ObjectProxy):
    """ A property set that cannot be written (via __setitem__) """
    def __setitem__(self, _key, _item):
        raise TranspilerAccessError("A TransformationPass should not modify property_set.")

class FencedDAGCircuit(wrapt.ObjectProxy):
    """ A dag circuit that cannot be modified. """
    def _remove_op_node(self, *_args, **_kwargs):
        raise TranspilerAccessError("An AnalysisPass should not modify DAGCircuit: _remove_op_node"
                                    "forbidden")

class TranspilerAccessError(Exception):
    """ Exception of access error in the transpiler passes. """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
