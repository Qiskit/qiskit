# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Fenced objects are Object Proxies for raising  TranspilerAccessError when they are modified."""

from ._transpilererror import TranspilerAccessError

class FencedObject():
    def __init__(self, instance, attributes_to_fence):
        self._wrapped = instance
        self._attributes_to_fence = attributes_to_fence

    def __getattribute__(self, name):
        object.__getattribute__(self, '_check_if_fenced')(name)
        return getattr(object.__getattribute__(self,'_wrapped'), name)

    def __getitem__(self, key):
        object.__getattribute__(self, '_check_if_fenced')('__getitem__')
        return object.__getattribute__(self,'_wrapped')[key]

    def __setitem__(self, key, value):
        object.__getattribute__(self, '_check_if_fenced')('__setitem__')
        object.__getattribute__(self, '_wrapped')[key] = value

    def _check_if_fenced(self, name):
        if name in object.__getattribute__(self,'_attributes_to_fence'):
            raise TranspilerAccessError("The fenced %s has the property %s protected" %
                                        (type(object.__getattribute__(self, '_wrapped')),name))

class FencedPropertySet(FencedObject):
    def __init__(self, PropertySetInstance):
        super().__init__(PropertySetInstance, ['__setitem__'])

class FencedDAGCircuit(FencedObject):
    def __init__(self, DagCircuitInstance):
        super().__init__(DagCircuitInstance, ['_remove_op_node'])