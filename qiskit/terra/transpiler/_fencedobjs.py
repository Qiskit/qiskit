# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Fenced objects are wraps for raising TranspilerAccessError when they are modified."""

from ._transpilererror import TranspilerAccessError


class FencedObject():
    """ Given an instance and a list of attributes to fence, raises a TranspilerAccessError when one
    of these attributes is accessed."""

    def __init__(self, instance, attributes_to_fence):
        self._wrapped = instance
        self._attributes_to_fence = attributes_to_fence

    def __getattribute__(self, name):
        object.__getattribute__(self, '_check_if_fenced')(name)
        return getattr(object.__getattribute__(self, '_wrapped'), name)

    def __getitem__(self, key):
        object.__getattribute__(self, '_check_if_fenced')('__getitem__')
        return object.__getattribute__(self, '_wrapped')[key]

    def __setitem__(self, key, value):
        object.__getattribute__(self, '_check_if_fenced')('__setitem__')
        object.__getattribute__(self, '_wrapped')[key] = value

    def _check_if_fenced(self, name):
        """
        Checks if the attribute name is in the list of attributes to protect. If so, raises
        TranspilerAccessError.

        Args:
            name (string): the attribute name to check

        Raises:
            TranspilerAccessError: when name is the list of attributes to protect.
        """
        if name in object.__getattribute__(self, '_attributes_to_fence'):
            raise TranspilerAccessError("The fenced %s has the property %s protected" %
                                        (type(object.__getattribute__(self, '_wrapped')), name))


class FencedPropertySet(FencedObject):
    """ A property set that cannot be written (via __setitem__) """
    def __init__(self, property_set_instance):
        super().__init__(property_set_instance, ['__setitem__'])


class FencedDAGCircuit(FencedObject):
    """ A dag circuit that cannot be modified (via _remove_op_node) """
    # FIXME: add more fenced methods of the dag after dagcircuit rewrite
    def __init__(self, dag_circuit_instance):
        super().__init__(dag_circuit_instance, ['_remove_op_node'])
