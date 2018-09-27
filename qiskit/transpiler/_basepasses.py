# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module implements the base pass."""

from abc import abstractmethod
from collections import OrderedDict


class MetaPass(type):
    """
    Enforces the creation of some fields in the pass
    while allowing passes to override __init__
    """

    def __call__(cls, *args, **kwargs):
        """ Called with __init__"""
        obj = type.__call__(cls, *args, **kwargs)
        obj._args = str(args)
        _kwargs = OrderedDict(sorted(kwargs.items(), key=lambda t: t[0]))
        obj._kwargs = '('+', '.join(["%s=%s" % (i, j) for i, j in _kwargs.items()])+')'
        obj._hash = hash(obj.__repr__())
        obj._defaults = {
            "idempotence": True,
            "ignore_requires": False,
            "ignore_preserves": False,

        }
        obj._settings = {}
        return obj


class BasePass(metaclass=MetaPass):
    """Base class for transpiler passes."""

    def __init__(self):
        self.requires = []  # List of passes that requires
        self.preserves = []  # List of passes that preserves
        self.idempotence = True  # By default, passes are idempotent
        self.ignore_preserves = False
        self.ignore_requires = False
        self.max_iteration = 1000

        self.property_set = {}
        self._hash = None

    def name(self):
        """ The name of the pass. """
        return self.__class__.__name__

    def __repr__(self):
        return self.name()+self._args+self._kwargs

    def __eq__(self, other):
        """
        Two passes are equal if and only if they are of the same class,
        and have the same arguments.
        """
        return self._hash == hash(other)

    def __hash__(self):
        return self._hash

    @abstractmethod
    def run(self, dag):
        """
        Run a pass on the DAGCircuit. This is implemented by the pass developer.
        Args:
            dag (DAGCircuit): the dag on which the pass is run.
        Raises:
            NotImplementedError: when this is left unimplemented for a pass.
        """
        raise NotImplementedError

    @property
    def is_transformation_pass(self):
        """ If the pass is a TransformationPass, that means that the pass can manipulate the DAG,
        but cannot modify the property set (but it can be read). """
        return isinstance(self, TransformationPass)

    @property
    def is_analysis_pass(self):
        """ If the pass is an AnalysisPass, that means that the pass can analyze the DAG and write
        the results of that analysis in the property set. Modifications on the DAG are not allowed
        by this kind of pass. """
        return isinstance(self, AnalysisPass)


class AnalysisPass(BasePass):  # pylint: disable=abstract-method
    """ An analysis pass: change property set, not DAG. """
    pass


class TransformationPass(BasePass):  # pylint: disable=abstract-method
    """ A transformation pass: change DAG, not property set. """
    pass
