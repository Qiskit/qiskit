# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module implements the base pass."""

from abc import abstractmethod
from collections.abc import Hashable
from inspect import signature
from ._propertyset import PropertySet


class MetaPass(type):
    """
    Enforces the creation of some fields in the pass
    while allowing passes to override __init__
    """

    def __call__(cls, *args, **kwargs):
        if '_pass_cache' not in cls.__dict__.keys():
            cls._pass_cache = {}
        args, kwargs = cls.normalize_parameters(*args, **kwargs)
        hash_ = hash(MetaPass._freeze_init_parameters(cls.__init__, args, kwargs))
        if hash_ not in cls._pass_cache:
            new_pass = type.__call__(cls, *args, **kwargs)
            cls._pass_cache[hash_] = new_pass
        return cls._pass_cache[hash_]

    @staticmethod
    def _freeze_init_parameters(init_method, args, kwargs):
        self_guard = object()
        init_signature = signature(init_method)
        bound_signature = init_signature.bind(self_guard, *args, **kwargs)
        arguments = []
        for name, value in bound_signature.arguments.items():
            if value == self_guard:
                continue
            if isinstance(value, Hashable):
                arguments.append((name, type(value), value))
            else:
                arguments.append((name, type(value), repr(value)))
        return frozenset(arguments)


class BasePass(metaclass=MetaPass):
    """Base class for transpiler passes."""

    def __init__(self):
        self.requires = []  # List of passes that requires
        self.preserves = []  # List of passes that preserves
        self.property_set = PropertySet()  # This pass's pointer to the pass manager's property set.

    @classmethod
    def normalize_parameters(cls, *args, **kwargs):
        """
        Because passes with the same args/kwargs are considered the same, this method allows to
        modify the args/kargs to respect that identity.
        Args:
            *args: args to normalize
            **kwargs: kwargs to normalize

        Returns:
            tuple: normalized (list(args), dict(kwargs))
        """
        return args, kwargs

    def name(self):
        """ The name of the pass. """
        return self.__class__.__name__

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
