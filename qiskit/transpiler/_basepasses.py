# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module implements the base pass."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from ._transpilererror import TranspilerUnknownOption

class BasePass(ABC):
    """Base class for transpiler passes."""

    requires = []  # List of passes that requires
    preserves = []  # List of passes that preserves

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self, *args, **kwargs):
        self._hash = hash(self.name + str(sorted(args)) + str(OrderedDict(kwargs)))
        self._defaults = {
            "idempotence": True,
            "ignore_requires": False,
            "ignore_preserves": False
        }
        self._settings = {}
        super().__init__()

    def __eq__(self, other):
        return self._hash == hash(other)

    def __hash__(self):
        return self._hash

    def set(self, **kwargs):
        """
        Sets a pass. `pass_.set(arg=value) equivalent to `pass_.arg = value`
        Args:
            TranspilerUnknownOption
            **kwargs (dict): arguments=value to set
        Raises:
            dict: current settings
        """
        for option in kwargs.keys():
            if option not in self._defaults:
                raise TranspilerUnknownOption("The option %s cannot be set in the pass %s",
                                              option, self.name)
        self._settings = kwargs

    @abstractmethod
    def run(self, dag, property_set=None):
        """
        Run a pass on the DAGCircuit. This is implemented by the pass developer.
        Args:
            dag:
            property_set:
        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @property
    def idempotence(self):
        return self._settings.get('idempotence', self._defaults['idempotence'])

    @property
    def ignore_requires(self):
        return self._settings.get('ignore_requires', self._defaults['ignore_requires'])

    @property
    def ignore_preserves(self):
        return self._settings.get('ignore_preserves', self._defaults['ignore_preserves'])


    @property
    def isTransformationPass(self):
        return isinstance(self, TransformationPass)

    @property
    def isAnalysisPass(self):
        return isinstance(self, AnalysisPass)


class AnalysisPass(BasePass):
    """ An analysis pass: change property set, not DAG. """
    pass


class TransformationPass(BasePass):
    """ A transformation pass: change DAG, not property set. """
    pass
