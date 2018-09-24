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
    Enforces the creation of some fields in the pass while allowing passes to override __init__
    """

    def __call__(cls, *args, **kwargs):
        """ Called with __init__"""
        obj = type.__call__(cls, *args, **kwargs)
        obj._hash = hash(cls.__name__ +
                         str(args) +
                         str(OrderedDict(sorted(kwargs.items(), key=lambda t: t[0]))))
        obj._defaults = {
            "idempotence": cls.idempotence,
            "ignore_requires": False,
            "ignore_preserves": False,

        }
        obj._settings = {}
        return obj


class BasePass(metaclass=MetaPass):
    """Base class for transpiler passes."""

    requires = []  # List of passes that requires
    preserves = []  # List of passes that preserves
    idempotence = True  # By default, passes are idempotent
    ignore_preserves = False
    ignore_requires = False
    max_iteration = 1000

    property_set = {}
    _hash = None

    def __init__(self):
        pass

    @property
    def name(self):
        """ The name of the pass. """
        return self.__class__.__name__

    def __eq__(self, other):
        return self._hash == hash(other)

    def __hash__(self):
        return self._hash

    @abstractmethod
    def run(self, dag):
        """
        Run a pass on the DAGCircuit. This is implemented by the pass developer.
        Args:
            dag (DAGCircuit): The dag in which the pass is run.
        Raises:
            NotImplementedError: Because YOU have to implement this :)
        """
        raise NotImplementedError

    @property
    def is_idempotent(self):
        """ A pass is idempotent when run several time is equivalent to run it once. In math terms,
        when `run(run(dag)) == run(dag)`. By default, the passes are idempotent. This allows to
        optimize the transpiler process when sequence of passes are repeated or when passes are
        preserves.
        """
        return self.idempotence

    @property
    def is_TransformationPass(self):  # pylint: disable=invalid-name
        """ If the pass is a TransformationPass, that means that the pass can manipulate the DAG,
        but cannot modified the property set (but it can be read). """
        return isinstance(self, TransformationPass)

    @property
    def is_AnalysisPass(self):  # pylint: disable=invalid-name
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
