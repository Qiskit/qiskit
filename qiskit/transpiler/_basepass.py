# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module implements the base pass.
"""
from abc import ABC, abstractmethod


class BasePass(ABC):
    """Base class for transpiler passes."""
    @abstractmethod
    def __init__(self):
        """Base class for passes.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Raises:
            FileNotFoundError: if referenced passes are not registered.
        """
        # list of other passes required before this pass can run. this instructs the
        # passmanger to ensure those other passes have run (and not been invalidated)
        # default: the pass is standalone and requires no other pass before it
        self._requires = None

        # list of other passes preserved by this pass. this hints at the passmanager
        # that it is safe to assume those other passes do not need to be rerun
        # default: the pass preseves no other pass; invalidates them all (conservative)
        self._preserves = None

    @abstractmethod
    def run(self, dag):
        """Run a pass on the DAGCircuit.
        """
        return dag

    @property
    def requires(self):
        """Return ``requires`` list"""
        return self._requires

    @property
    def preserves(self):
        """Return ``preserves`` list"""
        return self._preserves
