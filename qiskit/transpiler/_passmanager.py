# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module implements a passmanager
"""


class PassManager():
    """PassManager class for the transpiler."""
    def __init__(self, configuration=None):
        """Base class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            configuration (dict): configuration dictionary
        """
        self._configuration = configuration
        self._resources = {}
        self._passes = []

    def add_pass(self, pass_):
        """Schedule a pass in the passmanager."""
        self._passes.append(pass_)

    def passes(self):
        """Return list of passes scheduled."""
        return self._passes

    @property
    def configuration(self):
        """Return passmanager configuration"""
        return self._configuration
