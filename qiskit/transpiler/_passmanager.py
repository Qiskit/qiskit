# -*- coding: utf-8 -*-
# pylint: disable=redefined-builtin

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module implements a passmanager
"""


class PassManager():
    """PassManager class for the transpiler.

    A PassManager instance is responsible for launching the requisite
    analysis and transformation passes on the quantum circuit, and to
    do this correctly & efficiently
    (i.e. keep track of dependencies between passes)
    """
    def __init__(self):
        """Initialize an empty PassManager object
        (with no passes scheduled).
        """
        self._passes = []

    def add_pass(self, pass_):
        """Schedule a pass in the passmanager."""
        self._passes.append(pass_)

    def passes(self):
        """Return list of passes scheduled."""
        return self._passes
