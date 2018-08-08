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

    def __init__(self, basis_gates=None, initial_layout=None):
        """
        Initialize an empty PassManager object (with no passes scheduled).

        Args:
            basis_gates (str): basis gates for the quantum circuit.
            initial_layout (dict): A mapping of qubit to qubit::

                              {
                                ("q", start(int)): ("q", final(int)),
                                ...
                              }
                              eg.
                              {
                                ("q", 0): ("q", 0),
                                ("q", 1): ("q", 1),
                                ("q", 2): ("q", 2),
                                ("q", 3): ("q", 3)
                              }
        """
        self._passes = []
        self.shared_memory = {'basis': basis_gates.split(',') if basis_gates else [],
                              'layout': initial_layout or {}}

    def add_pass(self, pass_):
        """Schedule a pass in the passmanager."""
        pass_.shared_memory = self.shared_memory
        self._passes.append(pass_)

    def passes(self):
        """Return list of passes scheduled."""
        return self._passes
