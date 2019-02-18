# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unitary gate.
"""
from .instruction import Instruction


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name, params, circuit=None):
        """Create a new composite gate.

        name = instruction name string
        params = list of real parameters (will be converted to symbolic)
        circuit = QuantumCircuit containing this gate
        """
        self._decompositions = None
        self._matrix_rep = None

        super().__init__(name, params, circuit)

    def inverse(self):
        """Invert this gate."""
        raise NotImplementedError("inverse not implemented")

    def decompositions(self):
        """Returns a list of possible decompositions. """
        if self._decompositions is None:
            self._define_decompositions()
        return self._decompositions

    def _define_decompositions(self):
        """ Populates self.decompositions with way to decompose this gate"""
        raise NotImplementedError("No decomposition rules defined for %s" % self.name)

    @property
    def matrix_rep(self):
        """Return matrix representation if it exists else None"""
        return self._matrix_rep

    @matrix_rep.setter
    def matrix_rep(self, matrix):
        """Set matrix representation"""
        self._matrix_rep = matrix
