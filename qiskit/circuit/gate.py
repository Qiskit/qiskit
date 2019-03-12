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

    def __init__(self, name, num_qubits, params, circuit=None):
        """Create a new composite gate.

        name = instruction name string
        params = list of real parameters (will be converted to symbolic)
        circuit = QuantumCircuit containing this gate
        """
        # list of instructions (and their contexts) that this instruction is composed of
        # self.definition=None means opaque or fundamental
        self._definition = None
        self._matrix_rep = None

        super().__init__(name, num_qubits, 0, params, circuit)

    def reverse(self):
        """For a composite gate, reverse the order of sub-gates.

        This is done by recursively reversing all sub-gates. It does
        not invert any gate.

        Returns:
            Gate: a fresh gate with sub-gates reversed
        """
        if not self._definition:
            return self.copy()

        reverse_inst = self.copy(name=self.name+'_reverse')
        reverse_inst.definition = []
        for inst, qargs, cargs in reversed(self._definition):
            reverse_inst._definition.append((inst.reverse(), qargs, cargs))
        return reverse_inst

    def inverse(self):
        """Invert this gate.

        If the gate is composite (i.e. has a definition), then its definition
        will be recursively inverted.

        Special gates inheriting from Gate can implement their own inverse
        (e.g. T and Tdg)

        Returns:
            Gate: a fresh gate for the inverse

        Raises:
            NotImplementedError: if the gate is not composite and an inverse
                has not been implemented for it.
        """
        if not self.definition:
            raise NotImplementedError("inverse() not implemented for %s." %
                                      self.name)
        inverse_gate = self.copy(name=self.name+'_dg')
        inverse_gate._definition = []
        for inst, qargs, cargs in reversed(self._definition):
            inverse_gate._definition.append((inst.inverse(), qargs, cargs))
        return inverse_gate

    @property
    def definition(self):
        """Return definition in terms of other basic gates."""
        if self._definition is None:
            self._define()
        return self._definition

    @definition.setter
    def definition(self, array):
        """Set matrix representation"""
        self._definition = array

    @property
    def matrix_rep(self):
        """Return matrix representation if it exists else None"""
        return self._matrix_rep

    @matrix_rep.setter
    def matrix_rep(self, matrix):
        """Set matrix representation"""
        self._matrix_rep = matrix

    def _define(self):
        """Populates self.definition with a decomposition of this gate."""
        raise NotImplementedError("No definition for %s (cannot decompose)." %
                                  self.name)
