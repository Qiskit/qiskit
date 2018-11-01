# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A generic quantum instruction.

Instructions can be implementable on hardware (U, CX, etc.) or in simulation
(snapshot, noise, etc.).

Instructions can be unitary (a.k.a Gate) or non-unitary.

Instructions are identified by the following fields, and are serialized as such in Qobj.

    name: A string to identify the type of instruction.
          Used to request a specific instruction on the backend, or in visualizing circuits.

    param: List of parameters to specialize a specific intruction instance.

    arg: List of pairs (Register, index) that the instruction acts on.
"""
from sympy import Number, Basic

from ._qiskiterror import QISKitError
from ._register import Register


class Instruction(object):
    """Generic quantum instruction."""

    def __init__(self, name, param, arg, circuit=None):
        """Create a new instruction.

        Args:
            name (str): instruction name
            param (list[float]): list of real parameters
            arg (list[(Register, index)]): list of args
            circuit(QuantumCircuit or Instruction): where the instruction is attached

        Raises:
            QISKitError: when the register is not in the correct format.
        """
        for i in arg:
            if not isinstance(i[0], Register) or not isinstance(i[1], int):
                raise QISKitError("argument not (Register, int) tuple")
        self.name = name
        self.param = []
        for single_param in param:
            if not isinstance(single_param, (Basic, complex)):
                # If the item in param is not symbolic and not complex (used
                # by InitializeGate), make it symbolic.
                self.param.append(Number(single_param))
            else:
                self.param.append(single_param)
        self.arg = arg
        self.control = None  # tuple (ClassicalRegister, int) for "if"
        self.circuit = circuit

    def check_circuit(self):
        """Raise exception if self.circuit is None."""
        if self.circuit is None:
            raise QISKitError("Instruction's circuit not assigned")

    def c_if(self, classical, val):
        """Add classical control on register classical and value val."""
        self.check_circuit()
        self.circuit._check_creg(classical)
        if val < 0:
            raise QISKitError("control value should be non-negative")
        self.control = (classical, val)
        return self

    def _modifiers(self, gate):
        """Apply any modifiers of this instruction to another one."""
        if self.control is not None:
            self.check_circuit()
            if not gate.circuit.has_register(self.control[0]):
                raise QISKitError("control register %s not found"
                                  % self.control[0].name)
            gate.c_if(self.control[0], self.control[1])

    def _qasmif(self, string):
        """Print an if statement if needed."""
        if self.control is None:
            return string
        return "if(%s==%d) " % (self.control[0].name, self.control[1]) + string

    def qasm(self):
        """Return a default OpenQASM string for the instruction.

        Derived instructions may override this to print in a
        different format (e.g. measure).
        """
        name_and_param = self.name
        if self.param:
            name_and_param = "%s(%s)" % (name_and_param,
                                         ",".join([str(i)
                                                   for i in self.param]))

        return self._qasmif("%s %s;" % (name_and_param,
                                        ",".join(["%s[%d]" % (j[0].name, j[1])
                                                  for j in self.arg])))
