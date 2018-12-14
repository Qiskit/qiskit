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

    qargs: List of qubits (QuantumRegister, index) that the instruction acts on.

    cargs: List of clbits (ClassicalRegister, index) that the instruction acts on.

"""
import sympy

from qiskit.qasm._node import _node
from qiskit.qiskiterror import QiskitError
from .quantumregister import QuantumRegister
from .classicalregister import ClassicalRegister


class Instruction(object):
    """Generic quantum instruction."""

    def __init__(self, name, param, qargs, cargs, circuit=None):
        """Create a new instruction.
        Args:
            name (str): instruction name
            param (list[sympy.Basic|qasm.Node|int|float|complex|str]): list of parameters
            qargs (list[(QuantumRegister, index)]): list of quantum args
            cargs (list[(ClassicalRegister, index)]): list of classical args
            circuit (QuantumCircuit or Instruction): where the instruction is attached
        Raises:
            QiskitError: when the register is not in the correct format.
        """
        if not all((type(i[0]), type(i[1])) == (QuantumRegister, int) for i in qargs):
            raise QiskitError("qarg not (QuantumRegister, int) tuple")
        if not all((type(i[0]), type(i[1])) == (ClassicalRegister, int) for i in cargs):
            raise QiskitError("carg not (ClassicalRegister, int) tuple")
        self.name = name
        self.param = []  # a list of gate params stored as sympy objects
        for single_param in param:
            # example: u2(pi/2, sin(pi/4))
            if isinstance(single_param, sympy.Basic):
                self.param.append(single_param)
            # example: OpenQASM parsed instruction
            elif isinstance(single_param, _node.Node):
                self.param.append(single_param.sym())
            # example: u3(0.1, 0.2, 0.3)
            elif isinstance(single_param, (int, float)):
                self.param.append(sympy.Number(single_param))
            # example: Initialize([complex(0,1), complex(0,0)])
            elif isinstance(single_param, complex):
                self.param.append(single_param.real + single_param.imag * sympy.I)
            # example: snapshot('label')
            elif isinstance(single_param, str):
                self.param.append(sympy.Symbol(single_param))
            else:
                raise QiskitError("invalid param type {0} in instruction "
                                  "{1}".format(type(single_param), name))
        self.qargs = qargs
        self.cargs = cargs
        self.control = None  # tuple (ClassicalRegister, int) for "if"
        self.circuit = circuit

    def __eq__(self, other):
        """Two instructions are the same if they have the same name and same
        params.

        Args:
            other (instruction): other instruction

        Returns:
            bool: are self and other equal.
        """
        res = False
        if type(self) is type(other) and \
                self.name == other.name and \
                self.param == other.param:
            res = True
        return res

    def check_circuit(self):
        """Raise exception if self.circuit is None."""
        if self.circuit is None:
            raise QiskitError("Instruction's circuit not assigned")

    def c_if(self, classical, val):
        """Add classical control on register classical and value val."""
        self.check_circuit()
        self.circuit._check_creg(classical)
        if val < 0:
            raise QiskitError("control value should be non-negative")
        self.control = (classical, val)
        return self

    def _modifiers(self, gate):
        """Apply any modifiers of this instruction to another one."""
        if self.control is not None:
            self.check_circuit()
            if not gate.circuit.has_register(self.control[0]):
                raise QiskitError("control register %s not found"
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
        different format (e.g. measure q[0] -> c[0];).
        """
        name_param = self.name
        if self.param:
            name_param = "%s(%s)" % (name_param,
                                     ",".join([str(i) for i in self.param]))

        name_param_arg = "%s %s;" % (name_param,
                                     ",".join(["%s[%d]" % (j[0].name, j[1])
                                               for j in self.qargs + self.cargs]))
        return self._qasmif(name_param_arg)
