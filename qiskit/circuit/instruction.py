# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-boolean-expressions
"""
A generic quantum instruction.

Instructions can be implementable on hardware (u, cx, etc.) or in simulation
(snapshot, noise, etc.).

Instructions can be unitary (a.k.a Gate) or non-unitary.

Instructions are identified by the following:

    name: A string to identify the type of instruction.
          Used to request a specific instruction on the backend, or in visualizing circuits.

    num_qubits, num_clbits: dimensions of the instruction

    params: List of parameters to specialize a specific intruction instance.

Instructions do not have any context about where they are in a circuit (which qubits/clbits).
The circuit itself keeps this context.
"""
import copy
from itertools import zip_longest
import sympy
import numpy

from qiskit.qasm.node import node
from qiskit.exceptions import QiskitError
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.parameter import Parameter
from qiskit.qobj.models.qasm import QasmQobjInstruction

_CUTOFF_PRECISION = 1E-10


class Instruction:
    """Generic quantum instruction."""

    def __init__(self, name, num_qubits, num_clbits, params):
        """Create a new instruction.
        Args:
            name (str): instruction name
            num_qubits (int): instruction's qubit width
            num_clbits (int): instructions's clbit width
            params (list[sympy.Basic|qasm.Node|int|float|complex|str|ndarray]): list of parameters
        Raises:
            QiskitError: when the register is not in the correct format.
        """
        if not isinstance(num_qubits, int) or not isinstance(num_clbits, int):
            raise QiskitError("num_qubits and num_clbits must be integer.")
        if num_qubits < 0 or num_clbits < 0:
            raise QiskitError(
                "bad instruction dimensions: %d qubits, %d clbits." %
                num_qubits, num_clbits)
        self.name = name
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits

        self._params = []  # a list of gate params stored

        # tuple (ClassicalRegister, int) when the instruction has a conditional ("if")
        self.control = None
        # list of instructions (and their contexts) that this instruction is composed of
        # empty definition means opaque or fundamental instruction
        self._definition = None
        self.params = params

    def __eq__(self, other):
        """Two instructions are the same if they have the same name,
        same dimensions, and same params.

        Args:
            other (instruction): other instruction

        Returns:
            bool: are self and other equal.
        """
        if type(self) is not type(other) or \
                self.name != other.name or \
                self.num_qubits != other.num_qubits or \
                self.num_clbits != other.num_clbits or \
                self.definition != other.definition:
            return False

        for self_param, other_param in zip_longest(self.params, other.params):
            if self_param == other_param:
                continue

            try:
                if numpy.isclose(float(self_param), float(other_param),
                                 atol=_CUTOFF_PRECISION):
                    continue
            except TypeError:
                pass

            return False

        return True

    def _define(self):
        """Populates self.definition with a decomposition of this gate."""
        pass

    @property
    def params(self):
        """return instruction params"""
        return self._params

    @params.setter
    def params(self, parameters):
        self._params = []
        for single_param in parameters:
            # example: u2(pi/2, sin(pi/4))
            if isinstance(single_param, (Parameter, sympy.Basic)):
                self._params.append(single_param)
            # example: OpenQASM parsed instruction
            elif isinstance(single_param, node.Node):
                self._params.append(single_param.sym())
            # example: u3(0.1, 0.2, 0.3)
            elif isinstance(single_param, (int, float)):
                self._params.append(sympy.Number(single_param))
            # example: Initialize([complex(0,1), complex(0,0)])
            elif isinstance(single_param, complex):
                self._params.append(single_param.real +
                                    single_param.imag * sympy.I)
            # example: snapshot('label')
            elif isinstance(single_param, str):
                self._params.append(sympy.Symbol(single_param))
            # example: numpy.array([[1, 0], [0, 1]])
            elif isinstance(single_param, numpy.ndarray):
                self._params.append(single_param)
            # example: sympy.Matrix([[1, 0], [0, 1]])
            elif isinstance(single_param, sympy.Matrix):
                self._params.append(single_param)
            elif isinstance(single_param, sympy.Expr):
                self._params.append(single_param)
            elif isinstance(single_param, numpy.number):
                self._params.append(sympy.Number(single_param.item()))
            else:
                raise QiskitError("invalid param type {0} in instruction "
                                  "{1}".format(type(single_param), self.name))

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

    def assemble(self):
        """Assemble a QasmQobjInstruction"""
        instruction = QasmQobjInstruction(name=self.name)
        # Evaluate parameters
        if self.params:
            params = [
                x.evalf() if hasattr(x, 'evalf') else x for x in self.params
            ]
            params = [
                sympy.matrix2numpy(x, dtype=complex) if isinstance(
                    x, sympy.Matrix) else x for x in params
            ]
            instruction.params = params
        # Add placeholder for qarg and carg params
        if self.num_qubits:
            instruction.qubits = list(range(self.num_qubits))
        if self.num_clbits:
            instruction.memory = list(range(self.num_clbits))
        # Add control parameters for assembler. This is needed to convert
        # to a qobj conditional instruction at assemble time and after
        # conversion will be deleted by the assembler.
        if self.control:
            instruction._control = self.control
        return instruction

    def mirror(self):
        """For a composite instruction, reverse the order of sub-gates.

        This is done by recursively mirroring all sub-instructions.
        It does not invert any gate.

        Returns:
            Instruction: a fresh gate with sub-gates reversed
        """
        if not self._definition:
            return self.copy()

        reverse_inst = self.copy(name=self.name + '_mirror')
        reverse_inst.definition = []
        for inst, qargs, cargs in reversed(self._definition):
            reverse_inst._definition.append((inst.mirror(), qargs, cargs))
        return reverse_inst

    def inverse(self):
        """Invert this instruction.

        If the instruction is composite (i.e. has a definition),
        then its definition will be recursively inverted.

        Special instructions inheriting from Instruction can
        implement their own inverse (e.g. T and Tdg, Barrier, etc.)

        Returns:
            Instruction: a fresh instruction for the inverse

        Raises:
            QiskitError: if the instruction is not composite
                and an inverse has not been implemented for it.
        """
        if not self.definition:
            raise QiskitError("inverse() not implemented for %s." % self.name)
        inverse_gate = self.copy(name=self.name + '_dg')
        inverse_gate._definition = []
        for inst, qargs, cargs in reversed(self._definition):
            inverse_gate._definition.append((inst.inverse(), qargs, cargs))
        return inverse_gate

    def c_if(self, classical, val):
        """Add classical control on register classical and value val."""
        if not isinstance(classical, ClassicalRegister):
            raise QiskitError("c_if must be used with a classical register")
        if val < 0:
            raise QiskitError("control value should be non-negative")
        self.control = (classical, val)
        return self

    def copy(self, name=None):
        """
        shallow copy of the instruction.

        Args:
          name (str): name to be given to the copied circuit,
            if None then the name stays the same

        Returns:
          Instruction: a shallow copy of the current instruction, with the name
            updated if it was provided
        """
        cpy = copy.copy(self)
        if name:
            cpy.name = name
        return cpy

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
        if self.params:
            name_param = "%s(%s)" % (name_param, ",".join(
                [str(i) for i in self.params]))

        return self._qasmif(name_param)

    def broadcast_arguments(self, qargs, cargs):
        """
        Validation and handling of the arguments and its relationship. For example:
        `cx([q[0],q[1]], q[2])` means `cx(q[0], q[2]); cx(q[1], q[2])`. This method
        yields the arguments in the right grouping. In the example:
           in: [[q[0],q[1]], q[2]],[]
         outs: [q[0], q[2]], []
               [q[1], q[2]], []

        Args:
            qargs (List): List of quantum bit arguments.
            cargs (List): List of classical bit arguments.

        Yields:
            Tuple(List, List): A tuple with single arguments.

        Raises:
            QiskitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """
        if len(qargs) != self.num_qubits:
            raise QiskitError(
                'The amount of qubit arguments does not match the instruction expectation.')

        if len(cargs) != self.num_clbits:
            raise QiskitError(
                'The amount of clbit arguments does not match the instruction expectation.')

        if len(cargs) == len(qargs):
            #  [[q[0], q[1]], [c[0], c[1]]] -> [q[0]], [r[0]]
            #                               -> [q[1]], [r[1]]
            for qarg, carg in zip(qargs, cargs):
                yield [qarg], [carg]
        elif not cargs and len(qargs) == 1:
            #  [[q[0], q[1]], []] -> [q[0]], []]
            #                     -> [q[1]], []
            for qarg in qargs[0]:
                yield [qarg], []
        else:
            #  [[q[0], q[1]], [c[0], c[1]]] -> [q[0], r[0]], [q[1], r[1]]
            flat_qargs = [qarg for sublist in qargs for qarg in sublist]
            flat_cargs = [carg for sublist in cargs for carg in sublist]
            yield flat_qargs, flat_cargs
