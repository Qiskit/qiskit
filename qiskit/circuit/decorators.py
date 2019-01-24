# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
This module contains decorators for expanding register objects or
list of qubits into a series of single qubit/cbit instructions to be handled by
the wrapped operation.
"""

import functools
from qiskit.exceptions import QiskitError
from .instructionset import InstructionSet
from .quantumregister import QuantumRegister
from .register import Register


def _1q_gate(func):
    """Wrapper for one qubit gate"""
    @functools.wraps(func)
    def wrapper(self, *args):
        """Wrapper for one qubit gate"""
        params = args[0:-1] if len(args) > 1 else tuple()
        q = args[-1]
        if isinstance(q, QuantumRegister):
            q = [(q, j) for j in range(len(q))]

        if q and isinstance(q, list):
            instructions = InstructionSet()
            for qubit in q:
                self._check_qubit(qubit)
                instructions.add(func(self, *params, qubit))
            return instructions
        return func(self, *params, q)
    return wrapper


def _2q_gate(func=None, broadcastable=None):
    if func is None:
        return functools.partial(_2q_gate, broadcastable=broadcastable)

    @functools.wraps(func)
    def wrapper(self, *args):
        params = args[0:-2] if len(args) > 2 else tuple()
        qargs = args[-2:]
        if broadcastable is None:
            blist = [True] * len(qargs)
        else:
            blist = broadcastable
        if not all([isinstance(arg, tuple) for arg in qargs]):
            if any([not isinstance(arg, (tuple, list, Register))
                    for arg in qargs]):
                raise QiskitError('operation arguments must be qubits/cbits')
            broadcast_size = max(len(arg) for arg in qargs)
            expanded_qargs = []
            for arg, broadcast in zip(qargs, blist):
                if isinstance(arg, Register):
                    arg = [(arg, i) for i in range(len(arg))]
                elif isinstance(arg, tuple):
                    arg = [arg]
                # now we should have a list of qubits
                if isinstance(arg, list) and len(arg) == 1 and broadcast:
                    arg = arg * broadcast_size
                if len(arg) != broadcast_size:
                    raise QiskitError('register size error')
                expanded_qargs.append(arg)
            qargs = expanded_qargs
            if all([isinstance(arg, list) for arg in qargs]):
                if all(qargs):
                    instructions = InstructionSet()
                    for iqargs in zip(*qargs):
                        instructions.add(func(self, *params, *iqargs))
                    return instructions
                else:
                    raise QiskitError('empty control or target argument')
        return func(self, *params, *qargs)
    return wrapper


def _3q_gate(func=None, broadcastable=None):
    if func is None:
        return functools.partial(_3q_gate, broadcastable=broadcastable)

    @functools.wraps(func)
    def wrapper(self, *args):
        params = args[0:-3] if len(args) > 3 else tuple()
        qargs = args[-3:]
        if broadcastable is None:
            blist = [True] * len(qargs)
        else:
            blist = broadcastable
        if not all([isinstance(arg, tuple) for arg in qargs]):
            if any([not isinstance(arg, (tuple, list, Register))
                    for arg in qargs]):
                raise QiskitError('operation arguments must be qubits/cbits')
            broadcast_size = max(len(arg) for arg in qargs)
            expanded_qargs = []
            for arg, broadcast in zip(qargs, blist):
                if isinstance(arg, Register):
                    arg = [(arg, i) for i in range(len(arg))]
                elif isinstance(arg, tuple):
                    arg = [arg]
                # now we should have a list of qubits
                if isinstance(arg, list) and len(arg) == 1 and broadcast:
                    arg = arg * broadcast_size
                if len(arg) != broadcast_size:
                    raise QiskitError('register sizes should match or be one')
                expanded_qargs.append(arg)
            qargs = expanded_qargs
            if all([isinstance(arg, list) for arg in qargs]):
                if all(qargs):
                    instructions = InstructionSet()
                    for iqargs in zip(*qargs):
                        instructions.add(func(self, *params, *iqargs))
                    return instructions
                else:
                    raise QiskitError('empty control or target argument')
        return func(self, *params, *qargs)
    return wrapper
