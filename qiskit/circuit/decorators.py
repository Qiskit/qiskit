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


def _is_bit(obj):
    """Determine if obj is a bit"""
    # If there is a bit type this could be replaced by isinstance.
    if isinstance(obj, tuple) and len(obj) == 2:
        if isinstance(obj[0], Register) and isinstance(obj[1], int) and obj[1] < len(obj[0]):
            return True
    return False


def _op_expand(n_bits, func=None, broadcastable=None):
    """Decorator for expanding an operation across a whole register or register subset."""
    if func is None:
        return functools.partial(_op_expand, n_bits, broadcastable=broadcastable)

    @functools.wraps(func)
    def wrapper(self, *args):
        params = args[0:-n_bits] if len(args) > n_bits else tuple()
        rargs = args[-n_bits:]
        if broadcastable is None:
            blist = [True] * len(rargs)
        else:
            blist = broadcastable
        if not all([_is_bit(arg) for arg in rargs]):
            rarg_size = [1] * n_bits
            for iarg, arg in enumerate(rargs):
                if isinstance(arg, Register):
                    rarg_size[iarg] = len(arg)
                elif isinstance(arg, list) and all([_is_bit(bit) for bit in arg]):
                    rarg_size[iarg] = len(arg)
                elif isinstance(arg, tuple) and _is_bit(arg):
                    rarg_size[iarg] = 1
                else:
                    raise QiskitError('operation arguments must be qubits/cbits')
            broadcast_size = max(rarg_size)
            expanded_rargs = []
            for arg, broadcast in zip(rargs, blist):
                if isinstance(arg, Register):
                    arg = [(arg, i) for i in range(len(arg))]
                elif isinstance(arg, tuple):
                    arg = [arg]
                # now we should have a list of qubits
                if isinstance(arg, list) and len(arg) == 1 and broadcast:
                    arg = arg * broadcast_size
                if len(arg) != broadcast_size:
                    raise QiskitError('register size error')
                expanded_rargs.append(arg)
            rargs = expanded_rargs
            if all([isinstance(arg, list) for arg in rargs]):
                if all(rargs):
                    instructions = InstructionSet()
                    for irargs in zip(*rargs):
                        instructions.add(func(self, *params, *irargs))
                    return instructions
                else:
                    raise QiskitError('empty control or target argument')
        return func(self, *params, *rargs)
    return wrapper
